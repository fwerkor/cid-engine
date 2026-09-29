#include "cid_engine/ops.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <tuple>

namespace cid::engine {
namespace {

constexpr int kThreads = 1024;

__device__ bool better_max(
    const float candidate_value,
    const std::int64_t candidate_index,
    const float current_value,
    const std::int64_t current_index) {
  const bool candidate_nan = isnan(candidate_value);
  const bool current_nan = isnan(current_value);
  if (candidate_nan != current_nan) {
    return candidate_nan;
  }
  if (candidate_nan) {
    return candidate_index < current_index;
  }
  return candidate_value > current_value ||
      (candidate_value == current_value && candidate_index < current_index);
}

template <typename scalar_t, bool ApplyRepetition>
__global__ void display_token_statistics_kernel(
    const scalar_t* __restrict__ logits,
    const std::int64_t* __restrict__ token_ids,
    float* __restrict__ confidence,
    std::int64_t* __restrict__ predicted,
    float* __restrict__ current_confidence,
    const std::int64_t rows,
    const std::int64_t tokens_per_row,
    const std::int64_t vocab_size,
    const std::int64_t mask_token_id,
    const std::int64_t eos_token_id,
    const float log_multiplier) {
  const std::int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  std::int64_t left_id = -1;
  std::int64_t right_id = -1;
  int left_count = 0;
  int right_count = 0;
  if constexpr (ApplyRepetition) {
    const std::int64_t position = row % tokens_per_row;
    const std::int64_t sequence_start = row - position;
    const std::int64_t sequence_end = sequence_start + tokens_per_row;
    if (position > 0) {
      left_id = token_ids[row - 1];
      if (left_id != mask_token_id && left_id != eos_token_id) {
        left_count = 1;
        std::int64_t cursor = row - 2;
        while (cursor >= sequence_start && token_ids[cursor] == left_id) {
          ++left_count;
          --cursor;
        }
      } else {
        left_id = -1;
      }
    }
    if (position + 1 < tokens_per_row) {
      right_id = token_ids[row + 1];
      if (right_id != mask_token_id && right_id != eos_token_id) {
        right_count = 1;
        std::int64_t cursor = row + 2;
        while (cursor < sequence_end && token_ids[cursor] == right_id) {
          ++right_count;
          ++cursor;
        }
      } else {
        right_id = -1;
      }
    }
  }

  const auto adjusted_value = [&](const std::int64_t column, const float value) {
    if constexpr (!ApplyRepetition) {
      return value;
    }
    int exponent = 0;
    if (column == left_id) {
      exponent += left_count;
    }
    if (column == right_id) {
      exponent += right_count;
    }
    return value + static_cast<float>(exponent) * log_multiplier;
  };

  const int tid = threadIdx.x;
  const scalar_t* row_logits = logits + row * vocab_size;
  float local_max = -INFINITY;
  std::int64_t local_index = vocab_size;
  for (std::int64_t column = tid; column < vocab_size; column += blockDim.x) {
    const float value =
        adjusted_value(column, static_cast<float>(row_logits[column]));
    if (better_max(value, column, local_max, local_index)) {
      local_max = value;
      local_index = column;
    }
  }

  __shared__ float max_values[kThreads];
  __shared__ std::int64_t max_indices[kThreads];
  max_values[tid] = local_max;
  max_indices[tid] = local_index;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (tid < offset) {
      const float candidate_value = max_values[tid + offset];
      const std::int64_t candidate_index = max_indices[tid + offset];
      if (better_max(
              candidate_value,
              candidate_index,
              max_values[tid],
              max_indices[tid])) {
        max_values[tid] = candidate_value;
        max_indices[tid] = candidate_index;
      }
    }
    __syncthreads();
  }

  const float row_max = max_values[0];
  float local_sum = 0.0F;
  for (std::int64_t column = tid; column < vocab_size; column += blockDim.x) {
    const float value =
        adjusted_value(column, static_cast<float>(row_logits[column]));
    local_sum += expf(value - row_max);
  }

  __shared__ float sums[kThreads];
  sums[tid] = local_sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (tid < offset) {
      sums[tid] += sums[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float denominator = sums[0];
    const std::int64_t current_id = token_ids[row];
    const float current_logit = adjusted_value(
        current_id, static_cast<float>(row_logits[current_id]));
    confidence[row] = 1.0F / denominator;
    predicted[row] = max_indices[0];
    current_confidence[row] = expf(current_logit - row_max) / denominator;
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> display_token_statistics_cuda(
    const at::Tensor& token_ids,
    const at::Tensor& logits) {
  TORCH_CHECK(token_ids.is_cuda(), "token_ids must be CUDA tensors");
  TORCH_CHECK(logits.is_cuda(), "logits must be CUDA tensors");
  TORCH_CHECK(token_ids.device() == logits.device(), "token_ids and logits must share a device");
  TORCH_CHECK(token_ids.dim() == 2, "token_ids must have shape [batch, tokens]");
  TORCH_CHECK(token_ids.scalar_type() == at::kLong, "token_ids must use torch.int64");
  TORCH_CHECK(
      logits.dim() == 3 &&
          logits.size(0) == token_ids.size(0) &&
          logits.size(1) == token_ids.size(1),
      "logits must have shape [batch, tokens, vocab]");
  TORCH_CHECK(logits.size(-1) > 0, "logits vocabulary dimension must be non-empty");

  auto contiguous_logits = logits.contiguous();
  auto contiguous_ids = token_ids.contiguous();
  const std::int64_t rows = token_ids.numel();
  const std::int64_t vocab_size = logits.size(-1);

  auto confidence = at::empty(token_ids.sizes(), token_ids.options().dtype(at::kFloat));
  auto predicted = at::empty(token_ids.sizes(), token_ids.options().dtype(at::kLong));
  auto current_confidence =
      at::empty(token_ids.sizes(), token_ids.options().dtype(at::kFloat));

  if (rows == 0) {
    return {confidence, predicted, current_confidence};
  }

  const auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      contiguous_logits.scalar_type(),
      "cid_display_token_statistics_cuda",
      [&] {
        display_token_statistics_kernel<scalar_t, false>
            <<<rows, kThreads, 0, stream>>>(
                contiguous_logits.const_data_ptr<scalar_t>(),
                contiguous_ids.const_data_ptr<std::int64_t>(),
                confidence.mutable_data_ptr<float>(),
                predicted.mutable_data_ptr<std::int64_t>(),
                current_confidence.mutable_data_ptr<float>(),
                rows,
                token_ids.size(1),
                vocab_size,
                -1,
                -1,
                0.0F);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {confidence, predicted, current_confidence};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
display_token_statistics_with_repetition_cuda(
    const at::Tensor& token_ids,
    const at::Tensor& logits,
    const std::int64_t mask_token_id,
    const std::optional<std::int64_t>& eos_token_id,
    const double repetition_score_multiplier) {
  TORCH_CHECK(
      repetition_score_multiplier > 0.0 && repetition_score_multiplier <= 1.0,
      "repetition_score_multiplier must be in (0, 1]");
  TORCH_CHECK(token_ids.is_cuda(), "token_ids must be CUDA tensors");
  TORCH_CHECK(logits.is_cuda(), "logits must be CUDA tensors");
  TORCH_CHECK(token_ids.device() == logits.device(), "token_ids and logits must share a device");
  TORCH_CHECK(token_ids.dim() == 2, "token_ids must have shape [batch, tokens]");
  TORCH_CHECK(token_ids.scalar_type() == at::kLong, "token_ids must use torch.int64");
  TORCH_CHECK(
      logits.dim() == 3 &&
          logits.size(0) == token_ids.size(0) &&
          logits.size(1) == token_ids.size(1),
      "logits must have shape [batch, tokens, vocab]");
  TORCH_CHECK(logits.size(-1) > 0, "logits vocabulary dimension must be non-empty");

  auto contiguous_logits = logits.contiguous();
  auto contiguous_ids = token_ids.contiguous();
  const std::int64_t rows = token_ids.numel();
  const std::int64_t tokens_per_row = token_ids.size(1);
  const std::int64_t vocab_size = logits.size(-1);
  auto confidence = at::empty(token_ids.sizes(), token_ids.options().dtype(at::kFloat));
  auto predicted = at::empty(token_ids.sizes(), token_ids.options().dtype(at::kLong));
  auto current_confidence =
      at::empty(token_ids.sizes(), token_ids.options().dtype(at::kFloat));
  if (rows == 0) {
    return {confidence, predicted, current_confidence};
  }

  const float log_multiplier =
      static_cast<float>(std::log(repetition_score_multiplier));
  const std::int64_t eos_id = eos_token_id.value_or(-1);
  const auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      contiguous_logits.scalar_type(),
      "cid_display_token_statistics_with_repetition_cuda",
      [&] {
        display_token_statistics_kernel<scalar_t, true>
            <<<rows, kThreads, 0, stream>>>(
                contiguous_logits.const_data_ptr<scalar_t>(),
                contiguous_ids.const_data_ptr<std::int64_t>(),
                confidence.mutable_data_ptr<float>(),
                predicted.mutable_data_ptr<std::int64_t>(),
                current_confidence.mutable_data_ptr<float>(),
                rows,
                tokens_per_row,
                vocab_size,
                mask_token_id,
                eos_id,
                log_multiplier);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {confidence, predicted, current_confidence};
}

}  // namespace cid::engine

TORCH_LIBRARY_IMPL(cid_engine, CUDA, m) {
  m.impl(
      "display_token_statistics",
      TORCH_FN(cid::engine::display_token_statistics_cuda));
  m.impl(
      "display_token_statistics_with_repetition",
      TORCH_FN(cid::engine::display_token_statistics_with_repetition_cuda));
}
