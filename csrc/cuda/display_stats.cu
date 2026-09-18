#include "cid_engine/ops.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <tuple>

namespace cid::engine {
namespace {

constexpr int kThreads = 256;

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

template <typename scalar_t>
__global__ void display_token_statistics_kernel(
    const scalar_t* __restrict__ logits,
    const std::int64_t* __restrict__ token_ids,
    float* __restrict__ confidence,
    std::int64_t* __restrict__ predicted,
    float* __restrict__ current_confidence,
    const std::int64_t rows,
    const std::int64_t vocab_size) {
  const std::int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const int tid = threadIdx.x;
  const scalar_t* row_logits = logits + row * vocab_size;

  float local_max = -INFINITY;
  std::int64_t local_index = vocab_size;
  for (std::int64_t column = tid; column < vocab_size; column += blockDim.x) {
    const float value = static_cast<float>(row_logits[column]);
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
    const float value = static_cast<float>(row_logits[column]);
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
    const float current_logit = static_cast<float>(row_logits[current_id]);
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
        display_token_statistics_kernel<scalar_t>
            <<<rows, kThreads, 0, stream>>>(
                contiguous_logits.const_data_ptr<scalar_t>(),
                contiguous_ids.const_data_ptr<std::int64_t>(),
                confidence.mutable_data_ptr<float>(),
                predicted.mutable_data_ptr<std::int64_t>(),
                current_confidence.mutable_data_ptr<float>(),
                rows,
                vocab_size);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {confidence, predicted, current_confidence};
}

}  // namespace cid::engine

TORCH_LIBRARY_IMPL(cid_engine, CUDA, m) {
  m.impl(
      "display_token_statistics",
      TORCH_FN(cid::engine::display_token_statistics_cuda));
}
