#include "cid_engine/ops.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cfloat>
#include <climits>
#include <cstdint>
#include <tuple>

namespace cid::engine {
namespace {

constexpr int kThreads = 256;

__global__ void masked_diffusion_corrupt_kernel(
    const std::int64_t* clean_ids,
    const float* ratio_random,
    const float* mask_random,
    std::int64_t* corrupted,
    bool* masked,
    float* mask_ratio,
    float* metrics,
    const std::int64_t batch,
    const std::int64_t tokens,
    const std::int64_t mask_token_id,
    const float min_mask_ratio,
    const float max_mask_ratio) {
  const auto row = static_cast<std::int64_t>(blockIdx.x);
  if (row >= batch) {
    return;
  }

  const int lane = threadIdx.x;
  const float ratio =
      min_mask_ratio + (max_mask_ratio - min_mask_ratio) * ratio_random[row];
  const auto base = row * tokens;

  int local_count = 0;
  float local_min = FLT_MAX;
  int local_min_index = INT_MAX;
  for (std::int64_t column = lane; column < tokens; column += blockDim.x) {
    const auto index = base + column;
    const float draw = mask_random[index];
    const bool selected = draw < ratio;
    masked[index] = selected;
    corrupted[index] = selected ? mask_token_id : clean_ids[index];
    local_count += selected ? 1 : 0;
    if (draw < local_min ||
        (draw == local_min && column < local_min_index)) {
      local_min = draw;
      local_min_index = static_cast<int>(column);
    }
  }

  __shared__ int counts[kThreads];
  __shared__ float minima[kThreads];
  __shared__ int minimum_indices[kThreads];
  counts[lane] = local_count;
  minima[lane] = local_min;
  minimum_indices[lane] = local_min_index;
  __syncthreads();

  for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
    if (lane < offset) {
      counts[lane] += counts[lane + offset];
      const float candidate_min = minima[lane + offset];
      const int candidate_index = minimum_indices[lane + offset];
      if (candidate_min < minima[lane] ||
          (candidate_min == minima[lane] &&
           candidate_index < minimum_indices[lane])) {
        minima[lane] = candidate_min;
        minimum_indices[lane] = candidate_index;
      }
    }
    __syncthreads();
  }

  if (lane == 0) {
    int selected_count = counts[0];
    if (selected_count == 0) {
      const auto index = base + minimum_indices[0];
      masked[index] = true;
      corrupted[index] = mask_token_id;
      selected_count = 1;
    }
    mask_ratio[row] = ratio;
    atomicAdd(
        metrics,
        static_cast<float>(selected_count) /
            static_cast<float>(batch * tokens));
    atomicAdd(metrics + 1, ratio / static_cast<float>(batch));
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
masked_diffusion_corrupt_from_random_cuda(
    const at::Tensor& clean_ids_input,
    const at::Tensor& ratio_random_input,
    const at::Tensor& mask_random_input,
    const std::int64_t mask_token_id,
    const double min_mask_ratio,
    const double max_mask_ratio) {
  TORCH_CHECK(clean_ids_input.is_cuda(), "clean_ids must be CUDA");
  TORCH_CHECK(ratio_random_input.is_cuda(), "ratio_random must be CUDA");
  TORCH_CHECK(mask_random_input.is_cuda(), "mask_random must be CUDA");
  TORCH_CHECK(
      clean_ids_input.device() == ratio_random_input.device() &&
          clean_ids_input.device() == mask_random_input.device(),
      "masked diffusion inputs must share a CUDA device");
  TORCH_CHECK(clean_ids_input.dim() == 2, "clean_ids must have shape [batch, tokens]");
  TORCH_CHECK(clean_ids_input.scalar_type() == at::kLong, "clean_ids must use torch.int64");

  const auto batch = clean_ids_input.size(0);
  const auto tokens = clean_ids_input.size(1);
  TORCH_CHECK(batch > 0 && tokens > 0, "clean_ids dimensions must be non-empty");
  TORCH_CHECK(
      ratio_random_input.sizes() == at::IntArrayRef({batch, 1}),
      "ratio_random must have shape [batch, 1]");
  TORCH_CHECK(
      mask_random_input.sizes() == clean_ids_input.sizes(),
      "mask_random must match clean_ids shape");
  TORCH_CHECK(
      ratio_random_input.scalar_type() == at::kFloat &&
          mask_random_input.scalar_type() == at::kFloat,
      "CUDA random tensors must use torch.float32");
  TORCH_CHECK(
      min_mask_ratio > 0.0 &&
          min_mask_ratio <= max_mask_ratio &&
          max_mask_ratio <= 1.0,
      "mask ratio range must satisfy 0 < min <= max <= 1");

  const c10::cuda::CUDAGuard device_guard(clean_ids_input.device());
  auto clean_ids = clean_ids_input.contiguous();
  auto ratio_random = ratio_random_input.contiguous();
  auto mask_random = mask_random_input.contiguous();

  auto corrupted = at::empty_like(clean_ids);
  auto masked = at::empty(
      clean_ids.sizes(),
      clean_ids.options().dtype(at::kBool));
  auto mask_ratio = at::empty(
      {batch, 1},
      clean_ids.options().dtype(at::kFloat));
  auto metrics = at::zeros(
      {2},
      clean_ids.options().dtype(at::kFloat));

  const auto stream = at::cuda::getCurrentCUDAStream(clean_ids.get_device());
  masked_diffusion_corrupt_kernel<<<batch, kThreads, 0, stream>>>(
      clean_ids.const_data_ptr<std::int64_t>(),
      ratio_random.const_data_ptr<float>(),
      mask_random.const_data_ptr<float>(),
      corrupted.mutable_data_ptr<std::int64_t>(),
      masked.mutable_data_ptr<bool>(),
      mask_ratio.mutable_data_ptr<float>(),
      metrics.mutable_data_ptr<float>(),
      batch,
      tokens,
      mask_token_id,
      static_cast<float>(min_mask_ratio),
      static_cast<float>(max_mask_ratio));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {corrupted, masked, mask_ratio, metrics};
}

}  // namespace cid::engine

TORCH_LIBRARY_IMPL(cid_engine, CUDA, m) {
  m.impl(
      "masked_diffusion_corrupt_from_random",
      TORCH_FN(cid::engine::masked_diffusion_corrupt_from_random_cuda));
}
