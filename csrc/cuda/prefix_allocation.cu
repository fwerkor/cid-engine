#include "cid_engine/ops.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cstdint>

namespace cid::engine {
namespace {

template <typename occupancy_t, typename logits_t>
__global__ void prefix_allocation_kernel(
    const occupancy_t* __restrict__ occupancy,
    const logits_t* __restrict__ logits,
    bool* __restrict__ output,
    const std::int64_t slots,
    const float threshold,
    const std::int64_t max_allocations) {
  if (threadIdx.x != 0) {
    return;
  }

  const std::int64_t batch = blockIdx.x;
  const std::int64_t offset = batch * slots;
  for (std::int64_t slot = 0; slot < slots; ++slot) {
    output[offset + slot] = false;
  }

  std::int64_t allocations = 0;
  for (std::int64_t slot = 0; slot < slots; ++slot) {
    const auto index = offset + slot;
    if (static_cast<bool>(occupancy[index])) {
      continue;
    }

    const float value = static_cast<float>(logits[index]);
    const float probability = 1.0F / (1.0F + expf(-value));
    if (probability < threshold) {
      break;
    }
    if (allocations < max_allocations) {
      output[index] = true;
      ++allocations;
    }
  }
}

template <typename occupancy_t>
void launch_prefix_allocation(
    const at::Tensor& occupancy,
    const at::Tensor& logits,
    at::Tensor& output,
    const float threshold,
    const std::int64_t max_allocations) {
  const auto batch_size = occupancy.size(0);
  const auto slots = occupancy.size(1);
  const auto stream = at::cuda::getCurrentCUDAStream(occupancy.get_device());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      logits.scalar_type(),
      "cid_prefix_allocation_cuda_logits",
      [&] {
        prefix_allocation_kernel<occupancy_t, scalar_t>
            <<<batch_size, 1, 0, stream>>>(
                occupancy.const_data_ptr<occupancy_t>(),
                logits.const_data_ptr<scalar_t>(),
                output.mutable_data_ptr<bool>(),
                slots,
                threshold,
                max_allocations);
      });
}

}  // namespace

at::Tensor prefix_allocation_mask_cuda(
    const at::Tensor& occupancy_input,
    const at::Tensor& allocation_logits,
    const double threshold,
    const std::int64_t max_allocations) {
  TORCH_CHECK(occupancy_input.is_cuda(), "occupancy must be a CUDA tensor");
  TORCH_CHECK(allocation_logits.is_cuda(), "allocation_logits must be a CUDA tensor");
  TORCH_CHECK(
      occupancy_input.device() == allocation_logits.device(),
      "occupancy and allocation_logits must share a device");
  TORCH_CHECK(
      occupancy_input.dim() == 2 ||
          (occupancy_input.dim() == 3 && occupancy_input.size(-1) == 1),
      "occupancy must have shape [batch, slots] or [batch, slots, 1]");
  TORCH_CHECK(threshold >= 0.0 && threshold <= 1.0, "threshold must be in [0, 1]");
  TORCH_CHECK(max_allocations > 0, "max_allocations must be positive");

  auto occupancy =
      occupancy_input.dim() == 3 ? occupancy_input.squeeze(-1) : occupancy_input;
  TORCH_CHECK(
      allocation_logits.dim() == 2 &&
          allocation_logits.sizes() == occupancy.sizes(),
      "occupancy and allocation_logits must have matching shapes");

  occupancy = occupancy.contiguous();
  auto logits = allocation_logits.contiguous();
  auto output = at::empty(
      occupancy.sizes(),
      occupancy.options().dtype(at::kBool));

  if (occupancy.numel() == 0) {
    return output;
  }

  AT_DISPATCH_ALL_TYPES_AND3(
      at::kHalf,
      at::kBFloat16,
      at::kBool,
      occupancy.scalar_type(),
      "cid_prefix_allocation_cuda_occupancy",
      [&] {
        launch_prefix_allocation<scalar_t>(
            occupancy,
            logits,
            output,
            static_cast<float>(threshold),
            max_allocations);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace cid::engine

TORCH_LIBRARY_IMPL(cid_engine, CUDA, m) {
  m.impl(
      "prefix_allocation_mask",
      TORCH_FN(cid::engine::prefix_allocation_mask_cuda));
}
