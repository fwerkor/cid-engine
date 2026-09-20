#include "cid_engine/ops.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <tuple>

namespace cid::engine {
namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void rollout_slot_transition_kernel(
    const bool* __restrict__ occupancy,
    const scalar_t* __restrict__ allocation_logits,
    const scalar_t* __restrict__ lifecycle_logits,
    const scalar_t* __restrict__ revision_logits,
    const scalar_t* __restrict__ input_lifecycle,
    bool* __restrict__ next_occupancy,
    std::int64_t* __restrict__ lifecycle_indices,
    std::int64_t* __restrict__ revision_indices,
    std::int64_t* __restrict__ input_lifecycle_indices,
    bool* __restrict__ input_lifecycle_present,
    bool* __restrict__ live_slots,
    const std::int64_t batch,
    const std::int64_t slots,
    const std::int64_t lifecycles,
    const std::int64_t revisions,
    const float threshold,
    const std::int64_t max_allocations,
    const std::int64_t retired_index) {
  const auto row = static_cast<std::int64_t>(blockIdx.x);
  if (row >= batch) {
    return;
  }
  const auto base = row * slots;

  if (threadIdx.x == 0) {
    std::int64_t allocations = 0;
    bool blocked = false;
    for (std::int64_t slot = 0; slot < slots; ++slot) {
      const auto index = base + slot;
      const bool occupied = occupancy[index];
      bool selected = false;
      if (!occupied && !blocked) {
        const float value = static_cast<float>(allocation_logits[index]);
        const float probability = 1.0F / (1.0F + expf(-value));
        if (probability < threshold) {
          blocked = true;
        } else if (allocations < max_allocations) {
          selected = true;
          ++allocations;
        }
      }
      next_occupancy[index] = occupied || selected;
    }
  }
  __syncthreads();

  for (std::int64_t slot = threadIdx.x; slot < slots; slot += blockDim.x) {
    const auto index = base + slot;

    const auto lifecycle_base = index * lifecycles;
    float best_lifecycle =
        static_cast<float>(lifecycle_logits[lifecycle_base]);
    std::int64_t lifecycle_index = 0;
    for (std::int64_t candidate = 1; candidate < lifecycles; ++candidate) {
      const float value =
          static_cast<float>(lifecycle_logits[lifecycle_base + candidate]);
      if (value > best_lifecycle) {
        best_lifecycle = value;
        lifecycle_index = candidate;
      }
    }
    lifecycle_indices[index] = lifecycle_index;

    const auto revision_base = index * revisions;
    float best_revision =
        static_cast<float>(revision_logits[revision_base]);
    std::int64_t revision_index = 0;
    for (std::int64_t candidate = 1; candidate < revisions; ++candidate) {
      const float value =
          static_cast<float>(revision_logits[revision_base + candidate]);
      if (value > best_revision) {
        best_revision = value;
        revision_index = candidate;
      }
    }
    revision_indices[index] = revision_index;

    float best_input =
        static_cast<float>(input_lifecycle[lifecycle_base]);
    std::int64_t input_index = 0;
    bool input_present = best_input != 0.0F;
    for (std::int64_t candidate = 1; candidate < lifecycles; ++candidate) {
      const float value =
          static_cast<float>(input_lifecycle[lifecycle_base + candidate]);
      input_present = input_present || value != 0.0F;
      if (value > best_input) {
        best_input = value;
        input_index = candidate;
      }
    }
    input_lifecycle_indices[index] = input_index;
    input_lifecycle_present[index] = input_present;

    const bool occupied_before = occupancy[index];
    const bool occupied_now = next_occupancy[index];
    const bool newly_allocated = occupied_now && !occupied_before;
    const bool previous_retired =
        occupied_before && input_index == retired_index;
    const bool predicted_retired = lifecycle_index == retired_index;
    live_slots[index] =
        occupied_now &&
        !previous_retired &&
        (!predicted_retired || newly_allocated);
  }
}

}  // namespace

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
rollout_slot_transition_cuda(
    const at::Tensor& occupancy_input,
    const at::Tensor& allocation_logits_input,
    const at::Tensor& lifecycle_logits_input,
    const at::Tensor& revision_logits_input,
    const at::Tensor& input_lifecycle_input,
    const double threshold,
    const std::int64_t max_allocations,
    const std::int64_t retired_index) {
  TORCH_CHECK(occupancy_input.is_cuda(), "occupancy must be CUDA");
  TORCH_CHECK(allocation_logits_input.is_cuda(), "allocation_logits must be CUDA");
  TORCH_CHECK(lifecycle_logits_input.is_cuda(), "lifecycle_logits must be CUDA");
  TORCH_CHECK(revision_logits_input.is_cuda(), "revision_logits must be CUDA");
  TORCH_CHECK(input_lifecycle_input.is_cuda(), "input_lifecycle must be CUDA");

  const auto device = occupancy_input.device();
  TORCH_CHECK(
      allocation_logits_input.device() == device &&
          lifecycle_logits_input.device() == device &&
          revision_logits_input.device() == device &&
          input_lifecycle_input.device() == device,
      "rollout slot-state inputs must share a CUDA device");
  TORCH_CHECK(
      occupancy_input.dim() == 2 ||
          (occupancy_input.dim() == 3 && occupancy_input.size(-1) == 1),
      "occupancy must have shape [batch, slots] or [batch, slots, 1]");

  const c10::cuda::CUDAGuard device_guard(device);
  auto occupancy =
      occupancy_input.dim() == 3 ? occupancy_input.squeeze(-1) : occupancy_input;
  const auto batch = occupancy.size(0);
  const auto slots = occupancy.size(1);
  TORCH_CHECK(
      allocation_logits_input.sizes() == occupancy.sizes(),
      "allocation_logits must match occupancy shape");
  TORCH_CHECK(
      lifecycle_logits_input.dim() == 3 &&
          lifecycle_logits_input.size(0) == batch &&
          lifecycle_logits_input.size(1) == slots,
      "lifecycle_logits must have shape [batch, slots, lifecycles]");
  TORCH_CHECK(
      revision_logits_input.dim() == 3 &&
          revision_logits_input.size(0) == batch &&
          revision_logits_input.size(1) == slots,
      "revision_logits must have shape [batch, slots, revisions]");
  TORCH_CHECK(
      input_lifecycle_input.dim() == 3 &&
          input_lifecycle_input.size(0) == batch &&
          input_lifecycle_input.size(1) == slots,
      "input_lifecycle must have shape [batch, slots, lifecycles]");
  const auto lifecycles = lifecycle_logits_input.size(-1);
  const auto revisions = revision_logits_input.size(-1);
  TORCH_CHECK(lifecycles > 0, "lifecycle width must be non-empty");
  TORCH_CHECK(revisions > 0, "revision width must be non-empty");
  TORCH_CHECK(
      input_lifecycle_input.size(-1) == lifecycles,
      "predicted and input lifecycle widths must match");
  TORCH_CHECK(
      retired_index >= 0 && retired_index < lifecycles,
      "retired_index is outside lifecycle width");
  TORCH_CHECK(threshold >= 0.0 && threshold <= 1.0, "threshold must be in [0, 1]");
  TORCH_CHECK(max_allocations > 0, "max_allocations must be positive");

  const auto dtype = allocation_logits_input.scalar_type();
  const bool fused_dtype =
      occupancy.scalar_type() == at::kBool &&
      lifecycle_logits_input.scalar_type() == dtype &&
      revision_logits_input.scalar_type() == dtype &&
      input_lifecycle_input.scalar_type() == dtype &&
      dtype != at::kDouble;
  if (!fused_dtype) {
    return rollout_slot_transition(
        occupancy_input,
        allocation_logits_input,
        lifecycle_logits_input,
        revision_logits_input,
        input_lifecycle_input,
        threshold,
        max_allocations,
        retired_index);
  }

  occupancy = occupancy.contiguous();
  auto allocation_logits = allocation_logits_input.contiguous();
  auto lifecycle_logits = lifecycle_logits_input.contiguous();
  auto revision_logits = revision_logits_input.contiguous();
  auto input_lifecycle = input_lifecycle_input.contiguous();

  auto next_occupancy =
      at::empty(occupancy.sizes(), occupancy.options().dtype(at::kBool));
  auto lifecycle_indices =
      at::empty(occupancy.sizes(), occupancy.options().dtype(at::kLong));
  auto revision_indices =
      at::empty(occupancy.sizes(), occupancy.options().dtype(at::kLong));
  auto input_lifecycle_indices =
      at::empty(occupancy.sizes(), occupancy.options().dtype(at::kLong));
  auto input_lifecycle_present =
      at::empty(occupancy.sizes(), occupancy.options().dtype(at::kBool));
  auto live_slots =
      at::empty(occupancy.sizes(), occupancy.options().dtype(at::kBool));

  if (batch == 0 || slots == 0) {
    return {
        next_occupancy,
        lifecycle_indices,
        revision_indices,
        input_lifecycle_indices,
        input_lifecycle_present,
        live_slots};
  }

  const auto stream = at::cuda::getCurrentCUDAStream(occupancy.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      dtype,
      "cid_rollout_slot_transition_cuda",
      [&] {
        rollout_slot_transition_kernel<scalar_t><<<batch, kThreads, 0, stream>>>(
            occupancy.const_data_ptr<bool>(),
            allocation_logits.const_data_ptr<scalar_t>(),
            lifecycle_logits.const_data_ptr<scalar_t>(),
            revision_logits.const_data_ptr<scalar_t>(),
            input_lifecycle.const_data_ptr<scalar_t>(),
            next_occupancy.mutable_data_ptr<bool>(),
            lifecycle_indices.mutable_data_ptr<std::int64_t>(),
            revision_indices.mutable_data_ptr<std::int64_t>(),
            input_lifecycle_indices.mutable_data_ptr<std::int64_t>(),
            input_lifecycle_present.mutable_data_ptr<bool>(),
            live_slots.mutable_data_ptr<bool>(),
            batch,
            slots,
            lifecycles,
            revisions,
            static_cast<float>(threshold),
            max_allocations,
            retired_index);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {
      next_occupancy,
      lifecycle_indices,
      revision_indices,
      input_lifecycle_indices,
      input_lifecycle_present,
      live_slots};
}

}  // namespace cid::engine

TORCH_LIBRARY_IMPL(cid_engine, CUDA, m) {
  m.impl(
      "rollout_slot_transition",
      TORCH_FN(cid::engine::rollout_slot_transition_cuda));
}
