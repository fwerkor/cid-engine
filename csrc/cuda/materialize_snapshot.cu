#include "cid_engine/ops.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cstdint>

namespace cid::engine {
namespace {

constexpr int kThreads = 32;

template <typename scalar_t>
__global__ void materialize_cell_snapshot_kernel(
    const scalar_t* __restrict__ thought_semantic,
    const scalar_t* __restrict__ role_logits,
    const scalar_t* __restrict__ uncertainty,
    const scalar_t* __restrict__ noise_delta,
    const scalar_t* __restrict__ lifecycle_logits,
    const bool* __restrict__ selected,
    const std::int64_t* __restrict__ semantic_indices,
    float* __restrict__ output,
    const std::int64_t rows,
    const std::int64_t hidden,
    const std::int64_t roles,
    const std::int64_t lifecycles,
    const std::int64_t samples) {
  const std::int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const auto output_width = 4 + roles + samples;
  auto* out = output + row * output_width;

  if (threadIdx.x == 0) {
    out[0] = selected[row] ? 1.0F : 0.0F;

    float best_value = static_cast<float>(
        lifecycle_logits[row * lifecycles]);
    std::int64_t best_index = 0;
    for (std::int64_t index = 1; index < lifecycles; ++index) {
      const float value = static_cast<float>(
          lifecycle_logits[row * lifecycles + index]);
      if (value > best_value) {
        best_value = value;
        best_index = index;
      }
    }
    out[1] = static_cast<float>(best_index);
    out[2] = static_cast<float>(uncertainty[row]);
    out[3] = static_cast<float>(noise_delta[row]);
  }

  for (std::int64_t role = threadIdx.x; role < roles; role += blockDim.x) {
    const float value = static_cast<float>(
        role_logits[row * roles + role]);
    out[4 + role] = 1.0F / (1.0F + expf(-value));
  }

  for (std::int64_t sample = threadIdx.x; sample < samples; sample += blockDim.x) {
    const auto semantic_index = semantic_indices[sample];
    out[4 + roles + sample] = static_cast<float>(
        thought_semantic[row * hidden + semantic_index]);
  }
}

}  // namespace

at::Tensor materialize_cell_snapshot_cuda(
    const at::Tensor& thought_semantic_input,
    const at::Tensor& role_logits_input,
    const at::Tensor& uncertainty_input,
    const at::Tensor& noise_delta_input,
    const at::Tensor& lifecycle_logits_input,
    const at::Tensor& selected_input,
    const at::Tensor& semantic_indices_input) {
  TORCH_CHECK(thought_semantic_input.is_cuda(), "thought_semantic must be CUDA");
  TORCH_CHECK(role_logits_input.is_cuda(), "role_logits must be CUDA");
  TORCH_CHECK(uncertainty_input.is_cuda(), "uncertainty must be CUDA");
  TORCH_CHECK(noise_delta_input.is_cuda(), "noise_delta must be CUDA");
  TORCH_CHECK(lifecycle_logits_input.is_cuda(), "lifecycle_logits must be CUDA");
  TORCH_CHECK(selected_input.is_cuda(), "selected must be CUDA");
  TORCH_CHECK(semantic_indices_input.is_cuda(), "semantic_indices must be CUDA");

  const auto device = thought_semantic_input.device();
  TORCH_CHECK(
      role_logits_input.device() == device &&
          uncertainty_input.device() == device &&
          noise_delta_input.device() == device &&
          lifecycle_logits_input.device() == device &&
          selected_input.device() == device &&
          semantic_indices_input.device() == device,
      "materialization snapshot inputs must share a CUDA device");

  TORCH_CHECK(
      thought_semantic_input.dim() == 3,
      "thought_semantic must have shape [batch, slots, hidden]");
  const auto batch = thought_semantic_input.size(0);
  const auto slots = thought_semantic_input.size(1);
  const auto hidden = thought_semantic_input.size(2);
  TORCH_CHECK(
      role_logits_input.dim() == 3 &&
          role_logits_input.size(0) == batch &&
          role_logits_input.size(1) == slots,
      "role_logits must have shape [batch, slots, roles]");
  TORCH_CHECK(
      uncertainty_input.sizes() == at::IntArrayRef({batch, slots, 1}),
      "uncertainty must have shape [batch, slots, 1]");
  TORCH_CHECK(
      noise_delta_input.sizes() == at::IntArrayRef({batch, slots, 1}),
      "noise_delta must have shape [batch, slots, 1]");
  TORCH_CHECK(
      lifecycle_logits_input.dim() == 3 &&
          lifecycle_logits_input.size(0) == batch &&
          lifecycle_logits_input.size(1) == slots,
      "lifecycle_logits must have shape [batch, slots, lifecycles]");
  TORCH_CHECK(
      selected_input.sizes() == at::IntArrayRef({batch, slots}),
      "selected must have shape [batch, slots]");
  TORCH_CHECK(
      semantic_indices_input.dim() == 1 &&
          semantic_indices_input.scalar_type() == at::kLong,
      "semantic_indices must be a one-dimensional int64 tensor");

  const auto dtype = thought_semantic_input.scalar_type();
  TORCH_CHECK(
      role_logits_input.scalar_type() == dtype &&
          uncertainty_input.scalar_type() == dtype &&
          noise_delta_input.scalar_type() == dtype &&
          lifecycle_logits_input.scalar_type() == dtype,
      "native materialization snapshot tensors must share a floating dtype");
  TORCH_CHECK(
      selected_input.scalar_type() == at::kBool,
      "selected must use torch.bool");

  auto thought_semantic = thought_semantic_input.contiguous();
  auto role_logits = role_logits_input.contiguous();
  auto uncertainty = uncertainty_input.contiguous();
  auto noise_delta = noise_delta_input.contiguous();
  auto lifecycle_logits = lifecycle_logits_input.contiguous();
  auto selected = selected_input.contiguous();
  auto semantic_indices = semantic_indices_input.contiguous();

  const auto rows = batch * slots;
  const auto roles = role_logits.size(-1);
  const auto lifecycles = lifecycle_logits.size(-1);
  const auto samples = semantic_indices.numel();
  auto output = at::empty(
      {batch, slots, 4 + roles + samples},
      thought_semantic.options().dtype(at::kFloat));

  if (rows == 0) {
    return output;
  }

  const auto stream = at::cuda::getCurrentCUDAStream(
      thought_semantic.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      dtype,
      "cid_materialize_cell_snapshot_cuda",
      [&] {
        materialize_cell_snapshot_kernel<scalar_t>
            <<<rows, kThreads, 0, stream>>>(
                thought_semantic.const_data_ptr<scalar_t>(),
                role_logits.const_data_ptr<scalar_t>(),
                uncertainty.const_data_ptr<scalar_t>(),
                noise_delta.const_data_ptr<scalar_t>(),
                lifecycle_logits.const_data_ptr<scalar_t>(),
                selected.const_data_ptr<bool>(),
                semantic_indices.const_data_ptr<std::int64_t>(),
                output.mutable_data_ptr<float>(),
                rows,
                hidden,
                roles,
                lifecycles,
                samples);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace cid::engine

TORCH_LIBRARY_IMPL(cid_engine, CUDA, m) {
  m.impl(
      "materialize_cell_snapshot",
      TORCH_FN(cid::engine::materialize_cell_snapshot_cuda));
}
