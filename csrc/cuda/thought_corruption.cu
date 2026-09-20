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
constexpr float kHalfPi = 1.57079632679489661923F;

template <typename scalar_t>
__global__ void thought_corrupt_kernel(
    const scalar_t* __restrict__ semantic,
    const float* __restrict__ timesteps,
    const bool* __restrict__ occupancy,
    const scalar_t* __restrict__ epsilon,
    scalar_t* __restrict__ corrupted,
    scalar_t* __restrict__ local_noise,
    scalar_t* __restrict__ masked_epsilon,
    const std::int64_t rows,
    const std::int64_t hidden) {
  const auto index =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto total = rows * hidden;
  if (index >= total) {
    return;
  }

  const auto row = index / hidden;
  const bool occupied = occupancy[row];
  if (!occupied) {
    corrupted[index] = static_cast<scalar_t>(0.0F);
    masked_epsilon[index] = static_cast<scalar_t>(0.0F);
    if (index % hidden == 0) {
      local_noise[row] = static_cast<scalar_t>(0.0F);
    }
    return;
  }

  const float timestep = timesteps[row];
  const float cosine = cosf(timestep * kHalfPi);
  const scalar_t alpha_scalar = static_cast<scalar_t>(cosine * cosine);
  const float alpha = static_cast<float>(alpha_scalar);
  const scalar_t signal_scale_scalar = static_cast<scalar_t>(sqrtf(alpha));
  const scalar_t noise_variance_scalar = static_cast<scalar_t>(1.0F - alpha);
  const scalar_t noise_scale_scalar =
      static_cast<scalar_t>(sqrtf(static_cast<float>(noise_variance_scalar)));

  const scalar_t signal = static_cast<scalar_t>(
      static_cast<float>(signal_scale_scalar) *
      static_cast<float>(semantic[index]));
  const scalar_t noise = static_cast<scalar_t>(
      static_cast<float>(noise_scale_scalar) *
      static_cast<float>(epsilon[index]));
  corrupted[index] = static_cast<scalar_t>(
      static_cast<float>(signal) + static_cast<float>(noise));
  masked_epsilon[index] = epsilon[index];
  if (index % hidden == 0) {
    local_noise[row] = static_cast<scalar_t>(timestep);
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor>
thought_corrupt_from_epsilon_cuda(
    const at::Tensor& semantic_input,
    const at::Tensor& timesteps_input,
    const at::Tensor& occupancy_input,
    const at::Tensor& epsilon_input) {
  TORCH_CHECK(semantic_input.is_cuda(), "semantic must be CUDA");
  TORCH_CHECK(timesteps_input.is_cuda(), "timesteps must be CUDA");
  TORCH_CHECK(occupancy_input.is_cuda(), "occupancy must be CUDA");
  TORCH_CHECK(epsilon_input.is_cuda(), "epsilon must be CUDA");

  const auto device = semantic_input.device();
  TORCH_CHECK(
      timesteps_input.device() == device &&
          occupancy_input.device() == device &&
          epsilon_input.device() == device,
      "thought corruption inputs must share a CUDA device");
  TORCH_CHECK(
      semantic_input.dim() == 3,
      "semantic must have shape [batch, slots, hidden]");
  const auto batch = semantic_input.size(0);
  const auto slots = semantic_input.size(1);
  const auto hidden = semantic_input.size(2);
  TORCH_CHECK(
      timesteps_input.sizes() == at::IntArrayRef({batch, slots}),
      "timesteps must have shape [batch, slots]");
  TORCH_CHECK(
      occupancy_input.sizes() == at::IntArrayRef({batch, slots, 1}),
      "occupancy must have shape [batch, slots, 1]");
  TORCH_CHECK(
      epsilon_input.sizes() == semantic_input.sizes(),
      "epsilon must match semantic");
  TORCH_CHECK(
      epsilon_input.scalar_type() == semantic_input.scalar_type(),
      "epsilon must share semantic dtype");

  // The training scheduler normalizes timesteps to FP32 before corruption.
  // Other dtype combinations retain the Composite implementation's exact behavior.
  if (timesteps_input.scalar_type() != at::kFloat ||
      semantic_input.scalar_type() == at::kDouble) {
    return thought_corrupt_from_epsilon(
        semantic_input,
        timesteps_input,
        occupancy_input,
        epsilon_input);
  }

  const c10::cuda::CUDAGuard device_guard(device);
  auto semantic = semantic_input.contiguous();
  auto timesteps = timesteps_input.contiguous();
  auto occupancy = occupancy_input.to(at::kBool).contiguous();
  auto epsilon = epsilon_input.contiguous();

  auto corrupted = at::empty_like(semantic);
  auto local_noise = at::empty(
      {batch, slots, 1},
      semantic.options());
  auto masked_epsilon = at::empty_like(epsilon);
  const auto rows = batch * slots;
  const auto total = rows * hidden;
  if (total == 0) {
    return {corrupted, local_noise, masked_epsilon};
  }

  const auto blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  const auto stream = at::cuda::getCurrentCUDAStream(semantic.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      semantic.scalar_type(),
      "cid_thought_corrupt_from_epsilon_cuda",
      [&] {
        thought_corrupt_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
            semantic.const_data_ptr<scalar_t>(),
            timesteps.const_data_ptr<float>(),
            occupancy.const_data_ptr<bool>(),
            epsilon.const_data_ptr<scalar_t>(),
            corrupted.mutable_data_ptr<scalar_t>(),
            local_noise.mutable_data_ptr<scalar_t>(),
            masked_epsilon.mutable_data_ptr<scalar_t>(),
            rows,
            hidden);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {corrupted, local_noise, masked_epsilon};
}

}  // namespace cid::engine

TORCH_LIBRARY_IMPL(cid_engine, CUDA, m) {
  m.impl(
      "thought_corrupt_from_epsilon",
      TORCH_FN(cid::engine::thought_corrupt_from_epsilon_cuda));
}
