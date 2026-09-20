#include "cid_engine/ops.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <climits>
#include <cstdint>
#include <optional>
#include <tuple>

namespace cid::engine {
namespace {

constexpr int kThreads = 256;

__device__ std::int64_t replacement_token(
    const std::int64_t token,
    const std::int64_t offset,
    const std::int64_t vocab_size,
    const std::int64_t mask_token_id,
    const std::int64_t eos_token_id,
    const bool has_eos) {
  auto replacement = (token + offset) % vocab_size;
  for (int iteration = 0; iteration < 3; ++iteration) {
    const bool forbidden =
        replacement == mask_token_id ||
        replacement == token ||
        (has_eos && replacement == eos_token_id);
    if (!forbidden) {
      break;
    }
    replacement = (replacement + 1) % vocab_size;
  }
  return replacement;
}

__global__ void display_corrupt_kernel(
    const std::int64_t* __restrict__ token_ids,
    const float* __restrict__ timesteps,
    const bool* __restrict__ eligible,
    const float* __restrict__ corruption_random,
    const float* __restrict__ replacement_random,
    const std::int64_t* __restrict__ replacement_offsets,
    std::int64_t* __restrict__ corrupted,
    std::int64_t* __restrict__ labels,
    bool* __restrict__ masked,
    bool* __restrict__ replaced,
    const std::int64_t batch,
    const std::int64_t tokens,
    const std::int64_t mask_token_id,
    const std::int64_t eos_token_id,
    const bool has_eos,
    const std::int64_t vocab_size,
    const float replacement_fraction) {
  const auto row = static_cast<std::int64_t>(blockIdx.x);
  if (row >= batch) {
    return;
  }

  const int lane = threadIdx.x;
  const auto base = row * tokens;
  const float timestep = timesteps[row];
  int local_count = 0;
  int local_first_eligible = INT_MAX;

  for (std::int64_t column = lane; column < tokens; column += blockDim.x) {
    const auto index = base + column;
    const bool can_corrupt = eligible[index];
    if (can_corrupt && column < local_first_eligible) {
      local_first_eligible = static_cast<int>(column);
    }

    const bool selected =
        can_corrupt && corruption_random[index] < timestep;
    local_count += selected ? 1 : 0;

    bool replace = false;
    if (selected && replacement_fraction > 0.0F) {
      replace = replacement_random[index] < replacement_fraction;
    }
    const bool mask = selected && !replace;
    masked[index] = mask;
    replaced[index] = replace;
    labels[index] = selected ? token_ids[index] : -100;

    if (mask) {
      corrupted[index] = mask_token_id;
    } else if (replace) {
      corrupted[index] = replacement_token(
          token_ids[index],
          replacement_offsets[index],
          vocab_size,
          mask_token_id,
          eos_token_id,
          has_eos);
    } else {
      corrupted[index] = token_ids[index];
    }
  }

  __shared__ int counts[kThreads];
  __shared__ int first_eligible[kThreads];
  counts[lane] = local_count;
  first_eligible[lane] = local_first_eligible;
  __syncthreads();

  for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
    if (lane < offset) {
      counts[lane] += counts[lane + offset];
      if (first_eligible[lane + offset] < first_eligible[lane]) {
        first_eligible[lane] = first_eligible[lane + offset];
      }
    }
    __syncthreads();
  }

  if (lane == 0 &&
      counts[0] == 0 &&
      timestep > 0.0F &&
      first_eligible[0] != INT_MAX) {
    const auto index = base + first_eligible[0];
    bool replace = false;
    if (replacement_fraction > 0.0F) {
      replace = replacement_random[index] < replacement_fraction;
    }
    replaced[index] = replace;
    masked[index] = !replace;
    labels[index] = token_ids[index];
    corrupted[index] = replace
        ? replacement_token(
              token_ids[index],
              replacement_offsets[index],
              vocab_size,
              mask_token_id,
              eos_token_id,
              has_eos)
        : mask_token_id;
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
display_corrupt_from_random_cuda(
    const at::Tensor& token_ids_input,
    const at::Tensor& timesteps_input,
    const at::Tensor& eligible_mask_input,
    const at::Tensor& corruption_random_input,
    const std::optional<at::Tensor>& replacement_random_input,
    const std::optional<at::Tensor>& replacement_offsets_input,
    const std::int64_t mask_token_id,
    const std::optional<std::int64_t>& eos_token_id,
    const std::int64_t vocab_size,
    const double replacement_fraction) {
  TORCH_CHECK(token_ids_input.is_cuda(), "token_ids must be CUDA");
  TORCH_CHECK(timesteps_input.is_cuda(), "timesteps must be CUDA");
  TORCH_CHECK(eligible_mask_input.is_cuda(), "eligible_mask must be CUDA");
  TORCH_CHECK(corruption_random_input.is_cuda(), "corruption_random must be CUDA");

  const auto device = token_ids_input.device();
  TORCH_CHECK(
      timesteps_input.device() == device &&
          eligible_mask_input.device() == device &&
          corruption_random_input.device() == device,
      "display corruption inputs must share a CUDA device");
  TORCH_CHECK(token_ids_input.dim() == 2, "token_ids must have shape [batch, tokens]");
  TORCH_CHECK(token_ids_input.scalar_type() == at::kLong, "token_ids must use torch.int64");

  const auto batch = token_ids_input.size(0);
  const auto tokens = token_ids_input.size(1);
  TORCH_CHECK(
      timesteps_input.sizes() == at::IntArrayRef({batch}),
      "timesteps must have shape [batch]");
  TORCH_CHECK(
      eligible_mask_input.sizes() == token_ids_input.sizes(),
      "eligible_mask must match token_ids shape");
  TORCH_CHECK(
      corruption_random_input.sizes() == token_ids_input.sizes(),
      "corruption_random must match token_ids shape");
  TORCH_CHECK(
      timesteps_input.scalar_type() == at::kFloat &&
          corruption_random_input.scalar_type() == at::kFloat,
      "CUDA timesteps and corruption_random must use torch.float32");
  TORCH_CHECK(
      eligible_mask_input.scalar_type() == at::kBool,
      "CUDA eligible_mask must use torch.bool");
  TORCH_CHECK(
      replacement_fraction >= 0.0 && replacement_fraction <= 1.0,
      "replacement_fraction must be in [0, 1]");

  const bool use_replacement = replacement_fraction > 0.0;
  if (use_replacement) {
    TORCH_CHECK(vocab_size >= 3, "visible replacement corruption requires vocab_size >= 3");
    TORCH_CHECK(
        replacement_random_input.has_value() && replacement_offsets_input.has_value(),
        "replacement random tensors are required when replacement_fraction > 0");
    TORCH_CHECK(
        replacement_random_input->is_cuda() &&
            replacement_offsets_input->is_cuda() &&
            replacement_random_input->device() == device &&
            replacement_offsets_input->device() == device,
        "replacement random tensors must share the CUDA device");
    TORCH_CHECK(
        replacement_random_input->sizes() == token_ids_input.sizes() &&
            replacement_offsets_input->sizes() == token_ids_input.sizes(),
        "replacement random tensors must match token_ids shape");
    TORCH_CHECK(
        replacement_random_input->scalar_type() == at::kFloat,
        "CUDA replacement_random must use torch.float32");
    TORCH_CHECK(
        replacement_offsets_input->scalar_type() == at::kLong,
        "replacement_offsets must use torch.int64");
  }

  const c10::cuda::CUDAGuard device_guard(device);
  auto token_ids = token_ids_input.contiguous();
  auto timesteps = timesteps_input.contiguous();
  auto eligible = eligible_mask_input.contiguous();
  auto corruption_random = corruption_random_input.contiguous();
  at::Tensor replacement_random;
  at::Tensor replacement_offsets;
  if (use_replacement) {
    replacement_random = replacement_random_input->contiguous();
    replacement_offsets = replacement_offsets_input->contiguous();
  }

  auto corrupted = at::empty_like(token_ids);
  auto labels = at::empty_like(token_ids);
  auto masked = at::empty(token_ids.sizes(), token_ids.options().dtype(at::kBool));
  auto replaced = at::empty(token_ids.sizes(), token_ids.options().dtype(at::kBool));
  if (batch == 0 || tokens == 0) {
    return {corrupted, labels, masked, replaced};
  }

  const auto stream = at::cuda::getCurrentCUDAStream(token_ids.get_device());
  display_corrupt_kernel<<<batch, kThreads, 0, stream>>>(
      token_ids.const_data_ptr<std::int64_t>(),
      timesteps.const_data_ptr<float>(),
      eligible.const_data_ptr<bool>(),
      corruption_random.const_data_ptr<float>(),
      use_replacement ? replacement_random.const_data_ptr<float>() : nullptr,
      use_replacement ? replacement_offsets.const_data_ptr<std::int64_t>() : nullptr,
      corrupted.mutable_data_ptr<std::int64_t>(),
      labels.mutable_data_ptr<std::int64_t>(),
      masked.mutable_data_ptr<bool>(),
      replaced.mutable_data_ptr<bool>(),
      batch,
      tokens,
      mask_token_id,
      eos_token_id.value_or(-1),
      eos_token_id.has_value(),
      vocab_size,
      static_cast<float>(replacement_fraction));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {corrupted, labels, masked, replaced};
}

}  // namespace cid::engine

TORCH_LIBRARY_IMPL(cid_engine, CUDA, m) {
  m.impl(
      "display_corrupt_from_random",
      TORCH_FN(cid::engine::display_corrupt_from_random_cuda));
}
