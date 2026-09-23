#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include <cstdint>
#include <tuple>

void cid_batched_linear_assignment_kernel(
    std::uint32_t block_dim,
    void* l2ctrl,
    void* stream,
    std::uint8_t* costs,
    std::uint8_t* row_counts,
    std::uint8_t* output,
    std::int64_t batch,
    std::int64_t rows,
    std::int64_t columns);

void cid_masked_diffusion_f32_kernel(
    std::uint32_t block_dim,
    void* l2ctrl,
    void* stream,
    std::uint8_t* clean_ids,
    std::uint8_t* ratio_random,
    std::uint8_t* mask_random,
    std::uint8_t* corrupted,
    std::uint8_t* masked,
    std::uint8_t* mask_ratio,
    std::uint8_t* metrics,
    std::int64_t batch,
    std::int64_t tokens,
    std::int64_t mask_token_id,
    float min_mask_ratio,
    float max_mask_ratio);


void cid_display_corrupt_mask_kernel(
    std::uint32_t block_dim,
    void* l2ctrl,
    void* stream,
    std::uint8_t* token_ids,
    std::uint8_t* timesteps,
    std::uint8_t* eligible,
    std::uint8_t* corruption_random,
    std::uint8_t* corrupted,
    std::uint8_t* labels,
    std::uint8_t* masked,
    std::uint8_t* replaced,
    std::int64_t batch,
    std::int64_t tokens,
    std::int64_t mask_token_id);

void cid_display_corrupt_replace_kernel(
    std::uint32_t block_dim,
    void* l2ctrl,
    void* stream,
    std::uint8_t* token_ids,
    std::uint8_t* timesteps,
    std::uint8_t* eligible,
    std::uint8_t* corruption_random,
    std::uint8_t* replacement_random,
    std::uint8_t* replacement_offsets,
    std::uint8_t* corrupted,
    std::uint8_t* labels,
    std::uint8_t* masked,
    std::uint8_t* replaced,
    std::int64_t batch,
    std::int64_t tokens,
    std::int64_t mask_token_id,
    std::int64_t eos_token_id,
    bool has_eos,
    std::int64_t vocab_size,
    float replacement_fraction);

namespace cid::engine {
namespace {

constexpr std::int64_t kMaxColumns = 16;
constexpr std::int64_t kMaskedDiffusionFastPathElements = 8192;
constexpr std::int64_t kDisplayCorruptionFastPathElements = 8192;

at::Tensor batched_linear_assignment_cann(
    const at::Tensor& costs_input,
    const at::Tensor& row_counts_input) {
  TORCH_CHECK(
      costs_input.device().type() == c10::DeviceType::PrivateUse1,
      "costs must be an NPU tensor");
  TORCH_CHECK(
      row_counts_input.device().type() == c10::DeviceType::PrivateUse1,
      "row_counts must be an NPU tensor");
  TORCH_CHECK(
      costs_input.device() == row_counts_input.device(),
      "costs and row_counts must share a device");
  TORCH_CHECK(
      costs_input.dim() == 3,
      "costs must have shape [batch, rows, columns]");
  TORCH_CHECK(
      costs_input.scalar_type() == at::kFloat,
      "CANN batched_linear_assignment requires float32 costs");
  TORCH_CHECK(
      row_counts_input.dim() == 1 &&
          row_counts_input.size(0) == costs_input.size(0),
      "row_counts must have shape [batch]");
  TORCH_CHECK(
      row_counts_input.scalar_type() == at::kLong,
      "row_counts must use torch.int64");
  TORCH_CHECK(
      costs_input.size(1) <= costs_input.size(2),
      "assignment rows cannot exceed columns");
  TORCH_CHECK(
      costs_input.size(2) <= kMaxColumns,
      "batched_linear_assignment supports at most 16 columns");

  const c10::OptionalDeviceGuard device_guard(costs_input.device());
  auto costs = costs_input.contiguous();
  auto row_counts = row_counts_input.contiguous();
  const auto batch = costs.size(0);
  const auto rows = costs.size(1);
  const auto columns = costs.size(2);
  auto output = at::empty(
      {batch, rows},
      costs.options().dtype(at::kLong));
  if (batch == 0 || rows == 0) {
    return output;
  }

  auto stream = c10_npu::getCurrentNPUStream(costs.get_device()).stream(false);
  auto* costs_address =
      reinterpret_cast<std::uint8_t*>(costs.data_ptr<float>());
  auto* row_counts_address =
      reinterpret_cast<std::uint8_t*>(row_counts.data_ptr<std::int64_t>());
  auto* output_address =
      reinterpret_cast<std::uint8_t*>(output.data_ptr<std::int64_t>());
  auto launch = [=]() -> int {
    cid_batched_linear_assignment_kernel(
        static_cast<std::uint32_t>(batch),
        nullptr,
        stream,
        costs_address,
        row_counts_address,
        output_address,
        batch,
        rows,
        columns);
    return 0;
  };
  at_npu::native::OpCommand::RunOpApi(
      "CidBatchedLinearAssignment",
      launch);
  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
masked_diffusion_corrupt_from_random_cann(
    const at::Tensor& clean_ids_input,
    const at::Tensor& ratio_random_input,
    const at::Tensor& mask_random_input,
    const std::int64_t mask_token_id,
    const double min_mask_ratio,
    const double max_mask_ratio) {
  TORCH_CHECK(
      clean_ids_input.device().type() == c10::DeviceType::PrivateUse1,
      "clean_ids must be NPU");
  const auto device = clean_ids_input.device();
  TORCH_CHECK(
      ratio_random_input.device() == device &&
          mask_random_input.device() == device,
      "masked diffusion inputs must share an NPU device");
  TORCH_CHECK(
      clean_ids_input.dim() == 2,
      "clean_ids must have shape [batch, tokens]");
  TORCH_CHECK(
      clean_ids_input.scalar_type() == at::kLong,
      "clean_ids must use torch.int64");
  const auto batch = clean_ids_input.size(0);
  const auto tokens = clean_ids_input.size(1);
  TORCH_CHECK(
      batch > 0 && tokens > 0,
      "clean_ids dimensions must be non-empty");
  TORCH_CHECK(
      ratio_random_input.sizes() == at::IntArrayRef({batch, 1}),
      "ratio_random must have shape [batch, 1]");
  TORCH_CHECK(
      mask_random_input.sizes() == clean_ids_input.sizes(),
      "mask_random must match clean_ids shape");
  TORCH_CHECK(
      ratio_random_input.is_floating_point() &&
          mask_random_input.is_floating_point(),
      "random tensors must use floating dtypes");
  TORCH_CHECK(
      min_mask_ratio > 0.0 &&
          min_mask_ratio <= max_mask_ratio &&
          max_mask_ratio <= 1.0,
      "mask ratio range must satisfy 0 < min <= max <= 1");

  const c10::OptionalDeviceGuard device_guard(device);
  if (clean_ids_input.numel() > kMaskedDiffusionFastPathElements) {
    auto ratio_f32 = ratio_random_input.to(at::kFloat);
    auto mask_random_f32 = mask_random_input.to(at::kFloat);
    auto mask_ratio =
        min_mask_ratio + (max_mask_ratio - min_mask_ratio) * ratio_f32;
    auto masked = mask_random_f32.lt(mask_ratio);
    auto empty_rows = masked.any(1).logical_not();
    auto fallback_positions = std::get<1>(mask_random_f32.min(1));
    auto fallback = at::zeros_like(masked);
    fallback.scatter_(
        1,
        fallback_positions.unsqueeze(1),
        empty_rows.unsqueeze(1));
    masked.logical_or_(fallback);
    auto corrupted = clean_ids_input.masked_fill(masked, mask_token_id);
    auto metrics = at::stack(
        {masked.to(at::kFloat).mean(), mask_ratio.mean()});
    return {corrupted, masked, mask_ratio, metrics};
  }

  auto clean_ids = clean_ids_input.contiguous();
  auto ratio_random = ratio_random_input.to(at::kFloat).contiguous();
  auto mask_random = mask_random_input.to(at::kFloat).contiguous();

  auto corrupted = at::empty_like(clean_ids);
  auto masked = at::empty(
      clean_ids.sizes(),
      clean_ids.options().dtype(at::kBool));
  auto mask_ratio = at::empty(
      {batch, 1},
      clean_ids.options().dtype(at::kFloat));
  auto metrics = at::empty(
      {2},
      clean_ids.options().dtype(at::kFloat));

  const auto npu_stream =
      c10_npu::getCurrentNPUStream(clean_ids.get_device());
  auto stream = npu_stream.stream(false);
  auto* clean_ids_address =
      reinterpret_cast<std::uint8_t*>(clean_ids.data_ptr<std::int64_t>());
  auto* ratio_random_address =
      reinterpret_cast<std::uint8_t*>(ratio_random.data_ptr<float>());
  auto* mask_random_address =
      reinterpret_cast<std::uint8_t*>(mask_random.data_ptr<float>());
  auto* corrupted_address =
      reinterpret_cast<std::uint8_t*>(corrupted.data_ptr<std::int64_t>());
  auto* masked_address =
      reinterpret_cast<std::uint8_t*>(masked.data_ptr<bool>());
  auto* mask_ratio_address =
      reinterpret_cast<std::uint8_t*>(mask_ratio.data_ptr<float>());
  auto* metrics_address =
      reinterpret_cast<std::uint8_t*>(metrics.data_ptr<float>());

  auto launch = [=]() -> int {
    cid_masked_diffusion_f32_kernel(
        1,
        nullptr,
        stream,
        clean_ids_address,
        ratio_random_address,
        mask_random_address,
        corrupted_address,
        masked_address,
        mask_ratio_address,
        metrics_address,
        batch,
        tokens,
        mask_token_id,
        static_cast<float>(min_mask_ratio),
        static_cast<float>(max_mask_ratio));
    return 0;
  };
  at_npu::native::OpCommand::RunOpApi("CidMaskedDiffusion", launch);
  return {corrupted, masked, mask_ratio, metrics};
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
display_corrupt_from_random_cann(
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
  TORCH_CHECK(
      token_ids_input.device().type() == c10::DeviceType::PrivateUse1,
      "token_ids must be NPU");
  const auto device = token_ids_input.device();
  TORCH_CHECK(
      timesteps_input.device() == device &&
          eligible_mask_input.device() == device &&
          corruption_random_input.device() == device,
      "display corruption inputs must share an NPU device");
  TORCH_CHECK(
      token_ids_input.dim() == 2,
      "token_ids must have shape [batch, tokens]");
  TORCH_CHECK(
      token_ids_input.scalar_type() == at::kLong,
      "token_ids must use torch.int64");
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
      timesteps_input.is_floating_point() &&
          corruption_random_input.is_floating_point(),
      "timesteps and corruption_random must use floating dtypes");
  TORCH_CHECK(
      replacement_fraction >= 0.0 && replacement_fraction <= 1.0,
      "replacement_fraction must be in [0, 1]");

  const bool use_replacement = replacement_fraction > 0.0;
  if (use_replacement) {
    TORCH_CHECK(
        vocab_size >= 3,
        "visible replacement corruption requires vocab_size >= 3");
    TORCH_CHECK(
        replacement_random_input.has_value() &&
            replacement_offsets_input.has_value(),
        "replacement random tensors are required when replacement_fraction > 0");
    TORCH_CHECK(
        replacement_random_input->device() == device &&
            replacement_offsets_input->device() == device,
        "replacement random tensors must share the NPU device");
    TORCH_CHECK(
        replacement_random_input->sizes() == token_ids_input.sizes() &&
            replacement_offsets_input->sizes() == token_ids_input.sizes(),
        "replacement random tensors must match token_ids shape");
    TORCH_CHECK(
        replacement_random_input->is_floating_point(),
        "replacement_random must use a floating dtype");
    TORCH_CHECK(
        replacement_offsets_input->scalar_type() == at::kLong,
        "replacement_offsets must use torch.int64");
  }

  const c10::OptionalDeviceGuard device_guard(device);
  if (token_ids_input.numel() > kDisplayCorruptionFastPathElements) {
    const auto eligible = eligible_mask_input.to(at::kBool);
    const auto timestep_f32 = timesteps_input.to(at::kFloat);
    auto corrupted_positions =
        corruption_random_input.to(at::kFloat).lt(timestep_f32.unsqueeze(1));
    corrupted_positions.logical_and_(eligible);

    const auto empty_rows = corrupted_positions.any(1).logical_not();
    const auto fallback_rows =
        empty_rows.logical_and(timestep_f32.gt(0.0)).logical_and(eligible.any(1));
    const auto first_eligible =
        std::get<1>(eligible.to(at::kLong).max(1)).unsqueeze(1);
    auto fallback = at::zeros_like(corrupted_positions);
    fallback.scatter_(1, first_eligible, fallback_rows.unsqueeze(1));
    corrupted_positions.logical_or_(fallback);

    at::Tensor replaced;
    at::Tensor masked;
    at::Tensor corrupted;
    if (use_replacement) {
      replaced = corrupted_positions.logical_and(
          replacement_random_input->to(at::kFloat).lt(replacement_fraction));
      auto replacements =
          at::remainder(token_ids_input + *replacement_offsets_input, vocab_size);
      for (int iteration = 0; iteration < 3; ++iteration) {
        auto forbidden = replacements.eq(mask_token_id).logical_or(
            replacements.eq(token_ids_input));
        if (eos_token_id.has_value()) {
          forbidden.logical_or_(replacements.eq(*eos_token_id));
        }
        replacements = at::where(
            forbidden,
            at::remainder(replacements + 1, vocab_size),
            replacements);
      }
      masked = corrupted_positions.logical_and(replaced.logical_not());
      corrupted = token_ids_input.masked_fill(masked, mask_token_id);
      corrupted = at::where(replaced, replacements, corrupted);
    } else {
      replaced = at::zeros_like(corrupted_positions);
      masked = corrupted_positions;
      corrupted = token_ids_input.masked_fill(masked, mask_token_id);
    }

    auto labels = at::where(
        corrupted_positions,
        token_ids_input,
        at::full({}, -100, token_ids_input.options()));
    return {corrupted, labels, masked, replaced};
  }

  auto token_ids = token_ids_input.contiguous();
  auto timesteps = timesteps_input.to(at::kFloat).contiguous();
  auto eligible = eligible_mask_input.to(at::kBool).contiguous();
  auto corruption_random =
      corruption_random_input.to(at::kFloat).contiguous();
  at::Tensor replacement_random;
  at::Tensor replacement_offsets;
  if (use_replacement) {
    replacement_random =
        replacement_random_input->to(at::kFloat).contiguous();
    replacement_offsets = replacement_offsets_input->contiguous();
  }

  auto corrupted = at::empty_like(token_ids);
  auto labels = at::empty_like(token_ids);
  auto masked = at::empty(
      token_ids.sizes(),
      token_ids.options().dtype(at::kBool));
  auto replaced = at::empty(
      token_ids.sizes(),
      token_ids.options().dtype(at::kBool));
  if (batch == 0 || tokens == 0) {
    return {corrupted, labels, masked, replaced};
  }

  auto stream =
      c10_npu::getCurrentNPUStream(token_ids.get_device()).stream(false);
  auto* token_ids_address =
      reinterpret_cast<std::uint8_t*>(token_ids.data_ptr<std::int64_t>());
  auto* timesteps_address =
      reinterpret_cast<std::uint8_t*>(timesteps.data_ptr<float>());
  auto* eligible_address =
      reinterpret_cast<std::uint8_t*>(eligible.data_ptr<bool>());
  auto* corruption_random_address =
      reinterpret_cast<std::uint8_t*>(corruption_random.data_ptr<float>());
  auto* corrupted_address =
      reinterpret_cast<std::uint8_t*>(corrupted.data_ptr<std::int64_t>());
  auto* labels_address =
      reinterpret_cast<std::uint8_t*>(labels.data_ptr<std::int64_t>());
  auto* masked_address =
      reinterpret_cast<std::uint8_t*>(masked.data_ptr<bool>());
  auto* replaced_address =
      reinterpret_cast<std::uint8_t*>(replaced.data_ptr<bool>());

  auto launch = [=]() -> int {
    constexpr std::uint32_t block_dim = 1;
    if (use_replacement) {
      cid_display_corrupt_replace_kernel(
          block_dim,
          nullptr,
          stream,
          token_ids_address,
          timesteps_address,
          eligible_address,
          corruption_random_address,
          reinterpret_cast<std::uint8_t*>(
              replacement_random.data_ptr<float>()),
          reinterpret_cast<std::uint8_t*>(
              replacement_offsets.data_ptr<std::int64_t>()),
          corrupted_address,
          labels_address,
          masked_address,
          replaced_address,
          batch,
          tokens,
          mask_token_id,
          eos_token_id.value_or(-1),
          eos_token_id.has_value(),
          vocab_size,
          static_cast<float>(replacement_fraction));
    } else {
      cid_display_corrupt_mask_kernel(
          block_dim,
          nullptr,
          stream,
          token_ids_address,
          timesteps_address,
          eligible_address,
          corruption_random_address,
          corrupted_address,
          labels_address,
          masked_address,
          replaced_address,
          batch,
          tokens,
          mask_token_id);
    }
    return 0;
  };
  at_npu::native::OpCommand::RunOpApi("CidDisplayCorrupt", launch);
  return {corrupted, labels, masked, replaced};
}

}  // namespace
}  // namespace cid::engine

TORCH_LIBRARY_IMPL(cid_engine, PrivateUse1, m) {
  m.impl(
      "batched_linear_assignment",
      TORCH_FN(cid::engine::batched_linear_assignment_cann));
  m.impl(
      "masked_diffusion_corrupt_from_random",
      TORCH_FN(cid::engine::masked_diffusion_corrupt_from_random_cann));
  m.impl(
      "display_corrupt_from_random",
      TORCH_FN(cid::engine::display_corrupt_from_random_cann));
}
