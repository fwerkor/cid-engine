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

namespace cid::engine {
namespace {

constexpr std::int64_t kMaxColumns = 16;
constexpr std::int64_t kMaskedDiffusionFastPathElements = 8192;

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

}  // namespace
}  // namespace cid::engine

TORCH_LIBRARY_IMPL(cid_engine, PrivateUse1, m) {
  m.impl(
      "batched_linear_assignment",
      TORCH_FN(cid::engine::batched_linear_assignment_cann));
  m.impl(
      "masked_diffusion_corrupt_from_random",
      TORCH_FN(cid::engine::masked_diffusion_corrupt_from_random_cann));
}
