#include <ATen/Functions.h>
#include <torch/library.h>

namespace cid::engine {
namespace {

at::Tensor materialize_cell_snapshot_cann(
    const at::Tensor& thought_semantic,
    const at::Tensor& role_logits,
    const at::Tensor& uncertainty,
    const at::Tensor& noise_delta,
    const at::Tensor& lifecycle_logits,
    const at::Tensor& selected,
    const at::Tensor& semantic_indices) {
  TORCH_CHECK(
      thought_semantic.dim() == 3,
      "thought_semantic must have shape [batch, slots, hidden]");
  const auto batch = thought_semantic.size(0);
  const auto slots = thought_semantic.size(1);
  TORCH_CHECK(
      role_logits.dim() == 3 &&
          role_logits.size(0) == batch &&
          role_logits.size(1) == slots,
      "role_logits must have shape [batch, slots, roles]");
  TORCH_CHECK(
      uncertainty.sizes() == at::IntArrayRef({batch, slots, 1}),
      "uncertainty must have shape [batch, slots, 1]");
  TORCH_CHECK(
      noise_delta.sizes() == at::IntArrayRef({batch, slots, 1}),
      "noise_delta must have shape [batch, slots, 1]");
  TORCH_CHECK(
      lifecycle_logits.dim() == 3 &&
          lifecycle_logits.size(0) == batch &&
          lifecycle_logits.size(1) == slots,
      "lifecycle_logits must have shape [batch, slots, lifecycles]");
  TORCH_CHECK(
      selected.sizes() == at::IntArrayRef({batch, slots}),
      "selected must have shape [batch, slots]");
  TORCH_CHECK(
      selected.scalar_type() == at::kBool,
      "selected must use torch.bool");
  TORCH_CHECK(
      semantic_indices.dim() == 1 &&
          semantic_indices.scalar_type() == at::kLong,
      "semantic_indices must be a one-dimensional int64 tensor");

  auto selected_f32 = selected.to(at::kFloat).unsqueeze(-1);
  auto lifecycle = std::get<1>(lifecycle_logits.to(at::kFloat).max(-1))
                       .to(at::kFloat)
                       .unsqueeze(-1);
  auto roles = role_logits.to(at::kFloat).sigmoid();
  auto semantic = thought_semantic.to(at::kFloat).index_select(
      -1,
      semantic_indices.to(thought_semantic.device()));

  return at::cat(
      {
          selected_f32,
          lifecycle,
          uncertainty.to(at::kFloat),
          noise_delta.to(at::kFloat),
          roles,
          semantic,
      },
      -1);
}

}  // namespace
}  // namespace cid::engine

TORCH_LIBRARY_IMPL(cid_engine, PrivateUse1, m) {
  m.impl(
      "materialize_cell_snapshot",
      TORCH_FN(cid::engine::materialize_cell_snapshot_cann));
}
