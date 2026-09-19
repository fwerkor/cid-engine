#include "cid_engine/ops.hpp"

#include <torch/library.h>

TORCH_LIBRARY(cid_engine, m) {
  m.def(
      "live_slot_occupancy(Tensor slot_occupancy, Tensor? lifecycle_features, "
      "int retired_index) -> Tensor");
  m.def(
      "prefix_allocation_mask(Tensor occupancy, Tensor allocation_logits, "
      "float threshold, int max_allocations) -> Tensor");
  m.def(
      "batched_linear_assignment(Tensor costs, Tensor row_counts) -> Tensor");
  m.def(
      "thought_corrupt_from_epsilon(Tensor semantic, Tensor timesteps, "
      "Tensor occupancy, Tensor epsilon) -> (Tensor, Tensor, Tensor)");
  m.def(
      "display_token_statistics(Tensor token_ids, Tensor logits) "
      "-> (Tensor, Tensor, Tensor)");
  m.def(
      "refine_display_from_statistics(Tensor token_ids, Tensor confidence, "
      "Tensor predicted, Tensor current_confidence, int mask_token_id, "
      "int? eos_token_id, float reveal_fraction, float revision_fraction, "
      "float revision_margin) -> Tensor");
  m.def(
      "materialize_cell_snapshot(Tensor thought_semantic, Tensor role_logits, "
      "Tensor uncertainty, Tensor noise_delta, Tensor lifecycle_logits, "
      "Tensor selected, Tensor semantic_indices) -> Tensor");
}

TORCH_LIBRARY_IMPL(cid_engine, CompositeImplicitAutograd, m) {
  m.impl("live_slot_occupancy", TORCH_FN(cid::engine::live_slot_occupancy));
  m.impl("prefix_allocation_mask", TORCH_FN(cid::engine::prefix_allocation_mask));
  m.impl(
      "batched_linear_assignment",
      TORCH_FN(cid::engine::batched_linear_assignment));
  m.impl(
      "thought_corrupt_from_epsilon",
      TORCH_FN(cid::engine::thought_corrupt_from_epsilon));
  m.impl(
      "display_token_statistics",
      TORCH_FN(cid::engine::display_token_statistics));
  m.impl(
      "refine_display_from_statistics",
      TORCH_FN(cid::engine::refine_display_from_statistics));
  m.impl(
      "materialize_cell_snapshot",
      TORCH_FN(cid::engine::materialize_cell_snapshot));
}
