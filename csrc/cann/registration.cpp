#include "cid_engine/ops.hpp"

#include <torch/library.h>

TORCH_LIBRARY_IMPL(cid_engine, PrivateUse1, m) {
  m.impl("live_slot_occupancy", TORCH_FN(cid::engine::live_slot_occupancy));
  m.impl("prefix_allocation_mask", TORCH_FN(cid::engine::prefix_allocation_mask));
  m.impl(
      "batched_linear_assignment",
      TORCH_FN(cid::engine::batched_linear_assignment));
  m.impl(
      "rollout_slot_transition",
      TORCH_FN(cid::engine::rollout_slot_transition));
  m.impl(
      "thought_corrupt_from_epsilon",
      TORCH_FN(cid::engine::thought_corrupt_from_epsilon));
  m.impl(
      "display_corrupt_from_random",
      TORCH_FN(cid::engine::display_corrupt_from_random));
  m.impl(
      "masked_diffusion_corrupt_from_random",
      TORCH_FN(cid::engine::masked_diffusion_corrupt_from_random));
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
