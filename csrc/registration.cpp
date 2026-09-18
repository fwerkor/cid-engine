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
      "thought_corrupt_from_epsilon(Tensor semantic, Tensor timesteps, "
      "Tensor occupancy, Tensor epsilon) -> (Tensor, Tensor, Tensor)");
  m.def(
      "display_token_statistics(Tensor token_ids, Tensor logits) "
      "-> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(cid_engine, CompositeImplicitAutograd, m) {
  m.impl("live_slot_occupancy", TORCH_FN(cid::engine::live_slot_occupancy));
  m.impl("prefix_allocation_mask", TORCH_FN(cid::engine::prefix_allocation_mask));
  m.impl(
      "thought_corrupt_from_epsilon",
      TORCH_FN(cid::engine::thought_corrupt_from_epsilon));
  m.impl(
      "display_token_statistics",
      TORCH_FN(cid::engine::display_token_statistics));
}
