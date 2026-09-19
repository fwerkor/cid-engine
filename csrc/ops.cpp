#include "cid_engine/ops.hpp"

#include <ATen/Functions.h>

namespace cid::engine {

at::Tensor live_slot_occupancy(
    const at::Tensor& slot_occupancy,
    const std::optional<at::Tensor>& lifecycle_features,
    const std::int64_t retired_index) {
  TORCH_CHECK(
      slot_occupancy.dim() == 3 && slot_occupancy.size(-1) == 1,
      "slot_occupancy must have shape [batch, slots, 1]");

  auto occupancy = slot_occupancy.clamp(0.0, 1.0);
  if (!lifecycle_features.has_value()) {
    return occupancy;
  }

  const auto& lifecycle = *lifecycle_features;
  TORCH_CHECK(
      lifecycle.dim() == 3 &&
          lifecycle.size(0) == slot_occupancy.size(0) &&
          lifecycle.size(1) == slot_occupancy.size(1),
      "lifecycle_features must match batch and slot dimensions");
  TORCH_CHECK(
      retired_index >= 0 && retired_index < lifecycle.size(-1),
      "retired_index is outside lifecycle_features");

  auto retired = lifecycle
                     .slice(-1, retired_index, retired_index + 1)
                     .clamp(0.0, 1.0);
  return occupancy * (1.0 - retired);
}

at::Tensor prefix_allocation_mask(
    const at::Tensor& occupancy_input,
    const at::Tensor& allocation_logits,
    const double threshold,
    const std::int64_t max_allocations) {
  TORCH_CHECK(
      occupancy_input.dim() == 2 ||
          (occupancy_input.dim() == 3 && occupancy_input.size(-1) == 1),
      "occupancy must have shape [batch, slots] or [batch, slots, 1]");
  TORCH_CHECK(threshold >= 0.0 && threshold <= 1.0, "threshold must be in [0, 1]");
  TORCH_CHECK(max_allocations > 0, "max_allocations must be positive");

  auto occupancy = occupancy_input.dim() == 3 ? occupancy_input.squeeze(-1) : occupancy_input;
  TORCH_CHECK(
      allocation_logits.dim() == 2 && allocation_logits.sizes() == occupancy.sizes(),
      "occupancy and allocation_logits must have matching shapes");

  auto occupied = occupancy.to(at::kBool);
  auto eligible = allocation_logits.to(at::kFloat).sigmoid().ge(threshold);
  auto free = occupied.logical_not();
  auto blocked = (free.logical_and(eligible.logical_not())).cumsum(1).gt(0);
  auto selected = free.logical_and(eligible).logical_and(blocked.logical_not());
  auto allocation_rank = selected.cumsum(1);
  return selected.logical_and(allocation_rank.le(max_allocations));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> thought_corrupt_from_epsilon(
    const at::Tensor& semantic,
    const at::Tensor& timesteps,
    const at::Tensor& occupancy,
    const at::Tensor& epsilon) {
  TORCH_CHECK(semantic.dim() == 3, "semantic must have shape [batch, slots, hidden]");
  TORCH_CHECK(
      timesteps.dim() == 2 &&
          timesteps.size(0) == semantic.size(0) &&
          timesteps.size(1) == semantic.size(1),
      "timesteps must have shape [batch, slots]");
  TORCH_CHECK(
      occupancy.dim() == 3 &&
          occupancy.size(0) == semantic.size(0) &&
          occupancy.size(1) == semantic.size(1) &&
          occupancy.size(2) == 1,
      "occupancy must have shape [batch, slots, 1]");
  TORCH_CHECK(epsilon.sizes() == semantic.sizes(), "epsilon must match semantic");

  constexpr double half_pi = 1.57079632679489661923;
  auto alpha = at::cos(timesteps * half_pi)
                   .square()
                   .to(semantic.scalar_type())
                   .unsqueeze(-1);
  auto corrupted = alpha.sqrt() * semantic + (1.0 - alpha).sqrt() * epsilon;
  auto occupied = occupancy.to(at::kBool);
  auto zeros = at::zeros({}, semantic.options());
  corrupted = at::where(occupied, corrupted, zeros);
  auto masked_epsilon = at::where(occupied, epsilon, zeros);
  auto local_noise = timesteps.to(semantic.scalar_type()).unsqueeze(-1);
  local_noise = at::where(occupied, local_noise, zeros);
  return {corrupted, local_noise, masked_epsilon};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> display_token_statistics(
    const at::Tensor& token_ids,
    const at::Tensor& logits) {
  TORCH_CHECK(token_ids.dim() == 2, "token_ids must have shape [batch, tokens]");
  TORCH_CHECK(token_ids.scalar_type() == at::kLong, "token_ids must use torch.int64");
  TORCH_CHECK(
      logits.dim() == 3 &&
          logits.size(0) == token_ids.size(0) &&
          logits.size(1) == token_ids.size(1),
      "logits must have shape [batch, tokens, vocab]");
  TORCH_CHECK(logits.size(-1) > 0, "logits vocabulary dimension must be non-empty");

  auto logits_f32 = logits.to(at::kFloat);
  auto max_result = logits_f32.max(-1);
  auto max_logits = std::get<0>(max_result);
  auto predicted = std::get<1>(max_result);
  // Normalize after subtracting the row maximum.  Keeping a large common logit
  // offset out of logsumexp avoids catastrophic cancellation when the final
  // confidence is reconstructed (for example logits near +1e4).
  auto shifted_logits = logits_f32 - max_logits.unsqueeze(-1);
  auto log_normalizer = at::logsumexp(shifted_logits, {-1});
  auto confidence = at::exp(-log_normalizer);
  auto current_logits = logits_f32
                            .gather(-1, token_ids.unsqueeze(-1))
                            .squeeze(-1);
  auto current_confidence =
      at::exp(current_logits - max_logits - log_normalizer);
  return {confidence, predicted, current_confidence};
}


}  // namespace cid::engine
