#include "cid_engine/ops.hpp"

#include <ATen/Functions.h>

#include <limits>

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


at::Tensor batched_linear_assignment(
    const at::Tensor& costs_input,
    const at::Tensor& row_counts_input) {
  TORCH_CHECK(costs_input.dim() == 3, "costs must have shape [batch, rows, columns]");
  TORCH_CHECK(costs_input.is_floating_point(), "costs must use a floating dtype");
  TORCH_CHECK(
      costs_input.size(1) <= costs_input.size(2),
      "assignment rows cannot exceed columns");
  TORCH_CHECK(
      costs_input.size(2) <= 16,
      "batched_linear_assignment supports at most 16 columns");
  TORCH_CHECK(
      row_counts_input.dim() == 1 &&
          row_counts_input.size(0) == costs_input.size(0),
      "row_counts must have shape [batch]");
  TORCH_CHECK(
      row_counts_input.scalar_type() == at::kLong,
      "row_counts must use torch.int64");

  const auto original_device = costs_input.device();
  auto costs = costs_input.detach().to(at::kCPU, at::kFloat).contiguous();
  auto row_counts = row_counts_input.detach().to(at::kCPU, at::kLong).contiguous();
  const auto batch = costs.size(0);
  const auto rows = costs.size(1);
  const auto columns = costs.size(2);
  auto output = at::full(
      {batch, rows},
      -1,
      at::TensorOptions().dtype(at::kLong).device(at::kCPU));

  const auto* cost_ptr = costs.const_data_ptr<float>();
  const auto* count_ptr = row_counts.const_data_ptr<std::int64_t>();
  auto* output_ptr = output.mutable_data_ptr<std::int64_t>();

  for (std::int64_t batch_index = 0; batch_index < batch; ++batch_index) {
    const auto active_rows = count_ptr[batch_index];
    TORCH_CHECK(
        active_rows >= 0 && active_rows <= rows,
        "row_counts values must be in [0, rows]");
    if (active_rows == 0) {
      continue;
    }

    double u[17] = {};
    double v[17] = {};
    std::int64_t p[17] = {};
    std::int64_t way[17] = {};
    const auto batch_offset = batch_index * rows * columns;

    for (std::int64_t row = 1; row <= active_rows; ++row) {
      p[0] = row;
      std::int64_t column0 = 0;
      double minimum[17];
      bool used[17] = {};
      for (std::int64_t column = 0; column <= columns; ++column) {
        minimum[column] = std::numeric_limits<double>::infinity();
      }

      do {
        used[column0] = true;
        const auto row0 = p[column0];
        double delta = std::numeric_limits<double>::infinity();
        std::int64_t column1 = 0;
        for (std::int64_t column = 1; column <= columns; ++column) {
          if (used[column]) {
            continue;
          }
          const auto cost_index =
              batch_offset + (row0 - 1) * columns + (column - 1);
          const double current =
              static_cast<double>(cost_ptr[cost_index]) - u[row0] - v[column];
          if (current < minimum[column]) {
            minimum[column] = current;
            way[column] = column0;
          }
          if (minimum[column] < delta) {
            delta = minimum[column];
            column1 = column;
          }
        }

        for (std::int64_t column = 0; column <= columns; ++column) {
          if (used[column]) {
            u[p[column]] += delta;
            v[column] -= delta;
          } else {
            minimum[column] -= delta;
          }
        }
        column0 = column1;
      } while (p[column0] != 0);

      do {
        const auto column1 = way[column0];
        p[column0] = p[column1];
        column0 = column1;
      } while (column0 != 0);
    }

    for (std::int64_t column = 1; column <= columns; ++column) {
      if (p[column] > 0 && p[column] <= active_rows) {
        output_ptr[batch_index * rows + (p[column] - 1)] = column - 1;
      }
    }
  }

  return original_device.is_cpu() ? output : output.to(original_device);
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
display_corrupt_from_random(
    const at::Tensor& token_ids,
    const at::Tensor& timesteps,
    const at::Tensor& eligible_mask,
    const at::Tensor& corruption_random,
    const std::optional<at::Tensor>& replacement_random,
    const std::optional<at::Tensor>& replacement_offsets,
    const std::int64_t mask_token_id,
    const std::optional<std::int64_t>& eos_token_id,
    const std::int64_t vocab_size,
    const double replacement_fraction) {
  TORCH_CHECK(token_ids.dim() == 2, "token_ids must have shape [batch, tokens]");
  TORCH_CHECK(token_ids.scalar_type() == at::kLong, "token_ids must use torch.int64");
  const auto batch = token_ids.size(0);
  const auto tokens = token_ids.size(1);
  TORCH_CHECK(
      timesteps.dim() == 1 && timesteps.size(0) == batch,
      "timesteps must have shape [batch]");
  TORCH_CHECK(
      eligible_mask.sizes() == token_ids.sizes(),
      "eligible_mask must match token_ids shape");
  TORCH_CHECK(
      corruption_random.sizes() == token_ids.sizes(),
      "corruption_random must match token_ids shape");
  TORCH_CHECK(
      timesteps.is_floating_point() && corruption_random.is_floating_point(),
      "timesteps and corruption_random must use floating dtypes");
  TORCH_CHECK(
      replacement_fraction >= 0.0 && replacement_fraction <= 1.0,
      "replacement_fraction must be in [0, 1]");

  const auto eligible = eligible_mask.to(at::kBool);
  const auto timestep_f32 = timesteps.to(at::kFloat);
  auto corrupted_positions =
      corruption_random.to(at::kFloat).lt(timestep_f32.unsqueeze(1));
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
  if (replacement_fraction > 0.0) {
    TORCH_CHECK(
        vocab_size >= 3,
        "visible replacement corruption requires vocab_size >= 3");
    TORCH_CHECK(
        replacement_random.has_value() && replacement_offsets.has_value(),
        "replacement random tensors are required when replacement_fraction > 0");
    TORCH_CHECK(
        replacement_random->sizes() == token_ids.sizes(),
        "replacement_random must match token_ids shape");
    TORCH_CHECK(
        replacement_offsets->sizes() == token_ids.sizes(),
        "replacement_offsets must match token_ids shape");
    TORCH_CHECK(
        replacement_random->is_floating_point(),
        "replacement_random must use a floating dtype");
    TORCH_CHECK(
        replacement_offsets->scalar_type() == at::kLong,
        "replacement_offsets must use torch.int64");

    replaced = corrupted_positions.logical_and(
        replacement_random->to(at::kFloat).lt(replacement_fraction));
    auto replacements =
        at::remainder(token_ids + *replacement_offsets, vocab_size);
    for (int iteration = 0; iteration < 3; ++iteration) {
      auto forbidden =
          replacements.eq(mask_token_id).logical_or(replacements.eq(token_ids));
      if (eos_token_id.has_value()) {
        forbidden.logical_or_(replacements.eq(*eos_token_id));
      }
      replacements = at::where(
          forbidden,
          at::remainder(replacements + 1, vocab_size),
          replacements);
    }
    masked = corrupted_positions.logical_and(replaced.logical_not());
    corrupted = token_ids.masked_fill(masked, mask_token_id);
    corrupted = at::where(replaced, replacements, corrupted);
  } else {
    replaced = at::zeros_like(corrupted_positions);
    masked = corrupted_positions;
    corrupted = token_ids.masked_fill(masked, mask_token_id);
  }

  auto labels = at::where(
      corrupted_positions,
      token_ids,
      at::full({}, -100, token_ids.options()));
  return {corrupted, labels, masked, replaced};
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
masked_diffusion_corrupt_from_random(
    const at::Tensor& clean_ids,
    const at::Tensor& ratio_random,
    const at::Tensor& mask_random,
    const std::int64_t mask_token_id,
    const double min_mask_ratio,
    const double max_mask_ratio) {
  TORCH_CHECK(clean_ids.dim() == 2, "clean_ids must have shape [batch, tokens]");
  TORCH_CHECK(clean_ids.scalar_type() == at::kLong, "clean_ids must use torch.int64");
  const auto batch = clean_ids.size(0);
  const auto tokens = clean_ids.size(1);
  TORCH_CHECK(batch > 0 && tokens > 0, "clean_ids dimensions must be non-empty");
  TORCH_CHECK(
      ratio_random.sizes() == at::IntArrayRef({batch, 1}),
      "ratio_random must have shape [batch, 1]");
  TORCH_CHECK(
      mask_random.sizes() == clean_ids.sizes(),
      "mask_random must match clean_ids shape");
  TORCH_CHECK(
      ratio_random.is_floating_point() && mask_random.is_floating_point(),
      "random tensors must use floating dtypes");
  TORCH_CHECK(
      min_mask_ratio > 0.0 &&
          min_mask_ratio <= max_mask_ratio &&
          max_mask_ratio <= 1.0,
      "mask ratio range must satisfy 0 < min <= max <= 1");

  auto ratio_f32 = ratio_random.to(at::kFloat);
  auto mask_random_f32 = mask_random.to(at::kFloat);
  auto mask_ratio =
      min_mask_ratio + (max_mask_ratio - min_mask_ratio) * ratio_f32;
  auto masked = mask_random_f32.lt(mask_ratio);

  // Conditional on an empty Bernoulli row, argmin(mask_random) is uniformly
  // distributed over positions by exchangeability.  It therefore preserves
  // the original uniform fallback semantics without a host sync or extra RNG.
  auto empty_rows = masked.any(1).logical_not();
  auto fallback_positions = std::get<1>(mask_random_f32.min(1));
  auto fallback = at::zeros_like(masked);
  fallback.scatter_(
      1,
      fallback_positions.unsqueeze(1),
      empty_rows.unsqueeze(1));
  masked.logical_or_(fallback);

  auto corrupted = clean_ids.masked_fill(masked, mask_token_id);
  auto metrics = at::stack(
      {masked.to(at::kFloat).mean(), mask_ratio.mean()});
  return {corrupted, masked, mask_ratio, metrics};
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



at::Tensor materialize_cell_snapshot(
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
  const auto hidden = thought_semantic.size(2);
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
  if (semantic_indices.numel() != 0) {
    auto invalid = semantic_indices.lt(0).logical_or(semantic_indices.ge(hidden));
    TORCH_CHECK(
        !invalid.any().item<bool>(),
        "semantic_indices contain values outside semantic width");
  }

  auto selected_f32 = selected.to(at::kFloat).unsqueeze(-1);
  auto lifecycle = std::get<1>(lifecycle_logits.to(at::kFloat).max(-1))
                       .to(at::kFloat)
                       .unsqueeze(-1);
  auto uncertainty_f32 = uncertainty.to(at::kFloat);
  auto noise_delta_f32 = noise_delta.to(at::kFloat);
  auto roles = role_logits.to(at::kFloat).sigmoid();
  auto indexes = semantic_indices.to(thought_semantic.device());
  auto semantic = thought_semantic.to(at::kFloat).index_select(-1, indexes);
  return at::cat(
      {selected_f32, lifecycle, uncertainty_f32, noise_delta_f32, roles, semantic},
      -1);
}

}  // namespace cid::engine
