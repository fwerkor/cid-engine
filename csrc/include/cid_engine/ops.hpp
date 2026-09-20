#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <optional>
#include <tuple>

namespace cid::engine {

at::Tensor live_slot_occupancy(
    const at::Tensor& slot_occupancy,
    const std::optional<at::Tensor>& lifecycle_features,
    std::int64_t retired_index);

at::Tensor prefix_allocation_mask(
    const at::Tensor& occupancy,
    const at::Tensor& allocation_logits,
    double threshold,
    std::int64_t max_allocations);

at::Tensor batched_linear_assignment(
    const at::Tensor& costs,
    const at::Tensor& row_counts);

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
rollout_slot_transition(
    const at::Tensor& occupancy,
    const at::Tensor& allocation_logits,
    const at::Tensor& lifecycle_logits,
    const at::Tensor& revision_logits,
    const at::Tensor& input_lifecycle,
    double threshold,
    std::int64_t max_allocations,
    std::int64_t retired_index);

std::tuple<at::Tensor, at::Tensor, at::Tensor> thought_corrupt_from_epsilon(
    const at::Tensor& semantic,
    const at::Tensor& timesteps,
    const at::Tensor& occupancy,
    const at::Tensor& epsilon);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
display_corrupt_from_random(
    const at::Tensor& token_ids,
    const at::Tensor& timesteps,
    const at::Tensor& eligible_mask,
    const at::Tensor& corruption_random,
    const std::optional<at::Tensor>& replacement_random,
    const std::optional<at::Tensor>& replacement_offsets,
    std::int64_t mask_token_id,
    const std::optional<std::int64_t>& eos_token_id,
    std::int64_t vocab_size,
    double replacement_fraction);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
masked_diffusion_corrupt_from_random(
    const at::Tensor& clean_ids,
    const at::Tensor& ratio_random,
    const at::Tensor& mask_random,
    std::int64_t mask_token_id,
    double min_mask_ratio,
    double max_mask_ratio);

std::tuple<at::Tensor, at::Tensor, at::Tensor> display_token_statistics(
    const at::Tensor& token_ids,
    const at::Tensor& logits);

at::Tensor refine_display_from_statistics(
    const at::Tensor& token_ids,
    const at::Tensor& confidence,
    const at::Tensor& predicted,
    const at::Tensor& current_confidence,
    std::int64_t mask_token_id,
    const std::optional<std::int64_t>& eos_token_id,
    double reveal_fraction,
    double revision_fraction,
    double revision_margin);

at::Tensor materialize_cell_snapshot(
    const at::Tensor& thought_semantic,
    const at::Tensor& role_logits,
    const at::Tensor& uncertainty,
    const at::Tensor& noise_delta,
    const at::Tensor& lifecycle_logits,
    const at::Tensor& selected,
    const at::Tensor& semantic_indices);

}  // namespace cid::engine
