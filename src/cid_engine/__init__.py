from __future__ import annotations

import torch
from torch import Tensor

from cid_engine import _C  # noqa: F401
from cid_engine.activation_offload import (
    AsyncPinnedActivationOffloader,
    LayerActivationPrefetchController,
)
from cid_engine.checkpointing import (
    SelectiveCheckpointController,
    checkpoint_fraction_for_budget,
    select_checkpoint_layer_indices,
)
from cid_engine.frozen_shard import shard_frozen_transformer
from cid_engine.gradient_reduce import AsyncBucketedGradientReducer
from cid_engine.gradient_stash import AsyncPinnedGradientAccumulator

__version__ = "0.7.1"
CUDA_BACKEND_BUILT = bool(_C.cuda_backend_built)


def live_slot_occupancy(
    slot_occupancy: Tensor,
    lifecycle_features: Tensor | None,
    *,
    retired_index: int,
) -> Tensor:
    return torch.ops.cid_engine.live_slot_occupancy(
        slot_occupancy,
        lifecycle_features,
        retired_index,
    )


def prefix_allocation_mask(
    occupancy: Tensor,
    allocation_logits: Tensor,
    *,
    threshold: float,
    max_allocations: int,
) -> Tensor:
    return torch.ops.cid_engine.prefix_allocation_mask(
        occupancy,
        allocation_logits,
        threshold,
        max_allocations,
    )


def thought_corrupt_from_epsilon(
    semantic: Tensor,
    timesteps: Tensor,
    occupancy: Tensor,
    epsilon: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    return torch.ops.cid_engine.thought_corrupt_from_epsilon(
        semantic,
        timesteps,
        occupancy,
        epsilon,
    )


def display_corrupt_from_random(
    token_ids: Tensor,
    timesteps: Tensor,
    eligible_mask: Tensor,
    corruption_random: Tensor,
    replacement_random: Tensor | None,
    replacement_offsets: Tensor | None,
    *,
    mask_token_id: int,
    eos_token_id: int | None,
    vocab_size: int,
    replacement_fraction: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return torch.ops.cid_engine.display_corrupt_from_random(
        token_ids,
        timesteps,
        eligible_mask,
        corruption_random,
        replacement_random,
        replacement_offsets,
        mask_token_id,
        eos_token_id,
        vocab_size,
        replacement_fraction,
    )


def masked_diffusion_corrupt_from_random(
    clean_ids: Tensor,
    ratio_random: Tensor,
    mask_random: Tensor,
    *,
    mask_token_id: int,
    min_mask_ratio: float,
    max_mask_ratio: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return torch.ops.cid_engine.masked_diffusion_corrupt_from_random(
        clean_ids,
        ratio_random,
        mask_random,
        mask_token_id,
        min_mask_ratio,
        max_mask_ratio,
    )


def display_token_statistics(
    token_ids: Tensor,
    logits: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    return torch.ops.cid_engine.display_token_statistics(token_ids, logits)


def batched_linear_assignment(
    costs: Tensor,
    row_counts: Tensor,
) -> Tensor:
    return torch.ops.cid_engine.batched_linear_assignment(costs, row_counts)


def rollout_slot_transition(
    occupancy: Tensor,
    allocation_logits: Tensor,
    lifecycle_logits: Tensor,
    revision_logits: Tensor,
    input_lifecycle: Tensor,
    *,
    threshold: float,
    max_allocations: int,
    retired_index: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return torch.ops.cid_engine.rollout_slot_transition(
        occupancy,
        allocation_logits,
        lifecycle_logits,
        revision_logits,
        input_lifecycle,
        threshold,
        max_allocations,
        retired_index,
    )


def materialize_cell_snapshot(
    thought_semantic: Tensor,
    role_logits: Tensor,
    uncertainty: Tensor,
    noise_delta: Tensor,
    lifecycle_logits: Tensor,
    selected: Tensor,
    semantic_indices: Tensor,
) -> Tensor:
    return torch.ops.cid_engine.materialize_cell_snapshot(
        thought_semantic,
        role_logits,
        uncertainty,
        noise_delta,
        lifecycle_logits,
        selected,
        semantic_indices,
    )


def refine_display_from_statistics(
    token_ids: Tensor,
    confidence: Tensor,
    predicted: Tensor,
    current_confidence: Tensor,
    *,
    mask_token_id: int,
    eos_token_id: int | None,
    reveal_fraction: float,
    revision_fraction: float,
    revision_margin: float,
) -> Tensor:
    return torch.ops.cid_engine.refine_display_from_statistics(
        token_ids,
        confidence,
        predicted,
        current_confidence,
        mask_token_id,
        eos_token_id,
        reveal_fraction,
        revision_fraction,
        revision_margin,
    )


__all__ = [
    "AsyncBucketedGradientReducer",
    "AsyncPinnedActivationOffloader",
    "AsyncPinnedGradientAccumulator",
    "LayerActivationPrefetchController",
    "SelectiveCheckpointController",
    "CUDA_BACKEND_BUILT",
    "batched_linear_assignment",
    "checkpoint_fraction_for_budget",
    "display_corrupt_from_random",
    "display_token_statistics",
    "live_slot_occupancy",
    "materialize_cell_snapshot",
    "masked_diffusion_corrupt_from_random",
    "prefix_allocation_mask",
    "refine_display_from_statistics",
    "rollout_slot_transition",
    "select_checkpoint_layer_indices",
    "shard_frozen_transformer",
    "thought_corrupt_from_epsilon",
]
