from __future__ import annotations

import torch
from torch import Tensor

from cid_engine import _C  # noqa: F401

__version__ = "0.4.2"
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
    "CUDA_BACKEND_BUILT",
    "batched_linear_assignment",
    "display_token_statistics",
    "live_slot_occupancy",
    "materialize_cell_snapshot",
    "prefix_allocation_mask",
    "refine_display_from_statistics",
    "thought_corrupt_from_epsilon",
]
