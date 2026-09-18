from __future__ import annotations

import torch
from torch import Tensor

from cid_engine import _C  # noqa: F401

__version__ = "0.2.0"


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


__all__ = [
    "display_token_statistics",
    "live_slot_occupancy",
    "prefix_allocation_mask",
    "thought_corrupt_from_epsilon",
]
