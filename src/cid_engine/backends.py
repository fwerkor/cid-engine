from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

from torch import Tensor

from cid_engine import ops, reference

LiveSlotOp = Callable[[Tensor, Tensor | None, int], Tensor]
AllocationOp = Callable[[Tensor, Tensor, float, int], Tensor]
ThoughtCorruptionOp = Callable[
    [Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor],
]
DisplayStatisticsOp = Callable[[Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]


@dataclass(frozen=True)
class BackendOps:
    live_slot_occupancy: LiveSlotOp
    prefix_allocation_mask: AllocationOp
    thought_corrupt_from_epsilon: ThoughtCorruptionOp
    display_token_statistics: DisplayStatisticsOp


@cache
def get_backend(name: str) -> BackendOps:
    if name == "reference":
        return BackendOps(
            live_slot_occupancy=reference.live_slot_occupancy,
            prefix_allocation_mask=reference.prefix_allocation_mask,
            thought_corrupt_from_epsilon=reference.thought_corrupt_from_epsilon,
            display_token_statistics=reference.display_token_statistics,
        )
    if name == "torch":
        return BackendOps(
            live_slot_occupancy=ops.live_slot_occupancy,
            prefix_allocation_mask=ops.prefix_allocation_mask,
            thought_corrupt_from_epsilon=ops.thought_corrupt_from_epsilon,
            display_token_statistics=ops.display_token_statistics,
        )
    if name == "compile":
        import torch

        return BackendOps(
            live_slot_occupancy=torch.compile(ops.live_slot_occupancy, fullgraph=True),
            prefix_allocation_mask=torch.compile(ops.prefix_allocation_mask, fullgraph=True),
            thought_corrupt_from_epsilon=torch.compile(
                ops.thought_corrupt_from_epsilon,
                fullgraph=True,
            ),
            display_token_statistics=torch.compile(
                ops.display_token_statistics,
                fullgraph=True,
            ),
        )
    raise ValueError(f"unknown backend: {name!r}")
