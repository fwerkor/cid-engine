from __future__ import annotations

import torch
from torch import Tensor

from cid_engine.backends import get_backend


class CIDEngine:
    def __init__(self, backend: str = "torch") -> None:
        self.backend_name = backend
        self._ops = get_backend(backend)

    def live_slot_occupancy(
        self,
        slot_occupancy: Tensor,
        lifecycle_features: Tensor | None,
        *,
        retired_index: int,
    ) -> Tensor:
        if slot_occupancy.ndim != 3 or slot_occupancy.shape[-1] != 1:
            raise ValueError("slot_occupancy must have shape [batch, slots, 1]")
        if lifecycle_features is not None:
            if lifecycle_features.shape[:2] != slot_occupancy.shape[:2]:
                raise ValueError("lifecycle_features must match batch and slot dimensions")
            if not 0 <= retired_index < lifecycle_features.shape[-1]:
                raise ValueError("retired_index is outside lifecycle_features")
        return self._ops.live_slot_occupancy(
            slot_occupancy,
            lifecycle_features,
            retired_index,
        )

    def prefix_allocation_mask(
        self,
        occupancy: Tensor,
        allocation_logits: Tensor,
        *,
        threshold: float,
        max_allocations: int,
    ) -> Tensor:
        if occupancy.ndim == 3:
            if occupancy.shape[-1] != 1:
                raise ValueError(
                    "occupancy must have shape [batch, slots] or [batch, slots, 1]"
                )
            occupancy_shape = occupancy.shape[:2]
        elif occupancy.ndim == 2:
            occupancy_shape = occupancy.shape
        else:
            raise ValueError(
                "occupancy must have shape [batch, slots] or [batch, slots, 1]"
            )
        if allocation_logits.shape != occupancy_shape:
            raise ValueError("occupancy and allocation_logits must have matching shapes")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if max_allocations <= 0:
            raise ValueError("max_allocations must be positive")
        return self._ops.prefix_allocation_mask(
            occupancy,
            allocation_logits,
            threshold,
            max_allocations,
        )

    def thought_corrupt_from_epsilon(
        self,
        semantic: Tensor,
        timesteps: Tensor,
        occupancy: Tensor,
        epsilon: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if semantic.ndim != 3:
            raise ValueError("semantic must have shape [batch, slots, hidden]")
        if timesteps.shape != semantic.shape[:2]:
            raise ValueError("timesteps must have shape [batch, slots]")
        if occupancy.shape != (*semantic.shape[:2], 1):
            raise ValueError("occupancy must have shape [batch, slots, 1]")
        if epsilon.shape != semantic.shape:
            raise ValueError("epsilon must match semantic")
        return self._ops.thought_corrupt_from_epsilon(
            semantic,
            timesteps,
            occupancy,
            epsilon,
        )

    def display_token_statistics(
        self,
        token_ids: Tensor,
        logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, tokens]")
        if logits.ndim != 3 or logits.shape[:2] != token_ids.shape:
            raise ValueError("logits must have shape [batch, tokens, vocab]")
        if logits.shape[-1] <= 0:
            raise ValueError("logits vocabulary dimension must be non-empty")
        if token_ids.dtype != torch.long:
            raise ValueError("token_ids must use torch.int64")
        if bool(((token_ids < 0) | (token_ids >= logits.shape[-1])).any()):
            raise ValueError("token_ids contain values outside the logits vocabulary")
        return self._ops.display_token_statistics(token_ids, logits)
