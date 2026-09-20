from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def checkpoint_fraction_for_budget(
    *,
    full_activation_bytes: int,
    available_activation_bytes: int,
) -> float:
    """Estimate the layer fraction that must be recomputed to fit an activation budget."""

    if full_activation_bytes <= 0:
        raise ValueError("full_activation_bytes must be positive")
    if available_activation_bytes < 0:
        raise ValueError("available_activation_bytes must be non-negative")
    if available_activation_bytes >= full_activation_bytes:
        return 0.0
    return min(1.0, max(0.0, 1.0 - available_activation_bytes / full_activation_bytes))


def select_checkpoint_layer_indices(
    layer_count: int, checkpoint_fraction: float
) -> tuple[int, ...]:
    """Select a deterministic, approximately uniform subset of transformer layers."""

    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if not math.isfinite(checkpoint_fraction) or not 0.0 <= checkpoint_fraction <= 1.0:
        raise ValueError("checkpoint_fraction must be in [0, 1]")
    count = min(layer_count, int(math.ceil(layer_count * checkpoint_fraction)))
    if count == 0:
        return ()
    if count == layer_count:
        return tuple(range(layer_count))
    return tuple(
        min(layer_count - 1, ((2 * index + 1) * layer_count) // (2 * count))
        for index in range(count)
    )


@dataclass(slots=True)
class _WrappedForward:
    module: nn.Module
    original_forward: Any


class SelectiveCheckpointController:
    """Install non-reentrant activation checkpointing on selected transformer layers."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        *,
        checkpoint_fraction: float,
    ) -> None:
        if not layers:
            raise ValueError("selective checkpointing requires at least one layer")
        self.checkpointed_indices = select_checkpoint_layer_indices(
            len(layers), checkpoint_fraction
        )
        self._wrapped: list[_WrappedForward] = []
        for index in self.checkpointed_indices:
            module = layers[index]
            original_forward = module.forward

            def checkpointed_forward(
                bound_module: nn.Module,
                *args: object,
                __original_forward=original_forward,
                **kwargs: object,
            ) -> object:
                del bound_module
                if not torch.is_grad_enabled():
                    return __original_forward(*args, **kwargs)
                return checkpoint(
                    __original_forward,
                    *args,
                    use_reentrant=False,
                    preserve_rng_state=True,
                    **kwargs,
                )

            module.forward = MethodType(checkpointed_forward, module)
            self._wrapped.append(
                _WrappedForward(module=module, original_forward=original_forward)
            )

    def restore(self) -> None:
        while self._wrapped:
            wrapped = self._wrapped.pop()
            wrapped.module.forward = wrapped.original_forward

    def __enter__(self) -> SelectiveCheckpointController:
        return self

    def __exit__(self, *_: object) -> None:
        self.restore()
