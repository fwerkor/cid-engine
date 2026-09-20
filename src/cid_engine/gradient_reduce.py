from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.nn import Parameter


@dataclass(slots=True)
class _GradientBucket:
    parameters: tuple[Parameter, ...]
    names: tuple[str, ...]
    ready: set[int] = field(default_factory=set)
    work: Any | None = None
    launched: bool = False


class AsyncBucketedGradientReducer:
    """Launch deterministic gradient all-reduces while the final backward is running.

    Buckets are built in reverse parameter order, matching the usual transformer
    backward traversal.  Hooks may mark later buckets ready first, but collectives
    are launched only as a contiguous bucket prefix.  That fixed ordering keeps
    NCCL/Gloo collective sequences identical on every rank even when a parameter is
    unused on one rank.  ``finish`` fills missing gradients with zeros, launches the
    remaining buckets, waits for completion, and applies the world-size average.
    """

    def __init__(
        self,
        parameters: Iterable[tuple[str, Parameter]],
        *,
        bucket_cap_mb: float = 25.0,
        process_group: Any | None = None,
        average: bool = True,
    ) -> None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("async gradient reduction requires initialized distributed")
        if bucket_cap_mb <= 0:
            raise ValueError("bucket_cap_mb must be positive")
        named = tuple(
            (name, parameter) for name, parameter in parameters if parameter.requires_grad
        )
        if not named:
            raise ValueError("async gradient reduction requires trainable parameters")
        if any(not parameter.is_leaf for _, parameter in named):
            raise ValueError("async gradient reduction requires leaf parameters")

        self.process_group = process_group
        self.average = bool(average)
        self.world_size = torch.distributed.get_world_size(group=process_group)
        self.bucket_cap_bytes = max(1, int(bucket_cap_mb * 1024 * 1024))
        self._named = named
        self._buckets = self._build_buckets(named)
        self._parameter_bucket: dict[int, tuple[int, int]] = {}
        for bucket_index, bucket in enumerate(self._buckets):
            for parameter_index, parameter in enumerate(bucket.parameters):
                self._parameter_bucket[id(parameter)] = (bucket_index, parameter_index)
        self._hooks = [
            parameter.register_post_accumulate_grad_hook(self._make_hook(parameter))
            for _, parameter in named
        ]
        self._active = False
        self._next_launch = 0
        self.last_bucket_count = len(self._buckets)
        self.last_reduced_bytes = 0
        self.last_overlap_buckets = 0

    def _build_buckets(
        self, named: tuple[tuple[str, Parameter], ...]
    ) -> tuple[_GradientBucket, ...]:
        buckets: list[_GradientBucket] = []
        current_names: list[str] = []
        current_parameters: list[Parameter] = []
        current_bytes = 0
        current_key: tuple[torch.device, torch.dtype] | None = None

        def flush() -> None:
            nonlocal current_names, current_parameters, current_bytes, current_key
            if current_parameters:
                buckets.append(
                    _GradientBucket(tuple(current_parameters), tuple(current_names))
                )
            current_names = []
            current_parameters = []
            current_bytes = 0
            current_key = None

        # Leaf gradients in transformer-style networks are normally produced in
        # roughly reverse module/parameter order, so reverse the stable named list.
        for name, parameter in reversed(named):
            key = (parameter.device, parameter.dtype)
            size = parameter.numel() * parameter.element_size()
            if current_parameters and (
                key != current_key or current_bytes + size > self.bucket_cap_bytes
            ):
                flush()
            if size > self.bucket_cap_bytes:
                flush()
                buckets.append(_GradientBucket((parameter,), (name,)))
                continue
            if not current_parameters:
                current_key = key
            current_names.append(name)
            current_parameters.append(parameter)
            current_bytes += size
        flush()
        return tuple(buckets)

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)

    def _make_hook(self, parameter: Parameter):
        def ready(_: Tensor) -> None:
            if not self._active:
                return
            bucket_index, parameter_index = self._parameter_bucket[id(parameter)]
            bucket = self._buckets[bucket_index]
            bucket.ready.add(parameter_index)
            self._launch_ready_prefix(from_backward=True)

        return ready

    def start(self) -> None:
        if self._active:
            raise RuntimeError("async gradient reduction is already active")
        self._active = True
        self._next_launch = 0
        self.last_reduced_bytes = 0
        self.last_overlap_buckets = 0
        for bucket in self._buckets:
            bucket.ready.clear()
            bucket.work = None
            bucket.launched = False

    def _launch_bucket(self, bucket_index: int, *, from_backward: bool) -> None:
        bucket = self._buckets[bucket_index]
        gradients: list[Tensor] = []
        for parameter in bucket.parameters:
            gradient = parameter.grad
            if gradient is None:
                raise RuntimeError("cannot launch a gradient bucket before all gradients are ready")
            gradients.append(gradient)
            self.last_reduced_bytes += gradient.numel() * gradient.element_size()
        if len(gradients) == 1:
            bucket.work = torch.distributed.all_reduce(
                gradients[0],
                op=torch.distributed.ReduceOp.SUM,
                group=self.process_group,
                async_op=True,
            )
        else:
            bucket.work = torch.distributed.all_reduce_coalesced(
                gradients,
                op=torch.distributed.ReduceOp.SUM,
                group=self.process_group,
                async_op=True,
            )
        bucket.launched = True
        if from_backward:
            self.last_overlap_buckets += 1

    def _launch_ready_prefix(self, *, from_backward: bool) -> None:
        while self._next_launch < len(self._buckets):
            bucket = self._buckets[self._next_launch]
            if len(bucket.ready) != len(bucket.parameters):
                break
            self._launch_bucket(self._next_launch, from_backward=from_backward)
            self._next_launch += 1

    def finish(self) -> None:
        if not self._active:
            raise RuntimeError("async gradient reduction is not active")
        try:
            # Parameters absent from this backward still participate with an exact
            # zero contribution, preserving DDP semantics and collective order.
            for bucket_index in range(self._next_launch, len(self._buckets)):
                bucket = self._buckets[bucket_index]
                for parameter_index, parameter in enumerate(bucket.parameters):
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    bucket.ready.add(parameter_index)
                self._launch_ready_prefix(from_backward=False)

            for bucket in self._buckets:
                if not bucket.launched or bucket.work is None:
                    raise RuntimeError("gradient bucket was not launched")
                bucket.work.wait()
            if self.average and self.world_size > 1:
                scale = float(self.world_size)
                for _, parameter in self._named:
                    assert parameter.grad is not None
                    parameter.grad.div_(scale)
        finally:
            self._active = False

    def abort(self) -> None:
        """Stop hook-driven launching after an exceptional backward.

        Already launched collectives are waited so the process group is left in a
        usable state.  Gradients are not averaged because the optimizer must not step.
        """
        if not self._active:
            return
        for bucket in self._buckets:
            if bucket.work is not None:
                bucket.work.wait()
        self._active = False

    def close(self) -> None:
        self.abort()
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
