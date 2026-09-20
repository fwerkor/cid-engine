from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import Parameter


@dataclass(slots=True)
class _GradientBuffers:
    accumulator: Tensor
    staging: Tensor
    initialized: bool = False


class AsyncPinnedGradientAccumulator:
    """Accumulate CUDA gradients in reusable pinned host buffers.

    D2H copies run on a dedicated CUDA stream. After the first contribution for
    a parameter, later contributions land in a staging buffer and are folded
    into the persistent host accumulator on one background CPU worker. The
    staging buffer is reused only after that add completes, bounding host memory
    at two pinned gradient buffers per parameter.
    """

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("async pinned gradient accumulation requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self._buffers: dict[str, _GradientBuffers] = {}
        self._pending_add: Future[None] | None = None
        self._pending_copy: torch.cuda.Event | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="cid-gradient-stash",
        )
        self._closed = False

    @property
    def has_data(self) -> bool:
        return any(buffers.initialized for buffers in self._buffers.values())

    @property
    def allocated_bytes(self) -> int:
        return sum(
            (buffers.accumulator.numel() * buffers.accumulator.element_size()) * 2
            for buffers in self._buffers.values()
        )

    def _wait_pending_add(self) -> None:
        pending = self._pending_add
        if pending is None:
            return
        pending.result()
        self._pending_add = None

    def _buffers_for(self, name: str, gradient: Tensor) -> _GradientBuffers:
        buffers = self._buffers.get(name)
        if buffers is not None:
            if (
                buffers.accumulator.shape != gradient.shape
                or buffers.accumulator.dtype != gradient.dtype
            ):
                raise RuntimeError(f"gradient geometry changed for {name}")
            return buffers

        accumulator = torch.empty(
            gradient.shape,
            dtype=gradient.dtype,
            device="cpu",
            pin_memory=True,
        )
        staging = torch.empty_like(accumulator, pin_memory=True)
        buffers = _GradientBuffers(accumulator=accumulator, staging=staging)
        self._buffers[name] = buffers
        return buffers

    def _accumulate_staging(
        self,
        ready: torch.cuda.Event,
        names: tuple[str, ...],
    ) -> None:
        ready.synchronize()
        for name in names:
            buffers = self._buffers[name]
            buffers.accumulator.add_(buffers.staging)

    def stash(self, parameters: Iterable[tuple[str, Parameter]]) -> None:
        if self._closed:
            raise RuntimeError("gradient accumulator is closed")
        self._wait_pending_add()

        current = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current)
        staged_names: list[str] = []
        copied = False
        with torch.cuda.stream(self.copy_stream):
            for name, parameter in parameters:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.device != self.device:
                    raise ValueError(
                        f"gradient for {name} is on {gradient.device}, expected {self.device}"
                    )
                buffers = self._buffers_for(name, gradient)
                target = buffers.staging if buffers.initialized else buffers.accumulator
                target.copy_(gradient.detach(), non_blocking=True)
                gradient.record_stream(self.copy_stream)
                parameter.grad = None
                copied = True
                if buffers.initialized:
                    staged_names.append(name)
                else:
                    buffers.initialized = True

            if not copied:
                return
            ready = torch.cuda.Event()
            ready.record(self.copy_stream)

        self._pending_copy = ready
        if staged_names:
            self._pending_add = self._executor.submit(
                self._accumulate_staging,
                ready,
                tuple(staged_names),
            )

    def flush(self) -> None:
        self._wait_pending_add()
        pending_copy = self._pending_copy
        if pending_copy is not None:
            pending_copy.synchronize()
            self._pending_copy = None

    def restore(self, parameters: Iterable[tuple[str, Parameter]]) -> None:
        if not self.has_data:
            return
        self.flush()
        parameter_map = dict(parameters)
        current = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current)

        restored: list[Tensor] = []
        with torch.cuda.stream(self.copy_stream):
            for name, buffers in self._buffers.items():
                if not buffers.initialized:
                    continue
                parameter = parameter_map[name]
                gradient = torch.empty_like(parameter, device=self.device)
                gradient.copy_(buffers.accumulator, non_blocking=True)
                parameter.grad = gradient
                restored.append(gradient)
            ready = torch.cuda.Event()
            ready.record(self.copy_stream)

        current.wait_event(ready)
        for gradient in restored:
            gradient.record_stream(current)
        for buffers in self._buffers.values():
            buffers.initialized = False
        self._pending_copy = ready

    def snapshot(self) -> dict[str, Tensor]:
        self.flush()
        return {
            name: buffers.accumulator.clone()
            for name, buffers in self._buffers.items()
            if buffers.initialized
        }

    def clear(self) -> None:
        self.flush()
        for buffers in self._buffers.values():
            buffers.initialized = False

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._executor.shutdown(wait=True)
        self._closed = True
