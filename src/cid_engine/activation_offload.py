from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor


def _is_non_overlapping_layout(tensor: Tensor) -> bool:
    dimensions = [
        (abs(int(stride)), int(size))
        for size, stride in zip(tensor.size(), tensor.stride(), strict=True)
        if int(size) > 1
    ]
    dimensions.sort()
    storage_span = 1
    for stride, size in dimensions:
        if stride < storage_span:
            return False
        storage_span += (size - 1) * stride
    return True


@dataclass(slots=True)
class _PoolEntry:
    tensor: Tensor
    ready: torch.cuda.Event | None = None


class _PinnedTensorPool:
    def __init__(self) -> None:
        self._available: dict[
            tuple[torch.dtype, tuple[int, ...], tuple[int, ...]], list[_PoolEntry]
        ] = defaultdict(list)
        self.allocated_bytes = 0

    @staticmethod
    def _key(
        *,
        dtype: torch.dtype,
        size: tuple[int, ...],
        stride: tuple[int, ...],
    ) -> tuple[torch.dtype, tuple[int, ...], tuple[int, ...]]:
        return dtype, size, stride

    def acquire(self, source: Tensor) -> Tensor:
        size = tuple(source.size())
        stride = tuple(source.stride())
        key = self._key(dtype=source.dtype, size=size, stride=stride)
        available = self._available[key]
        for index in range(len(available) - 1, -1, -1):
            entry = available[index]
            if entry.ready is None or entry.ready.query():
                available.pop(index)
                return entry.tensor
        tensor = torch.empty_strided(
            size,
            stride,
            dtype=source.dtype,
            device="cpu",
            pin_memory=True,
        )
        self.allocated_bytes += tensor.numel() * tensor.element_size()
        return tensor

    def release(self, tensor: Tensor, ready: torch.cuda.Event | None = None) -> None:
        key = self._key(
            dtype=tensor.dtype,
            size=tuple(tensor.size()),
            stride=tuple(tensor.stride()),
        )
        self._available[key].append(_PoolEntry(tensor=tensor, ready=ready))


@dataclass(slots=True)
class _OffloadedActivation:
    cpu_tensor: Tensor
    device: torch.device
    d2h_ready: torch.cuda.Event
    index: int
    restored: Tensor | None = None
    h2d_ready: torch.cuda.Event | None = None
    reuse_ready: torch.cuda.Event | None = None
    layer_index: int | None = None


class AsyncPinnedActivationOffloader:
    """Offload saved CUDA activations through pinned host memory and prefetch backward use."""

    def __init__(
        self,
        device: torch.device | str,
        *,
        max_bytes: int,
        min_tensor_bytes: int = 1 << 20,
        prefetch_depth: int = 2,
        requires_grad_only: bool = False,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("activation offload requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if min_tensor_bytes <= 0:
            raise ValueError("min_tensor_bytes must be positive")
        if prefetch_depth <= 0:
            raise ValueError("prefetch_depth must be positive")
        self.max_bytes = int(max_bytes)
        self.min_tensor_bytes = int(min_tensor_bytes)
        self.prefetch_depth = int(prefetch_depth)
        self.requires_grad_only = bool(requires_grad_only)
        self.d2h_stream = torch.cuda.Stream(device=self.device)
        self.h2d_stream = torch.cuda.Stream(device=self.device)
        self._pool = _PinnedTensorPool()
        self.last_offloaded_bytes = 0
        self.last_offloaded_tensors = 0
        self.last_layer_prefetches = 0
        self._active_layer: int | None = None
        self._context_layer_handles: dict[int, list[_OffloadedActivation]] | None = None

    @property
    def pool_allocated_bytes(self) -> int:
        return self._pool.allocated_bytes

    def set_active_layer(self, layer_index: int | None) -> None:
        self._active_layer = layer_index

    def prefetch_layer(self, layer_index: int, *, depth: int = 1) -> None:
        if depth <= 0:
            raise ValueError("layer prefetch depth must be positive")
        layer_handles = self._context_layer_handles
        if layer_handles is None:
            return
        prefetched = False
        for candidate in range(layer_index, max(-1, layer_index - depth), -1):
            for handle in reversed(layer_handles.get(candidate, ())):
                if handle.restored is None:
                    self._prefetch(handle)
                    prefetched = True
        if prefetched:
            self.last_layer_prefetches += 1

    def _prefetch(self, handle: _OffloadedActivation) -> None:
        if handle.restored is not None:
            return
        restored = torch.empty_strided(
            tuple(handle.cpu_tensor.size()),
            tuple(handle.cpu_tensor.stride()),
            dtype=handle.cpu_tensor.dtype,
            device=handle.device,
        )
        with torch.cuda.stream(self.h2d_stream):
            self.h2d_stream.wait_event(handle.d2h_ready)
            restored.copy_(handle.cpu_tensor, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self.h2d_stream)
        handle.restored = restored
        handle.h2d_ready = ready

    @contextmanager
    def saved_tensors_context(
        self,
        *,
        excluded_storage_ptrs: Collection[int] = (),
    ) -> Iterator[None]:
        """Return saved-tensor hooks for one forward/backward region.

        The byte budget is per context. Tensors sharing parameter storage can be excluded
        so immutable model weights are never duplicated into host activation storage.
        """

        excluded = set(excluded_storage_ptrs)
        handles: list[_OffloadedActivation] = []
        layer_handles: dict[int, list[_OffloadedActivation]] = defaultdict(list)
        offloaded_bytes = 0
        if self._context_layer_handles is not None:
            raise RuntimeError("activation offloader contexts cannot be nested")
        self._context_layer_handles = layer_handles
        self.last_layer_prefetches = 0

        def pack(tensor: Tensor) -> object:
            nonlocal offloaded_bytes
            if tensor.device != self.device:
                return tensor
            if self.requires_grad_only and not tensor.requires_grad:
                return tensor
            if tensor.untyped_storage().data_ptr() in excluded:
                return tensor
            if not _is_non_overlapping_layout(tensor):
                return tensor
            tensor_bytes = tensor.numel() * tensor.element_size()
            if tensor_bytes < self.min_tensor_bytes:
                return tensor
            if offloaded_bytes + tensor_bytes > self.max_bytes:
                return tensor

            cpu_tensor = self._pool.acquire(tensor)
            current = torch.cuda.current_stream(self.device)
            self.d2h_stream.wait_stream(current)
            with torch.cuda.stream(self.d2h_stream):
                cpu_tensor.copy_(tensor.detach(), non_blocking=True)
                tensor.record_stream(self.d2h_stream)
                ready = torch.cuda.Event()
                ready.record(self.d2h_stream)
            handle = _OffloadedActivation(
                cpu_tensor=cpu_tensor,
                device=tensor.device,
                d2h_ready=ready,
                index=len(handles),
                layer_index=self._active_layer,
            )
            handles.append(handle)
            if handle.layer_index is not None:
                layer_handles[handle.layer_index].append(handle)
            offloaded_bytes += tensor_bytes
            return handle

        def unpack(packed: object) -> Tensor:
            if not isinstance(packed, _OffloadedActivation):
                if not isinstance(packed, Tensor):
                    raise TypeError("saved tensor hook received an unsupported packed value")
                return packed

            if packed.layer_index is not None:
                self.prefetch_layer(packed.layer_index, depth=self.prefetch_depth)
            stop = max(-1, packed.index - self.prefetch_depth)
            for index in range(packed.index, stop, -1):
                self._prefetch(handles[index])
            assert packed.restored is not None
            assert packed.h2d_ready is not None
            current = torch.cuda.current_stream(self.device)
            current.wait_event(packed.h2d_ready)
            restored = packed.restored
            restored.record_stream(current)
            packed.reuse_ready = packed.h2d_ready
            packed.restored = None
            packed.h2d_ready = None
            return restored

        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                yield
        finally:
            self.last_offloaded_bytes = offloaded_bytes
            self.last_offloaded_tensors = len(handles)
            self._context_layer_handles = None
            self._active_layer = None
            for handle in handles:
                pending = handle.reuse_ready or handle.h2d_ready or handle.d2h_ready
                handle.restored = None
                handle.h2d_ready = None
                handle.reuse_ready = None
                self._pool.release(handle.cpu_tensor, pending)


class LayerActivationPrefetchController:
    """Tag transformer-layer activations and prefetch them before layer backward."""

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        offloader: AsyncPinnedActivationOffloader,
        *,
        prefetch_layers: int = 2,
    ) -> None:
        if not layers:
            raise ValueError("layer activation prefetch requires at least one layer")
        if prefetch_layers <= 0:
            raise ValueError("prefetch_layers must be positive")
        self.layers = tuple(layers)
        self.offloader = offloader
        self.prefetch_layers = int(prefetch_layers)
        self._handles: list[object] = []
        for index, layer in enumerate(self.layers):
            self._handles.append(
                layer.register_forward_pre_hook(self._forward_pre_hook(index))
            )
            self._handles.append(layer.register_forward_hook(self._forward_post_hook))
            self._handles.append(
                layer.register_full_backward_pre_hook(self._backward_pre_hook(index))
            )

    def _forward_pre_hook(self, index: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[Tensor, ...]) -> None:
            self.offloader.set_active_layer(index)

        return hook

    def _forward_post_hook(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[Tensor, ...],
        output: object,
    ) -> object:
        self.offloader.set_active_layer(None)
        return output

    def _backward_pre_hook(self, index: int):
        def hook(_module: torch.nn.Module, _grad_output: tuple[Tensor | None, ...]) -> None:
            self.offloader.prefetch_layer(index, depth=self.prefetch_layers)

        return hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
