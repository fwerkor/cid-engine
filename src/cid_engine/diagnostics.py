from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from cid_engine._accelerator import accelerator_for_device


@dataclass(frozen=True, slots=True)
class AttentionBackendReport:
    backend: str
    events: tuple[str, ...]


def classify_attention_backend(events: tuple[str, ...] | list[str] | set[str]) -> str:
    names = tuple(str(event).lower() for event in events)
    if any(
        "aclnnflashattentionscore" in name or "npu_fusion_attention" in name
        for name in names
    ):
        return "npu-fused"
    if any("flash_attention" in name for name in names):
        return "flash"
    if any("efficient_attention" in name for name in names):
        return "memory-efficient"
    if any("cudnn_attention" in name for name in names):
        return "cudnn"
    if any("scaled_dot_product_attention_math" in name for name in names):
        return "math"
    if any("scaled_dot_product_attention" in name for name in names):
        return "sdpa-unknown"
    if any("softmax" in name for name in names) and any("matmul" in name for name in names):
        return "eager"
    return "unknown"


def profile_attention_backend(
    fn: Callable[[], Any],
    *,
    device: torch.device | str | None = None,
    warmup: int = 1,
) -> AttentionBackendReport:
    """Run ``fn`` once under torch.profiler and identify its attention backend.

    This is intentionally a diagnostic helper rather than a permanent profiler in the
    training loop.  It is suitable for validating that an actual model path reaches
    Flash SDPA instead of silently falling back to math/eager attention.
    """

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    resolved = torch.device(device) if device is not None else None
    for _ in range(warmup):
        fn()
    if resolved is not None and resolved.type in {"cuda", "npu"}:
        accelerator_for_device(resolved).synchronize(resolved)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if resolved is not None and resolved.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    elif resolved is not None and resolved.type == "npu":
        npu_activity = getattr(torch.profiler.ProfilerActivity, "NPU", None)
        if npu_activity is not None:
            activities.append(npu_activity)
    with torch.profiler.profile(activities=activities) as profiler:
        fn()
    if resolved is not None and resolved.type in {"cuda", "npu"}:
        accelerator_for_device(resolved).synchronize(resolved)

    events = tuple(sorted({event.key for event in profiler.key_averages()}))
    return AttentionBackendReport(
        backend=classify_attention_backend(events),
        events=events,
    )
