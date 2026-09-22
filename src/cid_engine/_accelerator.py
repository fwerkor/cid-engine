from __future__ import annotations

import importlib.util
from typing import Any

import torch


def load_npu_backend() -> bool:
    if hasattr(torch, "npu"):
        return True
    if importlib.util.find_spec("torch_npu") is None:
        return False
    import torch_npu  # noqa: F401

    return hasattr(torch, "npu")


def accelerator_for_device(device: torch.device | str) -> Any:
    if str(device).startswith("npu"):
        load_npu_backend()
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        return torch.cuda
    if resolved.type == "npu":
        if not load_npu_backend() or not torch.npu.is_available():
            raise RuntimeError("CANN/NPU is unavailable")
        return torch.npu
    raise ValueError("accelerator device must be CUDA or NPU")


def synchronize(device: torch.device | str) -> None:
    if str(device).startswith("npu"):
        load_npu_backend()
    resolved = torch.device(device)
    if resolved.type in {"cuda", "npu"}:
        accelerator_for_device(resolved).synchronize(resolved)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if load_npu_backend() and torch.npu.is_available():
        return torch.device("npu")
    return torch.device("cpu")