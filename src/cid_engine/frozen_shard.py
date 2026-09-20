from __future__ import annotations

from collections.abc import Iterable
from functools import partial

import torch
from torch import nn


def shard_frozen_transformer(
    module: nn.Module,
    *,
    transformer_layer_cls: type[nn.Module],
    device_id: int | torch.device,
    ignored_modules: Iterable[nn.Module] = (),
    compute_dtype: torch.dtype | None = None,
    process_group: object | None = None,
    aggressive_prefetch: bool = True,
) -> nn.Module:
    """FULL_SHARD a frozen transformer while preserving input gradients.

    Stage-A style training keeps the backbone frozen but still differentiates through it
    into trainable adapters. FSDP can shard those immutable parameters and materialize
    one transformer block at a time. Embedding modules that are read directly outside
    the decoder forward should be passed through ``ignored_modules`` so their parameters
    remain resident and safe to access.
    """

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("frozen transformer sharding requires initialized torch.distributed")
    if not isinstance(transformer_layer_cls, type) or not issubclass(
        transformer_layer_cls, nn.Module
    ):
        raise TypeError("transformer_layer_cls must be an nn.Module type")

    ignored = tuple(ignored_modules)
    ignored_parameter_ids = {
        id(parameter)
        for ignored_module in ignored
        for parameter in ignored_module.parameters(recurse=True)
    }
    trainable = [
        name
        for name, parameter in module.named_parameters()
        if id(parameter) not in ignored_parameter_ids and parameter.requires_grad
    ]
    if trainable:
        sample = ", ".join(trainable[:3])
        raise ValueError(
            f"sharded transformer parameters must be frozen; found trainable: {sample}"
        )

    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullyShardedDataParallel,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_cls},
    )
    mixed_precision = (
        None
        if compute_dtype is None
        else MixedPrecision(
            param_dtype=compute_dtype,
            reduce_dtype=compute_dtype,
            buffer_dtype=compute_dtype,
        )
    )
    return FullyShardedDataParallel(
        module,
        auto_wrap_policy=auto_wrap_policy,
        ignored_modules=list(ignored) or None,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_id=device_id,
        process_group=process_group,
        sync_module_states=False,
        backward_prefetch=(
            BackwardPrefetch.BACKWARD_PRE
            if aggressive_prefetch
            else BackwardPrefetch.BACKWARD_POST
        ),
        forward_prefetch=aggressive_prefetch,
        limit_all_gathers=not aggressive_prefetch,
        use_orig_params=True,
    )
