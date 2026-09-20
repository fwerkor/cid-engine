from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from cid_engine.frozen_shard import shard_frozen_transformer


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(hidden))


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(17, 8)
        self.layers = nn.ModuleList((_Block(), _Block()))
        self.norm = nn.LayerNorm(8)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


def test_frozen_shard_requires_distributed() -> None:
    module = _Decoder()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(RuntimeError, match="torch.distributed"):
        shard_frozen_transformer(
            module,
            transformer_layer_cls=_Block,
            device_id=torch.device("cpu"),
        )


def test_frozen_shard_rejects_trainable_transformer_parameter(tmp_path: Path) -> None:
    rendezvous = tmp_path / "dist-init"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        module = _Decoder()
        with pytest.raises(ValueError, match="must be frozen"):
            shard_frozen_transformer(
                module,
                transformer_layer_cls=_Block,
                device_id=torch.device("cpu"),
            )
    finally:
        torch.distributed.destroy_process_group()


def test_frozen_shard_preserves_input_gradient(tmp_path: Path) -> None:
    rendezvous = tmp_path / "dist-init"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        torch.manual_seed(3)
        reference = _Decoder()
        sharded_source = _Decoder()
        sharded_source.load_state_dict(reference.state_dict())
        for module in (reference, sharded_source):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        sharded = shard_frozen_transformer(
            sharded_source,
            transformer_layer_cls=_Block,
            ignored_modules=(sharded_source.embed,),
            device_id=torch.device("cpu"),
        )
        reference_input = torch.randn(2, 5, 8, requires_grad=True)
        sharded_input = reference_input.detach().clone().requires_grad_(True)
        reference_loss = reference(reference_input).square().mean()
        sharded_loss = sharded(sharded_input).square().mean()
        reference_loss.backward()
        sharded_loss.backward()
        torch.testing.assert_close(sharded_loss, reference_loss)
        torch.testing.assert_close(sharded_input.grad, reference_input.grad)
    finally:
        torch.distributed.destroy_process_group()
