from __future__ import annotations

import pytest
import torch
from torch import nn

from cid_engine.checkpointing import (
    SelectiveCheckpointController,
    checkpoint_fraction_for_budget,
    select_checkpoint_layer_indices,
)


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
        return torch.nn.functional.gelu(self.proj(hidden)) * scale


class _Stack(nn.Module):
    def __init__(self, width: int = 16, layers: int = 6) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Block(width) for _ in range(layers))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden, scale=0.75)
        return hidden


def test_checkpoint_fraction_for_budget() -> None:
    assert checkpoint_fraction_for_budget(
        full_activation_bytes=100,
        available_activation_bytes=100,
    ) == 0.0
    assert checkpoint_fraction_for_budget(
        full_activation_bytes=100,
        available_activation_bytes=25,
    ) == 0.75
    assert checkpoint_fraction_for_budget(
        full_activation_bytes=100,
        available_activation_bytes=0,
    ) == 1.0


def test_select_checkpoint_indices_are_uniform() -> None:
    assert select_checkpoint_layer_indices(8, 0.0) == ()
    assert select_checkpoint_layer_indices(8, 1.0) == tuple(range(8))
    selected = select_checkpoint_layer_indices(8, 0.5)
    assert selected == (1, 3, 5, 7)
    assert len(set(selected)) == len(selected)
    with pytest.raises(ValueError):
        select_checkpoint_layer_indices(8, 1.1)


def test_selective_checkpointing_preserves_outputs_and_gradients() -> None:
    torch.manual_seed(11)
    reference = _Stack()
    actual = _Stack()
    actual.load_state_dict(reference.state_dict())
    reference_input = torch.randn(3, 5, 16, requires_grad=True)
    actual_input = reference_input.detach().clone().requires_grad_(True)

    reference_loss = reference(reference_input).square().mean()
    reference_loss.backward()
    controller = SelectiveCheckpointController(actual.layers, checkpoint_fraction=0.5)
    try:
        actual_loss = actual(actual_input).square().mean()
        actual_loss.backward()
    finally:
        controller.restore()

    torch.testing.assert_close(actual_loss, reference_loss)
    torch.testing.assert_close(actual_input.grad, reference_input.grad)
    for actual_parameter, reference_parameter in zip(
        actual.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad)
    assert controller.checkpointed_indices == (1, 3, 5)
