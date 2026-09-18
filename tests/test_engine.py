from __future__ import annotations

import pytest
import torch

from cid_engine import CIDEngine


@pytest.fixture
def reference() -> CIDEngine:
    return CIDEngine("reference")


@pytest.fixture
def optimized() -> CIDEngine:
    return CIDEngine("torch")


def test_display_statistics_matches_reference(
    reference: CIDEngine,
    optimized: CIDEngine,
) -> None:
    generator = torch.Generator().manual_seed(7)
    token_ids = torch.randint(257, (2, 13), generator=generator)
    logits = torch.randn(2, 13, 257, generator=generator)

    expected = reference.display_token_statistics(token_ids, logits)
    actual = optimized.display_token_statistics(token_ids, logits)

    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual[2], expected[2], rtol=2e-5, atol=2e-6)


def test_prefix_allocation_matches_reference(
    reference: CIDEngine,
    optimized: CIDEngine,
) -> None:
    occupancy = torch.tensor(
        [
            [[1.0], [0.0], [0.0], [0.0], [0.0]],
            [[0.0], [1.0], [0.0], [0.0], [0.0]],
        ]
    )
    logits = torch.tensor(
        [
            [10.0, 10.0, 10.0, -10.0, 10.0],
            [10.0, -10.0, 10.0, 10.0, 10.0],
        ]
    )

    expected = reference.prefix_allocation_mask(
        occupancy,
        logits,
        threshold=0.5,
        max_allocations=2,
    )
    actual = optimized.prefix_allocation_mask(
        occupancy,
        logits,
        threshold=0.5,
        max_allocations=2,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_live_slot_occupancy_matches_reference(
    reference: CIDEngine,
    optimized: CIDEngine,
) -> None:
    occupancy = torch.tensor([[[1.0], [0.7], [0.0], [1.2]]])
    lifecycle = torch.zeros(1, 4, 3)
    lifecycle[0, 1, 2] = 1.0
    lifecycle[0, 3, 2] = 0.25

    expected = reference.live_slot_occupancy(
        occupancy,
        lifecycle,
        retired_index=2,
    )
    actual = optimized.live_slot_occupancy(
        occupancy,
        lifecycle,
        retired_index=2,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_thought_corruption_matches_reference(
    reference: CIDEngine,
    optimized: CIDEngine,
) -> None:
    generator = torch.Generator().manual_seed(9)
    semantic = torch.randn(2, 5, 16, generator=generator)
    epsilon = torch.randn(2, 5, 16, generator=generator)
    timesteps = torch.rand(2, 5, generator=generator)
    occupancy = torch.tensor(
        [
            [[1.0], [1.0], [0.0], [1.0], [0.0]],
            [[1.0], [0.0], [1.0], [1.0], [1.0]],
        ]
    )

    expected = reference.thought_corrupt_from_epsilon(
        semantic,
        timesteps,
        occupancy,
        epsilon,
    )
    actual = optimized.thought_corrupt_from_epsilon(
        semantic,
        timesteps,
        occupancy,
        epsilon,
    )
    for candidate, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, oracle, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["reference", "torch"])
def test_invalid_display_ids_are_rejected(backend: str) -> None:
    engine = CIDEngine(backend)
    token_ids = torch.tensor([[0, 3]])
    logits = torch.zeros(1, 2, 3)

    with pytest.raises(ValueError, match="outside"):
        engine.display_token_statistics(token_ids, logits)


def test_non_int64_display_ids_are_rejected() -> None:
    engine = CIDEngine("torch")
    token_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    logits = torch.zeros(1, 2, 3)

    with pytest.raises(ValueError, match="torch.int64"):
        engine.display_token_statistics(token_ids, logits)


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        CIDEngine("does-not-exist")
