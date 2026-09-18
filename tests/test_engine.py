from __future__ import annotations

import torch

import cid_engine
from cid_engine import reference


def test_display_statistics_matches_reference() -> None:
    generator = torch.Generator().manual_seed(7)
    token_ids = torch.randint(257, (2, 13), generator=generator)
    logits = torch.randn(2, 13, 257, generator=generator)

    expected = reference.display_token_statistics(token_ids, logits)
    actual = cid_engine.display_token_statistics(token_ids, logits)

    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual[2], expected[2], rtol=2e-5, atol=2e-6)


def test_prefix_allocation_matches_reference() -> None:
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

    expected = reference.prefix_allocation_mask(occupancy, logits, 0.5, 2)
    actual = cid_engine.prefix_allocation_mask(
        occupancy,
        logits,
        threshold=0.5,
        max_allocations=2,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_live_slot_occupancy_matches_reference() -> None:
    occupancy = torch.tensor([[[1.0], [0.7], [0.0], [1.2]]])
    lifecycle = torch.zeros(1, 4, 3)
    lifecycle[0, 1, 2] = 1.0
    lifecycle[0, 3, 2] = 0.25

    expected = reference.live_slot_occupancy(occupancy, lifecycle, 2)
    actual = cid_engine.live_slot_occupancy(
        occupancy,
        lifecycle,
        retired_index=2,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_live_slot_occupancy_without_lifecycle() -> None:
    occupancy = torch.tensor([[[1.2], [-0.1], [0.5]]])
    expected = reference.live_slot_occupancy(occupancy, None, 0)
    actual = cid_engine.live_slot_occupancy(
        occupancy,
        None,
        retired_index=0,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_thought_corruption_matches_reference() -> None:
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
    actual = cid_engine.thought_corrupt_from_epsilon(
        semantic,
        timesteps,
        occupancy,
        epsilon,
    )
    for candidate, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, oracle, rtol=0, atol=0)


def test_non_int64_display_ids_are_rejected() -> None:
    token_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    logits = torch.zeros(1, 2, 3)

    try:
        cid_engine.display_token_statistics(token_ids, logits)
    except RuntimeError as exc:
        assert "torch.int64" in str(exc)
    else:
        raise AssertionError("expected native dtype validation to fail")


def test_native_ops_are_registered() -> None:
    assert hasattr(torch.ops.cid_engine, "display_token_statistics")
    assert hasattr(torch.ops.cid_engine, "prefix_allocation_mask")


def test_cuda_build_flag_is_boolean() -> None:
    assert isinstance(cid_engine.CUDA_BACKEND_BUILT, bool)
