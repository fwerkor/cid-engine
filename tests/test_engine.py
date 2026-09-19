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


def _statistics_from_logits(
    token_ids: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return reference.display_token_statistics(token_ids, logits)


def test_native_refinement_reveals_and_revises() -> None:
    tokens = torch.tensor([[5, 9, 10, 11]])
    logits = torch.zeros(1, 4, 16)
    logits[0, 0, 5] = 20.0
    logits[0, 0, 7] = 10.0
    logits[0, 1, 12] = 10.0
    logits[0, 2, 10] = 10.0
    logits[0, 3, 13] = 9.0
    confidence, predicted, current = _statistics_from_logits(tokens, logits)

    refined = cid_engine.refine_display_from_statistics(
        tokens,
        confidence,
        predicted,
        current,
        mask_token_id=5,
        eos_token_id=None,
        reveal_fraction=1.0,
        revision_fraction=0.5,
        revision_margin=0.1,
    )

    assert refined.tolist() == [[5, 12, 10, 13]]


def test_native_refinement_matches_float32_margin_quantization() -> None:
    tokens = torch.tensor([[2]])
    confidence = torch.tensor([[0.25]])
    predicted = torch.tensor([[0]])
    current = torch.tensor([[0.25]])

    refined = cid_engine.refine_display_from_statistics(
        tokens,
        confidence,
        predicted,
        current,
        mask_token_id=0,
        eos_token_id=None,
        reveal_fraction=0.0,
        revision_fraction=1.0,
        revision_margin=float.fromhex("0x0.0000000000001p-1022"),
    )

    # torch.float32 >= scalar quantizes this subnormal double margin to zero.
    assert refined.tolist() == [[0]]


def test_native_refinement_expands_existing_eos() -> None:
    tokens = torch.tensor([[9, 2, 5, 5]])
    logits = torch.zeros(1, 4, 16)
    logits[0, 0, 9] = 30.0
    logits[0, 1, 7] = 30.0
    logits[0, 2, 8] = 30.0
    logits[0, 3, 2] = 30.0
    confidence, predicted, current = _statistics_from_logits(tokens, logits)

    refined = cid_engine.refine_display_from_statistics(
        tokens,
        confidence,
        predicted,
        current,
        mask_token_id=5,
        eos_token_id=2,
        reveal_fraction=1.0,
        revision_fraction=1.0,
        revision_margin=0.0,
    )

    assert refined.tolist() == [[9, 7, 8, 2]]


def test_native_refinement_splices_middle_insertion() -> None:
    tokens = torch.tensor([[9, 10, 11, 12, 13, 14, 2, 5, 5]])
    logits = torch.full((1, 9, 16), -20.0)
    proposal = [9, 10, 7, 11, 12, 13, 0, 0, 0]
    for position, token in enumerate(proposal):
        logits[0, position, token] = 20.0
    confidence, predicted, current = _statistics_from_logits(tokens, logits)

    refined = cid_engine.refine_display_from_statistics(
        tokens,
        confidence,
        predicted,
        current,
        mask_token_id=5,
        eos_token_id=2,
        reveal_fraction=1.0,
        revision_fraction=1.0,
        revision_margin=0.0,
    )

    assert refined.tolist() == [[9, 10, 7, 11, 12, 13, 14, 2, 5]]


def test_native_refinement_splices_middle_deletion() -> None:
    tokens = torch.tensor([[9, 10, 7, 11, 12, 13, 14, 2, 5]])
    logits = torch.full((1, 9, 16), -20.0)
    proposal = [9, 10, 11, 12, 13, 0, 0, 0, 0]
    for position, token in enumerate(proposal):
        logits[0, position, token] = 20.0
    confidence, predicted, current = _statistics_from_logits(tokens, logits)

    refined = cid_engine.refine_display_from_statistics(
        tokens,
        confidence,
        predicted,
        current,
        mask_token_id=5,
        eos_token_id=2,
        reveal_fraction=1.0,
        revision_fraction=1.0,
        revision_margin=0.0,
    )

    assert refined.tolist() == [[9, 10, 11, 12, 13, 14, 2, 5, 5]]


def test_materialize_cell_snapshot_matches_reference() -> None:
    generator = torch.Generator().manual_seed(31)
    semantic = torch.randn(2, 5, 17, generator=generator)
    roles = torch.randn(2, 5, 6, generator=generator)
    uncertainty = torch.rand(2, 5, 1, generator=generator)
    noise_delta = torch.randn(2, 5, 1, generator=generator)
    lifecycle = torch.randn(2, 5, 4, generator=generator)
    selected = torch.rand(2, 5, generator=generator) > 0.5
    indices = torch.tensor([0, 3, 8, 16], dtype=torch.long)

    expected = reference.materialize_cell_snapshot(
        semantic,
        roles,
        uncertainty,
        noise_delta,
        lifecycle,
        selected,
        indices,
    )
    actual = cid_engine.materialize_cell_snapshot(
        semantic,
        roles,
        uncertainty,
        noise_delta,
        lifecycle,
        selected,
        indices,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
