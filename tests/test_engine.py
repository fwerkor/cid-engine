from __future__ import annotations

import pytest
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


def test_rollout_slot_transition_matches_reference() -> None:
    generator = torch.Generator().manual_seed(433)
    occupancy = torch.randint(0, 2, (3, 17, 1), generator=generator).bool()
    allocation_logits = torch.randn(3, 17, generator=generator)
    lifecycle_logits = torch.randn(3, 17, 4, generator=generator)
    revision_logits = torch.randn(3, 17, 3, generator=generator)
    input_lifecycle = torch.randn(3, 17, 4, generator=generator)
    input_lifecycle[:, ::5] = 0

    expected = reference.rollout_slot_transition(
        occupancy,
        allocation_logits,
        lifecycle_logits,
        revision_logits,
        input_lifecycle,
        0.45,
        3,
        3,
    )
    actual = cid_engine.rollout_slot_transition(
        occupancy,
        allocation_logits,
        lifecycle_logits,
        revision_logits,
        input_lifecycle,
        threshold=0.45,
        max_allocations=3,
        retired_index=3,
    )
    for candidate, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, oracle, rtol=0, atol=0)


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


def test_display_corruption_matches_reference() -> None:
    generator = torch.Generator().manual_seed(177)
    token_ids = torch.randint(0, 257, (4, 41), generator=generator)
    timesteps = torch.rand(4, generator=generator)
    eligible = torch.rand(4, 41, generator=generator) > 0.2
    corruption_random = torch.rand(4, 41, generator=generator)
    replacement_random = torch.rand(4, 41, generator=generator)
    replacement_offsets = torch.randint(1, 257, (4, 41), generator=generator)

    expected = reference.display_corrupt_from_random(
        token_ids,
        timesteps,
        eligible,
        corruption_random,
        replacement_random,
        replacement_offsets,
        256,
        2,
        257,
        0.25,
    )
    actual = cid_engine.display_corrupt_from_random(
        token_ids,
        timesteps,
        eligible,
        corruption_random,
        replacement_random,
        replacement_offsets,
        mask_token_id=256,
        eos_token_id=2,
        vocab_size=257,
        replacement_fraction=0.25,
    )
    for candidate, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, oracle, rtol=0, atol=0)


def test_display_corruption_first_eligible_fallback() -> None:
    token_ids = torch.tensor([[10, 11, 12, 13]])
    timesteps = torch.tensor([1.0e-6])
    eligible = torch.tensor([[False, False, True, True]])
    corruption_random = torch.ones(1, 4)

    corrupted, labels, masked, replaced = cid_engine.display_corrupt_from_random(
        token_ids,
        timesteps,
        eligible,
        corruption_random,
        None,
        None,
        mask_token_id=99,
        eos_token_id=None,
        vocab_size=100,
        replacement_fraction=0.0,
    )

    assert corrupted.tolist() == [[10, 11, 99, 13]]
    assert labels.tolist() == [[-100, -100, 12, -100]]
    assert masked.tolist() == [[False, False, True, False]]
    assert not replaced.any()


def test_masked_diffusion_corruption_matches_reference() -> None:
    generator = torch.Generator().manual_seed(71)
    clean = torch.randint(0, 97, (4, 33), generator=generator)
    ratio_random = torch.rand(4, 1, generator=generator)
    mask_random = torch.rand(4, 33, generator=generator)

    expected = reference.masked_diffusion_corrupt_from_random(
        clean,
        ratio_random,
        mask_random,
        96,
        0.001,
        0.9,
    )
    actual = cid_engine.masked_diffusion_corrupt_from_random(
        clean,
        ratio_random,
        mask_random,
        mask_token_id=96,
        min_mask_ratio=0.001,
        max_mask_ratio=0.9,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)


def test_masked_diffusion_corruption_forces_one_mask_without_host_branch() -> None:
    clean = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    ratio_random = torch.zeros(2, 1)
    mask_random = torch.tensor(
        [[0.9, 0.7, 0.8, 0.6], [0.5, 0.4, 0.3, 0.2]],
        dtype=torch.float32,
    )

    corrupted, masked, ratios, metrics = cid_engine.masked_diffusion_corrupt_from_random(
        clean,
        ratio_random,
        mask_random,
        mask_token_id=99,
        min_mask_ratio=1.0e-6,
        max_mask_ratio=1.0e-6,
    )

    assert masked.sum(dim=1).tolist() == [1, 1]
    assert masked.tolist() == [
        [False, False, False, True],
        [False, False, False, True],
    ]
    assert corrupted.tolist() == [[1, 2, 3, 99], [5, 6, 7, 99]]
    torch.testing.assert_close(ratios, torch.full((2, 1), 1.0e-6))
    torch.testing.assert_close(metrics[0], torch.tensor(0.25))
    torch.testing.assert_close(metrics[1], torch.tensor(1.0e-6))


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
    assert hasattr(torch.ops.cid_engine, "display_corrupt_from_random")
    assert hasattr(torch.ops.cid_engine, "display_token_statistics")
    assert hasattr(torch.ops.cid_engine, "prefix_allocation_mask")
    assert hasattr(torch.ops.cid_engine, "masked_diffusion_corrupt_from_random")
    assert hasattr(torch.ops.cid_engine, "rollout_slot_transition")


def test_backend_build_flags_are_boolean() -> None:
    assert isinstance(cid_engine.CUDA_BACKEND_BUILT, bool)
    assert isinstance(cid_engine.CANN_BACKEND_BUILT, bool)


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


def test_materialize_cell_snapshot_rejects_non_bool_selection() -> None:
    semantic = torch.randn(1, 2, 4)
    roles = torch.randn(1, 2, 3)
    uncertainty = torch.rand(1, 2, 1)
    noise_delta = torch.randn(1, 2, 1)
    lifecycle = torch.randn(1, 2, 4)
    selected = torch.ones(1, 2)
    indices = torch.tensor([0, 3], dtype=torch.long)

    with pytest.raises(RuntimeError, match="selected must use torch.bool"):
        cid_engine.materialize_cell_snapshot(
            semantic,
            roles,
            uncertainty,
            noise_delta,
            lifecycle,
            selected,
            indices,
        )


def test_batched_linear_assignment_matches_reference() -> None:
    costs = torch.tensor(
        [
            [
                [4.0, 1.0, 3.0, 8.0],
                [2.0, 0.0, 5.0, 7.0],
                [3.0, 2.0, 2.0, 6.0],
                [9.0, 9.0, 9.0, 9.0],
            ],
            [
                [1.0, 1.0, 2.0, 3.0],
                [1.0, 1.0, 2.0, 3.0],
                [2.0, 2.0, 0.0, 0.0],
                [4.0, 3.0, 2.0, 1.0],
            ],
        ],
        dtype=torch.float32,
    )
    row_counts = torch.tensor([3, 4], dtype=torch.long)

    expected = reference.batched_linear_assignment(costs, row_counts)
    actual = cid_engine.batched_linear_assignment(costs, row_counts)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
