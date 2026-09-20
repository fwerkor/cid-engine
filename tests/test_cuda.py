from __future__ import annotations

import math

import pytest
import torch

import cid_engine
from cid_engine import reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not cid_engine.CUDA_BACKEND_BUILT,
    reason="native CUDA backend is unavailable",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_display_statistics_matches_reference(dtype: torch.dtype) -> None:
    generator = torch.Generator(device="cuda").manual_seed(123)
    token_ids = torch.randint(
        4096,
        (2, 64),
        device="cuda",
        generator=generator,
    )
    logits = torch.randn(
        2,
        64,
        4096,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )

    expected = reference.display_token_statistics(token_ids, logits)
    actual = cid_engine.display_token_statistics(token_ids, logits)

    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[0], expected[0], rtol=5e-5, atol=2e-7)
    torch.testing.assert_close(actual[2], expected[2], rtol=5e-5, atol=2e-7)


def test_cuda_argmax_keeps_first_tie() -> None:
    token_ids = torch.tensor([[2]], device="cuda")
    logits = torch.tensor([[[1.0, 5.0, 5.0, 2.0]]], device="cuda")

    _, predicted, _ = cid_engine.display_token_statistics(token_ids, logits)

    assert predicted.item() == 1


def test_cuda_statistics_and_native_refinement_preserve_structural_insertion() -> None:
    token_ids = torch.tensor([[9, 10, 11, 12, 13, 14, 2, 5, 5]], device="cuda")
    logits = torch.full((1, 9, 16), -20.0, device="cuda")
    proposal = [9, 10, 7, 11, 12, 13, 0, 0, 0]
    for position, token in enumerate(proposal):
        logits[0, position, token] = 20.0

    confidence, predicted, current = cid_engine.display_token_statistics(token_ids, logits)
    refined = cid_engine.refine_display_from_statistics(
        token_ids,
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


def test_cuda_masked_diffusion_corruption_matches_reference() -> None:
    generator = torch.Generator(device="cuda").manual_seed(211)
    clean = torch.randint(0, 32000, (8, 2048), device="cuda", generator=generator)
    ratio_random = torch.rand(8, 1, device="cuda", generator=generator)
    mask_random = torch.rand(8, 2048, device="cuda", generator=generator)

    expected = reference.masked_diffusion_corrupt_from_random(
        clean,
        ratio_random,
        mask_random,
        31999,
        0.001,
        1.0,
    )
    actual = cid_engine.masked_diffusion_corrupt_from_random(
        clean,
        ratio_random,
        mask_random,
        mask_token_id=31999,
        min_mask_ratio=0.001,
        max_mask_ratio=1.0,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], expected[2], rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(actual[3], expected[3], rtol=2e-5, atol=2e-6)


def test_cuda_display_corruption_matches_reference() -> None:
    generator = torch.Generator(device="cuda").manual_seed(913)
    token_ids = torch.randint(0, 32000, (8, 1536), device="cuda", generator=generator)
    timesteps = torch.rand(8, device="cuda", generator=generator)
    eligible = torch.rand(8, 1536, device="cuda", generator=generator) > 0.15
    corruption_random = torch.rand(8, 1536, device="cuda", generator=generator)
    replacement_random = torch.rand(8, 1536, device="cuda", generator=generator)
    replacement_offsets = torch.randint(
        1,
        32000,
        (8, 1536),
        device="cuda",
        generator=generator,
    )

    expected = reference.display_corrupt_from_random(
        token_ids,
        timesteps,
        eligible,
        corruption_random,
        replacement_random,
        replacement_offsets,
        31999,
        2,
        32000,
        0.25,
    )
    actual = cid_engine.display_corrupt_from_random(
        token_ids,
        timesteps,
        eligible,
        corruption_random,
        replacement_random,
        replacement_offsets,
        mask_token_id=31999,
        eos_token_id=2,
        vocab_size=32000,
        replacement_fraction=0.25,
    )
    for candidate, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, oracle, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_thought_corruption_training_path_matches_reference(dtype: torch.dtype) -> None:
    generator = torch.Generator(device="cuda").manual_seed(194)
    semantic = torch.randn(4, 128, 257, device="cuda", dtype=dtype, generator=generator)
    epsilon = torch.randn(4, 128, 257, device="cuda", dtype=dtype, generator=generator)
    timesteps = torch.rand(4, 128, device="cuda", dtype=torch.float32, generator=generator)
    occupancy = torch.rand(4, 128, 1, device="cuda", generator=generator) > 0.3

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

    rtol, atol = (4e-2, 5e-2) if dtype == torch.bfloat16 else (3e-3, 3e-4)
    torch.testing.assert_close(actual[0], expected[0], rtol=rtol, atol=atol)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_async_pinned_gradient_accumulator_preserves_sum_and_reuses_buffers() -> None:
    parameter = torch.nn.Parameter(torch.zeros(4096, device="cuda", dtype=torch.float32))
    accumulator = cid_engine.AsyncPinnedGradientAccumulator(parameter.device)
    expected = torch.zeros_like(parameter)

    try:
        for value in (1.0, 2.0, -0.5, 4.0):
            gradient = torch.full_like(parameter, value)
            expected.add_(gradient)
            parameter.grad = gradient
            accumulator.stash((("weight", parameter),))
            assert parameter.grad is None

        snapshot = accumulator.snapshot()
        torch.testing.assert_close(snapshot["weight"], expected.cpu(), rtol=0, atol=0)
        allocated = accumulator.allocated_bytes
        assert allocated == parameter.numel() * parameter.element_size() * 2

        accumulator.restore((("weight", parameter),))
        torch.cuda.synchronize()
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)

        accumulator.clear()
        parameter.grad = torch.ones_like(parameter)
        accumulator.stash((("weight", parameter),))
        accumulator.flush()
        assert accumulator.allocated_bytes == allocated
    finally:
        accumulator.close()


def test_cuda_masked_diffusion_corruption_forces_empty_row_fallback() -> None:
    clean = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device="cuda")
    ratio_random = torch.zeros(2, 1, device="cuda")
    mask_random = torch.tensor(
        [[0.9, 0.7, 0.8, 0.6], [0.5, 0.4, 0.3, 0.2]],
        dtype=torch.float32,
        device="cuda",
    )

    corrupted, masked, ratios, metrics = cid_engine.masked_diffusion_corrupt_from_random(
        clean,
        ratio_random,
        mask_random,
        mask_token_id=99,
        min_mask_ratio=1.0e-6,
        max_mask_ratio=1.0e-6,
    )

    assert masked.tolist() == [
        [False, False, False, True],
        [False, False, False, True],
    ]
    assert corrupted.tolist() == [[1, 2, 3, 99], [5, 6, 7, 99]]
    torch.testing.assert_close(ratios, torch.full((2, 1), 1.0e-6, device="cuda"))
    torch.testing.assert_close(metrics[0], torch.tensor(0.25, device="cuda"))
    torch.testing.assert_close(metrics[1], torch.tensor(1.0e-6, device="cuda"))


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("rank3", [False, True])
def test_cuda_prefix_allocation_matches_reference(
    dtype: torch.dtype,
    batch_size: int,
    rank3: bool,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(812)
    occupancy = torch.randint(
        -1,
        2,
        (batch_size, 128),
        device="cuda",
        generator=generator,
    ).float()
    if rank3:
        occupancy = occupancy.unsqueeze(-1)

    logits_storage = torch.randn(
        batch_size,
        256,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    logits = logits_storage[:, ::2]
    assert not logits.is_contiguous()

    expected = reference.prefix_allocation_mask(
        occupancy,
        logits,
        0.45,
        4,
    )
    actual = cid_engine.prefix_allocation_mask(
        occupancy,
        logits,
        threshold=0.45,
        max_allocations=4,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("threshold", [1.0e-6, 0.01, 0.5, 0.99, 1.0 - 1.0e-6])
def test_cuda_prefix_allocation_matches_float32_threshold_boundaries(
    threshold: float,
) -> None:
    boundary = math.log(threshold / (1.0 - threshold))
    logits = torch.tensor(
        [[
            math.nextafter(boundary, -math.inf),
            boundary,
            math.nextafter(boundary, math.inf),
            80.0,
            -80.0,
        ]],
        dtype=torch.float64,
        device="cuda",
    )
    occupancy = torch.zeros(1, logits.shape[1], device="cuda")

    expected = reference.prefix_allocation_mask(
        occupancy,
        logits,
        threshold,
        logits.shape[1],
    )
    actual = cid_engine.prefix_allocation_mask(
        occupancy,
        logits,
        threshold=threshold,
        max_allocations=logits.shape[1],
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_cuda_prefix_allocation_uses_tensor_device_not_current_device() -> None:
    current_device = torch.cuda.current_device()
    target_device = 1 if current_device == 0 else 0
    device = torch.device(f"cuda:{target_device}")
    occupancy = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device)
    logits = torch.tensor([[8.0, 7.0, -5.0, 6.0]], device=device)

    expected = reference.prefix_allocation_mask(occupancy, logits, 0.5, 2)
    actual = cid_engine.prefix_allocation_mask(
        occupancy,
        logits,
        threshold=0.5,
        max_allocations=2,
    )

    assert actual.device == device
    assert torch.cuda.current_device() == current_device
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_materialize_cell_snapshot_matches_reference(
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(902)
    semantic = torch.randn(2, 128, 257, device="cuda", dtype=dtype, generator=generator)
    roles = torch.randn(2, 128, 6, device="cuda", dtype=dtype, generator=generator)
    uncertainty = torch.rand(2, 128, 1, device="cuda", dtype=dtype, generator=generator)
    noise_delta = torch.randn(2, 128, 1, device="cuda", dtype=dtype, generator=generator)
    lifecycle = torch.randn(2, 128, 4, device="cuda", dtype=dtype, generator=generator)
    selected = torch.rand(2, 128, device="cuda", generator=generator) > 0.5
    indices = torch.tensor([0, 23, 47, 70, 93, 117, 140, 163, 187, 210, 233, 256], device="cuda")

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

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("columns", [4, 8])
def test_cuda_batched_linear_assignment_matches_reference(columns: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(1417 + columns)
    costs = torch.randn(
        128,
        columns,
        columns,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    row_counts = torch.randint(
        0,
        columns + 1,
        (128,),
        device="cuda",
        dtype=torch.long,
        generator=generator,
    )

    expected = reference.batched_linear_assignment(costs, row_counts)
    actual = cid_engine.batched_linear_assignment(costs, row_counts)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_rollout_slot_transition_matches_reference(dtype: torch.dtype) -> None:
    generator = torch.Generator(device="cuda").manual_seed(744)
    occupancy = torch.randint(0, 2, (8, 128), device="cuda", generator=generator).bool()
    allocation_logits = torch.randn(
        8, 128, device="cuda", dtype=dtype, generator=generator
    )
    lifecycle_logits = torch.randn(
        8, 128, 4, device="cuda", dtype=dtype, generator=generator
    )
    revision_logits = torch.randn(
        8, 128, 3, device="cuda", dtype=dtype, generator=generator
    )
    input_lifecycle = torch.randn(
        8, 128, 4, device="cuda", dtype=dtype, generator=generator
    )
    input_lifecycle[:, ::7] = 0

    expected = reference.rollout_slot_transition(
        occupancy,
        allocation_logits,
        lifecycle_logits,
        revision_logits,
        input_lifecycle,
        0.45,
        4,
        3,
    )
    actual = cid_engine.rollout_slot_transition(
        occupancy,
        allocation_logits,
        lifecycle_logits,
        revision_logits,
        input_lifecycle,
        threshold=0.45,
        max_allocations=4,
        retired_index=3,
    )
    for candidate, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, oracle, rtol=0, atol=0)


def test_cuda_batched_linear_assignment_keeps_first_tie() -> None:
    costs = torch.zeros(2, 8, 8, device="cuda")
    row_counts = torch.tensor([4, 8], device="cuda", dtype=torch.long)

    actual = cid_engine.batched_linear_assignment(costs, row_counts)

    assert actual[0, :4].tolist() == [0, 1, 2, 3]
    assert actual[1].tolist() == list(range(8))
