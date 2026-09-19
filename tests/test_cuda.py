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
