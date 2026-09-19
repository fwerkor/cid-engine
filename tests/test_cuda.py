from __future__ import annotations

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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 8])
def test_cuda_prefix_allocation_matches_reference(
    dtype: torch.dtype,
    batch_size: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(812)
    occupancy = (
        torch.rand(batch_size, 128, 1, device="cuda", generator=generator) > 0.6
    )
    logits = torch.randn(
        batch_size,
        128,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )

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
