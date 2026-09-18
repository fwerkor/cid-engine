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
