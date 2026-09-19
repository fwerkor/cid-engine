from __future__ import annotations

import math

import pytest
import torch

import cid_engine

pytestmark = pytest.mark.precision


def _high_precision_display_oracle(
    token_ids: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits64 = logits.to(torch.float64)
    probabilities = torch.softmax(logits64, dim=-1)
    predicted = logits.float().argmax(dim=-1)
    confidence = probabilities.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
    current = probabilities.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    return confidence, predicted, current


def test_display_prediction_uses_logits_argmax_before_softmax_rounding() -> None:
    token_ids = torch.tensor([[0]])
    logits = torch.tensor([[[0.0, 1.0e-8]]], dtype=torch.float32)

    # FP32 softmax rounds these to an exact tie, even though token 1 has the
    # strictly larger model logit.  Prediction semantics must follow logits.
    assert torch.softmax(logits, dim=-1).argmax(dim=-1).item() == 0
    assert logits.argmax(dim=-1).item() == 1

    confidence, predicted, current = cid_engine.display_token_statistics(
        token_ids,
        logits,
    )
    assert predicted.item() == 1
    assert confidence.item() == 0.5
    assert current.item() == 0.5


def _adversarial_logits(
    *,
    vocab: int,
    dtype: torch.dtype,
    pattern: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0xC1D + vocab)
    shape = (2, 3, vocab)
    if pattern == "random":
        logits = torch.randn(shape, generator=generator) * 7.0
    elif pattern == "wide":
        row = torch.linspace(-80.0, 80.0, vocab)
        logits = row.expand(shape).clone()
        logits[1] = logits[1].flip(-1)
    elif pattern == "near_tie":
        logits = torch.randn(shape, generator=generator) * 1.0e-3
        logits[..., 0] = 1.0
        if vocab > 1:
            logits[..., 1] = 1.0 - 2.0e-6
    elif pattern == "offset":
        logits = torch.randn(shape, generator=generator) * 0.25 + 10000.0
    else:
        raise AssertionError(f"unknown pattern: {pattern}")
    logits = logits.to(dtype)
    token_ids = torch.randint(vocab, shape[:2], generator=generator)
    return token_ids, logits


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("vocab", [1, 2, 31, 32, 33, 255, 1024, 4097, 65537])
@pytest.mark.parametrize("pattern", ["random", "wide", "near_tie", "offset"])
def test_cpu_display_statistics_precision(
    dtype: torch.dtype,
    vocab: int,
    pattern: str,
) -> None:
    token_ids, logits = _adversarial_logits(vocab=vocab, dtype=dtype, pattern=pattern)
    expected = _high_precision_display_oracle(token_ids, logits)
    confidence, predicted, current = cid_engine.display_token_statistics(token_ids, logits)

    torch.testing.assert_close(predicted, expected[1], rtol=0, atol=0)
    torch.testing.assert_close(
        confidence.double(),
        expected[0],
        rtol=2e-5,
        atol=3e-7,
    )
    torch.testing.assert_close(
        current.double(),
        expected[2],
        rtol=2e-5,
        atol=3e-7,
    )
    assert torch.isfinite(confidence).all()
    assert torch.isfinite(current).all()
    assert ((confidence >= 0.0) & (confidence <= 1.0)).all()
    assert ((current >= 0.0) & (current <= 1.0)).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("timestep", [0.0, 1.0e-7, 0.25, 0.5, 0.75, 1.0 - 1.0e-7, 1.0])
def test_thought_corruption_against_float64_oracle(
    dtype: torch.dtype,
    timestep: float,
) -> None:
    generator = torch.Generator().manual_seed(1701)
    semantic = (torch.randn(2, 5, 17, generator=generator) * 3.0).to(dtype)
    epsilon = (torch.randn(2, 5, 17, generator=generator) * 3.0).to(dtype)
    timesteps = torch.full((2, 5), timestep, dtype=dtype)
    occupancy = torch.randint(0, 2, (2, 5, 1), generator=generator).to(dtype)

    actual, local_noise, masked_epsilon = cid_engine.thought_corrupt_from_epsilon(
        semantic,
        timesteps,
        occupancy,
        epsilon,
    )

    semantic64 = semantic.double()
    epsilon64 = epsilon.double()
    timesteps64 = timesteps.double()
    alpha64 = torch.cos(timesteps64 * (math.pi / 2)).square().unsqueeze(-1)
    expected64 = alpha64.sqrt() * semantic64 + (1.0 - alpha64).sqrt() * epsilon64
    expected64 = torch.where(occupancy.bool(), expected64, torch.zeros_like(expected64))

    if dtype == torch.bfloat16:
        # The CID semantic path intentionally quantizes alpha to the semantic
        # tensor dtype before sqrt/mixing.  Bound that BF16 quantization error
        # against an ideal float64 computation rather than expecting FP32-like
        # accuracy from BF16 arithmetic.
        rtol, atol = 4e-2, 5e-2
    elif dtype == torch.float32:
        rtol, atol = 2e-5, 3e-6
    else:
        rtol, atol = 2e-12, 3e-13
    torch.testing.assert_close(actual.double(), expected64, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        local_noise,
        torch.where(
            occupancy.bool(),
            timesteps.unsqueeze(-1),
            torch.zeros((), dtype=dtype),
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        masked_epsilon,
        torch.where(occupancy.bool(), epsilon, torch.zeros((), dtype=dtype)),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("threshold", [1.0e-6, 0.01, 0.5, 0.99, 1.0 - 1.0e-6])
def test_prefix_allocation_precision_at_sigmoid_boundary(threshold: float) -> None:
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
    )
    occupancy = torch.zeros(1, logits.shape[1])

    actual = cid_engine.prefix_allocation_mask(
        occupancy,
        logits,
        threshold=threshold,
        max_allocations=logits.shape[1],
    )

    probabilities = torch.sigmoid(logits.float())
    eligible = probabilities >= threshold
    blocked = (~eligible).cumsum(dim=1) > 0
    expected = eligible & ~blocked
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not cid_engine.CUDA_BACKEND_BUILT,
    reason="CUDA cid-engine backend is unavailable",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [1, 31, 32, 33, 1023, 1024, 1025, 32000, 65536])
@pytest.mark.parametrize("pattern", ["random", "wide", "near_tie", "offset"])
def test_cuda_display_statistics_precision(
    dtype: torch.dtype,
    vocab: int,
    pattern: str,
) -> None:
    token_ids, logits = _adversarial_logits(vocab=vocab, dtype=dtype, pattern=pattern)
    token_ids = token_ids.cuda()
    logits = logits.cuda()
    expected = _high_precision_display_oracle(token_ids, logits)
    confidence, predicted, current = cid_engine.display_token_statistics(token_ids, logits)

    torch.testing.assert_close(predicted, expected[1], rtol=0, atol=0)
    torch.testing.assert_close(
        confidence.double(),
        expected[0],
        rtol=8e-5,
        atol=5e-7,
    )
    torch.testing.assert_close(
        current.double(),
        expected[2],
        rtol=8e-5,
        atol=5e-7,
    )
