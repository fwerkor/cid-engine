from __future__ import annotations

import os

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import cid_engine
from cid_engine import reference

pytestmark = pytest.mark.fuzz

_FUZZ_EXAMPLES = int(os.environ.get("CID_ENGINE_FUZZ_EXAMPLES", "200"))
_FUZZ_SETTINGS = settings(
    max_examples=_FUZZ_EXAMPLES,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)

_DTYPES = st.sampled_from([torch.float32, torch.float64, torch.bfloat16])
_LOGIT_DTYPES = st.sampled_from(
    [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
_SCALES = st.sampled_from([0.0, 1.0e-4, 0.1, 1.0, 20.0, 100.0])
_UNIT_INTERVAL = st.one_of(
    st.sampled_from([0.0, 1.0, 0.5, 1.0e-7, 1.0 - 1.0e-7]),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _strided_last_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[-1] == 0:
        return tensor
    storage = torch.empty(
        (*tensor.shape[:-1], tensor.shape[-1] * 2),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    storage[..., ::2] = tensor
    return storage[..., ::2]


@_FUZZ_SETTINGS
@given(
    batch=st.integers(0, 5),
    slots=st.integers(0, 48),
    features=st.integers(1, 8),
    seed=st.integers(0, 2**31 - 1),
    dtype=_DTYPES,
    with_lifecycle=st.booleans(),
)
def test_fuzz_live_slot_occupancy(
    batch: int,
    slots: int,
    features: int,
    seed: int,
    dtype: torch.dtype,
    with_lifecycle: bool,
) -> None:
    generator = _generator(seed)
    occupancy = (torch.randn(batch, slots, 1, generator=generator) * 1.5).to(dtype)
    lifecycle = None
    retired_index = 0
    if with_lifecycle:
        lifecycle = (torch.randn(batch, slots, features, generator=generator) * 1.5).to(dtype)
        retired_index = seed % features

    expected = reference.live_slot_occupancy(occupancy, lifecycle, retired_index)
    actual = cid_engine.live_slot_occupancy(
        occupancy,
        lifecycle,
        retired_index=retired_index,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@_FUZZ_SETTINGS
@given(
    batch=st.integers(0, 5),
    slots=st.integers(0, 64),
    seed=st.integers(0, 2**31 - 1),
    threshold=_UNIT_INTERVAL,
    max_extra=st.integers(1, 8),
    scale=_SCALES,
    rank3=st.booleans(),
    strided=st.booleans(),
)
def test_fuzz_prefix_allocation_mask(
    batch: int,
    slots: int,
    seed: int,
    threshold: float,
    max_extra: int,
    scale: float,
    rank3: bool,
    strided: bool,
) -> None:
    generator = _generator(seed)
    occupancy = torch.randint(-1, 2, (batch, slots), generator=generator).float()
    logits = torch.randn(batch, slots, generator=generator) * scale
    if rank3:
        occupancy = occupancy.unsqueeze(-1)
    if strided and slots:
        logits = _strided_last_dim(logits)

    max_allocations = 1 + (seed % max(1, slots + max_extra))
    expected = reference.prefix_allocation_mask(
        occupancy,
        logits,
        threshold,
        max_allocations,
    )
    actual = cid_engine.prefix_allocation_mask(
        occupancy,
        logits,
        threshold=threshold,
        max_allocations=max_allocations,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@_FUZZ_SETTINGS
@given(
    batch=st.integers(0, 4),
    slots=st.integers(0, 24),
    hidden=st.integers(1, 96),
    seed=st.integers(0, 2**31 - 1),
    dtype=_DTYPES,
)
def test_fuzz_thought_corruption(
    batch: int,
    slots: int,
    hidden: int,
    seed: int,
    dtype: torch.dtype,
) -> None:
    generator = _generator(seed)
    semantic = torch.randn(batch, slots, hidden, generator=generator).to(dtype)
    epsilon = torch.randn(batch, slots, hidden, generator=generator).to(dtype)
    timesteps = (torch.rand(batch, slots, generator=generator) * 1.5 - 0.25).to(dtype)
    occupancy = torch.randint(0, 2, (batch, slots, 1), generator=generator).to(dtype)

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


@_FUZZ_SETTINGS
@given(
    batch=st.integers(0, 4),
    tokens=st.integers(0, 32),
    vocab=st.integers(1, 1024),
    seed=st.integers(0, 2**31 - 1),
    dtype=_LOGIT_DTYPES,
    scale=_SCALES,
    strided=st.booleans(),
)
def test_fuzz_display_statistics(
    batch: int,
    tokens: int,
    vocab: int,
    seed: int,
    dtype: torch.dtype,
    scale: float,
    strided: bool,
) -> None:
    generator = _generator(seed)
    token_ids = torch.randint(vocab, (batch, tokens), generator=generator)
    logits = (torch.randn(batch, tokens, vocab, generator=generator) * scale).to(dtype)
    if strided:
        logits = _strided_last_dim(logits)

    expected = reference.display_token_statistics(token_ids, logits)
    actual = cid_engine.display_token_statistics(token_ids, logits)

    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[0], expected[0], rtol=3e-5, atol=2e-6)
    torch.testing.assert_close(actual[2], expected[2], rtol=3e-5, atol=2e-6)


@_FUZZ_SETTINGS
@given(
    batch=st.integers(1, 3),
    tokens=st.integers(1, 32),
    vocab=st.integers(4, 96),
    seed=st.integers(0, 2**31 - 1),
    reveal_fraction=_UNIT_INTERVAL,
    revision_fraction=_UNIT_INTERVAL,
    revision_margin=st.one_of(
        st.sampled_from([0.0, 1.0e-7, 0.01, 0.1, 1.0]),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ),
    with_eos=st.booleans(),
    scale=_SCALES,
)
def test_fuzz_display_refinement_matches_python_policy(
    batch: int,
    tokens: int,
    vocab: int,
    seed: int,
    reveal_fraction: float,
    revision_fraction: float,
    revision_margin: float,
    with_eos: bool,
    scale: float,
) -> None:
    generator = _generator(seed)
    mask_token_id = 0
    eos_token_id = 1 if with_eos else None
    token_ids = torch.randint(2, vocab, (batch, tokens), generator=generator)
    mask_positions = torch.rand(batch, tokens, generator=generator) < 0.3
    token_ids[mask_positions] = mask_token_id

    if eos_token_id is not None:
        for row in range(batch):
            eos_position = (seed + row * 17) % tokens
            token_ids[row, eos_position] = eos_token_id
            if eos_position + 1 < tokens:
                token_ids[row, eos_position + 1 :] = mask_token_id

    logits = torch.randn(batch, tokens, vocab, generator=generator) * scale
    confidence, predicted, current = reference.display_token_statistics(token_ids, logits)

    expected = reference.refine_display_from_statistics(
        token_ids,
        confidence,
        predicted,
        current,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
        reveal_fraction=reveal_fraction,
        revision_fraction=revision_fraction,
        revision_margin=revision_margin,
    )
    actual = cid_engine.refine_display_from_statistics(
        token_ids,
        confidence,
        predicted,
        current,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
        reveal_fraction=reveal_fraction,
        revision_fraction=revision_fraction,
        revision_margin=revision_margin,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@_FUZZ_SETTINGS
@given(
    active_tokens=st.integers(4, 24),
    capacity_extra=st.integers(0, 8),
    seed=st.integers(0, 2**31 - 1),
    mode=st.sampled_from(["insert", "delete"]),
)
def test_fuzz_structural_refinement(
    active_tokens: int,
    capacity_extra: int,
    seed: int,
    mode: str,
) -> None:
    generator = _generator(seed)
    mask_token_id = 0
    eos_token_id = 1
    capacity = active_tokens + capacity_extra
    content = torch.arange(2, active_tokens + 1, dtype=torch.long)
    token_ids = torch.full((1, capacity), mask_token_id, dtype=torch.long)
    token_ids[0, : active_tokens - 1] = content[: active_tokens - 1]
    token_ids[0, active_tokens - 1] = eos_token_id
    predicted = token_ids.clone()

    start = 1 + seed % max(1, active_tokens - 3)
    if mode == "insert" and capacity_extra > 0:
        inserted = int(torch.randint(64, 128, (), generator=generator))
        shifted = token_ids[0, start : active_tokens - 1].clone()
        predicted[0, start] = inserted
        copy_count = min(len(shifted), capacity - start - 1)
        predicted[0, start + 1 : start + 1 + copy_count] = shifted[:copy_count]
    elif mode == "delete" and start + 2 < active_tokens - 1:
        predicted[0, start : active_tokens - 2] = token_ids[0, start + 1 : active_tokens - 1]

    confidence = torch.full((1, capacity), 0.6)
    current = torch.full((1, capacity), 0.6)
    confidence[0, start] = 0.95
    current[0, start] = 0.05

    expected = reference.refine_display_from_statistics(
        token_ids,
        confidence,
        predicted,
        current,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
        reveal_fraction=1.0,
        revision_fraction=1.0,
        revision_margin=0.1,
    )
    actual = cid_engine.refine_display_from_statistics(
        token_ids,
        confidence,
        predicted,
        current,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
        reveal_fraction=1.0,
        revision_fraction=1.0,
        revision_margin=0.1,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@_FUZZ_SETTINGS
@given(
    threshold=st.one_of(
        st.floats(max_value=-1.0e-9, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.000000001, allow_nan=False, allow_infinity=False),
        st.just(float("nan")),
    ),
    max_allocations=st.integers(1, 16),
)
def test_fuzz_invalid_allocation_threshold_is_rejected(
    threshold: float,
    max_allocations: int,
) -> None:
    occupancy = torch.zeros(1, 4)
    logits = torch.zeros(1, 4)
    with pytest.raises(RuntimeError):
        cid_engine.prefix_allocation_mask(
            occupancy,
            logits,
            threshold=threshold,
            max_allocations=max_allocations,
        )
