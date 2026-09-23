from __future__ import annotations

import pytest
import torch

import cid_engine
from cid_engine import reference
from cid_engine.activation_offload import AsyncPinnedActivationOffloader
from cid_engine.gradient_stash import AsyncPinnedGradientAccumulator

NPU_READY = (
    hasattr(torch, "npu")
    and torch.npu.is_available()
    and cid_engine.CANN_BACKEND_BUILT
)
pytestmark = pytest.mark.skipif(
    not NPU_READY,
    reason="CANN cid-engine backend is unavailable",
)


def _device() -> torch.device:
    return torch.device("npu:0")


def test_cann_backend_build_flag_is_enabled() -> None:
    assert cid_engine.CANN_BACKEND_BUILT


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_npu_display_statistics_matches_reference(dtype: torch.dtype) -> None:
    device = _device()
    token_ids_cpu = torch.tensor([[1, 3, 2, 0]], dtype=torch.long)
    logits_cpu = torch.randn(1, 4, 17, dtype=dtype)
    expected = reference.display_token_statistics(token_ids_cpu, logits_cpu)
    actual = cid_engine.display_token_statistics(
        token_ids_cpu.to(device),
        logits_cpu.to(device),
    )
    torch.npu.synchronize(device)
    torch.testing.assert_close(actual[1].cpu(), expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[0].cpu(), expected[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual[2].cpu(), expected[2], rtol=2e-5, atol=2e-6)


def test_npu_display_statistics_preserves_float32_near_tie_argmax() -> None:
    device = _device()
    token_ids = torch.tensor([[0]], device=device, dtype=torch.long)
    logits = torch.tensor([[[0.0, 1.0e-8]]], device=device, dtype=torch.float32)

    _, predicted, _ = cid_engine.display_token_statistics(token_ids, logits)

    assert predicted.item() == 1


def test_npu_training_primitives_stay_on_device() -> None:
    device = _device()
    clean = torch.arange(32, device=device, dtype=torch.long).reshape(4, 8)
    ratio_random = torch.tensor([[0.0], [0.2], [0.6], [0.9]], device=device)
    mask_random = torch.linspace(0.0, 0.99, 32, device=device).reshape(4, 8)
    corrupted, masked, ratios, metrics = cid_engine.masked_diffusion_corrupt_from_random(
        clean,
        ratio_random,
        mask_random,
        mask_token_id=99,
        min_mask_ratio=1.0e-6,
        max_mask_ratio=1.0,
    )
    occupancy = torch.zeros(4, 8, device=device)
    logits = torch.linspace(-2.0, 2.0, 32, device=device).reshape(4, 8)
    allocation = cid_engine.prefix_allocation_mask(
        occupancy,
        logits,
        threshold=0.5,
        max_allocations=3,
    )
    torch.npu.synchronize(device)

    tensors = (corrupted, masked, ratios, metrics, allocation)
    assert all(tensor.device.type == "npu" for tensor in tensors)
    assert masked.any(dim=1).all().item()
    assert (allocation.sum(dim=1) <= 3).all().item()


@pytest.mark.parametrize("batch, size", [(1, 8), (17, 8), (32, 16)])
def test_npu_linear_assignment_fast_path_matches_reference(
    batch: int,
    size: int,
) -> None:
    device = _device()
    generator = torch.Generator().manual_seed(1931 + batch + size)
    costs = torch.randn(batch, size, size, generator=generator)
    row_counts = torch.randint(
        0,
        size + 1,
        (batch,),
        generator=generator,
        dtype=torch.long,
    )
    expected = reference.batched_linear_assignment(costs, row_counts)
    actual = cid_engine.batched_linear_assignment(
        costs.to(device),
        row_counts.to(device),
    )
    torch.npu.synchronize(device)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape",
    [(4, 257), (8, 1024), (16, 2048)],
)
def test_npu_masked_diffusion_hybrid_path_matches_reference(
    shape: tuple[int, int],
) -> None:
    device = _device()
    batch, tokens = shape
    generator = torch.Generator().manual_seed(2718 + batch + tokens)
    clean_ids = torch.randint(
        0,
        4096,
        shape,
        generator=generator,
        dtype=torch.long,
    )
    ratio_random = torch.rand(
        batch,
        1,
        generator=generator,
        dtype=torch.bfloat16,
    )
    mask_random = torch.rand(
        shape,
        generator=generator,
        dtype=torch.bfloat16,
    )
    expected = reference.masked_diffusion_corrupt_from_random(
        clean_ids,
        ratio_random,
        mask_random,
        mask_token_id=4095,
        min_mask_ratio=0.02,
        max_mask_ratio=0.8,
    )
    actual = cid_engine.masked_diffusion_corrupt_from_random(
        clean_ids.to(device),
        ratio_random.to(device),
        mask_random.to(device),
        mask_token_id=4095,
        min_mask_ratio=0.02,
        max_mask_ratio=0.8,
    )
    torch.npu.synchronize(device)
    for candidate, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            candidate.cpu(),
            oracle,
            rtol=2e-5 if candidate.is_floating_point() else 0,
            atol=2e-6 if candidate.is_floating_point() else 0,
        )


def test_npu_linear_assignment_roundtrip_is_semantically_correct() -> None:
    device = _device()
    costs = torch.tensor(
        [[[3.0, 1.0, 2.0], [1.0, 3.0, 2.0]]],
        device=device,
    )
    row_counts = torch.tensor([2], device=device, dtype=torch.long)
    actual = cid_engine.batched_linear_assignment(costs, row_counts)
    assert actual.device.type == "npu"
    assert actual.cpu().tolist() == [[1, 0]]


def test_npu_slot_state_ops_match_cpu_reference() -> None:
    device = _device()
    occupancy = torch.tensor(
        [[[1.0], [0.0], [1.0], [0.0]], [[0.0], [1.0], [1.0], [0.0]]]
    )
    lifecycle = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 0.0]],
        ]
    )
    expected_live = reference.live_slot_occupancy(occupancy, lifecycle, 2)
    actual_live = cid_engine.live_slot_occupancy(
        occupancy.to(device),
        lifecycle.to(device),
        retired_index=2,
    )
    torch.testing.assert_close(actual_live.cpu(), expected_live, rtol=0, atol=0)

    generator = torch.Generator().manual_seed(744)
    occupancy = torch.randint(0, 2, (2, 8), generator=generator).bool()
    allocation_logits = torch.randn(2, 8, generator=generator)
    lifecycle_logits = torch.randn(2, 8, 4, generator=generator)
    revision_logits = torch.randn(2, 8, 3, generator=generator)
    input_lifecycle = torch.randn(2, 8, 4, generator=generator)
    input_lifecycle[:, ::3] = 0
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
        occupancy.to(device),
        allocation_logits.to(device),
        lifecycle_logits.to(device),
        revision_logits.to(device),
        input_lifecycle.to(device),
        threshold=0.45,
        max_allocations=3,
        retired_index=3,
    )
    for candidate, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate.cpu(), oracle, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    ("shape", "replacement_fraction"),
    [
        ((1, 128), 0.25),
        ((4, 257), 1.0),
        ((8, 2048), 0.25),
    ],
)
def test_npu_display_corruption_hybrid_path_matches_reference(
    shape: tuple[int, int],
    replacement_fraction: float,
) -> None:
    device = _device()
    batch, tokens = shape
    generator = torch.Generator().manual_seed(811 + batch + tokens)
    token_ids = torch.randint(
        3,
        32000,
        shape,
        generator=generator,
        dtype=torch.long,
    )
    timesteps = torch.rand(batch, generator=generator)
    eligible = torch.rand(shape, generator=generator) > 0.2
    corruption_random = torch.rand(
        shape,
        generator=generator,
        dtype=torch.bfloat16,
    )
    replacement_random = torch.rand(
        shape,
        generator=generator,
        dtype=torch.bfloat16,
    )
    replacement_offsets = torch.randint(
        1,
        32000,
        shape,
        generator=generator,
        dtype=torch.long,
    )
    expected = reference.display_corrupt_from_random(
        token_ids,
        timesteps,
        eligible,
        corruption_random,
        replacement_random,
        replacement_offsets,
        mask_token_id=31999,
        eos_token_id=2,
        vocab_size=32000,
        replacement_fraction=replacement_fraction,
    )
    actual = cid_engine.display_corrupt_from_random(
        token_ids.to(device),
        timesteps.to(device),
        eligible.to(device),
        corruption_random.to(device),
        replacement_random.to(device),
        replacement_offsets.to(device),
        mask_token_id=31999,
        eos_token_id=2,
        vocab_size=32000,
        replacement_fraction=replacement_fraction,
    )
    torch.npu.synchronize(device)
    for candidate, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate.cpu(), oracle, rtol=0, atol=0)


def test_npu_corruption_ops_match_cpu_reference() -> None:
    device = _device()
    generator = torch.Generator().manual_seed(194)
    semantic = torch.randn(2, 6, 17, generator=generator)
    epsilon = torch.randn(2, 6, 17, generator=generator)
    timesteps = torch.rand(2, 6, generator=generator)
    occupancy = torch.rand(2, 6, 1, generator=generator) > 0.3
    expected_thought = reference.thought_corrupt_from_epsilon(
        semantic,
        timesteps,
        occupancy,
        epsilon,
    )
    actual_thought = cid_engine.thought_corrupt_from_epsilon(
        semantic.to(device),
        timesteps.to(device),
        occupancy.to(device),
        epsilon.to(device),
    )
    for candidate, oracle in zip(actual_thought, expected_thought, strict=True):
        torch.testing.assert_close(candidate.cpu(), oracle, rtol=2e-5, atol=2e-6)

    token_ids = torch.randint(0, 257, (2, 16), generator=generator)
    timesteps = torch.rand(2, generator=generator)
    eligible = torch.rand(2, 16, generator=generator) > 0.2
    corruption_random = torch.rand(2, 16, generator=generator)
    replacement_random = torch.rand(2, 16, generator=generator)
    replacement_offsets = torch.randint(1, 257, (2, 16), generator=generator)
    expected_display = reference.display_corrupt_from_random(
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
    actual_display = cid_engine.display_corrupt_from_random(
        token_ids.to(device),
        timesteps.to(device),
        eligible.to(device),
        corruption_random.to(device),
        replacement_random.to(device),
        replacement_offsets.to(device),
        mask_token_id=256,
        eos_token_id=2,
        vocab_size=257,
        replacement_fraction=0.25,
    )
    for candidate, oracle in zip(actual_display, expected_display, strict=True):
        torch.testing.assert_close(candidate.cpu(), oracle, rtol=0, atol=0)


def test_npu_materialization_and_refinement_match_cpu_reference() -> None:
    device = _device()
    generator = torch.Generator().manual_seed(902)
    semantic = torch.randn(2, 5, 17, generator=generator)
    roles = torch.randn(2, 5, 6, generator=generator)
    uncertainty = torch.rand(2, 5, 1, generator=generator)
    noise_delta = torch.randn(2, 5, 1, generator=generator)
    lifecycle = torch.randn(2, 5, 4, generator=generator)
    selected = torch.rand(2, 5, generator=generator) > 0.5
    indices = torch.tensor([0, 3, 8, 16], dtype=torch.long)
    expected_snapshot = reference.materialize_cell_snapshot(
        semantic,
        roles,
        uncertainty,
        noise_delta,
        lifecycle,
        selected,
        indices,
    )
    actual_snapshot = cid_engine.materialize_cell_snapshot(
        semantic.to(device),
        roles.to(device),
        uncertainty.to(device),
        noise_delta.to(device),
        lifecycle.to(device),
        selected.to(device),
        indices.to(device),
    )
    torch.testing.assert_close(actual_snapshot.cpu(), expected_snapshot, rtol=2e-5, atol=2e-6)

    token_ids = torch.tensor([[9, 10, 11, 12, 13, 14, 2, 5, 5]])
    logits = torch.full((1, 9, 16), -20.0)
    proposal = [9, 10, 7, 11, 12, 13, 0, 0, 0]
    for position, token in enumerate(proposal):
        logits[0, position, token] = 20.0
    confidence, predicted, current = reference.display_token_statistics(token_ids, logits)
    expected_refined = reference.refine_display_from_statistics(
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
    actual_refined = cid_engine.refine_display_from_statistics(
        token_ids.to(device),
        confidence.to(device),
        predicted.to(device),
        current.to(device),
        mask_token_id=5,
        eos_token_id=2,
        reveal_fraction=1.0,
        revision_fraction=1.0,
        revision_margin=0.0,
    )
    torch.testing.assert_close(actual_refined.cpu(), expected_refined, rtol=0, atol=0)


def test_npu_activation_offload_preserves_gradients() -> None:
    device = _device()
    torch.manual_seed(17)
    source = torch.randn(32, 64, device=device, requires_grad=True)
    weight = torch.randn(64, 64, device=device, requires_grad=True)
    reference_source = source.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)

    reference_loss = torch.nn.functional.gelu(reference_source @ reference_weight).square().mean()
    reference_loss.backward()

    offloader = AsyncPinnedActivationOffloader(
        device,
        max_bytes=8 << 20,
        min_tensor_bytes=1,
        prefetch_depth=2,
    )
    with offloader.saved_tensors_context(
        excluded_storage_ptrs={weight.untyped_storage().data_ptr()}
    ):
        actual_loss = torch.nn.functional.gelu(source @ weight).square().mean()
        actual_loss.backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual_loss.cpu(), reference_loss.cpu())
    torch.testing.assert_close(source.grad.cpu(), reference_source.grad.cpu(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(weight.grad.cpu(), reference_weight.grad.cpu(), rtol=1e-4, atol=1e-5)
    assert offloader.last_offloaded_tensors > 0


def test_npu_pinned_gradient_accumulator_restores_sum() -> None:
    device = _device()
    parameter = torch.nn.Parameter(torch.zeros(128, device=device))
    accumulator = AsyncPinnedGradientAccumulator(device)
    try:
        parameter.grad = torch.full_like(parameter, 2.0)
        accumulator.stash((("weight", parameter),))
        parameter.grad = torch.full_like(parameter, 3.0)
        accumulator.stash((("weight", parameter),))
        accumulator.restore((("weight", parameter),))
        torch.npu.synchronize(device)
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad.cpu(), torch.full((128,), 5.0))
    finally:
        accumulator.close()
