from __future__ import annotations

import pytest
import torch

from cid_engine.activation_offload import AsyncPinnedActivationOffloader


def test_activation_offloader_validates_configuration() -> None:
    with pytest.raises(ValueError, match="CUDA device"):
        AsyncPinnedActivationOffloader("cpu", max_bytes=1024)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_activation_offload_preserves_forward_and_gradient() -> None:
    device = torch.device("cuda", 0)
    torch.manual_seed(7)
    weight = torch.randn(256, 256, device=device, requires_grad=True)
    source = torch.randn(8, 256, device=device, requires_grad=True)
    reference_source = source.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)

    reference = torch.nn.functional.gelu(reference_source @ reference_weight).square().mean()
    reference.backward()

    offloader = AsyncPinnedActivationOffloader(
        device,
        max_bytes=16 << 20,
        min_tensor_bytes=1024,
        prefetch_depth=2,
    )
    with offloader.saved_tensors_context(
        excluded_storage_ptrs={weight.untyped_storage().data_ptr()}
    ):
        actual = torch.nn.functional.gelu(source @ weight).square().mean()
        actual.backward()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(source.grad, reference_source.grad)
    torch.testing.assert_close(weight.grad, reference_weight.grad)
    assert offloader.last_offloaded_tensors > 0
    assert offloader.last_offloaded_bytes > 0
    allocated = offloader.pool_allocated_bytes

    source2 = source.detach().clone().requires_grad_(True)
    with offloader.saved_tensors_context():
        torch.nn.functional.gelu(source2 @ weight.detach()).sum().backward()
    torch.cuda.synchronize(device)
    assert offloader.pool_allocated_bytes <= allocated + (8 << 20)
