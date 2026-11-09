from __future__ import annotations

import pytest
import torch

import cid_engine
from cid_engine import reference


def test_rocm_backend_flag() -> None:
    assert isinstance(cid_engine.ROCM_BACKEND_BUILT, bool)
    if cid_engine.ROCM_BACKEND_BUILT:
        assert torch.version.hip is not None
        # HIP tensors share PyTorch's CUDA dispatch key.
        assert cid_engine.CUDA_BACKEND_BUILT
    if torch.version.hip is None:
        assert not cid_engine.ROCM_BACKEND_BUILT


@pytest.mark.skipif(
    not cid_engine.ROCM_BACKEND_BUILT or not torch.cuda.is_available(),
    reason="ROCm GPU is unavailable",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rocm_noncontiguous_statistics(dtype: torch.dtype) -> None:
    # Non-power-of-two vocab and non-contiguous leading dimensions stress
    # HIPify's generated kernel launch and the RDNA workgroup reduction.
    generator = torch.Generator(device="cuda").manual_seed(1703)
    logits = torch.randn(
        2, 13, 4097, device="cuda", dtype=dtype, generator=generator
    ).transpose(0, 1)
    token_ids = torch.randint(
        4097, (2, 13), device="cuda", generator=generator
    ).transpose(0, 1)

    expected = reference.display_token_statistics(token_ids, logits)
    actual = cid_engine.display_token_statistics(token_ids, logits)

    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[0], expected[0], rtol=8e-5, atol=3e-7)
    torch.testing.assert_close(actual[2], expected[2], rtol=8e-5, atol=3e-7)
