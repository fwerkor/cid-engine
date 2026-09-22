from __future__ import annotations

import torch

from cid_engine.diagnostics import classify_attention_backend, profile_attention_backend


def test_classify_attention_backend_prefers_specific_sdpa_kernel() -> None:
    assert (
        classify_attention_backend(
            (
                "aten::scaled_dot_product_attention",
                "aten::_scaled_dot_product_flash_attention",
            )
        )
        == "flash"
    )
    assert (
        classify_attention_backend(
            (
                "aten::scaled_dot_product_attention",
                "aclnnFlashAttentionScore",
                "npu::npu_fusion_attention",
            )
        )
        == "npu-fused"
    )
    assert (
        classify_attention_backend(("aten::_scaled_dot_product_efficient_attention",))
        == "memory-efficient"
    )
    assert classify_attention_backend(("aten::_scaled_dot_product_attention_math",)) == "math"


def test_profile_attention_backend_detects_cpu_sdpa() -> None:
    query = torch.randn(2, 4, 8, 16)
    key = torch.randn(2, 4, 8, 16)
    value = torch.randn(2, 4, 8, 16)
    report = profile_attention_backend(
        lambda: torch.nn.functional.scaled_dot_product_attention(query, key, value),
        device="cpu",
        warmup=0,
    )
    assert report.backend in {"flash", "math", "sdpa-unknown"}
    assert any("scaled_dot_product_attention" in event for event in report.events)
