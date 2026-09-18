from __future__ import annotations

import torch

import cid_engine


def test_display_statistics_supports_fullgraph_compile() -> None:
    token_ids = torch.randint(97, (1, 8), dtype=torch.long)
    logits = torch.randn(1, 8, 97)

    compiled = torch.compile(
        cid_engine.display_token_statistics,
        fullgraph=True,
    )
    expected = cid_engine.display_token_statistics(token_ids, logits)
    actual = compiled(token_ids, logits)

    for candidate, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, reference)
