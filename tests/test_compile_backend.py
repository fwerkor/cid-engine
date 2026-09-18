from __future__ import annotations

import os

import pytest
import torch

from cid_engine import CIDEngine

pytestmark = pytest.mark.skipif(
    os.environ.get("CID_ENGINE_TEST_COMPILE") != "1",
    reason="torch.compile smoke test is opt-in because compiler startup dominates CPU CI",
)


def test_compile_display_statistics_matches_reference() -> None:
    generator = torch.Generator().manual_seed(11)
    token_ids = torch.randint(97, (1, 8), generator=generator)
    logits = torch.randn(1, 8, 97, generator=generator)

    expected = CIDEngine("reference").display_token_statistics(token_ids, logits)
    actual = CIDEngine("compile").display_token_statistics(token_ids, logits)

    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual[2], expected[2], rtol=2e-5, atol=2e-6)
