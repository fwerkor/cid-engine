from __future__ import annotations

import argparse
import time

import torch

import cid_engine
from cid_engine import reference
from cid_engine._accelerator import default_device, synchronize


def _time_call(fn, *, device: torch.device, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    synchronize(device)
    return (time.perf_counter() - started) * 1000.0 / iterations


def _assert_equivalent(
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    candidate: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    ref_confidence, ref_predicted, ref_current = reference
    confidence, predicted, current = candidate
    torch.testing.assert_close(predicted, ref_predicted, rtol=0, atol=0)
    torch.testing.assert_close(confidence, ref_confidence, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(current, ref_current, rtol=2e-5, atol=2e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark cid-engine C++ display statistics")
    parser.add_argument("--device", default=str(default_device()))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type in {"cuda", "npu"} else torch.float32
    token_ids = torch.randint(
        args.vocab,
        (args.batch, args.tokens),
        device=device,
        dtype=torch.long,
    )
    logits = torch.randn(
        args.batch,
        args.tokens,
        args.vocab,
        device=device,
        dtype=dtype,
    )

    reference_output = reference.display_token_statistics(token_ids, logits)
    cpp_output = cid_engine.display_token_statistics(token_ids, logits)
    synchronize(device)
    _assert_equivalent(reference_output, cpp_output)

    reference_ms = _time_call(
        lambda: reference.display_token_statistics(token_ids, logits),
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    cpp_ms = _time_call(
        lambda: cid_engine.display_token_statistics(token_ids, logits),
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    print(f"reference {reference_ms:9.3f} ms  1.00x")
    print(f"cpp       {cpp_ms:9.3f} ms  {reference_ms / cpp_ms:4.2f}x")


if __name__ == "__main__":
    main()
