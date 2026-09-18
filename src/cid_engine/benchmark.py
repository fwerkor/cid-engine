from __future__ import annotations

import argparse
import time

import torch

from cid_engine import CIDEngine


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(fn, *, device: torch.device, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    _synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    _synchronize(device)
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
    parser = argparse.ArgumentParser(description="Benchmark cid-engine display statistics")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
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

    engines = {name: CIDEngine(name) for name in ("reference", "torch", "compile")}
    outputs = {
        name: engine.display_token_statistics(token_ids, logits)
        for name, engine in engines.items()
    }
    _synchronize(device)
    for name in ("torch", "compile"):
        _assert_equivalent(outputs["reference"], outputs[name])

    reference_ms = _time_call(
        lambda: engines["reference"].display_token_statistics(token_ids, logits),
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    print(f"reference {reference_ms:9.3f} ms  1.00x")
    for name in ("torch", "compile"):
        elapsed = _time_call(
            lambda name=name: engines[name].display_token_statistics(token_ids, logits),
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(f"{name:9s} {elapsed:9.3f} ms  {reference_ms / elapsed:4.2f}x")


if __name__ == "__main__":
    main()
