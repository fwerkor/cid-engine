# cid-engine

`cid-engine` is the execution-engine project for **Continuous Interaction Diffusion (CID)**.

The goal is to move CID-specific tensor and runtime work out of generic framework code while
preserving the algorithmic semantics of the reference implementation. The project starts with a
small, testable tensor execution layer rather than attempting to reimplement autograd,
distributed training, or device libraries from day one.

## What exists now

The initial engine extracts CID hot-path primitives behind three backends:

- `reference`: deliberately straightforward PyTorch implementations used as the semantic oracle.
- `torch`: vectorized eager implementations that preserve the same discrete decisions.
- `compile`: the same optimized kernels wrapped by `torch.compile`.

The first primitives cover:

- live thought-slot occupancy with retired-slot masking;
- deterministic first-free prefix allocation;
- thought diffusion corruption from a supplied epsilon tensor;
- display-token confidence extraction without materializing the full softmax tensor.

The display primitive is especially relevant to CID decoding. The current reference path computes
a full probability tensor to obtain only three values per token: the predicted token, its
confidence, and the confidence of the current token. `cid-engine` computes those values from
`max`, `logsumexp`, and `gather`, reducing temporary memory while retaining the same mathematical
definition.

## Semantic contract

Optimizations in the default engine are allowed to change floating-point reduction order, but not
CID decisions or model/runtime policy.

The test suite therefore requires:

- exact equality for boolean masks, token IDs, allocation decisions, and argmax results;
- numerically close floating-point values for equivalent reductions;
- identical behavior across the reference and optimized backends for tested shapes and dtypes.

Algorithm-changing techniques such as quantization, approximate attention, skipped diffusion
steps, speculative updates, or new early-exit policies are intentionally outside this contract.

## Install

```bash
pip install -e .
```

For development:

```bash
pip install -e '.[dev]'
pytest
ruff check .
```

## Use

```python
import torch
from cid_engine import CIDEngine

engine = CIDEngine("torch")

token_ids = torch.randint(0, 32000, (1, 128), device="cuda")
logits = torch.randn(1, 128, 32000, device="cuda", dtype=torch.bfloat16)

confidence, predicted, current_confidence = engine.display_token_statistics(
    token_ids, logits
)
```

Switch to `CIDEngine("reference")` for the semantic oracle or `CIDEngine("compile")` to use
`torch.compile`.

## Benchmark

```bash
cid-engine-bench --device cuda --batch 1 --tokens 128 --vocab 65536
```

The benchmark reports latency for the reference, eager optimized, and compiled paths and verifies
their outputs before timing.

## Direction

The next layers should be added only when they have a measured CID workload:

1. fuse the extracted primitives with Triton/CUDA and Ascend kernels;
2. move tensorization and diffusion-state transitions into a CID execution plan;
3. add CID-aware buffer lifetime planning and stream scheduling;
4. overlap model work with asynchronous tool/source execution;
5. keep PyTorch as a reference/training frontend until replacing a layer has measured value.

This repository is Apache-2.0 licensed, matching the main CID codebase.
