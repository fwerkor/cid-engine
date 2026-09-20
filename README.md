# cid-engine

cid-engine is the native execution engine for Continuous Interaction Diffusion (CID).

The engine core is written in C++20. Python is a thin integration surface for the current CID
model and for semantic-reference tests; it is not the execution core.

## Install

PyPI releases can be installed with:

    pip install cid-engine

The current PyPI distribution is source-based so that installation can select the correct native
backend for the target machine. A C++20 compiler is required. If the installed PyTorch build has
CUDA support and `CUDA_HOME` points to a CUDA toolkit, the CUDA backend is built automatically;
otherwise cid-engine builds its CPU backend.

For an environment that already pins a particular PyTorch build, install against that exact local
PyTorch ABI with:

    pip install cid-engine --no-build-isolation

## Architecture

    CID Python/model frontend
              |
              v
       thin Python API
              |
              v
    +-----------------------+
    | cid-engine C++20 core |
    | ATen custom operators |
    +-----------------------+
              |
              +---- CPU
              +---- CUDA       (display, allocation, and materialization kernels)
              +---- Ascend     (planned)

The current C++ core owns CID-specific tensor primitives and the post-statistics display refinement policy:

- live thought-slot occupancy with retired-slot masking;
- deterministic first-free prefix allocation;
- thought diffusion corruption from a supplied epsilon tensor;
- display-token confidence/prediction statistics;
- C++ reveal/revision/EOS/structural-edit policy from those statistics;
- compact materialization snapshots for one-transfer TCT control-state decoding.
- fused masked-diffusion corruption from pre-generated random tensors for training, including
  empty-row fallback and device-resident mask-ratio metrics.

They are registered as torch.ops.cid_engine C++ operators, so tensors cross the Python/C++
boundary without NumPy copies. The pure-Python implementation remains only as a semantic oracle.

## Semantic contract

The default engine may change floating-point reduction order, but it must not change CID
algorithmic decisions or runtime policy.

Tests require exact equality for boolean masks, allocation decisions, token IDs, and argmax
results, plus numerically close floating-point outputs for mathematically equivalent reductions.

Hot-path native operators assume token IDs already satisfy the model vocabulary contract; CID validates
that invariant when constructing the display tensor, avoiding a device-to-host synchronization per step.

Quantization, skipped diffusion steps, approximate attention, speculative execution, or new
early-exit policies are outside this contract.

## Build

A C++20 compiler and PyTorch/LibTorch are required.

    pip install torch
    pip install -e '.[dev]' --no-build-isolation
    pytest
    ruff check .

The core can also be built and tested without the Python API:

    cmake -S . -B build -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
    cmake --build build
    ctest --test-dir build --output-on-failure

## Verification

`cid-engine` treats semantic and numerical equivalence as release requirements. The test stack has
three complementary layers:

- property-based differential fuzzing compares native operators with independent Python semantic
  oracles across randomized shapes, dtypes, strides, thresholds, EOS layouts, and structural edits;
- precision tests compare floating-point kernels with float64 oracles on adversarial logits,
  near-ties, large offsets, diffusion boundary timesteps, and vocabulary sizes up to 65,537;
- a Clang libFuzzer target drives the C++ core directly under AddressSanitizer and
  UndefinedBehaviorSanitizer.

The normal test suite uses a bounded fuzz budget. A scheduled stress workflow raises the Hypothesis
budget substantially and runs the coverage-guided native fuzzer. The same native target can be run
locally with Clang:

    CXX=clang++ cmake -S . -B build-fuzz \
      -DCID_ENGINE_BUILD_TESTS=OFF -DCID_ENGINE_BUILD_FUZZERS=ON \
      -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
    cmake --build build-fuzz --parallel 2
    cp -a tests/fuzz/corpus /tmp/cid-engine-fuzz-corpus
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ASAN_OPTIONS=detect_leaks=0 \
      ./build-fuzz/cid_engine_fuzz /tmp/cid-engine-fuzz-corpus -max_total_time=60

## Use

    import torch
    import cid_engine

    token_ids = torch.randint(0, 32000, (1, 128), device="cuda")
    logits = torch.randn(1, 128, 32000, device="cuda", dtype=torch.bfloat16)

    confidence, predicted, current_confidence = cid_engine.display_token_statistics(
        token_ids,
        logits,
    )

The underlying operator is also available directly through
torch.ops.cid_engine.display_token_statistics. The Python reference module is retained only for semantic tests and benchmarks.

## Training primitive

masked_diffusion_corrupt_from_random keeps CID's Stage-0 mask construction on device. The
caller supplies the two random tensors so RNG ownership stays with the training framework; the
CUDA backend fuses mask-ratio construction, Bernoulli mask application, the mandatory one-token
fallback for empty rows, corrupted token IDs, and mask/ratio metrics into the native path.

This operator does not approximate the objective. The fallback chooses the minimum pre-generated
mask draw on an otherwise-empty row; conditional on the row being empty, exchangeability makes
that position uniformly distributed, matching the previous fallback distribution without a
device-to-host branch.

### Training memory engine

Version 0.7 introduced Stage-A training-memory primitives; version 0.8 extends them into an
execution scheduler without changing the optimization objective:

- `shard_frozen_transformer` FULL_SHARDs immutable transformer blocks and overlaps layer
  materialization with compute, while leaving directly accessed embeddings resident;
- `AsyncPinnedActivationOffloader` moves selected saved activations through reusable pinned host
  buffers on dedicated CUDA streams; `LayerActivationPrefetchController` tags activations by
  transformer layer and starts H2D restoration before each layer backward;
- `SelectiveCheckpointController` checkpoints only a deterministic subset of transformer layers,
  with `checkpoint_fraction_for_budget` converting an activation-memory budget into a layer
  fraction;
- `AsyncBucketedGradientReducer` launches deterministic gradient all-reduces as buckets become
  ready during the final accumulation backward, overlapping communication with remaining compute;
- `profile_attention_backend` reports the actual SDPA backend used by a model path so Flash,
  memory-efficient, math, and eager fallbacks can be distinguished before adding custom kernels.

These primitives retain PyTorch autograd and distributed collectives as the semantic reference;
they optimize storage, transfer scheduling, communication overlap, and rematerialization rather
than changing model math.

## Benchmark

    cid-engine-bench --device cuda --batch 1 --tokens 128 --vocab 65536

The benchmark verifies C++/reference equivalence before reporting timing.

Moving Python tensor expressions into C++ does not by itself guarantee a speedup. If C++ launches
the same sequence of ATen kernels, device work is essentially unchanged. Establishing the native
core first gives CID a stable place for fused CUDA, CPU, and Ascend implementations and for future
scheduler and memory-planner logic.

## Roadmap

1. Keep model-visible TCT numeric state device-resident across CID steps.
2. Move the remaining materialization/control-state transitions into native state.
3. Expand fused CUDA coverage only where profiling shows a material gain.
4. Add CID-aware buffer lifetime and stream scheduling.
5. Overlap model/device work with asynchronous tool/source execution.
6. Add the Ascend backend behind the same C++ engine interface.
7. Keep PyTorch as the training/reference frontend until replacing a layer has measured value.

Apache-2.0, matching the main CID repository.
