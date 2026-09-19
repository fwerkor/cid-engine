# cid-engine

cid-engine is the native execution engine for Continuous Interaction Diffusion (CID).

The engine core is written in C++20. Python is a thin integration surface for the current CID
model and for semantic-reference tests; it is not the execution core.

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
              +---- CUDA       (fused display-statistics and prefix-allocation kernels)
              +---- Ascend     (planned)

The current C++ core owns CID-specific tensor primitives and the post-statistics display refinement policy:

- live thought-slot occupancy with retired-slot masking;
- deterministic first-free prefix allocation;
- thought diffusion corruption from a supplied epsilon tensor;
- display-token confidence/prediction statistics;
- C++ reveal/revision/EOS/structural-edit policy from those statistics.

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

## Benchmark

    cid-engine-bench --device cuda --batch 1 --tokens 128 --vocab 65536

The benchmark verifies C++/reference equivalence before reporting timing.

Moving Python tensor expressions into C++ does not by itself guarantee a speedup. If C++ launches
the same sequence of ATen kernels, device work is essentially unchanged. Establishing the native
core first gives CID a stable place for fused CUDA, CPU, and Ascend implementations and for future
scheduler and memory-planner logic.

## Roadmap

1. Profile and native-accelerate materialization/TCT state transitions.
2. Expand fused CUDA coverage across the remaining CID primitives.
3. Move diffusion-state transitions and TCT packing into an execution plan.
4. Add CID-aware buffer lifetime and stream scheduling.
5. Overlap model/device work with asynchronous tool/source execution.
6. Add the Ascend backend behind the same C++ engine interface.
7. Keep PyTorch as the training/reference frontend until replacing a layer has measured value.

Apache-2.0, matching the main CID repository.
