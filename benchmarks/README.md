# Benchmarks

The benchmark compares the native C++ path with the Python semantic reference and checks output
equivalence before timing.

    cid-engine-bench --device cuda --batch 1 --tokens 128 --vocab 65536

Record device, dtype, tensor shape, PyTorch/CUDA version, and commit when comparing results.

The initial C++20 implementation deliberately uses composite ATen operations. It establishes the
native execution boundary; it is not expected to outperform an equivalent PyTorch expression until
the relevant hot path is replaced by a fused backend kernel.
