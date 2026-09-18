# Benchmarks

The benchmark compares the native C++ path with the Python semantic reference and checks output
equivalence before timing.

    cid-engine-bench --device cuda --batch 1 --tokens 128 --vocab 65536

Record device, dtype, tensor shape, PyTorch/CUDA version, and commit when comparing results.

The initial C++20 implementation deliberately uses composite ATen operations. It establishes the
native execution boundary; it is not expected to outperform an equivalent PyTorch expression until
the relevant hot path is replaced by a fused backend kernel.

## A6000 CUDA microbenchmark

Measured on an NVIDIA RTX A6000 with BF16 logits and iLLaDA-8B's 155136-token vocabulary.
The reference is the equivalent PyTorch softmax/max/gather computation.

| Display slots | Reference | Fused CUDA | Speedup |
| ---: | ---: | ---: | ---: |
| 32 | 0.212 ms | 0.128 ms | 1.65x |
| 64 | 0.697 ms | 0.215 ms | 3.24x |
| 128 | 1.333 ms | 0.419 ms | 3.18x |
| 256 | 1.567 ms | 0.449 ms | 3.49x |

These are kernel-level measurements, not end-to-end CID latency.
