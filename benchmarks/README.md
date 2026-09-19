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


## Native full-refinement policy

With the C++ post-statistics policy enabled, the same A6000 / BF16 / 155136-vocabulary
microbenchmark measured the complete CIDDiffusionScheduler.refine_display path at 64 display
slots. Median latency over alternating reference/native runs was:

| Scenario | PyTorch policy | cid-engine | Speedup |
| --- | ---: | ---: | ---: |
| all masked, no revision | 0.660 ms | 0.182 ms | 3.62x |
| EOS at 16, no revision | 0.684 ms | 0.179 ms | 3.83x |
| EOS at 16, revision=1 | 1.044 ms | 0.178 ms | 5.86x |
| EOS at 48, revision=1 | 1.523 ms | 0.181 ms | 8.41x |

The GPU was shared, so occasional contention outliers were excluded by reporting medians. These
remain runtime microbenchmarks rather than end-to-end model-generation speedups.


## Fused prefix allocation

Measured on the same RTX A6000 with 128 thought slots. The reference is the current vectorized
PyTorch CID allocation policy, including FP32 sigmoid thresholding and first-free prefix semantics.

| Logit dtype | Batch | PyTorch | Fused CUDA | Speedup |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 1 | 0.079 ms | 0.006 ms | 13.04x |
| FP32 | 8 | 0.072 ms | 0.007 ms | 11.02x |
| BF16 | 1 | 0.094 ms | 0.007 ms | 13.17x |
| BF16 | 8 | 0.083 ms | 0.007 ms | 11.51x |

A 3000-case threshold-boundary differential test passed across FP32/BF16, batch sizes 1/3/8, and
thresholds from 0 to 1. Allocation masks were exactly equal to the reference.
