# Benchmarks

The installed `cid-engine-bench` command benchmarks the display-token statistics primitive.

Run on a CID GPU node with:

```bash
cid-engine-bench --device cuda --batch 1 --tokens 128 --vocab 65536
```

Always compare backends on the same device, dtype, shape, and software stack. The benchmark verifies
semantic equivalence before reporting timing.
