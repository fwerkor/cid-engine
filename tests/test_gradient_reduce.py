from __future__ import annotations

import os
import socket
from multiprocessing import get_context

import pytest
import torch
import torch.distributed as dist

from cid_engine.gradient_reduce import AsyncBucketedGradientReducer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(rank: int, world_size: int, port: int, queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(73)
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(32, 8, bias=False),
        )
        named = tuple(model.named_parameters())
        reducer = AsyncBucketedGradientReducer(named, bucket_cap_mb=0.001)
        try:
            x = torch.randn(4, 16) + rank
            reducer.start()
            model(x).square().mean().backward()
            overlap = reducer.last_overlap_buckets
            reducer.finish()
            flattened = torch.cat([p.grad.reshape(-1) for _, p in named])
            queue.put((rank, flattened.tolist(), overlap, reducer.bucket_count))
        finally:
            reducer.close()
    finally:
        dist.destroy_process_group()


def test_gradient_reducer_requires_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        pytest.skip("test requires no pre-existing process group")
    parameter = torch.nn.Parameter(torch.ones(4))
    with pytest.raises(RuntimeError, match="initialized distributed"):
        AsyncBucketedGradientReducer((("weight", parameter),))


def test_async_gradient_reducer_matches_global_average_and_launches_during_backward() -> None:
    ctx = get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    processes = [ctx.Process(target=_worker, args=(rank, 2, port, queue)) for rank in range(2)]
    for process in processes:
        process.start()
    results = [queue.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    results.sort()
    torch.testing.assert_close(torch.tensor(results[0][1]), torch.tensor(results[1][1]))
    assert all(overlap > 0 for _, _, overlap, _ in results)
    assert all(bucket_count >= 2 for _, _, _, bucket_count in results)
