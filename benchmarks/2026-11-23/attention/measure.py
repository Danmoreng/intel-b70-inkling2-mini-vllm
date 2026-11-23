"""Measure single-request B70 global and local decode, including the split merge."""

import argparse
import json
import statistics
import time

import torch


def graph_latency(call):
    call()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        output = call()
    graph.replay()
    torch.xpu.synchronize()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(12):
            graph.replay()
        torch.xpu.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / 12)
    return statistics.median(samples), output


def inputs(length, rows, local):
    def random(*shape, dtype=torch.float32):
        return torch.randn(shape, device="xpu", dtype=dtype)

    query = random(rows, 24, 128, dtype=torch.float16)
    table = random(rows, 24, 1024) * 0.7
    sink = random(rows, 24)
    positions = torch.arange(length - rows, length, device="xpu", dtype=torch.int32)
    seq_len = torch.tensor([length], device="xpu", dtype=torch.int32)
    if local:
        keys, values = (random(1088, 8, 128, dtype=torch.bfloat16) for _ in range(2))
        past = torch.arange(max(0, length - 1088), length, device="xpu", dtype=torch.int32)
        pages = torch.full((1, 1088), -1, device="xpu", dtype=torch.int32)
        pages[0, past.long().remainder(1088)] = past
    else:
        n = (length + 15) // 16
        cache = random(n, 8, 16, 256, dtype=torch.float16).to(torch.float8_e4m3fn)
        keys, values = cache.transpose(1, 2).split(128, -1)
        pages = torch.randperm(n, device="xpu", dtype=torch.int32).unsqueeze(0)
    return query, keys, values, pages, seq_len, positions, table, sink


def measure(length, rows, parts, local, simt):
    from inkling_xpu.split_attention import paged_splits

    tensors = inputs(length, rows, local)
    reference_parts = 16 if local else paged_splits(length, rows)
    candidates = [("xmx", reference_parts)]
    candidates.extend(("xmx", p) for p in parts if p != reference_parts)
    if simt and not local and rows == 1:
        candidates.extend(("simt", p) for p in parts)
    reference = None
    for impl, splits in candidates:
        partial = torch.empty(splits, rows, 24, 128, device="xpu")
        lse = torch.empty(splits, rows, 24, device="xpu")

        def call():
            if impl == "simt":
                q, k, v, pages, seq_len, _, table, sink = tensors
                torch.ops.inkling_hills.m1(q, k, v, pages, seq_len, table, sink, partial, lse)
            else:
                ops = torch.ops.inkling2_xpu
                kernel = ops.local_xmx if local else ops.paged_xmx
                kernel(*tensors, partial, lse)
            return torch.ops.inkling2_xpu.merge_attention(partial, lse)

        us, output = graph_latency(call)
        if reference is None:
            reference = output.clone()
        difference = (output - reference).float()
        error = (difference.norm() / reference.norm().clamp_min(1e-9)).item()
        absolute = difference.abs().max().item()
        if not (error < 0.003 and absolute < 0.004):
            raise RuntimeError(f"Output mismatch at {length=}, {rows=}, {splits=}, {error=}, {absolute=}")
        print(json.dumps(dict(ctx=length, rows=rows, impl=impl, parts=splits, local=local,
                              us=us, rel_l2=error, max_abs=absolute)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, help="Built inkling2-vllm.so")
    parser.add_argument("--control-library", required=True, help="Built inkling2-loop-control.so")
    parser.add_argument("--simt-library", help="Optional AOT simt_m1.so from this experiment")
    parser.add_argument("--context", nargs="+", type=int, default=[64504, 65537, 128701, 131075])
    parser.add_argument("--rows", nargs="+", type=int, default=list(range(1, 9)))
    parser.add_argument("--parts", nargs="+", type=int, required=True)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if min(args.parts) < 1 or min(args.rows) < 1 or max(args.rows) > 8 or min(args.context) < max(args.rows):
        parser.error("Expected positive split counts and 1 to 8 query rows within each context")
    torch.manual_seed(907)
    torch.ops.load_library(args.library)
    torch.ops.load_library(args.control_library)
    if args.simt_library:
        torch.ops.load_library(args.simt_library)
    for length in args.context:
        for rows in args.rows:
            measure(length, rows, args.parts, args.local, args.simt_library)


if __name__ == "__main__":
    main()
