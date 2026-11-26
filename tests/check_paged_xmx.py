"""Validate the B70 FP8 paged decode against dense FP32 attention and time each kernel."""

import argparse
import math
import statistics
import time

import torch

from inkling_xpu.paged_attention import decode_paged
from inkling_xpu.split_attention import merge_splits, paged_splits


def case(length, rows, page_size, layout, runs):
    splits = paged_splits(length, rows)
    num_pages = math.ceil(length / page_size)
    pages = torch.randperm(num_pages, device="xpu", dtype=torch.int32).unsqueeze(0)
    shape = (num_pages, 8, page_size, 256) if layout == "HND" else (num_pages, page_size, 8, 256)
    raw = torch.randn(shape, device="xpu", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    if layout == "HND":
        raw = raw.transpose(1, 2)
    k, v = raw.split(128, -1)
    q = torch.randn(rows, 24, 128, device="xpu").bfloat16()
    positions = torch.arange(length - rows, length, dtype=torch.int32, device="xpu")
    lens = torch.tensor([length], dtype=torch.int32, device="xpu")
    rel = torch.randn(rows, 24, 16, device="xpu") / 2
    proj = torch.randn(16, 1024, device="xpu") / 2
    sink = torch.randn(rows, 24, device="xpu") + 1
    partial = torch.empty(splits, rows, 24, 128, device="xpu")
    lse = torch.empty(splits, rows, 24, device="xpu")

    def fast():
        torch.ops.inkling2_xpu.paged_xmx(
            q.half(), k, v, pages, lens, positions, rel @ proj, sink, partial, lse
        )
        return merge_splits(partial, lse)

    def scalar():
        return decode_paged(
            q,
            k,
            v,
            pages,
            lens,
            positions,
            rel,
            proj,
            sink,
            queries_per_request=rows,
            num_splits=min(64, max(8, math.ceil(length / 1024))),
        )

    x = fast()
    if rows in (1, 2, 4, 8):
        y = scalar()
    else:
        # One-query scalar controls with the same page table and final length.
        y = torch.cat(
            [
                decode_paged(
                    q[i : i + 1],
                    k,
                    v,
                    pages,
                    lens,
                    positions[i : i + 1],
                    rel[i : i + 1],
                    proj,
                    sink[i : i + 1],
                    num_splits=8,
                )
                for i in range(rows)
            ]
        )

    abs_err = (x - y).abs().max().item()
    l2 = ((x - y).norm() / y.norm()).item()
    print(f"L={length} M={rows} page={page_size} {layout} abs={abs_err:.6g} rel={l2:.6g}", flush=True)
    assert abs_err < 0.003 and l2 < 0.002, (abs_err, l2)
    if runs:
        for label, fn in (("scalar", scalar), ("XMX", fast)):
            fn()
            torch.xpu.synchronize()
            times = []
            for _ in range(3):
                t = time.monotonic()
                for _ in range(runs):
                    fn()
                torch.xpu.synchronize()
                times.append((time.monotonic() - t) * 1e6 / runs)
            print(label, length, rows, round(statistics.median(times)), "us", flush=True)


def check_gated_merge():
    for parts in (1, 128, 192, 256, 512):
        shape = (parts, 2, 24)
        partial = torch.randn(*shape, 128, device="xpu")
        lse = torch.randn(shape, device="xpu")
        lse[:, 0, 0] = -torch.inf
        gate = torch.rand(2, 24 * 128, device="xpu")

        def reference():
            return torch.ops.inkling2_xpu.merge_attention(partial, lse).reshape_as(gate) * gate

        def fused():
            return torch.ops.inkling2_xpu.merge_attention_gated(partial, lse, gate)

        torch.testing.assert_close(fused(), reference(), rtol=0, atol=0)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            replayed = fused()
        gate.mul_(.5)
        graph.replay()
        torch.xpu.synchronize()
        torch.testing.assert_close(replayed, reference(), rtol=0, atol=0)
        torch.testing.assert_close(replayed[0, :128], torch.zeros_like(replayed[0, :128]))
        print(f"gated merge: {parts} splits, eager and graph FP32 parity", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    parser.add_argument("--control-library", help="Check the fused gate in the native parallel merge")
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()
    torch.ops.load_library(args.library)
    torch.manual_seed(37)
    if args.control_library:
        torch.ops.load_library(args.control_library)
        check_gated_merge()
    for page_size in (16, 32):
        for rows in range(1, 9):
            for length in (rows, 17, 1027):
                for layout in ("HND", "NHD"):
                    case(length, rows, page_size, layout, 0)
    if args.time:
        for length in (4096, 32768, 196608):
            for rows in (1, 4, 8):
                case(length, rows, 16, "HND", 8)
