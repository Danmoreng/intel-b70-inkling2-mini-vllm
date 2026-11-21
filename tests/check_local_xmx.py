"""Check the sliding window across ring wraps, stale positions and multirow verification."""

import sys
import torch

from inkling_xpu.split_attention import merge_splits

torch.manual_seed(73)
torch.set_num_threads(2)
torch.ops.load_library(sys.argv[1])
ring, window = 1088, 1024
for rows in range(1, 9):
    for start in (0, 17, 1020, 1083, 2171, 13265):
        end = start + rows
        pos = torch.arange(start, end, dtype=torch.int32)
        ids = torch.arange(max(0, end - ring), end, dtype=torch.int32)
        locs = ids.long() % ring
        stored = torch.full((ring,), -1, dtype=torch.int32)
        stored[locs] = ids
        if len(locs) > 128:
            stored[locs[-73]] = -1
        q = torch.randn(rows, 24, 128).bfloat16()
        k, v = (torch.randn(ring, 8, 128).bfloat16() for _ in range(2))
        table = torch.randn(rows, 24, 1024) / 2
        sink = torch.randn(rows, 24)
        delta = pos[:, None] - stored[None, :]
        mask = (delta >= 0) & (delta < window) & (stored[None, :] >= 0)
        scores = (
            torch.einsum("mghd,kgd->mghk", q.float().view(rows, 8, 3, 128), k.float()).reshape(rows, 24, ring)
            / 128**0.5
        )
        scores += table.gather(2, delta[:, None, :].long().clamp(0, 1023).expand(rows, 24, ring))
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        weights = torch.cat([scores, sink[..., None]], -1).softmax(-1)[..., :-1]
        expected = torch.einsum("mghk,kgd->mghd", weights.reshape(rows, 8, 3, ring), v.float()).reshape(
            rows, 24, 128
        )
        for parts in (8, 32):
            partial = torch.empty(parts, rows, 24, 128, device="xpu")
            lse = torch.empty(parts, rows, 24, device="xpu")
            torch.ops.inkling2_xpu.local_xmx(
                q.half().to("xpu"),
                k.to("xpu"),
                v.to("xpu"),
                stored[None].to("xpu"),
                torch.tensor([end], dtype=torch.int32, device="xpu"),
                pos.to("xpu"),
                table.to("xpu"),
                sink.to("xpu"),
                partial,
                lse,
            )
            actual = merge_splits(partial, lse).cpu()
            err = (actual - expected).abs().max().item()
            l2 = ((actual - expected).norm() / expected.norm()).item()
            print(rows, start, parts, err, l2, flush=True)
            assert err < 0.003 and l2 < 0.002
