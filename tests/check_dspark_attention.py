"""Compare the native eight-query draft block against noncausal FP32 attention."""

import sys
import torch

torch.manual_seed(61)
torch.set_num_threads(2)
torch.ops.load_library(sys.argv[1])
for window, capacity in [(0, 0), (1024, 1088)]:
    for start in (0, 17, 1030, 1086, 2181, 4191):
        first = max(0, start - window) if window else 0
        length = start - first + 8
        rows = 2 * capacity if capacity else start + 37
        q = torch.randn(8, 24, 128, dtype=torch.float16)
        k, v = [torch.randn(length, 8, 128, dtype=torch.float16) for _ in range(2)]
        caches = []
        for data in (k, v):
            cache = torch.full((rows, 8, 128), 121.0, dtype=torch.float16)
            indices = torch.arange(first, start + 8)
            if capacity:
                indices = indices % capacity
                cache[indices + capacity] = data
            cache[indices] = data
            caches.append(cache.to("xpu"))
        query = q.transpose(0, 1).reshape(8, 3, 8, 128).float()
        scores = query @ k.transpose(0, 1).unsqueeze(1).float().transpose(-1, -2) * 128**-0.5
        ref = (
            (scores.softmax(-1) @ v.transpose(0, 1).unsqueeze(1).float()).reshape(24, 8, 128).transpose(0, 1)
        )
        for splits in (16, 32):
            partial = torch.empty(splits, 8, 24, 128, dtype=torch.float32, device="xpu")
            lse = torch.empty(splits, 8, 24, dtype=torch.float32, device="xpu")
            torch.ops.inkling2_xpu.dspark_attention(
                q.to("xpu"),
                *caches,
                torch.tensor([start], dtype=torch.int32, device="xpu"),
                partial,
                lse,
                capacity,
                window,
            )
            actual = (partial * lse.softmax(0).unsqueeze(-1)).sum(0).cpu()
            rel = (actual - ref).norm() / ref.norm()
            print(window, start, splits, float(rel), flush=True)
            assert rel < 0.002 and abs(actual - ref).max() < 0.003
