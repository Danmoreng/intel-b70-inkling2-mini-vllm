"""Compare native decode depth mixing, including empty banks and normalized outputs."""

import sys
import torch
from inkling_trellis.inkling2.fused import mix

torch.manual_seed(37)
torch.ops.load_library(sys.argv[1])
for m in (1, 4, 8):
    buf = torch.randn(12, m, 2304, device="xpu")
    cur = torch.randn(m, 2304, device="xpu")
    for states in (0, 1, 5, 11):
        ids = torch.randperm(12, device="xpu", dtype=torch.int32)[:states]
        for num in (1, 3, 4):
            q = torch.randn(num, 2304, device="xpu") / 2304**0.5
            for w in (None, torch.randn_like(q)):
                expected = mix(buf, ids, cur, q, w)
                actual = torch.ops.inkling2_xpu.depth_mix(buf, ids, cur, q, w)
                torch.testing.assert_close(actual, expected, atol=0.0003, rtol=0.0008)
    print("depth bank", m, "passed", flush=True)
for rows in (1, 4, 8):
    for splits in (8, 32, 256, 512):
        x = torch.randn(splits, rows, 24, 128, device="xpu")
        lse = torch.randn(splits, rows, 24, device="xpu") * 8
        lse[2:4] = float("-inf")
        actual = torch.ops.inkling2_xpu.merge_attention(x, lse)
        expected = (x * lse.softmax(0)[..., None]).sum(0)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    print("split reduction", rows, "passed", flush=True)
