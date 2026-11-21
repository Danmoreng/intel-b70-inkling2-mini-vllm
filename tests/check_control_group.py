"""Compare batched B70 loop controllers with their independent FP32 reference."""

import sys
import torch
from inkling_trellis.inkling2.fused import mix
from inkling_xpu.controller import mix_bank, pack_cross_queries

torch.manual_seed(19)
torch.ops.load_library(sys.argv[1])
D, R = 2304, 64
device = "xpu"
dq = torch.randint(-127, 128, (4, R, 2 * D), dtype=torch.int8, device=device)
ds = torch.rand(4, R, 2 * D // 32, device=device).half() * 0.0005
uq = torch.randint(-127, 128, (4, 2, R, D), dtype=torch.int8, device=device)
us = torch.rand(4, 2, 2, D, device=device).half() * 0.003
down = (dq.reshape(4, R, -1, 32).float() * ds.float()[..., None]).reshape(4, R, -1)
up = (uq.reshape(4, 2, 2, 32, D).float() * us.float()[:, :, :, None, :]).reshape(4, 2, R, D)


def norm(x):
    return x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt()


for m in (1, 2, 4, 8):
    msgs = torch.randn(4, m, D, device=device)
    bank = list(range(6))
    buf = torch.randn(8, m, D, device=device)
    last = torch.randn(m, D, device=device)
    a, b = torch.randn(2, 24, D, device=device) / D**0.5
    packed_queries = pack_cross_queries(a, b)
    actual = mix_bank(buf, bank, last, packed_queries)
    idx = torch.tensor(bank, device=device, dtype=torch.int32)
    for il in (0, 6, 11):
        ref = mix(buf, idx, last, packed_queries[4 * il : 4 * il + 4])
        torch.testing.assert_close(actual[il], ref, atol=0.0006, rtol=0.001)
    for first, count, n in ((0, 3, 1), (0, 3, 3), (3, 1, 1)):
        cur = torch.randn(n, m, D, device=device)
        nw = torch.randn(n, D, device=device)
        x = torch.ops.inkling2_xpu.control_many(cur, msgs, nw, dq, ds, uq, us, first, count)
        ref = []
        for i in range(count):
            c = first + i
            a = cur[i % n]
            msg = msgs[c]
            h = torch.cat([norm(a), norm(msg)], dim=-1) @ down[c].T
            gain, forget = (h @ up[c]).tanh().unbind(0)
            ref.append(norm((1 - forget) * a + gain * msg) * nw[i % n])
        expected = torch.stack(ref)
        err = (x - expected).abs().max().item()
        print(m, first, count, n, err, flush=True)
        torch.testing.assert_close(x, expected, atol=0.001, rtol=0.001)
