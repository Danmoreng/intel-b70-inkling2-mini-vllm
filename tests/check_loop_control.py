"""Check decoded small-M loop controller boundaries on the B70."""

import sys
import torch

torch.ops.load_library(sys.argv[1])
torch.manual_seed(37)


def norm(x):
    return x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt()


for m in [1, 2, 4, 8]:
    for factor in [0, 1e-4, 1, 10]:
        cur, msg = (torch.randn(m, 2304, device="xpu") * factor for _ in range(2))
        gate = torch.randn(m, 4608, device="xpu") * factor
        weight = torch.randn(2304, device="xpu")
        pre = torch.ops.inkling2_xpu.control_input(cur, msg)
        post = torch.ops.inkling2_xpu.control_output(cur, msg, gate, weight)
        gain, forget = gate.tanh().chunk(2, -1)
        ref = norm((1 - forget) * cur + gain * msg) * weight
        torch.testing.assert_close(pre, torch.cat([norm(cur), norm(msg)], -1), atol=3e-6, rtol=2e-6)
        torch.testing.assert_close(post, ref, atol=3e-6, rtol=2e-6)
print("All 16 XPU control input/output cases passed.")
