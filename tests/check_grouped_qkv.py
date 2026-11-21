import json
import statistics
import time

import torch
from inkling_trellis.inkling2.dense import Q8Linear

from inkling_xpu.grouped_dense import GroupedQKV

torch.manual_seed(81)
torch.ops.load_library("/lab/.work/build/inkling2-qkv.so")
d = "xpu"
models = []
for rep in range(6):
    kv = 1056 if rep % 2 else 1024
    p = [Q8Linear(torch.randn(n, 2304, device=d).bfloat16(), keep_bf16=False) for n in (6552, kv, kv)]
    models.append((p, GroupedQKV(p, 2)))
for m in (1, 3, 4, 8):
    x = [torch.randn(m, 2304, device=d) for j in range(3)]
    ctrl = [[op(a) for op, a in zip(p, x)] for p, _ in models]
    for splits in (0, 1, 2, 4, 8):

        def run():
            result = []
            for p, qkv in models:
                qkv.splits = splits
                result.append(qkv(*x) if splits else [op(a) for op, a in zip(p, x)])
            return result

        y = run()
        e = max(float((a - b).norm() / a.norm()) for aa, bb in zip(ctrl, y) for a, b in zip(aa, bb))
        assert e < 0.0001, (m, splits, e)
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g):
            run()
        g.replay()
        times = []
        for i in range(7):
            torch.xpu.synchronize()
            start = time.monotonic()
            for j in range(6):
                g.replay()
            torch.xpu.synchronize()
            times.append((time.monotonic() - start) * 1e6 / (6 * len(models)))
        print(json.dumps(dict(m=m, splits=splits, us=statistics.median(times), relative_l2=e)), flush=True)
