import statistics
import time

import torch
from inkling_trellis.inkling2.attn_triton import conv_decode1
from inkling_trellis.inkling2.fused import attn_prep

torch.ops.load_library("/lab/.work/build/inkling2-fused-prep.so")
torch.manual_seed(35)
d = "xpu"
for m, start in [(1, 0), (2, 101), (4, 1062), (8, 9000)]:
    for local in (True, False):
        base = [
            torch.randn(m, 6552, device=d),
            *[torch.randn(m, 1024 if local else 1056, device=d) for _ in range(2)],
        ]
        qscale, rscale, kw, vw = [
            torch.randn(*shape, device=d) for shape in [(24, 128), (24, 16), (1024, 4), (1024, 4)]
        ]
        ring = torch.randn(1, 36, 6656, device=d)
        rp = torch.full((1, 36), -1, device=d, dtype=torch.long)
        pp = torch.arange(max(0, start - 5), start + m, device=d)
        rp[:, pp % 36] = pp
        pos = pp[-m:]
        slot = torch.zeros_like(pos)
        src = ring.clone()
        other = ring.clone()
        lkv = [torch.zeros(1056, 8, 128, device=d, dtype=torch.bfloat16) for _ in range(2)]
        vs = [t if local else t[:, :1024] for t in base[1:]]
        taps = [None if local else t[:, 1024:].contiguous() for t in base[1:]]

        def original():
            convs = [
                conv_decode1(x.contiguous(), w, tap, src, rp, slot, pos, i * 1024, m)
                for i, (x, tap, w) in enumerate(zip(vs, taps, [kw, vw]))
            ]
            out = attn_prep(
                base[0],
                *convs,
                pos,
                qscale,
                rscale,
                H=24,
                HKV=8,
                D=128,
                DR=16,
                qk_thr=22.62741699796952,
                rel_thr=8.0,
                tau_alpha=None if local else 0.1,
                tau_floor=None if local else 8192.0,
            )
            if local:
                for buf, x in zip(lkv, out[4:]):
                    buf[pos % 1056] = x
            return out

        def new():
            return torch.ops.inkling_prep.forward(
                *base,
                pos,
                qscale,
                rscale,
                kw,
                vw,
                other,
                rp,
                lkv[0] if local else None,
                lkv[1] if local else None,
                22.62741699796952,
                8.0,
                0.1,
                8192.0,
            )

        ref = original()
        for b in lkv:
            b.fill_(88)
        res = new()
        if local:
            res[4:] = [z[pos % 1056] for z in lkv]
        delta = [float((a.float() - b.float()).norm() / a.float().norm()) for a, b in zip(ref, res)]
        assert max(delta) < 0.0005 and torch.equal(src, other), (m, start, local, delta)
        times = []
        for call in (original, new):
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                call()
            graph.replay()
            r = []
            for i in range(4):
                torch.xpu.synchronize()
                now = time.monotonic()
                for j in range(50):
                    graph.replay()
                torch.xpu.synchronize()
                r.append((time.monotonic() - now) * 1e6 / 50)
            times.append(round(statistics.median(r), 2))
        print(m, start, local, delta, times, flush=True)
