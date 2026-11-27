"""Step 2b: per-(layer, expert) input Hessians from the dumped visit inputs, accumulated over ALL visits of the layer.

H13[e] = sum_t w_te^2 x_t x_t^T          (gate_up input, x = bf16 post-norm input, as the expert GEMM sees it)
H2[e]  = sum_t w_te^2 a_te a_te^T        (down input, a = bf16 activation computed with the reference NVFP4 weights)
w_te = routing weight (route_scale applied), so the GPTQ objective weights each token by its share of the layer output.
Unrotated; GPTQ rotates (R H R^T). Saved fp32 to hess/Lxx.pt with per-expert token counts.
usage: hess.py <layers|all> [device] [n_seqs]
"""
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

import iqrlib as L

layers = list(range(24)) if sys.argv[1] == "all" else [int(a) for a in sys.argv[1].split(",")]
dev = sys.argv[2] if len(sys.argv) > 2 else "xpu"
done = int(open(f"{L.IQR}/calib/done.txt").read())
n_seqs = int(sys.argv[3]) if len(sys.argv) > 3 else done
T = L.TEST_T or 2048
rows_all = n_seqs * T
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("highest")
import json
cfg = json.load(open(f"{L.MODEL}/config.json"))["text_config"]
f = safe_open(L.SHARD, "pt", device="cpu")
vis = L.visits()
B = 16384
for il in layers:
    t0 = time.time()
    gw = f.get_tensor(f"model.llm.layers.{il}.mlp.gate.weight").to(dev)
    gb = f.get_tensor(f"model.llm.layers.{il}.mlp.gate.bias").to(dev)
    w13, _ = L.dequant_layer(L.load_layer_raw(il, f), torch.bfloat16)
    w13 = w13.to(dev)
    H13 = torch.zeros(L.E, L.D, L.D, device=dev)
    H2 = torch.zeros(L.E, L.I, L.I, device=dev)
    cnt = torch.zeros(L.E, dtype=torch.int64, device=dev)
    wsum = torch.zeros(L.E, dtype=torch.float64, device=dev)
    for vi in [v for v, (l, _) in enumerate(vis) if l == il]:
        mm = np.memmap(f"{L.IQR}/calib/x_v{vi:02d}.bin", dtype=np.int16, mode="r").reshape(-1, L.D)
        assert mm.shape[0] >= rows_all
        for r0 in range(0, rows_all, B):
            x = torch.from_numpy(np.ascontiguousarray(mm[r0:r0 + B])).view(torch.bfloat16).to(dev)
            topi, topw = L.route(x, gw, gb, cfg)
            for e in topi.unique().tolist():
                rows, slot = (topi == e).nonzero(as_tuple=True)
                xe = x[rows]
                w = topw[rows, slot].float().unsqueeze(1)
                a = L.act_fn(torch.nn.functional.linear(xe, w13[e]).float(), cfg).to(torch.bfloat16).float() * w
                xw = xe.float() * w
                H13[e].addmm_(xw.t(), xw)
                H2[e].addmm_(a.t(), a)
                cnt[e] += rows.numel()
                wsum[e] += w.double().pow(2).sum()
    torch.save({"H13": H13.cpu(), "H2": H2.cpu(), "count": cnt.cpu(), "w2sum": wsum.cpu(), "n_seqs": n_seqs},
               f"{L.IQR}/hess/L{il:02d}.pt")
    print(f"layer {il:2d}: {time.time() - t0:.0f}s, tokens/expert min {int(cnt.min())} median {int(cnt.median())} "
          f"max {int(cnt.max())}", flush=True)
    del H13, H2, w13
