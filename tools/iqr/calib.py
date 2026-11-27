"""Step 2a (B70): run the plain-torch reference forward (Inkling2Ref, native NVFP4/MXFP8 experts dequantized exactly as
the runtime oracle) over the calibration sequences and append each visit's expert input x (post mlp_norm, bf16) to
calib/x_v{visit}.bin. Resumable: calib/done.txt holds the number of finished sequences.
usage: calib.py <n_seqs> [max_minutes]
"""
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import iqrlib as L

n_want = int(sys.argv[1])
deadline = time.time() + 60 * float(sys.argv[2]) if len(sys.argv) > 2 else float("inf")
out = f"{L.IQR}/calib"
os.makedirs(out, exist_ok=True)
seqs = np.load(f"{L.IQR}/corpus/calib.npy")
done = int(open(f"{out}/done.txt").read()) if os.path.exists(f"{out}/done.txt") else 0
if L.TEST_T:
    seqs = seqs[:, :L.TEST_T]
T = seqs.shape[1]
rowbytes = L.D * 2
for v in range(48):   # truncate partial appends from an interrupted run
    p = f"{out}/x_v{v:02d}.bin"
    if os.path.exists(p):
        os.truncate(p, done * T * rowbytes)
files = [open(f"{out}/x_v{v:02d}.bin", "ab") for v in range(48)]

torch.set_grad_enabled(False)
ref = L.Ref(L.DEV, load_experts=True)
buf = []
orig = ref.experts


def rec(il, x):
    buf.append(x.to(torch.bfloat16))
    return orig(il, x)


ref.experts = rec
nll_sum = n_tok = 0
t0 = time.time()
for s in range(done, min(n_want, len(seqs))):
    if time.time() > deadline:
        break
    ids = torch.from_numpy(seqs[s]).long().to(L.DEV)
    ref.reset()
    buf.clear()
    logits = ref.forward(ids)
    nll = F.cross_entropy(logits[:-1].float(), ids[1:], reduction="sum").item()
    assert len(buf) == 48
    for v in range(48):
        files[v].write(buf[v].view(torch.int16).cpu().numpy().tobytes())
    for fh in files:
        fh.flush()
    with open(f"{out}/done.txt", "w") as fh:
        fh.write(str(s + 1))
    nll_sum += nll; n_tok += T - 1
    if (s - done) % 10 == 0:
        el = time.time() - t0
        print(f"seq {s + 1}/{n_want}  {(s + 1 - done) * T / el:.0f} tok/s  running ppl {math.exp(nll_sum / n_tok):.3f}", flush=True)
print(f"end: done {open(f'{out}/done.txt').read()} seqs, ppl {math.exp(nll_sum / max(n_tok, 1)):.3f}, "
      f"{time.time() - t0:.0f}s", flush=True)
