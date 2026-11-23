"""Step 3: GPTQ (Hadamard-rotated) to IQR-style formats, per layer, all experts batched.

Format: W_r = W @ R (R = block-diag normalized WHT-128 over the input dim), codes c in a 2^b-entry table T,
one bf16 scale s per (output row, 128 of K): W_r ~= s * T[c].
Variants (name: table, gptq?):
  rtn4  : int -8..7, round-to-nearest (no GPTQ)    gptq4 : int -8..7
  gptq3 : int -4..3                                cb4   : 16-entry Lloyd table (int8 levels)   cb3 : 8-entry Lloyd
Scale per group: grid search s = f * absmax / max|T| over f in [0.55, 1.05], minimising the diag(H)-weighted error.
Writes q/<var>/Lxx.pt (codes uint8, scales bf16, table) and dense/<var>/Lxx.pt (fake-quant bf16, original basis).
usage: gptq.py <layers|all> <variants,comma> [device] [expert_chunk]
"""
import json
import os
import sys
import time

import torch
from safetensors import safe_open

import iqrlib as L

layers = list(range(24)) if sys.argv[1] == "all" else [] if sys.argv[1] == "tables" else [int(a) for a in sys.argv[1].split(",")]
variants = sys.argv[2].split(",")
dev = sys.argv[3] if len(sys.argv) > 3 else "cpu"
EC = int(sys.argv[4]) if len(sys.argv) > 4 else 32
GS = 128
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("highest")


def lloyd_table(levels):
    """Symmetric Lloyd-Max table for w/absmax(group) of rotated weights (fitted once on layers 0/12/20, cached)."""
    path = f"{L.IQR}/q/lloyd{levels}.json"
    if os.path.exists(path):
        return torch.tensor(json.load(open(path)))
    vals = []
    f = safe_open(L.SHARD, "pt", device="cpu")
    g = torch.Generator().manual_seed(0)
    for il in (0, 12, 20):
        w13, w2 = L.dequant_layer(L.load_layer_raw(il, f), torch.float32)
        for w in (w13[:16], w2[:16]):
            wr = L.rot_last(w).reshape(-1, GS)
            v = wr / wr.abs().amax(-1, keepdim=True).clamp(min=1e-30)
            vals.append(v.flatten()[torch.randint(0, v.numel(), (2_000_000,), generator=g)])
    x = torch.cat(vals).abs().double()
    half = levels // 2
    c = torch.quantile(x, (torch.arange(half, dtype=torch.float64) + 0.5) / half)
    for _ in range(200):
        mids = (c[1:] + c[:-1]) / 2
        idx = torch.bucketize(x, mids)
        c = torch.stack([x[idx == k].mean() for k in range(half)])
    t = torch.cat([-c.flip(0), c])
    t = torch.round(t / t.abs().max() * 127)       # int8 table, as the kernel's cb4 path stores it
    json.dump(t.tolist(), open(path, "w"))
    return t


TABLES = {"rtn4": torch.arange(-8, 8.), "gptq4": torch.arange(-8, 8.), "gptq3": torch.arange(-4, 4.)}
for name, lv in (("cb4", 16), ("cb3", 8)):
    if name in variants:
        TABLES[name] = lloyd_table(lv)
FACTORS = torch.linspace(0.55, 1.05, 21)
if sys.argv[1] == "tables":
    print({k: v.tolist() for k, v in TABLES.items()})
    sys.exit(0)


def nearest(v, mids):
    return torch.bucketize(v.contiguous(), mids)


def find_scale(Wb, hdiag, table, mids):
    """Wb [E, R, G], hdiag [E, G] -> s [E, R, 1] (bf16-rounded)."""
    amax = Wb.abs().amax(-1, keepdim=True).clamp(min=1e-12) / table.abs().max()
    best, best_s = None, None
    for f in FACTORS.tolist():
        s = (amax * f).to(torch.bfloat16).float()
        q = table[nearest(Wb / s, mids)] * s
        err = ((Wb - q) ** 2 * hdiag.unsqueeze(1)).sum(-1, keepdim=True)
        if best is None:
            best, best_s = err, s
        else:
            m = err < best
            best = torch.where(m, err, best)
            best_s = torch.where(m, s, best_s)
    return best_s


def hinv_chol(H, tag, damp=0.01, dtype=torch.float64):
    """Upper Cholesky factor of H^-1 (GPTQ), damping escalated on failure; zero-diag -> 1. Runs on H's device."""
    H = H.to(dtype).clone()
    d = H.diagonal(dim1=-2, dim2=-1)
    d[d <= 0] = 1.0
    eye = torch.eye(H.shape[-1], dtype=dtype, device=H.device)
    for dp in (damp, 0.1, 1.0):
        Hd = H + dp * d.mean(-1).view(-1, 1, 1) * eye
        Lc, info = torch.linalg.cholesky_ex(Hd)
        if int(info.max()) == 0:
            Hi = torch.cholesky_inverse(Lc)
            U, info2 = torch.linalg.cholesky_ex(Hi, upper=True)
            if int(info2.max()) == 0:
                if dp != damp:   # escalation changes the whole chunk's GPTQ: make it visible
                    print(f"  {tag}: cholesky needed damping {dp} (default {damp}, {dtype})", flush=True)
                return U.float()
        print(f"  {tag}: cholesky failed at damping {dp} ({dtype}), escalating", flush=True)
    raise RuntimeError(f"{tag}: cholesky failed")


def gptq(W, U, hdiag, table, use_gptq):
    """W [E, R, K] rotated fp32 (modified in place), U [E, K, K] upper chol of H^-1. -> codes uint8, scales."""
    E, Rr, K = W.shape
    table = table.to(W.device)
    mids = ((table[1:] + table[:-1]) / 2).contiguous()
    codes = torch.empty(E, Rr, K, dtype=torch.uint8, device=W.device)
    scales = torch.empty(E, Rr, K // GS, device=W.device)
    for b0 in range(0, K, GS):
        Wb = W[:, :, b0:b0 + GS].clone()
        s = find_scale(Wb, hdiag[:, b0:b0 + GS], table, mids)
        scales[:, :, b0 // GS] = s[..., 0]
        if not use_gptq:
            codes[:, :, b0:b0 + GS] = nearest(Wb / s, mids).to(torch.uint8)
            continue
        Ub = U[:, b0:b0 + GS, b0:b0 + GS]
        Err = torch.empty_like(Wb)
        for i in range(GS):
            w = Wb[:, :, i]
            c = nearest(w / s[..., 0], mids)
            codes[:, :, b0 + i] = c.to(torch.uint8)
            err = (w - table[c] * s[..., 0]) / Ub[:, i, i].unsqueeze(1)
            Wb[:, :, i:] -= err.unsqueeze(2) * Ub[:, i, i:].unsqueeze(1)
            Err[:, :, i] = err
        if b0 + GS < K:
            W[:, :, b0 + GS:] -= torch.bmm(Err, U[:, b0:b0 + GS, b0 + GS:])
    return codes, scales


def dequant(codes, scales, table):
    t = table.to(codes.device)
    wr = t[codes.long()] * scales.repeat_interleave(GS, -1)
    return L.rot_last(wr)          # back to the original basis (R is symmetric and self-inverse)


f = safe_open(L.SHARD, "pt", device="cpu")
for il in layers:
    t0 = time.time()
    w13, w2 = L.dequant_layer(L.load_layer_raw(il, f), torch.float32)
    hs = torch.load(f"{L.IQR}/hess/L{il:02d}.pt")
    res = {v: {} for v in variants}
    stats = {v: {} for v in variants}
    for proj, W0, Hk in (("w13", w13, "H13"), ("w2", w2, "H2")):
        K = W0.shape[-1]
        cod = {v: torch.empty(L.E, W0.shape[1], K, dtype=torch.uint8) for v in variants}
        sca = {v: torch.empty(L.E, W0.shape[1], K // GS, dtype=torch.bfloat16) for v in variants}
        num = {v: 0.0 for v in variants}; den = 0.0; wn = {v: 0.0 for v in variants}; wd = 0.0
        for e0 in range(0, L.E, EC):
            H = hs[Hk][e0:e0 + EC].float()
            nz = hs["count"][e0:e0 + EC] == 0
            tag = f"layer {il} {proj} experts {e0}-{e0 + EC - 1}"
            if nz.any():   # experts never routed in calibration: identity Hessian (plain RTN-like rounding)
                H[nz] = torch.eye(K)
                print(f"  {tag}: {int(nz.sum())} never-routed experts {(nz.nonzero().flatten() + e0).tolist()} "
                      "use an identity Hessian", flush=True)
            Hr = L.rot_hess(H.to(dev))
            try:     # fp32 Cholesky on the device (standard GPTQ practice); float64 CPU fallback
                U = hinv_chol(Hr, tag, dtype=torch.float32)
            except (RuntimeError, NotImplementedError) as ex:
                print("  device cholesky failed, CPU float64:", str(ex)[:80], flush=True)
                U = hinv_chol(Hr.cpu(), tag)
                U = U.to(dev)
            hdiag = Hr.diagonal(dim1=-2, dim2=-1).clamp(min=1e-12)
            Wr0 = L.rot_last(W0[e0:e0 + EC].to(dev))
            den += torch.einsum("erk,ekl,erl->", Wr0, Hr, Wr0).item()
            wd += Wr0.pow(2).sum().item()
            for v in variants:
                c, s = gptq(Wr0.clone(), U, hdiag, TABLES[v], v != "rtn4")
                cod[v][e0:e0 + EC] = c.cpu()
                sca[v][e0:e0 + EC] = s.to(torch.bfloat16).cpu()
                dW = TABLES[v].to(dev)[c.long()] * s.to(torch.bfloat16).float().repeat_interleave(GS, -1) - Wr0
                num[v] += torch.einsum("erk,ekl,erl->", dW, Hr, dW).item()
                wn[v] += dW.pow(2).sum().item()
        for v in variants:
            res[v][proj + "_codes"], res[v][proj + "_scales"] = cod[v], sca[v]
            stats[v][proj] = (num[v] / den, (wn[v] / wd) ** 0.5)
    for v in variants:
        os.makedirs(f"{L.IQR}/q/{v}", exist_ok=True)
        os.makedirs(f"{L.IQR}/dense/{v}", exist_ok=True)
        packed = {k: (t[..., 0::2] | (t[..., 1::2] << 4)) if k.endswith("_codes") else t for k, t in res[v].items()}
        torch.save({**packed, "table": TABLES[v], "codes_packing": "two 4-bit codes per byte along K (low nibble = even k)"},
                   f"{L.IQR}/q/{v}/L{il:02d}.pt")
        torch.save({"w13": dequant(res[v]["w13_codes"], res[v]["w13_scales"].float(), TABLES[v]).to(torch.bfloat16),
                    "w2": dequant(res[v]["w2_codes"], res[v]["w2_scales"].float(), TABLES[v]).to(torch.bfloat16)},
                   f"{L.IQR}/dense/{v}/L{il:02d}.pt")
        st = stats[v]
        print(f"layer {il:2d} {v:6s}: proxy loss w13 {st['w13'][0]:.5f} w2 {st['w2'][0]:.5f} | rel weight err "
              f"w13 {st['w13'][1]:.4f} w2 {st['w2'][1]:.4f}", flush=True)
    print(f"layer {il:2d} done in {time.time() - t0:.0f}s", flush=True)
