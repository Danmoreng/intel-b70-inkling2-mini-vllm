"""Stage 1 output: IQR expert shards for models/iqr4-unc, and their validation against the q/ fake-quant.

write <var> <out_dir>    : iqr-layer-XX.safetensors (codes/scales copied from q/<var>/Lxx.pt), iqr_config.json, and
                           model.safetensors.index.json (exl3-unc's dense entries + the IQR expert tensors)
validate <var> <out_dir> : per layer and expert, (a) checkpoint tensors == q/<var>/Lxx.pt, (b) the checkpoint decoded
                           with gptq.py's own dequant == dense/<var>/Lxx.pt (the bf16 experts evalkl.py measured), bitwise
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import iqrlib as L

mode, var, out = sys.argv[1], sys.argv[2], sys.argv[3]
SRC = os.environ["IQR_DENSE_SRC"]   # EXL3 checkout whose dense/MTP files the model dir reuses
NAMES = ("w13_codes", "w13_scales", "w2_codes", "w2_scales")
GS = 128
torch.set_grad_enabled(False)


def key(il, n):
    return f"model.llm.layers.{il}.mlp.experts.{n}"


def dequant(codes, scales, table):     # == gptq.py dequant (codes unpacked uint8 [E, R, K], scales fp32)
    wr = table[codes.long()] * scales.repeat_interleave(GS, -1)
    return L.rot_last(wr)


if mode == "write":
    os.makedirs(out, exist_ok=True)
    with open(f"{SRC}/model.safetensors.index.json") as fh:
        wm = {k: v for k, v in json.load(fh)["weight_map"].items() if not v.startswith("exl3-layer-")}
    total = 0
    for il in range(24):
        q = torch.load(f"{L.IQR}/q/{var}/L{il:02d}.pt")
        if not torch.equal(q["table"], torch.arange(-8, 8.)):
            raise ValueError(f"q/{var}/L{il:02d}.pt: IQR i4 expects the signed int4 table, got {q['table'].tolist()}")
        t = {key(il, n): q[n].contiguous() for n in NAMES}
        fn = f"iqr-layer-{il:02d}.safetensors"
        save_file(t, f"{out}/{fn}", metadata={"format": "iqr-gptq4-v1", "layer": str(il)})
        os.chmod(f"{out}/{fn}", 0o644)
        for k, v in t.items():
            wm[k] = fn
            total += v.numel() * v.element_size()
        print(f"layer {il:2d} written", flush=True)
    with open(f"{out}/model.safetensors.index.json", "w") as fh:
        json.dump({"metadata": {"iqr_expert_bytes": total}, "weight_map": dict(sorted(wm.items()))}, fh, indent=2)
    with open(f"{out}/iqr_config.json", "w") as fh:
        json.dump({
            "format": "iqr-gptq4-v1",
            "experts": "routed experts of all 24 layers (128 each); dense, MTP and vision tensors are exl3-unc's BF16",
            "source_experts": "OpenUncensored Inkling 2 Mini NVFP4 (layers 0-19) / MXFP8 (20-23), unpacked losslessly",
            "rotation": "W_r = W R, R = block-diagonal normalized Walsh-Hadamard (128) over the input dim (symmetric, self-inverse)",
            "codes": "uint8, two 4-bit codes per byte along K, low nibble = even k; value = code - 8 (-8..7)",
            "scales": "bf16 per (output row, 128 of K): W_r[r, k] = scale[r, k // 128] * value[r, k]",
            "w13_rows": "2*I rows, interleaved as in the source checkpoint: row 2j = gate j, row 2j+1 = up j",
            "quantizer": "GPTQ (tools/iqr/gptq.py variant gptq4): rotated Hessian, damp 0.01, group-128 scale grid search",
            "calibration": os.environ.get("IQR_CALIB_NOTE", "corpus/calib.npy, Hessians over all 48 visits, routing-weighted"),
            "kernel_arithmetic": "int8 rotated activations per (row, 128), int8 x u4 DPAS (inkling_xpu/csrc/iqr_moe.sycl)",
        }, fh, indent=2)
    print("WRITE-DONE", total, flush=True)
else:
    table = torch.arange(-8, 8.)
    ok = True
    for il in range(24):
        q = torch.load(f"{L.IQR}/q/{var}/L{il:02d}.pt")
        dn = torch.load(f"{L.IQR}/dense/{var}/L{il:02d}.pt")
        with safe_open(f"{out}/iqr-layer-{il:02d}.safetensors", "pt", device="cpu") as f:
            c = {n: f.get_tensor(key(il, n)) for n in NAMES}
        same_q = all(torch.equal(c[n], q[n]) for n in NAMES)
        res = []
        for p in ("w13", "w2"):
            pk = c[p + "_codes"]
            codes = torch.stack([pk & 15, pk >> 4], -1).reshape(*pk.shape[:2], -1)
            w = dequant(codes, c[p + "_scales"].float(), table).to(torch.bfloat16)
            diff = (w != dn[p]).flatten(1).sum(1)          # per expert
            res.append(int((diff > 0).sum()))
        good = same_q and res == [0, 0]
        ok &= good
        print(f"layer {il:2d}: tensors == q/{var}: {same_q}; experts whose decode != dense/{var}: w13 {res[0]}/128 "
              f"w2 {res[1]}/128  {'OK' if good else 'MISMATCH'}", flush=True)
    print("VALIDATE", "ALL IDENTICAL" if ok else "FAILED", flush=True)
