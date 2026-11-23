"""Shared pieces of the IQR quantizer: reference model loader, NVFP4/MXFP8 unpack, block WHT, ref expert math.
IQR_MODEL = the native checkpoint (e.g. OpenUncensored/Inkling-2-Mini-Uncensored, one model-00001-of-00001 shard),
IQR_DIR = work dir (corpus/calib.npy in, calib/ hess/ q/ dense/ out)."""
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from inkling_trellis.inkling2 import ref as R

MODEL = os.environ["IQR_MODEL"]
SHARD = f"{MODEL}/model-00001-of-00001.safetensors"
IQR = os.environ["IQR_DIR"]
DEV = os.environ.get("IQR_DEV", "xpu")
TEST_T = int(os.environ.get("IQR_TEST_T", "0"))   # >0: truncate sequences (CPU smoke tests)
D, I, E = 2304, 1152, 128


class Ref(R.Inkling2Ref):
    """Inkling2Ref that can skip loading the expert tensors (they are then supplied per visit by an `experts` override)."""

    def __init__(self, device="xpu", load_experts=True):
        self.dev, self.dtype = torch.device(device), torch.bfloat16
        cfg = json.loads((Path(MODEL) / "config.json").read_text())["text_config"]
        self.cfg = cfg
        self.D, self.H, self.Hkv, self.hd = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.local = set(cfg["local_layer_ids"])
        lc = cfg["loop_config"]
        self.core = range(lc["core_start"], lc["core_end"])
        self.passes = lc["num_passes"]
        self.n_layers = cfg["num_hidden_layers"]
        self.window, self.rel_extent, self.d_rel = cfg["sliding_window_size"], cfg["rel_extent"], cfg["d_rel"]
        self.block = cfg["n_layers_per_attn_res_block"]
        self.w = {}
        with safe_open(SHARD, "pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("model.visual") or (not load_experts and ".mlp.experts." in k):
                    continue
                self.w[k.removeprefix("model.llm.")] = f.get_tensor(k).to(self.dev)
        self.visits = [(il, 0) for il in range(self.core.start)]
        for p in range(self.passes):
            self.visits += [(il, p) for il in self.core]
        self.visits += [(il, 0) for il in range(self.core.stop, self.n_layers)]
        self.reset()


def visits():
    v = [(il, 0) for il in range(6)] + [(il, p) for p in range(3) for il in range(6, 18)] + [(il, 0) for il in range(18, 24)]
    assert len(v) == 48
    return v


def route(x, gw, gb, cfg):
    """Exactly Inkling2Ref.experts' router. x [T, D] fp32-able."""
    logits = F.linear(x.float(), gw.float())
    probs = torch.sigmoid(logits)
    topi = (probs + gb.float()).topk(cfg["num_experts_per_tok"], -1).indices
    topw = probs.gather(1, topi)
    return topi, topw / topw.sum(-1, keepdim=True) * cfg["route_scale"]


def act_fn(h, cfg):
    gte, up = h[:, 0::2], h[:, 1::2]
    gc, uc = cfg["mlp_gate_softcap"], cfg["mlp_up_softcap"]
    return gc * torch.tanh(gte / gc) * torch.sigmoid(gte) * (uc * torch.tanh(up / uc))


def load_layer_raw(il, f=None):
    """Raw checkpoint expert tensors of one layer (CPU)."""
    own = f is None
    if own:
        f = safe_open(SHARD, "pt", device="cpu")
    p = f"model.llm.layers.{il}.mlp.experts."
    out = {k.removeprefix(p): f.get_tensor(k) for k in f.keys() if k.startswith(p)}
    return out


def dequant_layer(raw, dtype=torch.float32):
    """-> (w13 [E, 2I, D], w2 [E, D, I]) in `dtype`. fp32 is exact for MXFP8; for NVFP4 it is e2m1*e4m3 (exact) * alpha
    (one fp32 rounding). dtype=bf16 reproduces Inkling2Ref.experts bit for bit."""
    if "w13_weight_amax" in raw:
        a13 = raw["w13_weight_amax"] / raw["w13_weight_outer_scale_denominator"][0]
        a2 = raw["w2_weight_amax"] / raw["w2_weight_outer_scale_denominator"][0]
        return (R.nvfp4_dequant(raw["w13_weight"], raw["w13_weight_scale"], a13, dtype),
                R.nvfp4_dequant(raw["w2_weight"], raw["w2_weight_scale"], a2, dtype))
    return R.mxfp8_dequant(raw["w13_weight"], raw["w13_weight_scale"], dtype), R.mxfp8_dequant(raw["w2_weight"], raw["w2_weight_scale"], dtype)


_H = {}


def had128(device, dtype=torch.float32):
    key = (str(device), dtype)
    if key not in _H:
        h = torch.ones((1, 1), dtype=torch.float64)
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _H[key] = (h / math.sqrt(128)).to(device=device, dtype=dtype)
    return _H[key]


def rot_last(x):
    """Block-WHT (normalized, symmetric, self-inverse) on the last dim in blocks of 128."""
    shp = x.shape
    return (x.reshape(-1, 128) @ had128(x.device, x.dtype)).reshape(shp)


def rot_hess(H):
    """H [..., K, K] of unrotated inputs -> Hessian of rotated inputs R H R^T (R block-diag WHT)."""
    return rot_last(rot_last(H).transpose(-1, -2)).transpose(-1, -2)


def a8(x):
    """IQR activation fake-quant: rotate, int8 per (row, 128) with absmax/127 scale, dequantize, rotate back."""
    xr = rot_last(x.float()).reshape(*x.shape[:-1], -1, 128)
    s = xr.abs().amax(-1, keepdim=True).clamp(min=1e-30) / 127.0
    q = torch.round(xr / s).clamp(-127, 127) * s
    return rot_last(q.reshape(x.shape))
