"""Compare real-code teacher-forced target logits one token at a time and in verification blocks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from tokenizers import Tokenizer

from inkling_trellis.inkling2.sgl_model import Inkling2LoopForConditionalGeneration
from inkling_xpu.split_attention import local_xmx
from inkling_xpu.vllm_model import Inkling2B70ForCausalLM


def shards(checkpoint):
    for path in sorted(checkpoint.glob("dense-*.safetensors")):
        with safe_open(path, framework="pt") as shard:
            for name in shard.keys():
                yield name, shard.get_tensor(name)


def make_model(checkpoint):
    config = json.loads((checkpoint / "config.json").read_text())["text_config"]
    model = Inkling2B70ForCausalLM.__new__(Inkling2B70ForCausalLM)
    Inkling2LoopForConditionalGeneration.__init__(
        model, SimpleNamespace(inkling_text=config, inkling_vision=None), external_slots=1
    )
    model.cache_dtype = torch.float8_e4m3fn
    model.page_tokens = 16
    model.prefix_block = 0
    model.checkpoint = checkpoint
    model.want_dspark = False
    model.logit_cap = config["final_logit_softcapping"]
    model.packed_kv = os.environ.get("INKLING2_PACKED_KV") == "1"
    model.use_paged_xmx = os.environ.get("INKLING2_PAGED_XMX") == "1"
    model.local_xmx = local_xmx if os.environ.get("INKLING2_LOCAL_XMX") == "1" else None
    model.load_weights(shards(checkpoint))
    return model


def score(ref, other, truth):
    p = ref.float().log_softmax(-1)
    q = other.float().log_softmax(-1)
    ids = torch.arange(truth.numel())
    return dict(
        positions=len(ref),
        same_top1=int((p.argmax(-1) == q.argmax(-1)).sum()),
        mean_forward_kl=float((p.exp() * (p - q)).sum(-1).mean()),
        reference_nll=float(-p[ids, truth].mean()),
        candidate_nll=float(-q[ids, truth].mean()),
        max_abs_logit_delta=float((ref - other).abs().max()),
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("text", type=Path)
    p.add_argument("--prefix", type=int, default=1021)
    p.add_argument("--tokens", type=int, default=96)
    p.add_argument("--save", type=Path)
    p.add_argument("--reference", type=Path)
    args = p.parse_args()
    if args.tokens < 8 or args.tokens % 8 or args.prefix < 1:
        p.error("--tokens must be a positive multiple of eight and --prefix must be positive")
    torch.set_num_threads(3)
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    text = args.text.read_bytes()
    ids = tokenizer.encode(text.decode(), add_special_tokens=False).ids
    if len(ids) < args.prefix + args.tokens + 1:
        raise ValueError("The code sample is shorter than the requested scoring span")
    ids = torch.tensor(ids, dtype=torch.long, device="xpu")
    model = make_model(args.checkpoint)
    npages = (args.prefix + args.tokens + 15) // 16
    caches = [
        torch.empty((npages, 8, 16, 256), device="xpu", dtype=model.cache_dtype)
        for _ in range(model.n_global)
    ]
    pages = torch.arange(npages, device="xpu", dtype=torch.int32)[None]

    def batch(start, rows):
        return SimpleNamespace(
            caches=caches,
            block_table=pages,
            seq_lens=torch.tensor([start + rows], dtype=torch.int32, device="xpu"),
            out_cache_loc=torch.arange(start, start + rows, device="xpu"),
            total_tokens=start + rows,
            first_position=start,
        )

    pre = batch(0, args.prefix)
    pos = pre.out_cache_loc
    reqs = [(0, 0, args.prefix, 0)]
    model._static = False
    model._reqs = reqs
    model._conv_pos_update = []
    model._prepare(pre, reqs, pos, kv_pool=True)
    model._forward_body(ids[: args.prefix], pos, pre, reqs, None, return_hidden=True)
    model._static = True
    logits = {rows: [] for rows in (8, 4, 1)}
    for base in range(args.prefix, args.prefix + args.tokens, 8):
        # These writes touch only the eight current slots. The ordinary pass
        # runs last to commit the same reference history for the next block.
        for rows in logits:
            for start in range(base, base + 8, rows):
                b = batch(start, rows)
                hidden = model._static_forward(ids[start : start + rows], b.out_cache_loc, b)
                logits[rows].append(model.compute_logits(hidden).cpu())
    logits = {rows: torch.cat(values) for rows, values in logits.items()}
    truth = ids[args.prefix + 1 : args.prefix + args.tokens + 1].cpu()
    report = dict(
        text_sha256=hashlib.sha256(text).hexdigest(),
        prefix_tokens=args.prefix,
        scored_tokens=args.tokens,
        against_ordinary={str(rows): score(logits[1], logits[rows], truth) for rows in (4, 8)},
    )
    if args.reference:
        baseline = torch.load(args.reference, map_location="cpu", weights_only=True)
        if baseline["text_sha256"] != report["text_sha256"] or not torch.equal(baseline["truth"], truth):
            raise ValueError("The saved reference used different teacher-forced tokens")
        report["against_reference"] = score(baseline["logits"], logits[1], truth)
    if args.save:
        torch.save(dict(logits=logits[1], truth=truth, text_sha256=report["text_sha256"]), args.save)
    print(json.dumps(report), flush=True)
    if any(
        v["mean_forward_kl"] > 0.01 or v["same_top1"] < 0.9 * args.tokens
        for v in report["against_ordinary"].values()
    ):
        raise RuntimeError("Large single-row versus multi-row target disagreement")


if __name__ == "__main__":
    main()
