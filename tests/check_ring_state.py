"""Compare the target's saved ring state with an uninterrupted branch on shuffled KV pages."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer

from check_verifier import make_model, score


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("text", type=Path)
    args = parser.parse_args()
    torch.manual_seed(832)
    torch.set_num_threads(3)
    model = make_model(args.checkpoint)
    prefix, discard, followup = 8192, 128, 23
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    corpus = tokenizer.encode(args.text.read_text(), add_special_tokens=False).ids
    count = prefix + discard + followup + 1
    tokens = torch.tensor((corpus * (count // len(corpus) + 1))[:count], device="xpu")
    npages = (count + 15) // 16
    pages = torch.randperm(npages, dtype=torch.int32, device="xpu")[None]
    caches = [
        torch.empty(npages, 8, 16, 256, device="xpu", dtype=model.cache_dtype) for _ in range(model.n_global)
    ]

    def forward(ids, start):
        size = len(ids)
        pos = torch.arange(start, start + size, device="xpu")
        batch = SimpleNamespace(
            caches=caches,
            block_table=pages,
            seq_lens=torch.tensor([start + size], dtype=torch.int32, device="xpu"),
            out_cache_loc=pages[0, pos // 16].long() * 16 + pos % 16,
            first_position=start,
            total_tokens=start + size,
        )
        model._static = size == 4
        if model._static:
            return model._static_forward(ids, pos, batch)
        reqs = [(0, 0, size, start)]
        model._reqs = reqs
        model._conv_pos_update = []
        model._prepare(batch, reqs, pos, kv_pool=True)
        return model._forward_body(ids, pos, batch, reqs, None, return_hidden=True)

    model.saved_rings = []
    for start in range(0, prefix, 2048):
        forward(tokens[start : start + 2048], start)
    model._save_history(prefix)
    question = tokens[prefix + discard : prefix + discard + followup]
    truth = tokens[prefix + discard + 1 : prefix + discard + followup + 1].cpu()
    original = model.compute_logits(forward(question, prefix)).cpu()
    # The first branch touched 23 positions. Restore the saved frontier, as
    # a client would do after discarding a much longer generated continuation.
    model.ring_high = prefix + discard
    model._restore_history(prefix)
    for offset in range(0, discard, 4):
        forward(tokens[prefix + offset : prefix + offset + 4], prefix + offset)
    model.ring_high = prefix + discard
    model._restore_history(prefix)
    for live, saved in zip(model._ring_buffers(), model.saved_rings, strict=True):
        assert torch.equal(live.view(torch.uint8), saved.view(torch.uint8))
    replay = model.compute_logits(forward(question, prefix)).cpu()
    result = score(original, replay, truth)
    print(json.dumps(dict(prefix=prefix, discarded=discard, followup=followup, **result)), flush=True)
    if result["mean_forward_kl"] > 0.001 or result["same_top1"] != followup:
        raise RuntimeError("Restored target branch disagrees with the uninterrupted control")


if __name__ == "__main__":
    main()
