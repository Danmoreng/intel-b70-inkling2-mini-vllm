#!/usr/bin/env python3
"""Measure one uncached synthetic prefill and a fixed decode at each exact prompt length."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import time
import urllib.request
from uuid import uuid4

counters = runpy.run_path(str(Path(__file__).with_name("bench-frozen-serving.py")))["counters"]


def post(base, path, data):
    request = urllib.request.Request(base + path, json.dumps(data).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=150)


def measure(base, model, ids, output):
    before = counters(base)
    payload = dict(model=model, prompt=ids, max_tokens=output, temperature=1.0, top_p=0.95, top_k=20,
                   seed=42, ignore_eos=True, cache_salt=uuid4().hex, stream=True, stream_options={"include_usage": True})
    start = time.monotonic()
    first = usage = None
    with post(base, "/v1/completions", payload) as response:
        for line in response:
            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                continue
            packet = json.loads(line[6:])
            if packet.get("error"):
                raise RuntimeError(packet["error"])
            if packet.get("usage"):
                usage = packet["usage"]
            if first is None and any(c.get("text") for c in packet.get("choices", [])):
                first = time.monotonic() - start
    wall = time.monotonic() - start
    for _ in range(20):
        after = counters(base)
        delta = {key: after[key] - value for key, value in before.items()}
        if delta["request_decode_time_seconds_count"]:
            break
        time.sleep(0.25)
    if (first is None or usage is None or usage["prompt_tokens"] != len(ids)
            or usage["completion_tokens"] != output or delta["prompt_tokens_cached_total"]
            or delta["request_decode_time_seconds_count"] != 1 or delta["spec_decode_num_drafts_total"] <= 0):
        raise RuntimeError(f"Incomplete, cached or non-speculative measurement: {usage}, {delta}")
    return dict(prompt_tokens=len(ids), output_tokens=output, cached_tokens=0, first_seconds=first,
                prefill_tps=len(ids)/delta["request_prefill_time_seconds_sum"],
                decode_tps=(output-1)/delta["request_decode_time_seconds_sum"],
                client_decode_tps=(output-1)/(wall-first), wall_seconds=wall, counters=delta,
                prompt_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("corpus", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--url", default="http://127.0.0.1:8093")
    p.add_argument("--model", default="inkling2-mini-exl3-b70")
    args = p.parse_args()
    corpus = args.corpus.read_text()[:600000]
    question = "Review this source archive and explain in detail how a maintainer could test its most important invariants."
    with post(args.url, "/tokenize", dict(model=args.model, messages=[dict(role="user", content=corpus+"\n\n"+question)],
                                         add_generation_prompt=True, chat_template_kwargs={"reasoning_effort":"medium"})) as response:
        original = json.load(response)["tokens"]
    if len(original) < 128000:
        raise ValueError("The tokenized corpus must contain at least 128,000 tokens")
    # Preserve the rendered question and assistant header when shortening the source archive.
    suffix = original[-128:]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        for length in (1024, 2000, 4000, 16000, 32000, 64000, 128000):
            row = measure(args.url, args.model, original[:length-len(suffix)] + suffix, 256)
            if length != 1024:
                print(json.dumps(row), file=output, flush=True)
            print(length, round(row["prefill_tps"], 2), round(row["decode_tps"], 2), flush=True)


if __name__ == "__main__":
    main()
