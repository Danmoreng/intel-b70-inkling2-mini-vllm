#!/usr/bin/env python3
"""Replay the archived single-request Qwen B70 source-review prompts byte-for-byte."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request
from uuid import uuid4


def counters(base):
    with urllib.request.urlopen(base + "/metrics", timeout=8) as f:
        raw = f.read().decode()
    names = (
        "request_prefill_time_seconds_sum",
        "request_decode_time_seconds_sum",
        "request_decode_time_seconds_count",
        "prompt_tokens_cached_total",
        "spec_decode_num_draft_tokens_total",
        "spec_decode_num_accepted_tokens_total",
    )
    return {
        name: sum(float(value) for value in re.findall(rf"^vllm:{name}\{{[^\n]+\}} ([\d.e+\-]+)", raw, re.M))
        for name in names
    }


def replay(args, folder):
    recorded = json.loads((folder / "request-1.json").read_text())
    prompt = (folder / "prompt-1.txt").read_text()
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    if digest != recorded["prompt_sha256"]:
        raise ValueError("Archived prompt differs from the recorded request")
    payload = {
        key: recorded[key] for key in ("max_tokens", "temperature", "top_p", "top_k", "seed", "ignore_eos")
    }
    payload.update(
        cache_salt=uuid4().hex,
        model=args.model,
        messages=[dict(role="user", content=prompt)],
        reasoning_effort="none",
        stream=True,
        stream_options=dict(include_usage=True),
    )
    req = urllib.request.Request(
        args.url + "/v1/chat/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    before = counters(args.url)
    t = time.monotonic()
    ttft = None
    usage = None
    generated = []
    with urllib.request.urlopen(req, timeout=300) as resp:
        for entry in resp:
            if not entry.startswith(b"data: ") or entry[6:].strip() == b"[DONE]":
                continue
            packet = json.loads(entry[6:])
            if "error" in packet:
                raise RuntimeError(packet["error"])
            if packet.get("usage"):
                usage = packet["usage"]
            for choice in packet.get("choices") or []:
                delta = choice["delta"]
                if delta.get("content"):
                    generated.append(delta["content"])
                if ttft is None and any(
                    delta.get(k) for k in ("content", "reasoning", "reasoning_content", "tool_calls")
                ):
                    ttft = time.monotonic() - t
    wall = time.monotonic() - t
    for _ in range(20):
        after = counters(args.url)
        if after["request_decode_time_seconds_count"] > before["request_decode_time_seconds_count"]:
            break
        time.sleep(0.25)
    if after["request_decode_time_seconds_count"] - before["request_decode_time_seconds_count"] != 1:
        raise RuntimeError("Expected metrics for exactly one completed request")
    stat = {k: v - before[k] for k, v in after.items()}
    if ttft is None or usage is None or usage["completion_tokens"] != payload["max_tokens"]:
        raise RuntimeError("Missing usage or incomplete fixed-length output")
    tokens = usage["prompt_tokens"]
    cached = stat["prompt_tokens_cached_total"]
    if cached:
        raise RuntimeError(f"Expected a cold review, observed {cached} cached prompt tokens")
    review = "".join(generated)
    return dict(
        case=folder.name,
        prompt_sha256=digest,
        response_sha256=hashlib.sha256(review.encode()).hexdigest(),
        reviewed_files_mentioned=[
            name for name in recorded.get("source_paths", []) if Path(name).name in review
        ],
        input_tokens=tokens,
        output_tokens=usage["completion_tokens"],
        cached=cached,
        wall_seconds=wall,
        first_output_seconds=ttft,
        client_decode_tps=(usage["completion_tokens"] - 1) / (wall - ttft),
        prefill_tps=(tokens - cached) / stat["request_prefill_time_seconds_sum"],
        decode_tps=(usage["completion_tokens"] - 1) / stat["request_decode_time_seconds_sum"],
        drafted=stat["spec_decode_num_draft_tokens_total"],
        accepted=stat["spec_decode_num_accepted_tokens_total"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="inkling2-mini-exl3-b70")
    p.add_argument("--url", default="http://127.0.0.1:8093")
    args = p.parse_args()
    with args.output.open("x") as f:
        for name in args.cases:
            row = replay(args, args.archive / name)
            print(json.dumps(row), file=f, flush=True)
            print(name, row["input_tokens"], round(row["decode_tps"], 2), flush=True)


if __name__ == "__main__":
    main()
