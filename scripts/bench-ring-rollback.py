#!/usr/bin/env python3
"""Measure a coding-agent branch that discards a long generated answer and reuses the prompt."""

import argparse
import hashlib
import importlib.util
import json
import re
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

spec = importlib.util.spec_from_file_location("baseline", Path(__file__).with_name("bench-sycl-baseline.py"))
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def complete(base, model, ids, count, salt, expected, case):
    payload = dict(
        model=model,
        prompt=ids,
        max_tokens=count,
        seed=37,
        temperature=0,
        ignore_eos=True,
        cache_salt=salt,
        stream=True,
        stream_options={"include_usage": True},
    )
    request = urllib.request.Request(
        base + "/v1/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    before = baseline.vllm_metrics(base, model)
    start = time.monotonic()
    first = None
    chunks = []
    usage = None
    with urllib.request.urlopen(request, timeout=310) as response:
        for line in response:
            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                continue
            packet = json.loads(line[6:])
            if packet.get("error"):
                raise RuntimeError(packet["error"])
            if packet.get("usage"):
                usage = packet["usage"]
            for choice in packet.get("choices", []):
                if text := choice.get("text"):
                    if first is None:
                        first = time.monotonic() - start
                    chunks.append(text)
    elapsed = time.monotonic() - start
    if (
        not usage
        or usage["prompt_tokens"] != len(ids)
        or usage["completion_tokens"] != count
        or first is None
    ):
        raise RuntimeError(f"Incomplete {case} request")
    dep = "vllm:request_decode_time_seconds"
    pre = "vllm:request_prefill_time_seconds"
    for _ in range(20):
        after = baseline.vllm_metrics(base, model)
        if after[dep + "_count"] != before[dep + "_count"]:
            break
        time.sleep(0.25)
    if any(after[k + "_count"] - before[k + "_count"] != 1 for k in (pre, dep)):
        raise RuntimeError("Expected one completed request in the engine counters")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    generated = "".join(chunks)
    match = re.search(r"\d+", generated)
    result = dict(
        case=case,
        prompt_tokens=len(ids),
        completion_tokens=count,
        cached_tokens=cached,
        engine_prefill_tps=(len(ids) - cached) / (after[pre + "_sum"] - before[pre + "_sum"]),
        engine_decode_tps=(count - 1) / (after[dep + "_sum"] - before[dep + "_sum"]),
        first_output_seconds=first,
        wall_seconds=elapsed,
        client_decode_tps=(count - 1) / (elapsed - first),
        expected_code=expected,
        retrieval_pass=bool(match and match.group() == expected),
        completion_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        excerpt=generated[:300],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8093")
    parser.add_argument("--model", default="inkling2-mini-exl3-b70")
    parser.add_argument("--length", default=8192, type=int)
    parser.add_argument("--phase", choices=["all", "initial", "rollback", "isolated"], default="all")
    parser.add_argument("--salt", help="Shared session salt when executing each phase separately")
    parser.add_argument("--branch", choices=["question", "audit"], default="question")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ids, code = baseline.prompt_tokens(args.url, args.length, 37, args.model)
    extra = baseline.request(
        args.url,
        "tokenize",
        dict(
            model=args.model,
            prompt=(
                "\nUse the record above. What is the validation code for warehouse Vireo? Reply with the digits."
                if args.branch == "question"
                else "\nFor the audit report, fill in the value from the logbook above.\nWarehouse Vireo validation code (seven digits):"
            ),
            add_special_tokens=False,
        ),
        10,
    )["tokens"]
    if args.phase != "all" and not args.salt:
        parser.error("A shared --salt is required for individual phases")
    salt = args.salt or uuid4().hex
    failures = []
    with args.output.open("x") as out:
        for case, prompt, scope in (
            ("initial", ids, salt),
            ("rollback", ids + extra, salt),
            ("isolated", ids + extra, salt + "-cold"),
        ):
            if args.phase not in ("all", case):
                continue
            result = complete(args.url, args.model, prompt, 128, scope, code, case)
            result["branch"] = args.branch
            print(json.dumps(result), file=out, flush=True)
            print(
                case,
                result["cached_tokens"],
                round(result["first_output_seconds"], 3),
                result["retrieval_pass"],
                flush=True,
            )
            if not result["retrieval_pass"]:
                failures.append(f"{case} returned an incorrect validation code")
            cached = result["cached_tokens"]
            if case == "rollback" and not args.length - 32 <= cached <= args.length:
                failures.append(f"Expected the saved prefill ring for {args.length} tokens, got {cached}")
            if case != "rollback" and cached:
                failures.append(f"Unexpected prefix hit for {case}")
    if failures:
        raise RuntimeError(", ".join(failures))


if __name__ == "__main__":
    main()
