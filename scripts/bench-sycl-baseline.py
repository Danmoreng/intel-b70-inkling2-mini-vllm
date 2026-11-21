#!/usr/bin/env python3
"""Measure cold prefill plus fixed-length decode and check retrieval at the same input length."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import time
import urllib.request


def request(base, endpoint, payload, timeout):
    req = urllib.request.Request(
        f"{base}/{endpoint}", json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def prompt_tokens(base, length, seed, model=None):
    rng = random.Random(seed)
    code = str(rng.randrange(1000000, 9999999))
    prefix = "Read the following logbook and remember any record labelled KEY.\n\n"
    needle = f"\nKEY: The exact validation code for warehouse Vireo is {code}.\n"
    suffix = "\nWhat is the exact validation code for warehouse Vireo? Reply with the seven digits only.\nAnswer:"
    filler = "\n".join(
        f"Row {i:06d}: Truck {rng.randrange(10, 99)} reported {rng.randrange(100, 900)} units "
        f"of material at depot {rng.choice(('North', 'South', 'Bridge', 'Central'))}. Inspection completed."
        for i in range(length // 12 + 128)
    )

    def tokenize(text):
        if model is not None:
            return request(base, "tokenize", {"model": model, "prompt": text, "add_special_tokens": False}, 30)["tokens"]
        return request(base, "tokenize", {"content": text, "add_special": False}, 30)["tokens"]

    before, target, after, padding = map(tokenize, (prefix, needle, suffix, filler))
    needed = length - len(before) - len(target) - len(after)
    if needed > len(padding) or needed < 0:
        raise ValueError("Cannot construct the requested prompt length")
    offset = needed // 2
    tokens = before + padding[:offset] + target + padding[offset:needed] + after
    return tokens, code


def record(base, tokens, code, seed, output_tokens, timeout, reuse=False):
    settings = {
        "prompt": tokens, "n_predict": output_tokens, "temperature": 0, "seed": seed,
        "ignore_eos": True, "cache_prompt": reuse, "return_tokens": True,
    }
    start = time.monotonic()
    response = request(base, "completion", settings, timeout)
    elapsed = time.monotonic() - start
    timings = response["timings"]
    first_number = re.search(r"\d+", response["content"])
    return {
        "input_tokens": len(tokens), "input_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "output_tokens": len(response["tokens"]), "seed": seed, "reuse": reuse,
        "code": code, "retrieval_pass": bool(first_number and first_number[0] == code),
        "answer_excerpt": response["content"][:320], "wall_seconds": elapsed, "timings": timings,
        "truncated": response.get("truncated", False), "response_model": response.get("model"),
    }


def vllm_metrics(base, model):
    with urllib.request.urlopen(f"{base}/metrics", timeout=5) as response:
        content = response.read().decode()
    names = {f"vllm:request_{phase}_time_seconds_{suffix}" for phase in ("prefill", "decode")
             for suffix in ("sum", "count")}
    metrics = {}
    for line in content.splitlines():
        name = line.split("{", 1)[0]
        if name not in names or f'model_name="{model}"' not in line:
            continue
        metrics[name] = float(line.split("}", 1)[1].split()[0])
    return metrics


def record_vllm(base, model, tokens, code, seed, output_tokens, timeout):
    prefill = "vllm:request_prefill_time_seconds"
    decode = "vllm:request_decode_time_seconds"
    before = vllm_metrics(base, model)
    start = time.monotonic()
    response = request(base, "v1/completions", {
        "model": model, "prompt": tokens, "max_tokens": output_tokens, "temperature": 0,
        "ignore_eos": True, "seed": seed,
    }, timeout)
    elapsed = time.monotonic() - start
    for _ in range(20):
        after = vllm_metrics(base, model)
        if after.get(decode + "_count", 0) - before.get(decode + "_count", 0) >= 1:
            break
        time.sleep(0.25)
    if any(after.get(metric + "_count", 0) - before.get(metric + "_count", 0) != 1 for metric in (prefill, decode)):
        raise RuntimeError("Expected metrics for exactly one completed vLLM request")
    prefill_seconds = after[prefill + "_sum"] - before.get(prefill + "_sum", 0)
    decode_seconds = after[decode + "_sum"] - before.get(decode + "_sum", 0)
    prompt_n, predicted_n = (response["usage"][key] for key in ("prompt_tokens", "completion_tokens"))
    answer = response["choices"][0]["text"]
    first_number = re.search(r"\d+", answer)
    return {
        "input_tokens": len(tokens), "input_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "output_tokens": predicted_n, "seed": seed, "reuse": False, "code": code,
        "retrieval_pass": bool(first_number and first_number[0] == code),
        "answer_excerpt": answer[:320], "wall_seconds": elapsed, "truncated": prompt_n != len(tokens),
        "response_model": response["model"],
        "timings": {"prompt_n": prompt_n, "prompt_ms": prefill_seconds * 1000,
                    "prompt_per_second": prompt_n / prefill_seconds,
                    "predicted_n": predicted_n, "predicted_ms": decode_seconds * 1000,
                    "predicted_per_second": (predicted_n - 1) / decode_seconds if decode_seconds > 0 else None},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8092")
    parser.add_argument("--vllm-model", help="Name of a vLLM model served at --url")
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096])
    parser.add_argument("--decode", type=int, default=128)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output:
        for length in args.lengths:
            for index in range(args.replicates):
                seed = 37 + index
                tokens, code = prompt_tokens(args.url, length, seed, args.vllm_model)
                if args.vllm_model:
                    row = record_vllm(args.url, args.vllm_model, tokens, code, seed, args.decode, args.timeout)
                else:
                    row = record(args.url, tokens, code, seed, args.decode, args.timeout)
                print(json.dumps(row), file=output, flush=True)
                print(json.dumps(row), flush=True)
                if row["truncated"] or not row["timings"]["prompt_n"] >= length:
                    raise RuntimeError(f"The server did not evaluate all {length} cold prompt tokens")
                if row["output_tokens"] != args.decode:
                    raise RuntimeError(f"The server returned {row['output_tokens']} of {args.decode} output tokens")


if __name__ == "__main__":
    main()
