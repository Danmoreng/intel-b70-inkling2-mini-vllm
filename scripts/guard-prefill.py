#!/usr/bin/env python3
"""Bound a single cold vLLM benchmark by KV allocation progress and elapsed time."""

import argparse
import re
import subprocess
import time
import urllib.request


def metrics(url):
    with urllib.request.urlopen(url + "/metrics", timeout=4) as response:
        raw = response.read().decode()
    values = {}
    for name in (
        "kv_cache_usage_perc",
        "generation_tokens_total",
        "prompt_tokens_total",
        "num_requests_running",
    ):
        matches = re.findall(rf"^vllm:{name}\{{[^\n]+\}} ([0-9.eE+\-]+)", raw, re.M)
        if len(matches) != 1:
            raise ValueError(f"Expected a single engine's {name} series")
        values[name] = float(matches[0])
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8093")
    parser.add_argument("--cache-tokens", type=int, default=262144)
    parser.add_argument("--floor", type=float, default=650)
    parser.add_argument("--wall", type=int, default=310)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or min(args.cache_tokens, args.floor, args.wall) <= 0:
        parser.error("A command and positive limits are required")
    before = metrics(args.url)
    started = prev_time = time.monotonic()
    prev_pos = slow_intervals = 0
    process = subprocess.Popen(command)
    try:
        while True:
            try:
                return process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            stats = metrics(args.url)
            generated = stats["generation_tokens_total"] - before["generation_tokens_total"]
            declared = stats["prompt_tokens_total"] - before["prompt_tokens_total"]
            allocated = round(stats["kv_cache_usage_perc"] * args.cache_tokens)
            current = max(prev_pos, declared, allocated)
            rate = (current - prev_pos) / (now - prev_time)
            print(
                f"{now - started:.1f} s, approx. {current:.0f} prompt positions, {rate:.0f} tok/s, "
                f"{generated:.0f} generated",
                flush=True,
            )
            if now - started >= 30 and stats["num_requests_running"] and not generated:
                slow_intervals = slow_intervals + 1 if rate < args.floor else 0
            if now - started >= args.wall or slow_intervals >= 2:
                raise RuntimeError("Elapsed time or minimum prefill throughput limit reached")
            prev_time, prev_pos = now, current
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
