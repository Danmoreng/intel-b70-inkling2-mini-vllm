#!/usr/bin/env python3
"""Warm the coding endpoint's chat, sampling and verification kernels with isolated requests."""

import argparse
import json
import time
import urllib.request
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8093")
    parser.add_argument("--model", default="inkling2-mini-exl3-b70")
    parser.add_argument("--output", help="Append JSON measurements to this file")
    args = parser.parse_args()
    source = "\n".join(f"def case_{i}(value): return value + {i}" for i in range(96))
    tool = dict(
        type="function",
        function=dict(
            name="read_file",
            description="Read a project file.",
            parameters=dict(type="object", properties={"path": {"type": "string"}}, required=["path"]),
        ),
    )
    for temperature in (1.0, 0.0, 1.0):
        payload = dict(
            model=args.model,
            temperature=temperature,
            top_k=20,
            top_p=0.95,
            seed=73,
            max_tokens=32,
            ignore_eos=True,
            reasoning_effort="none",
            tools=[tool],
            messages=[
                {"role": "user", "content": f"Review this Python source and explain the pattern.\n{source}"}
            ],
            stream=True,
            stream_options={"include_usage": True},
            cache_salt=uuid4().hex,
        )
        req = urllib.request.Request(
            args.url + "/v1/chat/completions",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        start = time.monotonic()
        first = None
        usage = None
        with urllib.request.urlopen(req, timeout=180) as response:
            for line in response:
                if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                    continue
                packet = json.loads(line[6:])
                if packet.get("error"):
                    raise RuntimeError(packet["error"])
                if packet.get("usage"):
                    usage = packet["usage"]
                for choice in packet.get("choices") or []:
                    if first is None and any(
                        choice["delta"].get(k)
                        for k in ("content", "reasoning", "reasoning_content", "tool_calls")
                    ):
                        first = time.monotonic() - start
        if not usage or first is None or usage["completion_tokens"] != 32:
            raise RuntimeError(f"Incomplete warm-up response: {usage}")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if cached:
            raise RuntimeError("Warm-up cache salt unexpectedly produced a prefix hit")
        out = dict(
            temperature=temperature,
            prompt_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            cached_tokens=cached,
            first_output_seconds=first,
            wall_seconds=time.monotonic() - start,
        )
        line = json.dumps(out)
        print(line, flush=True)
        if args.output:
            with open(args.output, "a", encoding="utf-8") as f:
                print(line, file=f)


if __name__ == "__main__":
    main()
