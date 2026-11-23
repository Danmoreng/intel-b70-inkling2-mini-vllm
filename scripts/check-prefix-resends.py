#!/usr/bin/env python3
"""Replay one frozen Inkling chat prompt, comparing cold and cached token IDs."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4


def generate(base, payload):
    req = Request(
        base + "/v1/chat/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    start = time.monotonic()
    ids, text = [], []
    usage = finish = first_ids = first_text = None
    with urlopen(req, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                continue
            event = json.loads(line[6:])
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event["choices"]:
                now = time.monotonic() - start
                if tokens := choice.get("token_ids"):
                    ids.extend(tokens)
                    if first_ids is None:
                        first_ids = now
                if content := choice["delta"].get("content"):
                    text.append(content)
                    if first_text is None:
                        first_text = now
                finish = choice.get("finish_reason") or finish
    if (
        not usage
        or len(ids) != usage["completion_tokens"]
        or len(ids) != payload["max_tokens"]
        or finish != "length"
    ):
        raise ValueError("Incomplete response or token accounting")
    return {
        "input": usage["prompt_tokens"],
        "cached": usage["prompt_tokens_details"]["cached_tokens"],
        "output": len(ids),
        "first_ids_s": first_ids,
        "first_text_s": first_text,
        "wall_s": time.monotonic() - start,
        "content_chunks": len(text),
        "token_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "text_sha256": hashlib.sha256("".join(text).encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        required=True,
        type=Path,
        help="Inkling serving wave containing request-1.json and prompt-1.txt",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((args.fixture / "request-1.json").read_bytes())
    prompt = (args.fixture / "prompt-1.txt").read_bytes()
    if hashlib.sha256(prompt).hexdigest() != original["prompt_sha256"]:
        raise ValueError("Archived input text differs from request digest")
    keys = (
        "model",
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "seed",
        "ignore_eos",
        "chat_template_kwargs",
        "reasoning_effort",
        "thinking_token_budget",
    )
    payload = {k: original[k] for k in keys}
    payload.update(
        messages=[{"role": "user", "content": prompt.decode()}],
        stream=True,
        stream_options={"include_usage": True},
        return_token_ids=True,
    )
    scope = uuid4().hex
    rows = []
    with args.output.open("x") as out:
        for phase in ("cold", "resend-1", "resend-2", "new-salt"):
            row = generate(args.url, {**payload, "cache_salt": scope if phase != "new-salt" else uuid4().hex})
            rows.append(row)
            print(json.dumps({"phase": phase, **row}), file=out, flush=True)
    for i, row in enumerate(rows):
        cached = row["cached"]
        if i in (1, 2):
            cache_ok = cached > 0 and 0 < row["input"] - cached <= 32
        else:
            cache_ok = cached == 0
        if not cache_ok:
            raise RuntimeError(f"Unexpected prefix count in phase {i + 1}: {cached}")
        if row["content_chunks"] < 2 or any(
            row[k] != rows[0][k] for k in ("token_sha256", "text_sha256", "input")
        ):
            raise RuntimeError(f"Cold/cached token or streaming mismatch in phase {i + 1}")


if __name__ == "__main__":
    main()
