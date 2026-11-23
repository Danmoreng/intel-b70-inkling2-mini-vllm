#!/usr/bin/env python3
"""Check simultaneous decode, prefix ownership and abort/reuse on a live Inkling worker."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier, Event
import time
from urllib.request import Request, urlopen
from uuid import uuid4

MODEL = "inkling2-mini-exl3-b70"


def completion(base, word, count, salt, tokens, *, barrier=None, started=None, abort_after=None):
    prompt = "Continue the sequence: " + (word + ", ") * count + word + ","
    payload = dict(model=MODEL, prompt=prompt, temperature=0, seed=648, max_tokens=tokens,
                   ignore_eos=True, return_token_ids=True, cache_salt=salt, stream=True,
                   stream_options={"include_usage": True})
    request = Request(base + "/v1/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"})
    if barrier:
        barrier.wait(timeout=10)
    ids = []
    first = usage = None
    with urlopen(request, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: {"):
                continue
            event = json.loads(line[6:])
            if "error" in event:
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                ids.extend(choice.get("token_ids") or [])
            if first is None and ids:
                first = time.monotonic()
                if started:
                    started.set()
            if abort_after and len(ids) >= abort_after:
                return {"ids": ids}
    if not usage or len(ids) != tokens or usage["completion_tokens"] != tokens:
        raise RuntimeError("Incomplete stream or incorrect output count")
    return dict(ids=ids, cached=usage["prompt_tokens_details"]["cached_tokens"],
                first=first, end=time.monotonic())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8093")
    parser.add_argument("--clients", type=int, choices=(2, 3, 4), default=4)
    args = parser.parse_args()
    clients = args.clients
    words = ["apple", "banana", "fig", "mango"][:clients]
    counts = [131 + 19 * i for i in range(clients)]
    salts = [uuid4().hex for _ in words]

    def run(i, limit=128, **kwargs):
        return completion(args.url, words[i], counts[i], salts[i], limit, **kwargs)

    refs = [run(i) for i in range(clients)]
    barrier = Barrier(clients)
    with ThreadPoolExecutor(clients) as pool:
        futures = [pool.submit(run, i, barrier=barrier) for i in range(clients)]
        joint = [f.result() for f in futures]
    overlap = min(r["end"] for r in joint) - max(r["first"] for r in joint)
    if (any(r["cached"] for r in refs) or any(not r["cached"] for r in joint)
            or not all(a["ids"] == b["ids"] for a, b in zip(refs, joint)) or overlap <= 0):
        raise RuntimeError("Cache, greedy parity or stream overlap mismatch")

    # Force cold admissions while one stream is decoding, then close a response
    # before EOS. The survivor and replacement must each keep their own history.
    ready = Event()
    salts = [uuid4().hex for _ in words]
    with ThreadPoolExecutor(2) as pool:
        keep = pool.submit(run, 0, started=ready)
        if not ready.wait(60):
            raise TimeoutError("The survivor emitted no token")
        aborted = run(1, 4096, abort_after=6)
        salts[1] = uuid4().hex
        replacement = run(1, 48)
        survivor = keep.result()
    if survivor["ids"] != refs[0]["ids"] or replacement["ids"] != refs[1]["ids"][:48]:
        raise RuntimeError("An abort affected another request or its replacement")
    if replacement["cached"] or survivor["cached"]:
        raise RuntimeError("A changed cache salt reused stale history")
    # Six arrivals on a four-slot worker force turnover within a single wave.
    # A different salt on every request includes repeated texts with no legal hit.
    wave_size = clients + 2
    barrier = Barrier(wave_size)
    with ThreadPoolExecutor(wave_size) as pool:
        queued = []
        for i in range(wave_size):
            row, length = i % clients, 32 + 16 * i
            future = pool.submit(completion, args.url, words[row], counts[row], uuid4().hex, length,
                                 barrier=barrier)
            queued.append((future, refs[row]["ids"][:length]))
        for future, expected in queued:
            result = future.result()
            if result["ids"] != expected or result["cached"]:
                raise RuntimeError("Request turnover changed the output or reused the wrong salt")
    print(json.dumps(dict(clients=clients, tokens_per_client=128, overlap_s=overlap,
                          turnover_wave=wave_size,
                          cached_tokens=[r["cached"] for r in joint], aborted_after=len(aborted["ids"]),
                          greedy_parity=True, survivor_and_replacement=True)))


if __name__ == "__main__":
    main()
