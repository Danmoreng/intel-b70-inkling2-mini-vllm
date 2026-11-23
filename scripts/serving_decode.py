"""Measure generation only while every member of a serving wave is decoding."""

import json
from pathlib import Path


def active_decode(folder: Path, summary: dict) -> dict:
    clients = summary["scenario"]["concurrency"]
    requests = summary["requests"]
    last_events = []
    for i in range(1, clients + 1):
        last = None
        with (folder / f"stream-{i}.jsonl").open() as events:
            for line in events:
                event = json.loads(line)
                if any(choice["delta"].get("content") or choice["delta"].get("reasoning")
                       or choice["delta"].get("reasoning_content")
                       for choice in event["event"].get("choices", []) if choice.get("delta")):
                    last = event["elapsed_s"]
        if last is None:
            raise ValueError(f"No output in {folder} for client {i}")
        last_events.append(last)

    # Each stream measures from the start barrier. Include the observed wave
    # setup time on the left edge and trim one monitor sample at either edge.
    setup = max(0., summary["batch_wall_s"] - max(r["wall_s"] for r in requests))
    lo = max(r["ttft_s"] for r in requests) + setup
    hi = min(last_events)
    samples = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    window = [s for s in samples if lo <= s["elapsed_s"] <= hi][1:-1]
    tokens = intervals = 0
    seconds = 0.
    for a, b in zip(window, window[1:]):
        if any("error" in s or s["running"] != clients or s["waiting"] for s in (a, b)):
            continue
        if any(a[key] != b[key] for key in ("prefill_computed", "finished_requests", "preemptions")):
            continue
        delta = b["generation_tokens"] - a["generation_tokens"]
        elapsed = b["elapsed_s"] - a["elapsed_s"]
        if delta < 0 or elapsed <= 0:
            raise ValueError(f"Non-monotonic counters in {folder}")
        tokens += int(delta)
        seconds += elapsed
        intervals += 1
    rate = tokens / seconds if seconds else None
    return dict(clients=clients, generated_tokens=tokens, sampled_seconds=seconds,
                intervals=intervals, aggregate_decode_tps=rate,
                mean_decode_tps_per_client=rate / clients if rate is not None else None)
