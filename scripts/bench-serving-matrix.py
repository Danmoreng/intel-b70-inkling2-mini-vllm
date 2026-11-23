#!/usr/bin/env python3
"""Replay the Qwen EXL3 v8 serving matrix on the Inkling B70 serving recipe."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from serving_decode import active_decode

MODEL = "inkling2-mini-exl3-b70"
BERLIN = ZoneInfo("Europe/Berlin")
REFERENCE = "0356a370da5f73f6b0a4eab613d7103a2356b513"
PINNED = {
    "scripts/current-profile-benchmark.py": "01105118930b595aaf31b2bf58f3427b76a3b94721ceeabd947b432105ad3345",
    "scripts/meaningful_benchmark.py": "df202c54b66fb3504db2161a99eef13646e4a3ccc3e5a3ad8ffdb9ab7f03085a",
    "scripts/decode_overlap.py": "233f5424137c9078e87a2bf974e956d826bb0ce54c6c5203ed47a8afee84d56f",
    "benchmarks/current-profile-scenarios.json": "b58c43bb445f3dd4cae12eb7e088f331a13cd17e3b4842238079d5400224ccf1",
}
# Digest over every R12 request JSON and prompt, including unmeasured warmup bytes.
INVENTORIES = {
    "matrix": (248, "baa86391880b3dc9efed2e27a9cafa4e9b5a99668c39f1c202001376c6518cbf"),
    "warmups": (70, "36c7fc604f08ef3dcc509a6722d03477c07e7d2da5e02f442d0d5ac7bbc60a85"),
}


def runtime_receipt(container):
    """Capture the image, selected native libraries and mounted vLLM overrides."""
    item = json.loads(subprocess.check_output(["docker", "inspect", container]))[0]
    env = dict(v.split("=", 1) for v in item["Config"]["Env"] if v.startswith("INKLING2_") and "=" in v)
    volumes = {v["Destination"]: Path(v["Source"]) for v in item["Mounts"]}
    repo = volumes["/lab"]
    files = {dest: src for dest, src in volumes.items() if src.is_file()}
    for value in env.values():
        if value.startswith("/lab/") and value.endswith(".so"):
            files[value] = volumes["/lab"] / value.removeprefix("/lab/")
    for path in [
        *repo.glob("inkling_xpu/**/*.py"),
        repo / ".work/vllm-inkling/config/config.json",
        repo / "models/exl3/checkpoint.json",
    ]:
        files[f"/lab/{path.relative_to(repo)}"] = path
    return {
        "image_id": item["Image"],
        "command": item["Config"]["Cmd"],
        "recipe_revision": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "environment": env,
        "file_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(files.items())
        },
    }


def prepare_fixtures(source, dest, group):
    root = source / "warmups" if group == "warmups" else source
    digest = hashlib.sha256()
    count = 0
    for request in sorted(root.glob("*/request-*.json")):
        prompt = request.with_name(request.name.replace("request-", "prompt-").replace(".json", ".txt"))
        for path in (request, prompt):
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            digest.update(f"{path.relative_to(source)} {checksum}\n".encode())
            count += 1
        payload = json.loads(request.read_bytes())
        assert payload["model"] == "Qwen3.8-27B" and payload["chat_template_kwargs"] == {
            "enable_thinking": False
        }
        assert payload["max_tokens"] == 1024 and payload["ignore_eos"]
        assert payload["prompt_sha256"] == checksum
        payload.update(
            model=MODEL,
            reasoning_effort="none",
            thinking_token_budget=0,
            chat_template_kwargs={"reasoning_effort": "none"},
        )
        # run_once reads frozen files for nonnegative repeat indices.
        name = request.parent.name
        if group == "warmups":
            assert name.endswith("-r0")
            name = name[:-3] + "-r1"
        folder = dest / group / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / prompt.name).symlink_to(prompt.resolve())
        (folder / request.name).write_text(json.dumps(payload, indent=2))
    if (count, digest.hexdigest()) != INVENTORIES[group]:
        raise ValueError(f"{group} differs from the archived R12 input set")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", required=True, type=Path)
    parser.add_argument("--reference-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    parser.add_argument("--container", default="inkling2-vllm-exl3")
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--batching-4k", action="store_true",
                        help="Qualify the four-slot recipe at 4K, from one through four simultaneous clients.")
    subset.add_argument(
        "--isolated-prefix",
        action="store_true",
        help="Only the three 64K cold/warm requests; start a fresh engine first.",
    )
    parser.add_argument("--ordinary", action="store_true", help="Use ordinary decode for the four-slot 4K sweep.")
    args = parser.parse_args()
    if args.ordinary and not args.batching_4k:
        parser.error("--ordinary requires --batching-4k")
    for filename, expected in PINNED.items():
        if hashlib.sha256((args.reference_repo / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reference harness mismatch: {filename}")
    sys.path.insert(0, str((args.reference_repo / "scripts").resolve()))
    spec = importlib.util.spec_from_file_location(
        "qwen_serving_matrix", args.reference_repo / "scripts/current-profile-benchmark.py"
    )
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    if args.ordinary:
        # The pinned speculative-round estimator divides by the proposal width.
        # Ordinary decode has no such estimator. run_once still computes all
        # timing metrics and aggregate decode from the same counter samples.
        bench.summarize_overlap = lambda *args: {}
    bench.MODEL = MODEL
    bench.CHAT_TEMPLATE_KWARGS = {"reasoning_effort": "none"}
    cases = bench.load_scenarios(args.reference_repo / "benchmarks/current-profile-scenarios.json")
    if args.isolated_prefix:
        cases = [s for s in cases if s.name == "prefix-64k-cold-warm"]
    if args.batching_4k:
        cases = [s for s in cases if s.name == "phase-4k-c1" or s.name.startswith("concurrency-4k-")]
        if sorted(s.concurrency for s in cases) != [1, 2, 3, 4]:
            raise ValueError("Incomplete 4K reference sweep")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    fixtures = out / "adapted-inputs"
    for group in INVENTORIES:
        prepare_fixtures(args.reference_run, fixtures, group)
    runtime = runtime_receipt(args.container)
    env = runtime["environment"]
    if env.get("INKLING2_VISION") != "1" or (not args.ordinary and env.get("INKLING2_DSPARK_ADAPTIVE") != "1"):
        raise ValueError("Expected vision and the requested decode profile")
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2))
    config = bench.inspect_container(args.container)
    models = bench.http_json(args.base, "/v1/models")
    if not any(m["id"] == MODEL for m in models["data"]):
        raise ValueError("Selected endpoint does not serve Inkling")
    slots = 4 if args.batching_4k else 1
    width = None if args.ordinary else 7
    if (config["max_num_seqs"], config["num_speculative_tokens"], config["max_model_len"]) != (slots, width, 262144):
        raise ValueError(f"Expected {slots} slots, draft width {width}, and 262144 context")
    power = [
        p
        for p in Path("/sys/class/hwmon").glob("hwmon*/power1_cap")
        if (p.parent / "name").read_text().strip() == "xe"
    ]
    if len(power) != 1 or int(power[0].read_text()) != 180000000:
        raise ValueError("Expected a 180 W Intel GPU power cap")
    counters, _ = bench.metric_snapshot(args.base)
    if any(counters[k] for k in ("running", "waiting", "finished_requests")):
        raise ValueError("Start with an idle fresh engine, without prior completions")
    manifest = {
        "start": datetime.now(BERLIN).isoformat(),
        "reference_revision": REFERENCE,
        "fixture_inventories": INVENTORIES,
        "container": config,
        "power_w": 180,
        "models": models,
        "reasoning_effort": "none",
        "thinking_token_budget": 0,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for case in cases:
        for group, repeats in [("warmups", case.warmups), ("matrix", case.repeats)]:
            for rep in range(repeats):
                if datetime.now(BERLIN).strftime("%H:%M") >= "22:35":
                    raise RuntimeError("Daily shutdown margin reached")
                print(datetime.now(BERLIN).strftime("%H:%M:%S"), group, case.name, rep + 1, flush=True)
                row = bench.run_once(
                    args.base,
                    args.container,
                    case,
                    rep,
                    out / group,
                    [],
                    "",
                    True,
                    fixtures / group,
                    config["num_speculative_tokens"],
                )
                if row["preemptions"] or any(r["reasoning_chars"] for r in row["requests"]):
                    raise RuntimeError("Unexpected preemption or reasoning text")
                if args.batching_4k and not all(row[key] for key in (
                    "all_outputs_nonempty", "all_prompt_counts_match", "all_completion_counts_exact",
                    "all_finish_reasons_length", "admission_expectation_met"
                )):
                    raise RuntimeError("Incomplete or incorrectly accounted serving wave")
                if args.batching_4k or case.concurrency > 1:
                    wave = out / group / f"{case.name}-r{rep + 1}"
                    decode = active_decode(wave, row)
                    if args.batching_4k and (row["peak_running"] != case.concurrency or not decode["aggregate_decode_tps"]):
                        raise RuntimeError(f"Missing simultaneous decode at C{case.concurrency}")
                    row = dict(scenario=case.name, repeat=rep + 1, **decode)
                with (out / (group + ".jsonl")).open("a") as f:
                    print(json.dumps(row), file=f)
    if bench.inspect_container(args.container)["id"] != config["id"]:
        raise RuntimeError("Engine restarted during the run")
    final, raw_metrics = bench.metric_snapshot(args.base)
    (out / "final-metrics.prom").write_text(raw_metrics)
    expected = sum((case.repeats + case.warmups) * case.concurrency for case in cases)
    if final["finished_requests"] != expected or final["running"] or final["waiting"] or final["preemptions"]:
        raise RuntimeError(f"Unexpected final counters for {expected} benchmark requests")
    (out / "completed.json").write_text(json.dumps({"end": datetime.now(BERLIN).isoformat()}))


if __name__ == "__main__":
    main()
