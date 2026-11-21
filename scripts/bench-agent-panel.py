#!/usr/bin/env python3
"""Run the two published B70 coding fixtures and grade their hidden tests after the final reply."""

import argparse
import importlib.util
import json
import re
from pathlib import Path
import shutil
import tempfile
import time
import urllib.request


SYSTEM = (
    "You are a software engineer working in a small Python repository. Use tools to inspect files, "
    "implement the requested change and run visible tests. Make the smallest complete implementation. "
    "When done, return a concise final message without tool calls. Hidden acceptance tests are not available."
)
IMAGE = "sha256:131dd9e11809de2f3136547618f130ee238f0b90cd5718567bcdd0dad5360e19"


def request(args, messages, tools, seed, salt, remaining):
    payload = dict(
        model=args.model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        seed=seed,
        cache_salt=salt,
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        max_tokens=args.max_output,
        reasoning_effort=args.effort,
    )
    if args.thinking_budget is not None:
        payload["thinking_token_budget"] = args.thinking_budget
    req = urllib.request.Request(
        args.url + "/v1/chat/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=max(1, remaining)) as f:
        return json.load(f)


def run_task(args, runner, fixture, seed, project):
    helper = runner.K
    identity, digest = helper.verify_fixture(fixture)
    authorized_seeds = json.loads((fixture / "manifest.json").read_text())["seeds"]
    if seed not in authorized_seeds:
        raise ValueError(f"Seed {seed} lies outside the fixture manifest")
    shutil.copytree(fixture / "project", project)
    editable = {p.name for p in project.glob("*.py")}
    tools = json.loads(json.dumps(helper.TOOLS))
    tools[0]["function"]["description"] = "List every file in the current coding project."
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (fixture / "instruction.txt").read_text()},
    ]
    turns = []
    complete = False
    visible_pass = False
    error = ""
    metrics_before = helper.snapshot(args.url)
    start = time.monotonic()
    try:
        for step in range(24):
            remaining = args.wall - (time.monotonic() - start)
            if remaining < 1:
                raise TimeoutError("task time limit")
            turn_start = time.monotonic()
            result = request(args, messages, tools, seed + step, identity + str(seed), remaining)
            choice = result["choices"][0]
            msg = choice["message"]
            calls = msg.get("tool_calls") or []
            turns.append(
                dict(
                    usage=result["usage"],
                    finish=choice["finish_reason"],
                    request_wall_seconds=time.monotonic() - turn_start,
                    tools=[c["function"]["name"] for c in calls],
                )
            )
            history = {
                k: msg[k]
                for k in ("role", "content", "reasoning", "reasoning_content", "tool_calls")
                if msg.get(k) is not None
            }
            messages.append(history)
            for call in calls:
                name = call["function"]["name"]
                try:
                    arguments = json.loads(call["function"]["arguments"])
                    if name == "run_tests":
                        checked = runner.isolated(project, args.image, None)
                        visible_pass = checked["passed"]
                        value = checked
                        value["output"] = re.sub(
                            r"Ran (\d+) tests? in [0-9.]+s", r"Ran \1 tests in <elapsed>s", value["output"]
                        )
                    elif name == "write_file" and arguments.get("path") not in editable:
                        value = {"error": "Only the fixture's implementation file is editable."}
                    elif name in ("read_file", "list_files", "write_file"):
                        value = helper.execute_tool(project, name, arguments)
                        if name == "write_file" and "error" not in value:
                            visible_pass = False
                    else:
                        value = {"error": "Unknown tool"}
                except (TypeError, ValueError, KeyError) as exc:
                    value = {"error": str(exc)}
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(value)})
            if not calls and choice["finish_reason"] != "length":
                complete = True
                break
            if not calls:
                messages.append(
                    {"role": "user", "content": "Continue the implementation and use tools as needed."}
                )
    except Exception as exc:
        error = str(exc)
    elapsed = time.monotonic() - start
    after = helper.snapshot(args.url)
    delta = {k: after[k] - v for k, v in metrics_before.items()}
    hidden = runner.isolated(project, args.image, fixture / "acceptance/test_acceptance.py")
    generated = sum(t["usage"]["completion_tokens"] for t in turns)
    return dict(
        fixture=identity,
        manifest_sha256=digest,
        seed=seed,
        reasoning_effort=args.effort,
        temperature=args.temperature,
        max_output=args.max_output,
        thinking_budget=args.thinking_budget,
        wall_seconds=elapsed,
        completed=complete,
        error=error,
        visible_pass=visible_pass,
        hidden_pass=hidden["passed"],
        correct=complete and hidden["passed"],
        hidden_tests=hidden["tests_run"],
        turns=turns,
        output_tokens=generated,
        engine_metrics=delta,
        decode_tps=(generated - len(turns)) / delta["decode_seconds"] if delta["decode_seconds"] else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-repo", required=True, type=Path)
    parser.add_argument("--fixtures", type=Path, help="Directory containing sealed coding fixtures")
    parser.add_argument("--tasks", nargs="+", default=["rangeset", "dependencyplan"])
    parser.add_argument("--url", default="http://127.0.0.1:8093")
    parser.add_argument("--model", default="inkling2-mini-exl3-b70")
    parser.add_argument("--effort", default="none", choices=["none", "low", "medium"])
    parser.add_argument("--thinking-budget", type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--max-output", type=int, default=8192)
    parser.add_argument("--wall", type=int, default=450)
    parser.add_argument("--seeds", type=int, nargs="+", default=[202611117, 202611119])
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = args.qwen_repo / "benchmarks/experiments/vllm-033-upgrade/agent-panel"
    spec = importlib.util.spec_from_file_location("b70_panel", panel / "run_task.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    with args.output.open("x") as output:
        fixtures = args.fixtures or panel / "fixtures"
        for name in args.tasks:
            for seed in args.seeds:
                with tempfile.TemporaryDirectory() as directory:
                    row = run_task(args, runner, fixtures / name, seed, Path(directory) / "project")
                print(json.dumps(row), file=output, flush=True)
                print(name, seed, row["correct"], row["error"], round(row["wall_seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
