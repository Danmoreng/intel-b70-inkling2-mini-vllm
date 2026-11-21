#!/usr/bin/env python3
"""Measure a deterministic two-turn read/repair loop and grade the submitted function offline."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request

IMAGE = "sha256:131dd9e11809de2f3136547618f130ee238f0b90cd5718567bcdd0dad5360e19"
ROOT = Path(__file__).resolve().parents[1]
CASES = [
    (
        "spans",
        "Merge unsorted nonempty half-open integer intervals. Merge touching endpoints, return a fresh sorted list of tuples, and leave the input list unchanged.",
        "def merge_spans(spans):\n    return sorted(spans)\n",
        """s=[(7,10),(0,2),(2,6),(3,4),(9,12)]
old=s.copy()
assert merge_spans(s)==[(0,6),(7,12)] and s==old
assert merge_spans([])==[] and merge_spans([(1,5),(1,5)])==[(1,5)]""",
    ),
    (
        "paths",
        "Implement stable dependency order. The input maps a node to a list of prerequisites. Return each node once, prerequisites first. Use insertion order for roots and listed order for dependencies. Include dependency-only nodes. Raise ValueError for any cycle.",
        "def order_nodes(graph):\n    return list(graph)\n",
        """assert order_nodes({'b':['d','c'],'a':['c'],'c':['d']})==['d','c','b','a']
assert order_nodes({})==[]
for g in [{'x':['x']},{'x':['y'],'y':['x']}]:
    try: order_nodes(g)
    except ValueError: pass
    else: raise AssertionError('cycle')""",
    ),
    (
        "chunks",
        "Yield lists of at most size items from a one-shot iterable. Read lazily, keep the final partial chunk, and raise ValueError for size <= 0 even when the iterable is empty.",
        "def chunks(items, size):\n    return [items[i:i+size] for i in range(0,len(items),size)]\n",
        """items=iter(range(8)); out=iter(chunks(items,3)); assert next(out)==[0,1,2]
assert next(items)==3
assert list(out)==[[4,5,6],[7]]
assert list(chunks([],3))==[]
try: list(chunks([],0))
except ValueError: pass
else: raise AssertionError('size')""",
    ),
]


def post(base, path, data):
    req = urllib.request.Request(base + path, json.dumps(data).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def turn(args, prompt, limit):
    start = time.monotonic()
    if args.model:
        data = post(
            args.url,
            "/v1/completions",
            dict(
                model=args.model,
                prompt=prompt,
                temperature=0,
                max_tokens=limit,
                stop=["<|end_message|>"],
                seed=37,
            ),
        )
        text = data["choices"][0]["text"]
        counts = data["usage"]
        timings = {}
    else:
        data = post(
            args.url,
            "/completion",
            dict(
                prompt=prompt,
                temperature=0,
                n_predict=limit,
                stop=["<|end_message|>"],
                seed=37,
                cache_prompt=args.reuse,
            ),
        )
        text = data["content"]
        timings = data["timings"]
        counts = {"prompt_tokens": data["tokens_evaluated"], "completion_tokens": data["tokens_predicted"]}
    return text, dict(
        seconds=time.monotonic() - start,
        **counts,
        timings=timings,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    )


def chat_turn(args, messages, tools, limit):
    payload = dict(
        model=args.model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        reasoning_effort="none",
        temperature=0,
        max_tokens=limit,
        seed=37,
        stream=True,
        stream_options={"include_usage": True},
    )
    req = urllib.request.Request(
        args.url + "/v1/chat/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    started = time.monotonic()
    first = None
    usage = {}
    reasoning = []
    content = []
    calls = {}
    finish = None
    with urllib.request.urlopen(req, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            packet = line[6:].strip()
            if packet == b"[DONE]":
                break
            data = json.loads(packet)
            if "error" in data:
                raise RuntimeError(data["error"])
            if data.get("usage"):
                usage = data["usage"]
            for choice in data.get("choices", []):
                delta = choice["delta"]
                if first is None and any(
                    delta.get(key) for key in ("content", "tool_calls", "reasoning", "reasoning_content")
                ):
                    first = time.monotonic() - started
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning") or delta.get("reasoning_content") or "")
                finish = choice.get("finish_reason") or finish
                for call in delta.get("tool_calls") or []:
                    item = calls.setdefault(
                        call["index"],
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if call.get("id"):
                        item["id"] = call["id"]
                    for key in ("name", "arguments"):
                        item["function"][key] += (call.get("function") or {}).get(key) or ""
    answer = {"role": "assistant", "content": "".join(content) or None}
    if any(reasoning):
        answer["reasoning"] = "".join(reasoning)
    if calls:
        answer["tool_calls"] = [calls[i] for i in sorted(calls)]
    return answer, dict(
        **usage, seconds=time.monotonic() - started, first_output_seconds=first, finish_reason=finish
    )


def tool_schema(name, description, fields):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    key: {"type": "string", "description": value} for key, value in fields.items()
                },
                "required": list(fields),
                "additionalProperties": False,
            },
        },
    }


def tool_case(args, name, task, source, tests, background, rep):
    path = f"lib/{name}.py"
    tools = [
        tool_schema("read_file", "Read one repository source file.", {"path": "Relative repository path."}),
        tool_schema(
            "submit_file",
            "Submit the complete corrected file for testing.",
            {"path": "Relative repository path.", "source": "Complete Python source code."},
        ),
    ]
    messages = [
        {
            "role": "system",
            "content": "Use read_file to inspect the file before editing. Then use "
            "submit_file with the complete corrected Python source. Respond by calling the appropriate tool.",
        },
        {"role": "user", "content": f"Repair {path}. {task}"},
    ]
    if background:
        messages[-1]["content"] += "\nAdditional repository context:\n" + background
    started = time.monotonic()
    turns = []
    error = ""
    passed = False
    submitted = 0
    code = ""
    for step in range(3 if args.retry else 2):
        message, meta = chat_turn(args, messages, tools, 1280)
        calls = message.get("tool_calls", [])
        turns.append(dict(**meta, message=message))
        if len(calls) != 1:
            error = "Expected one tool call."
            break
        call = calls[0]
        fn = call["function"]
        try:
            arguments = json.loads(fn["arguments"])
        except ValueError:
            error = "Invalid JSON tool arguments."
            break
        if not isinstance(arguments, dict) or arguments.get("path") != path:
            error = "Unexpected path."
            break
        if step == 0 and fn["name"] == "read_file":
            result = source
        elif step > 0 and fn["name"] == "submit_file" and isinstance(arguments.get("source"), str):
            code = arguments["source"]
            submitted += 1
            passed, error = grade(code, tests)
            if passed:
                break
            result = "The submitted function failed verification. Test output:\n" + error
        else:
            error = "Unexpected tool name or arguments."
            break
        messages.extend([message, {"role": "tool", "tool_call_id": call["id"], "content": result}])
    return dict(
        task=name,
        rep=rep,
        correct=passed,
        submissions=submitted,
        error=error,
        code=code,
        wall_seconds=time.monotonic() - started,
        turns=turns,
        context_source=args.context_source,
    )


def grade(code, tests):
    match = re.fullmatch(r"```(?:python)?\s*\n(.*)\n```", code.strip(), re.S)
    code = match[1] if match else code
    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-i",
                "--network",
                "none",
                "--read-only",
                "--user",
                "65534:65534",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--memory",
                "128m",
                "--pids-limit",
                "12",
                "--cpus",
                "1",
                "--ulimit",
                "cpu=2:3",
                "--entrypoint",
                "/usr/bin/python3",
                IMAGE,
                "-IS",
                "-",
            ],
            capture_output=True,
            text=True,
            input=code + "\n" + tests + "\n",
            timeout=12,
        )
        return result.returncode == 0, result.stderr[-320:]
    except subprocess.TimeoutExpired:
        return False, "grader timeout"


def message(role, text):
    return f"<|message_{role}|><|content_text|>{text}<|end_message|>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8092")
    parser.add_argument("--model", help="vLLM model name, otherwise use llama.cpp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reuse", action="store_true", help="allow llama.cpp to reuse the previous slot prefix"
    )
    parser.add_argument(
        "--retry", action="store_true", help="return failed checks to the agent once for a second submission"
    )
    parser.add_argument(
        "--context-source", action="store_true", help="include the repository benchmark source as background"
    )
    parser.add_argument("--tools", action="store_true", help="run streamed OpenAI tool calls against vLLM")
    parser.add_argument("--replicates", type=int, default=1)
    args = parser.parse_args()
    if args.tools and not args.model:
        parser.error("--tools requires --model")
    background = (ROOT / "scripts/bench-sycl-baseline.py").read_text() if args.context_source else ""
    answer = "<|message_model|><|content_text|>"
    with args.output.open("x") as out:
        for rep in range(args.replicates):
            for name, task, source, tests in CASES:
                if args.tools:
                    row = tool_case(args, name, task, source, tests, background, rep)
                    print(json.dumps(row), file=out, flush=True)
                    print(name, row["correct"], round(row["wall_seconds"], 2), row["error"], flush=True)
                    continue
                path = f"lib/{name}.py"
                system = "You are a coding agent. First read the relevant file by replying with just READ and the path. After seeing its contents, submit only the complete corrected Python function, with any necessary imports."
                user = f"Repair {path}.\n{task}"
                if background:
                    user += "\nAdditional repository context from a benchmark script:\n" + background
                history = (
                    message("system", system)
                    + message("system", "Thinking effort level: 0")
                    + message("user", user)
                )
                start = time.monotonic()
                read, first = turn(args, history + answer, 32)
                ok = read.strip() == f"READ {path}"
                second = retry = None
                code = err = ""
                if ok:
                    history += (
                        message("model", read)
                        + "<|content_model_end_sampling|>"
                        + message(
                            "user", f"Contents of {path}:\n{source}\nSubmit the corrected function now."
                        )
                    )
                    code, second = turn(args, history + answer, 384)
                    ok, err = grade(code, tests)
                    if not ok and args.retry:
                        history += message("model", code) + "<|content_model_end_sampling|>"
                        failure = {
                            "chunks": "The generator exhausted the entire input before yielding its first chunk.",
                            "spans": "At least one merge, input preservation, or empty-input check failed.",
                            "paths": "An ordering, cycle detection, or dependency-only node check failed.",
                        }
                        history += message(
                            "user",
                            "The submitted function failed verification. "
                            + failure[name]
                            + " Recheck the requirements and submit the complete function.",
                        )
                        code, retry = turn(args, history + answer, 384)
                        ok, err = grade(code, tests)
                row = dict(
                    task=name,
                    rep=rep,
                    correct=ok,
                    wall_seconds=time.monotonic() - start,
                    read=read,
                    code=code,
                    error=err,
                    first=first,
                    second=second,
                    retry=retry,
                    reuse=args.reuse,
                    context_source=args.context_source,
                )
                print(json.dumps(row), file=out, flush=True)
                print(name, ok, round(row["wall_seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
