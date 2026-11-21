# Single-agent decode on Arc Pro B70

One Intel Arc Pro B70, 32 GB at 180 W, ran one text sequence at a time. Both modes used the same 4-bit EXL3 experts, target vocabulary, FP8 E4M3 global KV and BF16 local KV. DSpark read the 3-layer drafter from the EXL3 checkpoint. The target verified every proposal. All measurements use the pinned vLLM 0.33.0 runtime from the setup guide. [Setup and checkpoint revisions](../../../docs/SETUP.md)

## What the measurements cover

- **Coding tasks.** The model inspected a small Python repository, edited its implementation, could run visible tests and ended the task itself. The harness ran hidden tests in an isolated container after the clock stopped. Wall time includes model requests and tool execution. Each fixture and seed is retained, including failed tasks. The two original tasks are `rangeset` and `dependencyplan`. [Per-task measurements](coding-tasks.json) · [Frozen Qwen fixtures and harness](https://github.com/Danmoreng/intel-b70-qwen38-vllm/tree/8fdf9260c3526dc42970b9bf6153c9df197c8bd2/benchmarks/experiments/vllm-033-upgrade/agent-panel)
- **Source reviews.** We replayed the archived Qwen source text byte-for-byte and checked its SHA-256 before each request. Each model used its own tokenizer and chat renderer. Each request generated 1,024 tokens at temperature 1, top-p 0.95 and top-k 20 with the recorded seed. Client decode speed divides 1,023 by the time from the first streamed output to completion. Cold-cache counts, first-output times and target/draft counters are recorded separately. The review text has no correctness grade. [Per-request measurements](source-review.json)
- **Context and cache.** The fixed-length logbook prompts contain a seven-digit code in the middle. Retrieval passes when the first number in the answer matches the code. A separate branching test discards 128 generated tokens and appends a question to the original prompt. The control uses the same extended prompt with a fresh cache salt. The 262,144-token boundary consists of 262,016 input and 128 output tokens. [Context and retrieval measurements](context.json)
- **Kernel screening.** Two seeded 4,096-token logbook prompts, each with 128 greedy outputs, establish short-context throughput after warm-up. Runs use vLLM's native prefill and decode timers. Token predictability changes the draft acceptance rate. [Selected kernel measurements](kernel-screen.json)

The Qwen agent reference ran four successful fixture/seed combinations in 145.88 seconds at temperature 1, medium reasoning, an 8,192-token thinking budget and a 16,384-token output limit. The benchmark records the requested reasoning effort and the usage counters on each turn. Inkling can emit reasoning tokens with the requested effort set to `none`. A medium-reasoning request can generate a different number of tokens on each model. The archived Qwen configuration supported 16 slots. These comparisons use a single active request. [Qwen qualification record](https://github.com/Danmoreng/intel-b70-qwen38-vllm/blob/8fdf9260c3526dc42970b9bf6153c9df197c8bd2/benchmarks/results/direct-xpu-20261114/qualification/m04-fixed-r12/gates/agents/summary.json)

The published checkout was rebuilt from the pinned Trellis sources and tested with both full-size server profiles. Each server answered two cold 4,096-token prompts with 128 output tokens after warm-up. The DSpark server also reused exactly 8,192 tokens after the client discarded its first continuation. With an audit-field follow-up both warm and cold requests retrieved the code. A repeated-question variant missed the code once with a cache hit, and once in the isolated cold control. Complete continuations can differ with these kernels. The records retain every observed miss. [Checkout verification](checkpoint-smoke.json)

## Decode experiments

On three short source reviews, the grouped QKV and fused attention-preparation kernels delivered 69.8 client tok/s with the original expert decode. Using the eight-row XMX expert path for four verified positions raised the median to 74.5. The shared group-32 INT8 vocabulary head and the drafter's reference rounding together reached 83.3. Proposal acceptance rose from 45.3% to 51.0% in that last comparison. These screening runs used a 32K serving profile. The full-context coding launcher also enables synchronous prefix reuse.

A sparse Gumbel implementation matched 360 seeded reference draws and reduced the time for three draws from 44.2 to 9.9 microseconds. The paired source-review median was 69.7 versus 69.8 tok/s, so the launcher uses vLLM's existing sampler. Shortening the drafter's global attention history reduced throughput on the longer source reviews. The launcher retains the whole history. [Kernel comparison](decode-screen.json) · [Source-review records](source-review.json)

## Target and cache checks

On a frozen code sample, the optimized target scored the same 96 next tokens in single-row decode and in four- and eight-row verification blocks. The highest-probability token agreed at all 96 positions for both block widths. A separate check used shuffled physical KV pages, discarded 128 tokens after an 8,192-token prefix, and compared 23 follow-up positions with an uninterrupted control. Their highest-probability tokens also agreed at every position. The numerical record includes KL divergence and reconstruction errors. [Target and kernel checks](target-parity.json)

## Reproduce a measurement

Run one server with `bash scripts/serve-coding-agent.sh dspark` or `bash scripts/serve-coding-agent.sh ordinary` from the root of this recipe. Use a fresh output file for each run.

```bash
python3 scripts/warm-coding-agent.py
python3 scripts/bench-agent-panel.py --qwen-repo /path/to/intel-b70-qwen38-vllm \
  --effort none --output results/agent-panel.jsonl

python3 scripts/bench-frozen-serving.py --archive /path/to/archived/serving/run \
  --cases phase-4k-c1-r1 phase-4k-c1-r2 phase-4k-c1-r3 \
  --output results/review-4k.jsonl

python3 scripts/guard-prefill.py --floor 650 --wall 310 -- \
  python3 scripts/bench-sycl-baseline.py --url http://127.0.0.1:8093 \
    --vllm-model inkling2-mini-exl3-b70 --lengths 196608 --decode 128 \
    --timeout 310 --output results/cold-192k.jsonl
```

Use `--fixtures /path/to/fixtures --tasks routefix streamcodec --seeds 911701 912101` for the additional coding tasks. The runner verifies the fixture manifest and copies the project into a fresh directory for each seed. The original fixture seeds are 202611117 and 202611119. The source-review harness requires the corresponding `prompt-1.txt` and `request-1.json` files from the serving archive.

The first API request for a new execution shape can include compilation and graph capture. The warm-up client exercises tool-enabled chat and both greedy and stochastic sampling, using a new cache salt each time. The optional persistent cache holds Triton artifacts. The pinned SYCL runtime crashed in its persistent-device-code-cache lookup during startup, so the launcher keeps the SYCL disk cache disabled.
