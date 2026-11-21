# Coding decode on the Arc Pro B70 with medium reasoning

The server had a 262,144-token context, four-bit EXL3 experts, one active request and prefix caching. Each frozen source request used a fresh cache salt, medium reasoning, an 8,192-token thinking budget, temperature 1, top-p 0.95 and top-k 20. The client measured decode from the first streamed token to the last. The fixed-length runs timed 1,024 generated tokens, including reasoning. Coding correctness is assessed separately with tool calls and hidden tests.

## Ordinary EXL3 expert projection

For a single BF16 input row with eight selected experts, the B70 kernel computes each expert's scaled Hadamard transform once for gate and once for up. The projection's nine output groups and two strips read the resulting FP16 tiles. The decode kernel accumulates those tiles in the original order.

| Three cold source prompts, 4,980–5,514 input tokens | Median client decode, tok/s |
| --- | ---: |
| Released kernel, first control | 58.03 |
| New kernel, first server | 58.31 |
| New kernel, second server | 58.36 |
| Released kernel, second control | 57.54 |

Against the faster control, the two candidate medians improved by 0.48% and 0.57%. The second control overlapped a CPU kernel build on its first two requests. All three generated reasoning streams matched their respective control hashes. The isolated expert layer went from 117.85 to 115.06 µs. On 64 teacher-forced positions after 4,093 code tokens, the single-row target logits matched exactly. The rebuilt library passed all six FP16/BF16 checkpoint reference checks at one, four and eight rows. [Per-request measurements and numerical checks](ordinary-experts.json)

Build and test the kernel with `INKLING2_EXL3_TILE_ROWS=32 bash scripts/check-sycl.sh exl3`, then start `bash scripts/serve-coding-agent.sh ordinary`. To reproduce the timed requests with the Qwen serving archive, run this command from the repository root.

```bash
python3 scripts/bench-frozen-serving.py --archive "$QWEN_SERVING_ARCHIVE" \
  --cases phase-4k-c1-r1 phase-4k-c1-r2 phase-4k-c1-r3 \
  --effort medium --thinking-budget 8192 --output ordinary-review.jsonl
```

## Thinking budget across tool calls

Inkling can produce multiple thinking messages before returning a tool call or a final answer. The original vLLM V2 sampler counted from the most recent thinking opener. On the held-out routing task with seed 912101, one API response reported 11,518 reasoning tokens against a requested budget of 8,192. The B70 sampler now carries the cumulative count through every thinking message in a response. It counts accepted history and only the draft prefix for the row under verification.

On the same fixture and seed, the corrected server reached exactly 8,192 thinking tokens in its longest response and passed the hidden suite. Total task time was 153.27 s versus 164.20 s in the original run. Both ordinary and DSpark also returned exactly the requested reasoning counts in six short API checks at budgets 0, 8, 32 and 96. The tool-call sequence changed in the routing task. This pair measures the behavior of the cap and illustrates the variability of full-task timings. A native regression test checks multiple thinking spans, a rejected draft prefix, prompt resumption, the end-of-sampling token, a recycled slot and an 8,192-token boundary. [API counts and tool turns](thinking-budget.json)

## Single-row paged attention

The B70 XMX kernel processes 16 keys per tile. Short ordinary requests benefit from more independent partitions. At 192K, reducing the previous 512 partitions to 192 cuts the cost of merging partial results. The recipe uses this schedule for one query row up to 8,192 and from 131,072 tokens. The original middle range performed better on a 35,914-token source prompt. Four-row DSpark verification uses the original schedule.

| Matched input | Released ordinary | Ordinary with both kernel changes |
| --- | ---: | ---: |
| Code, 5,292 tokens | 58.14 tok/s | 59.16 tok/s |
| Code, 5,514 tokens | 57.77 tok/s | 58.82 tok/s |
| Code, 35,914 tokens | 54.76 tok/s | 55.07 tok/s |
| Code, 77,274 tokens | 50.50 tok/s | 50.98 tok/s |
| Logbook, 196,608 tokens | 42.80 tok/s | 43.22 tok/s |

The code rows use client decode timing and 1,024 generated tokens. Each logbook run used vLLM's decode timer, generated 256 tokens and retrieved the embedded code. Cold prefill measured 1,391 and 1,394 tok/s respectively. The final ordinary server passed all four original coding fixtures in 304.43 s with medium reasoning and the 8,192-token cap. The attention-only sweep also tested the previous middle-range split rule and a candidate for all lengths. At 4,093 prefix tokens the candidate agreed on the most likely token at all 64 teacher-forced positions with mean forward KL 0.00097. [Controls, intermediate variants and complete recipe measurements](ordinary-attention.json)
