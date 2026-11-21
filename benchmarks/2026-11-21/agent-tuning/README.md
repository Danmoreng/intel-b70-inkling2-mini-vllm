# Coding decode on the Arc Pro B70 with medium reasoning

The server had a 262,144-token context, four-bit EXL3 experts, one active request and prefix caching. Each frozen source request used a fresh cache salt, medium reasoning, an 8,192-token thinking budget, temperature 1, top-p 0.95 and top-k 20. The client measured decode from the first streamed token to the last. The fixed-length runs timed 1,024 generated tokens, including reasoning. Coding correctness is assessed separately with tool calls and hidden tests.

## Ordinary EXL3 expert projection

For a single BF16 input row with eight selected experts, the B70 kernel computes each expert's scaled Hadamard transform once for gate and once for up. The projection's nine output groups and two strips read the resulting FP16 tiles. The checkpoint, weight layout, accumulation order and prefill kernels stay the same.

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
