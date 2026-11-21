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

Inkling can produce multiple thinking messages before returning a tool call or a final answer. The original vLLM V2 sampler counted from the most recent thinking opener. On the held-out routing task with seed 912101, one API response reported 11,518 reasoning tokens against a requested budget of 8,192. A fresh control run exposed the two thinking messages directly in the returned token IDs, with 8,192 and 170 reasoning tokens in one response. The B70 sampler now carries the cumulative count through every thinking message in a response. It counts accepted history and only the draft prefix for the row under verification.

On the same fixture and seed, the corrected server reached exactly 8,192 thinking tokens in its longest response and passed the hidden suite. Total task time was 153.27 s versus 164.20 s in the original run. Both ordinary and DSpark also returned exactly the requested reasoning counts in six short API checks at budgets 0, 8, 32 and 96. The tool-call sequence changed in the routing task. This pair measures the behavior of the cap and illustrates the variability of full-task timings. A native regression test checks multiple thinking spans, a rejected draft prefix, prompt resumption, the end-of-sampling token, a recycled slot and an 8,192-token boundary. [API counts and tool turns](thinking-budget.json)

## Single-row paged attention

The B70 XMX kernel processes 16 keys per tile. Short ordinary requests benefit from more independent partitions. At 192K, reducing the previous 512 partitions to 192 cuts the cost of merging partial results. The recipe uses this schedule for one query row up to 8,192 and from 131,072 tokens. The original middle range performed better on a 35,914-token source prompt. The subsequent four-row DSpark comparison below covers its separate long-context setting.

| Matched input | Released ordinary | Ordinary with both kernel changes |
| --- | ---: | ---: |
| Code, 5,292 tokens | 58.14 tok/s | 59.16 tok/s |
| Code, 5,514 tokens | 57.77 tok/s | 58.82 tok/s |
| Code, 35,914 tokens | 54.76 tok/s | 55.07 tok/s |
| Code, 77,274 tokens | 50.50 tok/s | 50.98 tok/s |
| Logbook, 196,608 tokens | 42.80 tok/s | 43.22 tok/s |

The code rows use client decode timing and 1,024 generated tokens. Each logbook run used vLLM's decode timer, generated 256 tokens and retrieved the embedded code. Cold prefill measured 1,391 and 1,394 tok/s respectively. The final ordinary server passed all four original coding fixtures in 304.43 s with medium reasoning and the 8,192-token cap. The attention-only sweep also tested the previous middle-range split rule and a candidate for all lengths. At 4,093 prefix tokens the candidate agreed on the most likely token at all 64 teacher-forced positions with mean forward KL 0.00097. [Controls, intermediate variants and complete recipe measurements](ordinary-attention.json)

## Coding-agent comparisons

Each configuration ran two seeds on each of two editing tasks and two harder held-out tasks. The harness started each task in a clean workspace and used a unique prefix-cache salt. The model could reuse its own conversation prefix across turns. A task counted as solved when it finished and its edited repository passed hidden tests. The 240-second task deadline includes tool calls. Output tokens include model reasoning. A client timeout can leave the final response without usage counters, so reported output totals for timed-out tasks are lower bounds.

The medium profile requested 16,384 output tokens per turn, including up to 8,192 reasoning tokens. Sampling controls varied one setting at a time relative to temperature 1, top-p 0.95 and top-k 20. The open sampler removed both truncation filters. The cooler sampler changed temperature to 0.6. Adaptive verification kept the reference sampling settings and used the largest probability within the drafter's top-20 distribution to select how many proposals to verify. The target always verified the first proposal. This prototype still computed the full draft, so the experiment measures variable target verification width, including its scheduling overhead.

| DSpark configuration | Original solved | Wall seconds | Output tokens | Held-out solved | Wall seconds | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed, standard sampling (control) | 4/4 | 173.07 | 14,261 | 2/4 | 576.27 | 49,338 |
| Fixed, open sampling | 4/4 | 216.39 | 18,383 | 3/4 | 618.62 | 51,429 |
| Fixed, temperature 0.6 | 4/4 | 224.72 | 19,693 | 2/4 | 697.03 | 42,798 |
| Adaptive, threshold 0.52 | 4/4 | 195.80 | 16,832 | 2/4 | 527.04 | 45,461 |
| Adaptive, threshold 0.75 | 4/4 | 153.78 | 13,305 | — | — | — |
| Rebuilt, tensor draft mix | 4/4 | 206.34 | 17,382 | 3/4 | 536.91 | 45,695 |
| Rebuilt, fused draft mix | 4/4 | 190.27 | 16,219 | 3/4 | 566.55 | 47,274 |
| Rebuilt, adaptive threshold 0.75 | 4/4 | 174.19 | 14,993 | 4/4 | 452.83 | 39,131 |

The archived Qwen reference solved the same four original tasks in 145.88 s using 8,130 generated tokens. On the rebuilt Inkling recipe, the four original tasks used 16,219 tokens with fused DSpark, including all tool turns. For this panel the target tokenizers and generation trajectories differ. [Archived Qwen run and fixture details](../agent-decode/README.md)

The single-row attention and EXL3 changes above affect ordinary decode. The drafter tests below use three proposal positions. The graph-input experiment copied IDs, positions, sequence lengths, KV slots and the block table in a single SYCL launch. The four-tap fusion computed the three draft-layer inputs and the Markov-head input with one kernel. The original FP32 tensor expressions serve as the numerical reference. With four rows, fusion reduced the isolated feature mix and RMS normalization from 62.68 to 7.79 µs. The timing inputs at one, four and eight rows differed by at most 3.51e-9 in relative L2. A separate seeded comparison of the rebuilt public kernel exercised one, two, three, four and eight rows and measured a maximum relative L2 difference of 2.37e-5. [Kernel timings and numerical checks](kernel-sweeps.json)

To compare a sampler on the original panel, start the DSpark server and use the same two seeds, task deadline and per-turn budget. For the open sampler, pass `--top-p 1 --top-k -1`. For the cooler sampler, pass `--temperature 0.6`. To repeat the held-out panel, add `--fixtures /path/to/held-out/fixtures --tasks routefix streamcodec --seeds 911701 912101`. The JSON records include the fixture manifest hashes.

```bash
python3 scripts/bench-agent-panel.py --qwen-repo "$QWEN_REPO" \
  --effort medium --thinking-budget 8192 --max-output 16384 \
  --temperature 1 --top-p 0.95 --top-k 20 --wall 240 \
  --output coding-medium.jsonl
```

The open sampler completed both streaming-codec seeds and one routing seed on the harder set. The standard completed one of each. One open-sampler routing edit passed the hidden tests but the agent missed its task deadline. At temperature 0.6, the same number of harder cases finished correctly as with the standard sampler, and total harder-panel wall time increased from 576 to 697 s. The second cooler codec seed timed out at 240.10 s and its final engine metrics were unavailable.

### Verification width

At a 0.52 confidence threshold, both repeats on the three short code prompts accepted 1,799 of 3,061 offered tokens (58.8%). Fixed-width DSpark accepted 1,847 of 3,678 (50.2%). The higher acceptance fraction came with fewer accepted tokens for the same output length. Median client decode moved from 81.24 to 81.52 and 81.60 tok/s. On the original editing tasks both configurations solved four cases and adaptive verification took 195.80 s versus 173.07 s. On the harder panel both solved two cases. The adaptive pair of successful runs took 313.08 s in total versus 233.21 s for the fixed-width pair. They solved different routing seeds. A separate 0.75 threshold reached 84.23, 95.94 and 80.54 tok/s on the three frozen prompts. The late fixed-width control measured 85.04, 81.22 and 74.49. On the four original coding cases the 0.75 prototype solved all four in 153.78 s. That prototype used the earlier kernel stack and did not run the harder panel at this threshold. The default coding recipe uses three fixed proposals. [Per-request acceptance and decode](serving.json) · [Coding-task outcomes](coding-panel.json)

For the optional rebuilt variant, start `INKLING2_DSPARK_ADAPTIVE=1 bash scripts/serve-coding-agent.sh dspark`. The synchronous scheduler receives the live prefix from the drafter. Low confidence at one position suppresses later proposals, and the target captures graphs for the intermediate verify widths. On the four original real coding cases, this rebuilt configuration solved 4/4 in 174.19 s with 14,993 generated tokens, versus 190.27 s with fixed width on this same stack. The earlier 153.78 s prototype and this final combination are separate measurements. The optional rebuilt profile completed 4/4 held-out tasks correctly in 452.83 s across all attempts. The setting can change sampled outputs, tool turns and task success. [Individual fixtures and settings](coding-panel.json)

### Draft feature fusion and graph inputs

Three fixed greedy logbook prompts used 4,096 input tokens, 512 output tokens and the same seven-digit retrieval task. The early and late unfused medians were 122.04 and 122.05 engine tok/s. Two independent servers with fused taps reached 122.60 and 122.31. Graph input staging alone reached 121.99, and combining staging with feature fusion reached 122.22. All retrieved the correct code. Full completion hashes were captured for the second fusion run, the late control and both staging variants, and matched for all three prompts. On the final rebuilt stack, identical greedy completions measured 123.26, 121.17 and 120.73 tok/s with fusion and 123.87, 121.96 and 121.49 with the tensor implementation. This reverses the small timing advantage from the earlier isolated servers. On the original medium-reasoning editing tasks the fused and tensor runs both solved four, in 190.27 and 206.34 s respectively. They used 16,219 and 17,382 generated tokens. Across both panels they solved the same seven of eight tasks. The seven completed, correct tasks took 516.80 s with fusion versus 604.65 s without it. Including the eighth task yields 756.81 versus 743.25 s, because the fused run timed out and the tensor run submitted an incorrect solution sooner. The coding launcher enables fusion by default. The recipe retains the existing graph input copies. Set `INKLING2_DRAFT_FUSED_TAPS=0` to select the tensor implementation. [Per-prompt throughput and hashes](serving.json)

### Long-context draft and target attention

The alternative FP16 XMX drafter read the complete global KV history. Its isolated 196,608-key operation went from 1,899 to 1,762 µs, with relative L2 of 0.00015. On four cold code reviews at 35,914, 77,274, 37,535 and 78,242 input tokens, baseline client decode was 85.39, 82.53, 77.97 and 77.19 tok/s. The alternative gave 87.03, 72.00, 75.43 and 75.29. A fresh control returned 85.33, 82.54, 77.95 and 77.12. The production drafter retains its current attention kernel. [Kernel sweep](kernel-sweeps.json) · [Source-review timings](serving.json)

For the four-row target verifier, a 128-partition schedule reduced the isolated global attention operation at 196,608 tokens from 965.52 to 905.55 µs. With a broad split policy, the cold 196,608-token logbook went from 61.40 to 65.06 engine decode tok/s and both runs retrieved the code. On the 77,274-token source review, however, decode fell from 82.53 to 60.30 client tok/s. On two larger concatenations of archived code, both verification rounds got faster and the number of rounds changed substantially. Each prompt produced 1,024 tokens under medium reasoning.

| Cold input tokens | Original / broad split, client tok/s | Original / broad split, ms per verification round | Original / broad split, rounds |
| ---: | ---: | ---: | ---: |
| 231,858 | 46.53 / 68.82 | 43.11 / 42.47 | 510 / 350 |
| 232,055 | 59.75 / 55.22 | 42.80 / 41.82 | 400 / 443 |

Round time divides the client decode interval by the engine's verification-round counter. Different rounding can change sampled continuations, proposal acceptance and the number of rounds. Each cold prefill exceeded 1,310 tok/s. The fixed-length source tests time model generation, including reasoning. The hidden-test panel measures whether an editing task reaches a correct completion. [Per-request timings, proposal counts and source digests](serving.json)

The raw coding record also includes the earlier per-thinking-span controls, identified by `budget_scope`. Two HTTP 500 infrastructure failures in the early open-sampler run were repeated with the same codec fixture and seeds. A metrics scrape interrupted the cooler cumulative-budget run after the routing pair. Fresh servers repeated the two codec seeds. The task harness now retains the hidden-test result and measured wall time if the final metrics scrape fails. Valid task timeouts remain in the score. [Coding records](coding-panel.json)


### Rebuilt long-context recipe

Enabling 128 partitions only for four target verification rows at or above 196,608 input tokens improved all three matched long cases on the rebuilt fused DSpark recipe. The guarded cold logbook retrieved the correct code with both settings. The two code requests used the same prompts, seeds, medium reasoning and 1,024 generated tokens. The ordinary schedule is independent.

| Input | Original / new DSpark decode, tok/s | New cold prefill, tok/s | Original / new ms per round |
| --- | ---: | ---: | ---: |
| Logbook, 196,608 | 61.75 / 64.71 | 1,389 | — |
| Code, 231,858 | 64.46 / 76.73 | 1,312 | 43.01 / 42.46 |
| Code, 232,055 | 61.43 / 68.21 | 1,316 | 42.70 / 41.89 |

The coding launcher enables this long-context schedule for DSpark. Set `INKLING2_DSPARK_LONG_SPLITS=0` before starting the server to reproduce the original verifier schedule. The 8K prefix-branch check with fused taps retrieved the expected code on all three requests. The rollback reused exactly 8,192 cached tokens and returned its first output in 1.22 s, compared with 4.74 s for the isolated cold branch. [Rebuilt runs and branch audit](serving.json)
