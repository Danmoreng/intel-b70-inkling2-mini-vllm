# Repeated long-prefix hits on Inkling

On a single Arc Pro B70 at 180 W, we replayed the original 64K Qwen serving case against the Inkling EXL3 adaptive DSpark-7 worker. The frozen source text produces 74,690 Inkling input tokens. The November 23 serving report contains the original 70-wave qualification and its figures. [Serving matrix](../serving/README.md)

## Measurements

The unmodified streaming harness ran the three original seeds on a fresh worker before and after the fix. Timing starts at client submission and ends at the first visible text. The first request on each fresh worker also includes startup warm-up.

| 74,690-token request | Cached tokens before | Cached tokens after | First visible text before | First visible text after |
| --- | ---: | ---: | ---: | ---: |
| First pass | 0 | 0 | 47.08 s | 47.01 s |
| First resend | 74,688 | 74,672 | 18.31 s | 2.55 s |
| Second resend | 0 | 74,672 | 27.16 s | 0.70 s |

In the separate matched-seed test, both resends reproduced all 1,024 sampled token IDs and the decoded text of the cold request. Seed 770100 generates a header before the visible text opener. The API emitted the first generated token IDs in about 0.1 s on the warm path. Visible text arrived later and continued streaming in 340 content chunks. The 128-token greedy test also reproduced the original cold output exactly.

At 74,690 tokens a changed instruction in the last page requested a different seven-digit number. The engine returned the updated number on the first changed request, on its repeat and with a new cache salt. A separate 196,608-token branch test asked for a number with an appended question. Cached and cold continuations gave the same incorrect answer byte-for-byte, while the initial request retrieved the number. That case passed the cache parity comparison and failed the retrieval test.

In the 16K sampled panel, both resends exactly matched the originating cold output. A later independent cold control differed. Four further cold-only runs with the same seed and distinct salts produced four different token hashes. All reported zero cached tokens. Greedy decoding produced identical tokens on the cold request and both resends.

A 73,730-token control ended with two prompt tokens after a page-aligned, 73,728-token checkpoint. Both resends reused that checkpoint and retrieved the expected number. In the production medium-thinking vision profile, red and blue images were each named correctly on the cold request and its repeat. Changing the image gave zero cached tokens. A separate cold, short-output vision probe with thinking disabled produced malformed text.

[Measured counts, timings and response hashes](measurements.json)

## Causes

The original scheduler admitted 74,688 cached tokens on the first resend. Only two prompt tokens reached the model. The worker treats blocks up to eight rows as decode or speculative verification. It captured a prompt snapshot only after larger forwards. Changing the request owner cleared the scheduler's saved frontier. After the first warm request generated 1,024 tokens, the next request could reach neither the live decode history nor a registered prompt snapshot. It had to prefill all 74,690 tokens again.

The cache miss and the delayed first visible text had different causes. With seed 770100, a cold control and the original cache hit first diverged at generated token 26. The cold stream eventually opened a visible text block. The cached stream lacked a visible text opener throughout the 1,024-token budget. The chat parser, initially expecting an assistant message header, delivered the buffered text at completion. Returning raw token IDs exposed generation during that interval.

Recomputing an 18-token suffix restored consecutive cache hits, but did not reproduce the cold token sequence. A diagnostic that replayed the original 962-token final prefill chunk from token 73,728 did reproduce all 1,024 token IDs on both resends. This isolates a dependency on how the final prompt chunk is computed. Attributing the divergence to an individual arithmetic kernel would require a separate trace.

## Fix

The scheduler normally leaves at least nine prompt rows for a new request. At the release's 16-token page size it can usually reuse all but 9–24 tokens of a fully matching prompt. When the original prefill ended in a tiny chunk immediately after an aligned saved checkpoint, the scheduler retains that authenticated checkpoint across owners and replays the chunk at its original size. This gives changed suffixes a regular prefill forward and a fresh ring snapshot. Block hashes, including the cache salt, still bound what vLLM can reuse.

The worker also saves up to 32 text-token IDs, their final hidden states and their target and global draft KV rows with its existing local-ring snapshot. For an exact resend ending at the saved position, it checks the suffix IDs, copies the exact KV values into the slots assigned to this request and returns the saved hidden rows to vLLM for sampling. The drafter and sampler generate the reply with the new request's settings. A different suffix runs through the model. Embedded multimodal inputs and the experimental packed-KV layout use the regular computation path.

## Reproduce

Start the production Inkling recipe and use the archived wave with seed 770100. The checker creates a unique cache salt, submits a cold request and two resends, then forces a second cold control with another salt. It checks exact token IDs, decoded text, visible streaming, completion length and cache counters.

```bash
python3 -m unittest discover -s tests -p test_ring_prefix.py
# Run the tensor test in an environment with PyTorch.
python3 -m unittest discover -s tests -p test_prefix_tail.py
python3 scripts/check-prefix-resends.py \
  --fixture results/serving-20261123/full/matrix/prefix-64k-cold-warm-r2 \
  --output results/prefix-resend-check.jsonl
```
