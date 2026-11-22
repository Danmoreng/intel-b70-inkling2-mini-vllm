# DSpark verification width, prefix workspace and screenshot encoding on the B70

One 32 GB Arc Pro B70 at 180 W served the EXL3 checkpoint with the vision-enabled optimized prefill profile, 6,144-token scheduler chunks, FP8 global KV, a BF16 local sliding-window ring and a 262,144-token context. All agent measurements used real Pi 1.5.0 sessions with automatic compaction, medium reasoning and an 8,192-token thinking budget. [Per-run counters, file hashes and samples](recipe-comparison.json)

## Adaptive DSpark verification

The launcher defaults to three proposals. Set `INKLING2_DSPARK_ADAPTIVE=1 INKLING2_DRAFT_MAX=7` for up to seven. Both profiles applied the same 0.75 confidence threshold and 256-candidate shortlist. Each task used a fresh checkout and a 240-second wall limit. The grader applied the resulting patch and ran ten hidden checks independently.

| Pi task | Seed | Cap | Decode tok/s | Proposed / accepted per round | Output tokens | Task seconds | Passed within deadline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| routefix | 911701 | 3 | 92.84 | 2.35 / 1.74 | 12,451 | 141.0 | Yes |
| routefix | 911701 | 7 | 102.62 | 3.90 / 2.67 | 21,596 | 215.0 | Yes |
| streamcodec | 911701 | 3 | 91.33 | 2.34 / 1.69 | 9,847 | 113.2 | No |
| streamcodec | 911701 | 7 | 95.74 | 3.61 / 2.31 | 11,862 | 129.3 | Yes |
| streamcodec | 912101 | 3 | 95.16 | 2.49 / 1.84 | 6,816 | 77.3 | No |
| streamcodec | 912101 | 7 | 98.66 | 3.97 / 2.51 | 9,060 | 96.9 | No |

Decode tok/s pools completed streaming responses. The numerator sums output tokens after the first token per response and the denominator sums time from first content, reasoning or tool delta to stream completion. Task time also includes prefill, tools and thinking. Paths can diverge under probabilistic speculation, even at a matched seed. For seed 911701 the seven-proposal engine ran first. For the codec repeat with seed 912101 the three-proposal engine ran first. Text-only repairs never invoked the vision encoder.

In the earlier 35,914-token serving measurement, 676 accepted proposals across 348 rounds gave 1.94 accepted draft tokens per round. Including the output token from the target gives 2.94 tokens per round. [Historical serving counters](../2026-11-21/agent-tuning/serving.json)

Across the 3 matched comparisons, client decode rose from 92.85 to 99.76 tok/s (+7.5%). Verification rounds took about 29.69 and 35.45 ms, computed from the engine decode timer divided by speculative rounds. The target accepted the seventh proposal in 12.1% of all wide rounds. The wide runs generated 42,518 tokens in 441.2 s of task time and passed 2/3. The narrow runs generated 29,114 tokens in 331.5 s and passed 1/3. We keep three as the coding default. The evidence comes from two distinct tasks.

## Reusing the global K/V prefix buffer

The candidate stores one 32 MiB FP16 K/V pair per active XPU stream. Every full or partial 8,192-token block reuses that storage with a contiguous view sized for its actual length. The FP8 gather, head ordering and oneDNN attention are identical. Five full and partial gathers from shuffled FP8 pages matched bit for bit on the B70.

The experimental patch is archived at [K/V workspace candidate](../../patches/experimental-inkling-prefix-workspace.patch). Apply it to a clean checkout with `git apply patches/experimental-inkling-prefix-workspace.patch` to reproduce the comparison. The two fresh three-proposal engines used identical settings except for `INKLING2_PREFILL_KV_WORKSPACE=0/1`. Each received a short Pi warmup and one warmup at each prompt shape. The table gives the median of three further uncached sessions on each engine. Each run used a fresh cache salt and reported zero cached prompt tokens.

| Cold input tokens | First output, separate / reused | TTFT change | Engine prefill, separate / reused (tok/s) |
| ---: | ---: | ---: | ---: |
| 32,849 | 13.304 / 13.313 s | +0.066% | 2,487.3 / 2,485.8 |
| 95,679 | 42.723 / 42.721 s | -0.004% | 2,247.9 / 2,248.1 |

The minimum free VRAM sampled once per second during measured prefills was 3.36 GiB in the control and 3.25 GiB with the workspace. These are whole-card readings including allocator reservations.

Reproduce the two lengths from the repository root with `python benchmarks/2026-11-22/prefill/probe.py long.txt OUTPUT --url http://127.0.0.1:8093 --model inkling2-mini-exl3-b70 --chars 110000 333000 --repeats 3 --warm-shapes`. Use separate output directories and restart the server between variants. [Frozen corpus and Pi environment](quality/replay/README.md)

## Fused vision Q/K/V

Set `INKLING2_VISION=1 INKLING2_VISION_FUSED_QKV=1` to enable the candidate. It concatenates the 23 layers’ projection weights and biases once, after loading the checkpoint. Each layer then uses one BF16 projection and skips the single-image attention-output concatenation. These timings span the full encoder, with preprocessed inputs on the B70 and synchronization around each call. We used two warmups and 16 measured runs per variant and input, alternating which variant ran first. The images came from the earlier support web app built by Qwen 27B. The combined row exercises the encoder’s two-image path. The serving adapter encodes each image individually.

| Screenshot input | Separate (ms) | Fused (ms) | Time saved (ms) | Bitwise equal output |
| --- | ---: | ---: | ---: | --- |
| 1280 × 1535 | 109.016 | 108.733 | 0.283 | Yes |
| 390 × 1601 | 25.454 | 25.339 | 0.115 | Yes |
| 1280 × 1535 + 390 × 1601 | 142.856 | 142.705 | 0.151 | Yes |

Both serving variants also received the same Chromium screenshot through Pi’s file-read tool. The provider payload matched its SHA-256, and both correctly identified the two colors and number. The isolated encoder measurements exclude image preprocessing, tool execution and language-model time. Run `python benchmarks/2026-11-22/vision/compare_qkv.py CHECKPOINT DESKTOP.png MOBILE.png` in the patched serving runtime to repeat the operator comparison.

On the first image-bearing turn Pi saw a first delta after 0.895 s with separate projections and 0.929 s with fused projections. Each number comes from one fresh server. This check establishes screenshot correctness, with insufficient repeats to resolve a sub-millisecond end-to-end difference.

The released launcher also passed two fresh, uncached Pi prompts with the seven-proposal setting, fused vision and the standard K/V allocation. First output arrived at 13.74 s for 32,849 input tokens and 42.55 s for 95,679. These were one-shot startup checks.
