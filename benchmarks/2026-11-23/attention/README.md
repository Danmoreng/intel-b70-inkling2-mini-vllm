# Long-context attention on the Arc Pro B70

This study compares three changes on one 32 GB B70 at 180 W. The control is the EXL3 recipe at `281141d`, with its optimized 6,144-token prefill chunk, FP8 global KV, BF16 sliding-window KV, vision loaded and a 262,144-token context. Ordinary decode uses sampling. Adaptive DSpark offers up to seven draft tokens. All client runs use Pi with its tools and automatic compaction, medium thinking, an 8,192-token thinking budget, temperature 1, top-p 0.95 and top-k 20.

## Prefix block size

Each engine first processed both prompt shapes. Each reported value is the mean of two subsequent Pi requests with fresh cache salts, identical input hashes across arms and zero cached tokens. TTFT measures the interval to the first streamed output, including reasoning.

| Prefix block | 64,209 input, tok/s | TTFT, s | 128,368 input, tok/s | TTFT, s |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 2,383 | 27.06 | 2,150 | 59.90 |
| 16,384 | 2,367 | 27.25 | 2,152 | 59.84 |
| 32,768 | 2,369 | 27.22 | 2,155 | 59.76 |

At 128K, 32,768 saved 0.15 s to first output. At 64K it cost 0.16 s. The released 8,192 default remains appropriate across both sizes.

An isolated 6,144-row chunk shows where the difference comes from. At 131,072 context, a full 8,192-key block took 0.107 ms to gather, 8.04 ms for oneDNN SDPA and 0.378 ms to merge. A 32,768-key block took 0.420, 32.71 and 0.383 ms respectively. Fewer merges reduced the full older-prefix sweep from 131.38 to 126.72 ms. This covers one global visit of the final chunk. The complete request also processes the other layers and earlier chunks. Relative L2 against the 8,192-key sweep stayed below 0.00037. Prefix measurements use synchronized eager submissions, as in the prefill path. Decode measurements below use graph replay.

To reproduce TTFT, select the block size with `INKLING2_ONEDNN_PREFIX_BLOCK` on the optimized coding launcher. Repeat the [Pi prefill probe](../../2026-11-22/prefill/probe.py) with `--warm-shapes --repeats 2 --chars 222000 444000` using the [frozen corpus](../../2026-11-22/quality/replay/README.md).

## Ordinary decode with AOT SIMT

The donor's AOT kernel processes a single query using four SYCL subgroups, a shared FP8 decode table and groups of four keys. Our port consumes the existing HND pages, FP16 query and relative-bias table, includes the attention sink, and emits a natural-log normalizer for the existing native merge. The donor's published speedup uses a different starting kernel. [Donor implementation and measurements](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/af03fa9/docs/inkling2-mini/INKLING2-MINI-VLLM-XPU.md)

| Cached tokens | Existing XMX, µs | SIMT with donor's 32 splits, µs | Fastest tested SIMT, µs |
| ---: | ---: | ---: | ---: |
| 1,027 | 23 | 20 | 20 |
| 64,504 | 296 | 368 | 325 |
| 128,701 | 627 | 769 | 684 |
| 196,613 | 816 | 1,122 | 1,065 |
| 255,947 | 1,025 | 1,407 | 1,352 |

Each observation includes one global attention and the native reduction. The time is the median of five synchronized batches of 12 graph replays. Single-row parity was checked against dense FP32 at 17 and 1,027 tokens and against XMX through 255,947 tokens, with shuffled physical page IDs and partial final pages. The standalone [SIMT source](simt_m1.cpp) is included for reproduction.

On the frozen stream-codec repair, ordinary XMX delivered **51.52 and 45.66 decode tok/s** at the two starting contexts. The 256-split SIMT port delivered **50.98 and 44.83 tok/s**. All four Pi sessions completed. XMX passed eight of ten hidden tests at each length, SIMT seven. The SIMT sessions emitted fewer tokens (3,403 vs 8,580 at 64K and 5,661 vs 9,062 at 128K). Their shorter wall times include this difference in the work performed.

Run `bash benchmarks/2026-11-23/attention/build-simt.sh` from the checkout to build the experimental library. With the B70 available to the recipe's runtime container and the checkout mounted at `/lab`, the graph benchmark can be run as follows.

```sh
cd /lab
export PYTHONPATH=/lab:/lab/.work/python
python benchmarks/2026-11-23/attention/measure.py \
  --library .work/build/inkling2-vllm.so --control-library .work/build/inkling2-loop-control.so \
  --simt-library .work/build/inkling2-bench-simt-m1.so \
  --context 1027 64504 128701 196613 255947 --rows 1 --parts 32 64 128 256
```

For target verification, omit `--simt-library`, supply `--rows 2 3 4 5 6 7 8` and use `--parts 64 128 192 256 384 512`. For local attention, add `--local` and use `--parts 8 16 24 32 48 64 96`.

## Adaptive DSpark verification

Each target verification consumes one anchor plus the offered proposals. All widths occurred in the two control repairs. At 64K, 22.8% of rounds offered one proposal and 23.9% offered seven. At 128K the shares were 21.2% and 25.0%. The mean number of accepted proposals per round was 2.43 and 2.45 respectively.

The split sweep includes the reduction cost. At 64,504 tokens and two target rows, 96 partitions reduced local attention from 48.3 to 25.7 µs in a fresh process. With four rows, 48 partitions reduced it from 57.9 to 38.9 µs. The local kernel reads a 1,024-token window. A 45-token check showed that large partitions could cost time before the window fills, so the candidate retains 16 splits below 4,096 tokens. Global attention uses shuffled FP8 page IDs and a separate split count. With eight rows at 131,075 tokens, 256 partitions took 958 µs against 1,025 µs at the released 512. Experiments at 50K, 74K, 82K, 90K, 98K, 100K and 150K checked the range surrounding the two Pi workloads.

### Time inside an actual Pi session

The profiler placed XPU events around one in every 19 target replays per width and context bucket, and around the corresponding sample of seven-proposal draft graph replays. The table weights the per-width target medians with the first control repair's width histogram. That holds the amount of verification work constant as the sampled code and acceptances vary between sessions. The target graph includes the model's expert and dense operations. The separate draft graph includes proposal generation and draft sampling. Client overhead and work outside these two graphs are excluded.

| Stream-codec starting context | Target, control → tuned | Draft, control → tuned | Target + draft, control → tuned |
| ---: | ---: | ---: | ---: |
| 64,503 | 31.82 → 30.53 ms | 4.76 → 4.80 ms | 36.58 → 35.34 ms |
| 128,662 | 35.22 → 34.05 ms | 5.37 → 5.41 ms | 40.59 → 39.46 ms |

This gives a **4.1% / 3.3% reduction in target replay time** at 64K / 128K. With the drafter included, the reductions are 3.4% / 2.8%. Each figure describes the measured workload and width mix. At 128K the control's seven-row target median was 44.23 ms and the tuned one was 45.16 ms. The other six widths improved there.

| Stream-codec starting context | Control, decode tok/s | Instrumented control | Tuned | Hidden tests, control → tuned |
| ---: | ---: | ---: | ---: | ---: |
| 64,503 | 86.46 | 85.48 | 92.71 | 8/10 → 10/10 |
| 128,662 | 79.68 | 83.52 | 82.83 | 7/10 → 8/10 |

All sessions used the same seed and completed. The 128K control repeat accepted 2.71 draft tokens per round, compared with 2.45 on the first control and 2.56 on the candidate. This illustrates why raw decode tok/s can vary more than the fixed-width effect. Engine decode rates sum the completion tokens after the first token of each request and divide by total engine decode time.

### Second repair on the release launcher

The route-fix task used another seed (912101) and ran without diagnostic overlays or event timers, in tuned-then-control order. All four patches passed all ten independent tests. The new split counts increased raw decode throughput by **3.7% at 64K** and **4.5% at 128K**.

| Actual starting tokens | Decode tok/s, control → tuned | Generated tokens, control → tuned | Pi wall seconds, control → tuned |
| ---: | ---: | ---: | ---: |
| 64,602 | 83.11 → 86.15 | 14,436 → 17,679 | 206.5 → 238.3 |
| 128,761 | 78.58 → 82.12 | 14,363 → 9,432 | 247.7 → 178.3 |

The longer sampled repair at 64K took 31.8 s more wall time despite the faster engine. At 128K, Pi generated fewer tokens and finished 69.5 s sooner. Two fixtures leave the broader coding-quality comparison open.

The optimized DSpark coding launcher enables this policy automatically. Set `INKLING2_DSPARK_VERIFY_TUNING=0` for the previous policy. The global table applies at 50,000–149,999 tokens with separate choices on either side of 65,536. The local table applies from 4,096 tokens with two through eight verification rows. Single-row decode and the packed-KV experiment use the previous partition policy. The graph cache key includes the selected local and global split counts.

### Full-context capacity

The candidate served a single cold request with **262,016 input + 128 output = 262,144 tokens**, with speculative decoding active and vision loaded. This text-only maximum-context check ran directly against the server so Pi could not compact the input. Prefill was **1,793.5 tok/s**, decode **36.57 tok/s**, and the first output arrived after 146.24 s. The run recorded 60 draft rounds, 140 offered tokens and 66 accepted suggestions.

[Machine-readable measurements, per-width replay samples and independent grade counts](measurements.json)
