# Cold EXL3 prefill at 180 W and 230 W

One Arc Pro B70 ran the optimized recipe at `27ea9c1`, with 4-bit EXL3 experts, a 6,144-token scheduler chunk, 8,192-token attention prefix blocks, FP8 global KV, vision loaded and adaptive DSpark seven. The same live engine ran at 180, 230, 230 and 180 W in that order. Each pass covered all three lengths. Length order alternated. The real Pi Coding Agent used medium thinking, automatic compaction and an 8,192-token thinking budget. Every prompt was cold, and all arms at a given length had the same prompt hash.

| Input tokens | Engine tok/s, 180 → 230 W | Gain | First streamed output, 180 → 230 W | Client input tok/s at 230 W |
| ---: | ---: | ---: | ---: | ---: |
| 32,849 | 2,507 → 2,818 | 12.4% | 13.19 → 11.74 s | 2,799 |
| 65,324 | 2,376 → 2,681 | 12.8% | 27.61 → 24.48 s | 2,669 |
| 130,986 | 2,142 → 2,428 | 13.4% | 61.34 → 54.11 s | 2,421 |

Each entry is the mean of two requests at each power limit. Engine throughput divides prompt tokens by the vLLM prefill timer. Client input throughput divides them by time to the first streamed output, including reasoning. The Linux Xe driver accepted both card limits. The card energy counter over the central prefill interval measured 180.2–180.4 W and 230.0–230.1 W across the three means. Mean GPU clocks rose from 2.11–2.13 GHz to 2.48–2.50 GHz. The highest sampled package temperature during a 230 W baseline prefill was 70 °C. The higher limit cost roughly 13% more card energy per input token, excluding CPU and system power.

## Comparison with the original checkpoint

The cookbook's 230 W BF16-KV speed profile reports 3,330 and 2,891 input tok/s at 32,768 and 65,536 tokens, measured with one output token. Our comparable client rates at 230 W are 2,799 and 2,669. The cookbook also reports 2,701 and 2,195 tok/s at 64K and 128K with FP8 KV, vision and DSpark in its chunked-attention discussion. The profiles and prompts differ, so this cross-recipe comparison cannot isolate a kernel. Moving our card from 180 to 230 W closes much of the 64K gap and brings our 128K result above the cited FP8 figure. [Pinned cookbook, cold anchors and FP8 discussion](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/af03fa9829008dab082c797236293e86ab86226a/docs/inkling2-mini/INKLING2-MINI-VLLM-XPU.md)

The cookbook's grouped expert GEMMs unpack NVFP4 and MXFP8 weights. EXL3 reconstructs trellis weights and applies Hadamard transforms. Our long-prefix path already uses oneDNN beyond the relative-bias window. The expert implementation offers two testable ideas while preserving EXL3: change how routed tokens are grouped, and reuse each reconstructed weight tile across more routed rows.

## Scheduler chunk at 230 W

At 32,849 input tokens, Sergio's 2,048-token scheduler setting gave 2,702 engine tok/s versus 2,818 for our 6,144-token setting. At 65,324 it gave 2,584 versus 2,681. This arm had a separate engine start, a shape warmup and two cold Pi requests at each length. The larger scheduler setting remains the faster choice on the current EXL3 stack.

## Native EXL3 expert kernels

The production scheduler feeds up to 6,144 tokens to an expert layer. Processing 48 routed rows per XMX tile lets more tokens share the cost of reconstructing the same four-bit expert weights. The new library selects this tile for batches of at least 2,048 input rows on the Inkling 2 Mini expert shape (hidden size 2,304 and intermediate size 1,152). Smaller prefill, ordinary single-token decode and the adaptive verifier use their established tile sizes. Workgroup dimensions follow the selected kernel, including the verifier's eight-row specialization.

On all 128 real layer-zero experts at 230 W, the isolated 6,144-token operator fell from 24.11 to 21.29 ms with uniform routing and from 24.20 to 21.42 ms with skewed routing. Outputs matched the 32-row kernel bit for bit. Each timing averages the medians from two runs in reversed order. Each median covers seven batches of five calls. An alternative three-stage SYCL route sort (counts, prefix sums and scatter) added about 1.7–1.8% to the operator time at this batch size.

Across 15 whole-layer shapes with the final adaptive tile, both changed shapes (2,048 and 6,144) and all verifier widths (1–8) were bit-identical to the previous library. The 47-row grouped kernel had a relative L2 difference of 0.000927, within the 0.001040 observed between two runs of the previous library on identical input.

## Cold Pi requests with adaptive expert tiles

Both power limits use the same hybrid library. The 32K and 64K controls were also rerun from another engine start at 230 W (2,820 and 2,679 engine tok/s). The 128K hybrid sweep used the public optimized launcher, a shape warmup and the same 180/230/230/180 W order as the control. Values below average two cold requests per cell, with the 180 W result first.

| Input tokens | Prior engine tok/s, 180 / 230 W | New engine tok/s, 180 / 230 W | Kernel gain, 180 / 230 W | New client TTFT, 180 / 230 W |
| ---: | ---: | ---: | ---: | ---: |
| 32,849 | 2,507 / 2,818 | 2,663 / 3,015 | +6.2% / +7.0% | 12.43 s / 10.99 s |
| 65,324 | 2,376 / 2,681 | 2,513 / 2,852 | +5.8% / +6.4% | 26.12 s / 23.03 s |
| 130,986 | 2,142 / 2,428 | 2,257 / 2,560 | +5.4% / +5.5% | 58.21 s / 51.32 s |

On the updated recipe the client measured 2,990 and 2,837 input tok/s at 230 W, versus the cookbook's 3,330 and 2,891 in its speed profile. At 128K, our client rate was 2,552 tok/s with FP8 KV, vision loaded and adaptive DSpark. All runs used a single sequence.

An independent dense FP32 CPU reconstruction passed for 12 real experts with FP16 and BF16 activations at 4, 7, 8, 47, 2,047 and 2,048 rows (maximum relative L2 error 0.00186). At 180 W the real Pi agent returned a patch on the 64,602-input-token route repair in 196.1 s and passed all ten hidden tests. A separate capacity request processed 262,016 input and 128 output tokens without a prefix cache hit, reaching 1,881 engine prefill tok/s and 46.9 engine decode tok/s. That request qualifies capacity only. [Per-request timings, hardware counters, layer comparisons and quality results](measurements.json)

## Reproduction

Use the [cold Pi prefill probe](../../2026-11-22/prefill/probe.py) and its [frozen source corpus](../../2026-11-22/quality/replay/README.md) with character lengths 110000, 226500 and 453000. Start the optimized vision-enabled DSpark launcher. Change only the discrete B70 card's Xe hwmon `power1_cap` between measured passes, read back the requested limit, and restore the initial value after the experiment. The card-level energy readings use `energy1_input` with label `card`. The measurement window excludes the first and final 250 ms of client TTFT. Approximate GPU joules per input token use average card watts divided by engine prefill throughput.

Build both libraries as shown in the repository README. The optimized launcher selects the adaptive tile when the new library is present. To repeat the 32-row control set `INKLING2_EXL3_MOE_LIB=/lab/.work/build/inkling2-exl3-moe-mb32.so`. The launcher does not set a power limit.
