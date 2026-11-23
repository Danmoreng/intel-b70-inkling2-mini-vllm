# Cold 32K prefill at 180 W

This study targets one 32 GB Arc Pro B70 at its 180 W card limit. All serving arms use the same EXL3 checkpoint, native Intel XPU kernels, FP8 global KV, loaded vision encoder, adaptive DSpark with up to seven proposals and a 262,144-token context. The real Pi Coding Agent uses medium reasoning and an 8,192-token thinking budget. Each 32K timing uses the same 32,849-token source excerpt with a fresh cache salt. A shape warmup precedes three measured requests per engine start.

The original optimized recipe at commit `0a93121` spent 6.27 s in the expert calls out of 12.26 s in a synchronized forward pass. Local attention took 1.07 s plus 0.68 s for the relative bias. A 3,000 engine tok/s result requires a prefill timer of at most 10.950 s on this prompt. The client threshold also includes scheduling and the first streamed output.

## Results

On the rebuilt release at 180 W, engine prefill at 32,849 tokens measured **3,114, 3,083, 3,078 tok/s**. The median was **3,083 tok/s**, a **14.7%** gain over the previous optimized profile on the same prompt. The median client throughput including time to first output was **3,056 tok/s**. A separate build of the winning combination measured 3,101, 3,087, 3,082 engine tok/s.

| Configuration | Input tokens | Timed requests | Engine prefill (tok/s) | Client to first output (tok/s) | First output (s) |
|---|---:|---:|---:|---:|---:|
| Previous optimized recipe, 48-row expert tile | 32,849 | 3 | 2,688 | 2,671 | 12.297 |
| 64-row tile with SLM reuse | 32,849 | 3 | 2,771 | 2,754 | 11.927 |
| 64-row tile with per-assignment Hadamard staging | 32,849 | 3 | 2,713 | 2,697 | 12.182 |
| 48-row tile with fused local bias | 32,849 | 3 | 2,839 | 2,821 | 11.645 |
| 64-row tile with fused local bias | 32,849 | 3 | 2,948 | 2,929 | 11.217 |
| Add checkpoint-shared Gate/Up input transforms | 32,849 | 3 | 3,087 | 3,065 | 10.716 |
| Final release build | 32,849 | 3 | 3,083 | 3,056 | 10.749 |
| Final release build | 65,324 | 2 | 2,894 | 2,878 | 22.698 |

At the same 180 W limit, testing scheduler chunks of 6,144, 7,680 and 8,192 on the 64-row fused-bias implementation gave 32K medians within 0.1%. The qualified recipe uses 6,144. These are warm-engine, cold-prefix measurements. Each timed Pi request reported zero cached tokens. A fresh engine start includes one additional untimed request at each shape. The release build's first 32K shape warmup ran at 2,938 engine tok/s.

A blind routing repair through Pi started with 64,602 tokens and completed in 238.8 s. The independently applied patch passed all 10 hidden tests. On the same task and seed, the [earlier DSpark verification study](../attention/README.md) recorded 238.3 s. This task checks functional behavior and shows similar total time across those two runs. A separate capacity request filled the entire 262,144-token window with 262,016 input tokens and 128 generated tokens. At that boundary the engine measured 2,090 input tok/s and 43.9 output tok/s. The capacity request and the Pi repair both ran with the new public libraries. [Raw trials, kernel timings, power cap, binary hashes and checks](measurements.json)

## Expert tiles and shared local memory

Each Gate/Up workgroup loads transformed inputs into shared local memory (SLM). After the last matrix product, the workgroup reaches a barrier and uses SLM to apply the gated activation. The new kernel reuses the same bytes during these two phases. Reuse at the previous 48 rows had little effect. With 64 rows, each reconstruction of the four-bit weights serves more tokens while the workgroup uses 64 KiB of SLM. The dispatch changes only for four-bit Inkling experts with at least 2,048 input rows. The verifier and single-token expert kernels retain their original geometry.

Tests used all 128 experts from checkpoint layer zero, BF16 activations, eight experts per token and both uniform and skewed routing. We checked outputs against the old library on identical tensors, warmed the device and timed seven batches of five calls per shape. The larger shapes correspond to the five 6,144-token chunks and the 2,129-token tail of the 32K prompt. The 80-, 96- and 128-row variants lost throughput at 180 W.

## Hadamard transformations

A separate SYCL kernel prepared the Gate/Up Hadamard inputs once per token–expert assignment for all three output partitions. It wrote the operands in the layout the matrix instructions consume. The additional scratch write and subsequent reads outweighed the avoided recomputation. At 6,144 skewed input rows the 64-row precomputation took 23.63 ms versus 22.81 ms for the direct 64-row path. Both reproduced the reference BF16 outputs exactly. A single large Gate/Up workgroup with the original per-thread geometry exceeded the device's 64-work-item kernel limit. Alternate one- and two-partition geometries did not finish the operator check within the trial window.

In the released EXL3 checkpoint, all 128 experts share one Gate input-scale vector and one Up input-scale vector within each of the 24 layers. The loader compares the packed FP16 scales for every expert. When they match, the native kernel scales and transforms each input token once for Gate and once for Up. All eight selected experts and the output partitions read those two results. At 6,144 rows the temporary uses 54 MiB, compared with 432 MiB for per-assignment staging. With skewed routing the two independent shared-input trials took 20.97 and 21.03 ms. The direct 64-row kernel took 22.89 and 22.74 ms. Both routing distributions and both prefill shapes matched the reference BF16 outputs bit for bit. The release operator also matched across FP16 and BF16 at 2,048, 2,129 and 6,144 rows. Layers with different scales take the direct path. Set `INKLING2_EXL3_SHARED_INPUT=0` to select the direct path on this checkpoint.

## Local relative bias

The previous kernel wrote a BF16 tensor of shape `[6144, 24, 1024]` before sliding-window attention. The new SYCL kernel computes bias for the current group of query rows directly inside attention. It uses XMX matrix products and maintains the BF16 rounding of the bias before the softmax. It packs the 16-by-1,024 checkpoint projection once per local layer, reverses the distance axis and pads both ends for masked lanes. A second FP16 plane carries the projection residual when needed. The query features retain the high and low FP16 parts. The ordinary global attention, decode and DSpark verification paths use their previous routines.

Testing a ring of overlapping distance tiles and query workgroup widths of 1, 2, 4 and 8 favored the direct kernel at width four. At 6,144 query rows with 1,023 preceding keys, the previous table and attention took 7.91 ms together. Direct projection took 4.32 ms, including K/V conversion. On the 2,129-row tail the times were 2.27 and 1.53 ms. These checks used the real layer-zero projection and compared against the previous attention output on the same tensors, including ragged boundaries. The operator allows a nonzero projection residual, which is covered by a separate regression test.

## Reproduce

Build the new libraries in the existing checkout and start the vision-enabled optimized profile on the B70.

```bash
INKLING2_EXL3_TILE_ROWS=32-64 bash scripts/check-sycl.sh exl3 --small-dpas --tokens 4 7 8 47 2047 2048 2129 6144
bash scripts/check-sycl.sh paged-fused
INKLING2_PREFILL_PROFILE=optimized INKLING2_VISION=1 INKLING2_VISION_FUSED_QKV=1 bash scripts/serve-coding-agent.sh dspark
```

The release launcher selects the new libraries when available. For the previous expert kernel set `INKLING2_EXL3_MOE_LIB=/lab/.work/build/inkling2-exl3-moe-mb32-48.so`. For the previous bias path set `INKLING2_PREFILL_LOCAL_BIAS=table`. Use `INKLING2_VLLM_BATCH=6144` to reproduce the isolated kernel comparisons. The power limit is set outside the launcher. Use the [Pi cold prompt probe](../../2026-11-22/prefill/probe.py) with 110000 characters of the [frozen corpus](../../2026-11-22/quality/replay/README.md) at 32K and 226500 characters at 64K.

To repeat the cold timing runs, supply the same frozen corpus and a fresh output directory. The report retains all warmup observations separately (repeat zero). The timed subset used three requests at 32K and two at 64K.

```bash
python3 benchmarks/2026-11-22/prefill/probe.py "$corpus" "$output" \
  --url http://127.0.0.1:8093 --model inkling2-mini-exl3-b70 \
  --chars 110000 226500 --warm-shapes --repeats 3
```
