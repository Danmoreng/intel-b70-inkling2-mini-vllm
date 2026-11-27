# IQR 4-bit experts (opt-in)

[Back to README](../README.md)

`INKLING2_EXPERTS=iqr4` replaces the EXL3 routed experts with a Hadamard-rotated GPTQ 4-bit format read by its own
SYCL library, `inkling_xpu/csrc/iqr_moe.sycl`. EXL3 stays the default. Everything else (attention, DSpark, prefill
profiles, batching) is unchanged.

## Format

Per layer, `iqr-layer-XX.safetensors` (metadata `format: iqr-gptq4-v1`), prefix `model.llm.layers.L.mlp.experts.`:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `w13_codes` | uint8 [128, 2304, 1152] | two 4-bit codes per byte along K, low nibble = even k, value = code - 8; row 2j = gate j, 2j+1 = up j |
| `w13_scales` | bf16 [128, 2304, 18] | one scale per (row, 128 of K) |
| `w2_codes` | uint8 [128, 2304, 576] | down projection, same encoding |
| `w2_scales` | bf16 [128, 2304, 9] | |

`W[r, :] = (scale[r, k // 128] * value[r, k]) R`, with `R` the block-diagonal normalised Walsh-Hadamard transform of
width 128 over the input dimension. 4.125 bits per weight including scales (EXL3 4bpw: 4.0). The model directory
holds these shards next to the EXL3 checkout's dense shards, `mtp.safetensors`, tokenizer and configs, and an
`iqr_config.json` describing the format.

The kernel rotates and quantises each activation row to int8 (one scale per 128), runs int8 x int4 DPAS and folds
the scales per 128-wide group. Decode (M <= 8 rows) groups the (row, expert) pairs inside the kernel and reads each
touched expert once. M > 8 (prefill, and batched decode across slots) uses tiled int8-DPAS kernels with 32-row
blocks. Both paths are deterministic (fixed-order fp32 combine, no atomics) and safe under graph replay. Expert ids
outside [0, 128) or more than eight pairs on one expert in a decode call write NaN rows instead of reading out of
bounds; `INKLING2_IQR_CHECK_IDS=1` adds a host check that raises (eager only). The loader checks the shard metadata,
tensor shapes and the model constants the kernel hardcodes (hidden 2304, intermediate 1152, top-8, softcaps 4 / 25).

The format follows the IQR idea from [Radiance](https://codeberg.org/StillDeadcode/radiance) (Apache-2.0). The code
here is an independent implementation.

## Build, check and serve

```bash
hf download davetha/Inkling-2-Mini-Uncensored-IQR4-B70 --local-dir models/iqr4-unc
bash scripts/build-iqr-moe.sh
docker run --rm --network none --device "$(readlink -f "$INKLING2_RENDER_NODE")" \
  --mount "type=bind,source=$PWD,target=/lab,readonly" \
  --env PYTHONPATH=/lab:/lab/.work/python:/lab/.work/trellis-serve/core/src \
  --env EXL3_MOE_LIB=/lab/.work/build/inkling2-exl3-moe-mb32.so \
  --entrypoint /opt/venv/bin/python "${INKLING2_RUNTIME:-sha256:131dd9e11809de2f3136547618f130ee238f0b90cd5718567bcdd0dad5360e19}" \
  /lab/tests/check_iqr_moe.py check --model /lab/models/iqr4-unc --layer 5
INKLING2_EXPERTS=iqr4 INKLING2_MODEL_DIR=models/iqr4-unc bash scripts/serve-coding-agent.sh dspark
```

`check` compares the kernel with an fp64 reference of the same arithmetic for M = 1 to 2048 (including 24 and 32,
the batched four-slot verify sizes), decode rows against the same rows in a prefill call, graph replay against
eager, forced routing, invalid ids and repeatability; it prints `CHECK ALL PASS`. `bench` times IQR against the EXL3
library on random arenas. `INKLING2_IQR_MOE_LIB` overrides the library path.

## Rebuilding the checkpoint

The published checkpoint is the normal input. `tools/iqr` holds the quantizer that produced it. It runs the
plain-torch reference forward of the Trellis checkout on the native NVFP4/MXFP8 checkpoint, so it needs the B70
(about 100 GB host RAM for calibration):

```bash
export IQR_MODEL=/abs/models/nvfp4-unc        # OpenUncensored/Inkling-2-Mini-Uncensored (or the original)
export IQR_DENSE_SRC=/abs/models/exl3-unc     # EXL3 checkout with the matching dense files
export IQR_DIR=/abs/iqr-work                  # needs corpus/calib.npy: int token ids [N, 2048]
tools/iqr/run.sh gpu 16 100g calib.py 491           # expert inputs of every layer visit (about 50 min)
tools/iqr/run.sh gpu 16 100g hess.py all xpu        # routing-weighted Hessians (about 20 min)
tools/iqr/run.sh gpu 16 100g gptq.py all gptq4 xpu 64   # GPTQ, all 24 layers x 128 experts (about 20 min)
tools/iqr/run.sh cpu 16 64g write_ckpt.py write gptq4 "$IQR_DIR/ckpt"
tools/iqr/run.sh cpu 16 64g write_ckpt.py validate gptq4 "$IQR_DIR/ckpt"
tools/iqr/assemble.sh "$IQR_DIR/ckpt" models/iqr4-unc
```

The published build used 491 x 2,048 tokens of chat, code, tool calls and multilingual text. The uncensored dense
files come from `davetha/Inkling-2-Mini-Uncensored-EXL3-B70`. For the original model, use
`thinkingmachines/Inkling-2-Mini` and `models/exl3`.

## Quality

Teacher-forced full-vocabulary KL against the native NVFP4/MXFP8 experts, 24,564 positions of chat, code and docs:

| Experts | Mean KL | Top-1 | PPL (native 6.2693) |
| --- | ---: | ---: | ---: |
| EXL3 4bpw | 0.0406 | 95.28% | 6.2469 |
| IQR GPTQ 4-bit | 0.0407 | 95.09% | 6.2526 |
| IQR GPTQ 4-bit + int8 activations (kernel arithmetic) | 0.0408 | 95.16% | |
