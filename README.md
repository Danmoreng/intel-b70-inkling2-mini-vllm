# Inkling 2 Mini on Intel Arc Pro B70

Serve [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
on a single **32 GB Intel Arc Pro B70** using vLLM 0.33, 4-bit EXL3
experts and native SYCL kernels. A separate llama.cpp recipe runs the
Unsloth 4-bit GGUF with the DSpark drafter.

## Validated profile

| Setting | Reference configuration |
| --- | --- |
| GPU | 1 × Intel Arc Pro B70, 32 GB, 180 W |
| Checkpoint | [Inkling 2 Mini EXL3 4bpw](https://huggingface.co/grant-ai/Inkling-2-Mini-EXL3-4bpw/tree/c266f76b68429255e1d6ec151d0aa15f48f6abcc), MCG expert weights, BF16 dense weights |
| Runtime | vLLM 0.33.0, PyTorch 2.15.0+xpu, Level Zero |
| Context | 262,144 input and output tokens combined |
| Prefill | Grouped XMX expert GEMM, 2,048-token chunks, oneDNN prefix SDPA |
| KV cache | FP8 E4M3, 4 GiB + 16 MiB |
| Serving | Text, one sequence, TP1, eager, prefix caching off |
| API | `http://127.0.0.1:8093/v1`, model `inkling2-mini-exl3-b70` |

## Measured performance

Measured on November 21, 2026 with one request at a time and 128 completion
tokens. The A/B rows use the same tokenized prompts. The 4K row is the second
control request. The 256K row fills the context with 262,016 input tokens
and 128 output tokens.

| Context | ESIMD prefill tok/s | Hybrid oneDNN prefill tok/s | Hybrid decode tok/s |
| --- | ---: | ---: | ---: |
| 4K | 1,847 | 1,866 | 20.1 |
| 32K | 1,652–1,661 | 1,718–1,755 | 20.0 |
| 192K | 953 | **1,393** | 19.3 |
| 256K | — | **1,290** | 19.2 |

At 192K, hybrid attention reduced prefill time from 206.33 s to 141.15 s
(+46.2% throughput). It was inactive at 4K because that prompt had no eligible
8,192-token prefix block. The 192K and 256K requests retrieved a validation
code inserted in the middle of a synthetic prompt. [Method and raw measurements](benchmarks/2026-11-21/README.md)

## Build and serve

Start with the [setup instructions](docs/SETUP.md) for container images, device
selection and checkpoint downloads. The exact reference image IDs correspond
to the local artifacts from the
[Qwen B70 recipe](https://github.com/Danmoreng/intel-b70-qwen38-vllm/tree/0356a370da5f73f6b0a4eab613d7103a2356b513).
The setup also describes a source-build option for a new machine.

From the repository root, after preparing the images and checkpoint:

```bash
bash scripts/check-sycl.sh
bash scripts/check-sycl.sh exl3
INKLING2_EXL3_TILE_ROWS=32 bash scripts/check-sycl.sh exl3 --tokens 256 512

INKLING2_VLLM_BATCH=2048 INKLING2_EXL3_PREFILL_MIN=256 INKLING2_VLLM_KV_TYPE=fp8_e4m3 \
  INKLING2_ONEDNN_PREFIX_BLOCK=8192 \
  INKLING2_EXL3_MOE_LIB=/lab/.work/build/inkling2-exl3-moe-mb32.so \
  bash scripts/serve-vllm-exl3.sh 262144 --kv-cache-memory-bytes 4311744512
```

In a second terminal, check readiness and run a short benchmark:

```bash
curl -fsS http://127.0.0.1:8093/health
python3 scripts/bench-sycl-baseline.py --url http://127.0.0.1:8093 \
  --vllm-model inkling2-mini-exl3-b70 --lengths 4096 --replicates 2 \
  --timeout 90 --output results/vllm-4k.jsonl
```

For DSpark, use the [llama.cpp build and launch commands](docs/SETUP.md#llamacpp-and-dspark).
On the same B70 at 4K, the GGUF recipe measured **33.3 tok/s decode with
DSpark** and 26.5 tok/s with ordinary decode. The vLLM numbers above use
ordinary decode.

## How the kernels work

- The [EXL3 patch](patches/trellis-inkling-exl3-mcg-situ.patch) adds the MCG
  codebook and Inkling's SiTU activation to Trellis. The grouped prefill
  kernel decodes 4-bit expert weights into XMX tiles. The tuned recipe uses
  32 activation rows per tile.
- The [paged attention patch](patches/trellis-vllm-block-table.patch) lets
  the SYCL decode kernel read vLLM's physical KV pages in BF16 or FP8 E4M3.
  The vLLM adapter keeps the per-visit sliding-window and convolution state
  in device rings. Some supporting operations use XPU Triton.
- The [hybrid prefill](inkling_xpu/prefix_attention.py) reuses the
  [Qwen oneDNN operator](https://github.com/Danmoreng/intel-b70-qwen38-vllm/blob/0356a370da5f73f6b0a4eab613d7103a2356b513/engine/direct-xpu/runtime/csrc/exl3_ops.sycl)
  for older keys that are visible to every query and outside Inkling's
  relative-bias band. The ESIMD kernel handles the causal tail, relative
  bias and attention sink. The adapter merges the outputs using FP32
  softmax normalizers. Set `INKLING2_ONEDNN_PREFIX_BLOCK=0` for the full ESIMD
  control. The optional W8A8 experiment targets dense projections and gave
  comparable throughput to BF16 at 16K.

Kernel sources are pinned to [Trellis Serve](https://github.com/sybil-solutions/trellis-serve/tree/8f8e2fccf0c82c9715b774812f30d9d52f78a5fb)
and [Unsloth llama.cpp](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8).
Model files and built images are obtained separately. [Third-party notices](NOTICE.md)
