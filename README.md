# Inkling 2 Mini on Intel Arc Pro B70

Serve [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
on a single **32 GB Intel Arc Pro B70** using vLLM 0.33, 4-bit EXL3
experts and native SYCL kernels. The vLLM endpoint supports ordinary decoding,
the checkpoint's DSpark drafter, streamed tool calls and single-sequence
prefix reuse. A second recipe runs the Unsloth 4-bit GGUF in llama.cpp.

## Validated profile

| Setting | Reference configuration |
| --- | --- |
| GPU | 1 × Intel Arc Pro B70, 32 GB, 180 W |
| Checkpoint | [Inkling 2 Mini EXL3 4bpw](https://huggingface.co/grant-ai/Inkling-2-Mini-EXL3-4bpw/tree/c266f76b68429255e1d6ec151d0aa15f48f6abcc), MCG expert weights, BF16 dense weights |
| Runtime | vLLM 0.33.0, PyTorch 2.15.0+xpu, Level Zero |
| Context | 262,144 input and output tokens combined |
| Prefill | Grouped XMX expert GEMM, 2,048-token chunks, oneDNN prefix SDPA |
| KV cache | FP8 E4M3, 4 GiB + 16 MiB |
| Serving | Text and tools, one sequence, TP1, native XPU decode graphs, validated prefix reuse |
| API | `http://127.0.0.1:8093/v1`, model `inkling2-mini-exl3-b70` |

## Coding and decode measurements

The current full-context DSpark launcher delivered **82.6, 64.9 and 56.7 client output tokens/s** on the three frozen source-review lengths (about 5K, 77K and 155K Inkling input tokens). Each value is a median of three seeded 1,024-token continuations. The archived Qwen medians were 60.6, 53.3 and 44.6. The native 4K logbook timing was **106.7–114.8 decode tok/s**. At 196,608 input tokens the native rates were **1,381 prefill / 57.3 decode**. At 262,016 input plus 128 output tokens they were **1,285 / 40.6**. Both initial long prompts retrieved the code. A 196,608-token cache hit answered the follow-up correctly and took 0.88 s to first output. A cold control on that follow-up repeated the question and missed the code.

DSpark completed the four original coding fixtures/seeds in **113.7 seconds** with requested effort `none` and **194.2 seconds** with `medium`. All four passed in both settings. Archived Qwen passed all four in **145.9 seconds** at medium effort. The held-out routing and streaming-codec tasks passed once in four runs on Inkling with effort `none`. [Results, settings and raw data](benchmarks/2026-11-21/agent-decode/README.md)

### Previous public checkpoint

In a single-agent replay of three short source reviews (about 5,000 Inkling prompt tokens and 1,024 generated tokens each), the B70 delivered **62.9 output tokens/s** at the streaming client using probabilistic DSpark proposals and block verification. The first public DSpark checkout repeated these three reviews at **63.7 tok/s**. The archived Qwen median for the same source text was 60.6. An earlier greedy DSpark profile reached 53.1 tok/s on these reviews. In a separate warm, greedy-output 4K kernel screen, native depth mixing and grouped controllers reached **59.9 tok/s ordinary** and **96.3–101.7 tok/s with DSpark**.

The earlier long-context profile measured **1,389 prefill / 45.1 decode tok/s** with DSpark at 196,608 input tokens and **1,288 / 37.7** at 262,016. Ordinary decode was **38.4** and **31.1 tok/s**, respectively. All four prompts retrieved the inserted code. The coding fixtures also check task completion with hidden tests. [Measurements, settings and reproduction](benchmarks/2026-11-21/agent-decode/README.md)

That earlier full-context checkout with tool support and prefix reuse reached **57.7–57.9 tok/s ordinary** and **93.5–99.0 tok/s with DSpark** on the two warm 4K kernel-screen prompts. [Full-profile verification and cache results](benchmarks/2026-11-21/agent-decode/checkpoint-smoke.json)

## Original prefill measurements

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

bash scripts/build-loop-control.sh
bash scripts/serve-coding-agent.sh dspark
```

In a second terminal, check readiness and run a short benchmark:

```bash
curl -fsS http://127.0.0.1:8093/health
python3 scripts/bench-sycl-baseline.py --url http://127.0.0.1:8093 \
  --vllm-model inkling2-mini-exl3-b70 --lengths 4096 --replicates 2 \
  --timeout 90 --output results/vllm-4k.jsonl
```

Use `bash scripts/serve-coding-agent.sh ordinary` to run the same endpoint
with ordinary decoding. Both modes support the OpenAI-compatible chat and
completion endpoints, including automatic tool calls on chat requests.
The 262,144-token limit includes the generated continuation.

For the earlier GGUF comparison, the llama.cpp recipe measured 33.3 tok/s
with DSpark and 26.5 tok/s with ordinary decode at 4K.
[llama.cpp build and launch commands](docs/SETUP.md#llamacpp-and-dspark)

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
