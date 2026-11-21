# Setup and source builds

[Back to README](../README.md)

Use a Linux x86-64 host with an Intel Arc Pro B70, a working DRM render node,
Docker, Bash, GNU coreutils, Git, curl, Python 3 and the Hugging Face `hf`
CLI. Both serving stacks also use `patch` and `tar`. Allow space for Docker
images, compiler outputs and about 16 GiB per checkpoint. Give the launchers
exclusive access to the GPU during model tests.

## Container images and device

The default image references identify the two Docker images used on the
reference host. These IDs are local Docker config digests.

| Purpose | Tested image ID |
| --- | --- |
| oneAPI 2026.1.1 / PyTorch 2.13 XPU builder | `sha256:ae6950731b3c031f812a95c0eb615a572239426440bf67fd1b3c9c9cbfce9eca` |
| Qwen direct-XPU vLLM 0.33 / PyTorch 2.15 XPU runtime | `sha256:131dd9e11809de2f3136547618f130ee238f0b90cd5718567bcdd0dad5360e19` |

With those images already available, use the defaults. For different local
tags, export `INKLING2_RUNTIME` and `INKLING2_BUILDER`. The runtime needs the
Qwen native library at `/opt/exl3xpu/exl3xpu/_C.so`, which provides oneDNN
SDPA. The builder needs the Intel SYCL compiler and XPU PyTorch headers.
The llama.cpp build additionally uses the SYCL, oneDNN and oneMKL libraries
from these images.

For a fresh runtime, follow the pinned
[Qwen direct-XPU build](https://github.com/Danmoreng/intel-b70-qwen38-vllm/blob/0356a370da5f73f6b0a4eab613d7103a2356b513/engine/direct-xpu/README.md).
Its compiler Dockerfile can prepare a local Inkling builder from the rebuilt
runtime. From the Inkling checkout:

```bash
mkdir -p .work
git clone https://github.com/Danmoreng/intel-b70-qwen38-vllm.git .work/qwen-build
git -C .work/qwen-build checkout --detach 0356a370da5f73f6b0a4eab613d7103a2356b513
(
  cd .work/qwen-build
  python3 scripts/build-engine.py \
    --tag local/b70-qwen38-vllm:inkling-runtime \
    --output benchmark-results/inkling-runtime
)
export INKLING2_RUNTIME=local/b70-qwen38-vllm:inkling-runtime
docker build --build-arg "BASE_IMAGE=$INKLING2_RUNTIME" \
  --tag local/inkling2-oneapi:2026.1.1 \
  -f .work/qwen-build/benchmarks/experiments/m17-oneapi-recompile/Dockerfile.compiler \
  .work/qwen-build/benchmarks/experiments/m17-oneapi-recompile
export INKLING2_BUILDER=local/inkling2-oneapi:2026.1.1
```

The source-build route above uses the rebuilt runtime's PyTorch headers.
Run the numerical checks on the rebuilt images before comparing throughput.
The published measurements use the two exact reference images listed above.

Select your render node when it differs from the reference PCI address
(`0000:03:00.0`):

```bash
ls -l /dev/dri/by-path/
export INKLING2_RENDER_NODE=/dev/dri/renderD128
```

The scripts use the reference host's Europe/Berlin completion window. The
llama.cpp launcher also checks the selected device's 180 W power cap.

## EXL3 checkpoint for vLLM

```bash
hf download grant-ai/Inkling-2-Mini-EXL3-4bpw \
  --revision c266f76b68429255e1d6ec151d0aa15f48f6abcc \
  --local-dir models/exl3
```

Continue with the [build and serve commands](../README.md#build-and-serve).
The patch script fetches Trellis Serve at
`8f8e2fccf0c82c9715b774812f30d9d52f78a5fb`, applies the matching patch and
compiles the operators into `.work/build/`. The model adapter streams the
expert checkpoint shards into their final packed GPU arenas. DSpark reads
`mtp.safetensors` from the same EXL3 checkout. The launcher rejects a
missing drafter before starting vLLM. Build the coding kernels and check the grouped projections and local cache writes on your card.

```bash
bash scripts/build-loop-control.sh
bash scripts/check-coding-kernels.sh
```

Decode groups Q, K and V into one Intel XMX projection. It splits each activation into two signed INT8 terms against the existing group-32 weights. A second SYCL kernel computes the short convolutions, clips the attention heads and writes the local KV ring. The prefill path retains its original weight representation.

The API listens on loopback port 8093. For text with tools, use
`/v1/chat/completions`, the model name `inkling2-mini-exl3-b70`, and standard
OpenAI tool definitions. Set the `reasoning_effort` field explicitly when
comparing coding workflows. Prefix reuse tracks the most recent sequence
and is scoped to vLLM's block hashes and caller-provided cache salt.

## llama.cpp and DSpark

```bash
hf download unsloth/Inkling-2-Mini-GGUF \
  Inkling-2-Mini-UD-Q4_K_M.gguf dspark-Inkling-2-Mini-Q8_0.gguf \
  --revision 8b882751d7f2e7e4fff4f49c886ecc1f2eaf1af7 \
  --local-dir models/unsloth

sha256sum -c <<'CHECKSUMS'
f90c5a4df44d335ac9f9eea2a15c31d6c5a377b1b3cd9ff7fc2cf123f3cdf7e1  models/unsloth/Inkling-2-Mini-UD-Q4_K_M.gguf
2ec78510da837e8fc3be54a14757a0c5fbd2d8d9e85a0d92853fafdf674f3773  models/unsloth/dspark-Inkling-2-Mini-Q8_0.gguf
CHECKSUMS

INKLING2_BUILD_DIR="$PWD/.work/llama-unsloth/mask-skip" \
  INKLING2_BUILD_DSPARK_DEVICE=1 INKLING2_BUILD_BANDED_DECODE=1 \
  INKLING2_BUILD_NVFP4_PREFILL=1 INKLING2_BUILD_SWA_PREFILL=1 INKLING2_BUILD_MASK_SKIP=1 \
  bash scripts/build-sycl-baseline.sh

INKLING2_SERVER=/lab/.work/llama-unsloth/mask-skip/build/bin/llama-server \
  INKLING2_KV_TYPE=q8_0 INKLING2_BATCH=2048 INKLING2_UBATCH=1024 \
  INKLING2_SWA_BIAS=1 INKLING2_MASK_SKIP=1 INKLING2_MMID_SCHED=3 \
  INKLING2_DSPARK_DEVICE=1 INKLING2_BANDED_DECODE=1 \
  bash scripts/serve-sycl-baseline.sh dspark 8192 --no-warmup --ctx-checkpoints 0
```

Run the benchmark from another terminal after `/health` responds:

```bash
python3 scripts/bench-sycl-baseline.py --lengths 4096 --replicates 2 \
  --timeout 120 --output results/dspark-4k.jsonl
```

Use `ordinary` as the launcher mode for the non-speculative control.
