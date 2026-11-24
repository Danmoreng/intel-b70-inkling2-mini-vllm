#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
context=${1:?context length}
shift
port=${INKLING2_VLLM_PORT:-8093}
batch=${INKLING2_VLLM_BATCH:-128}
slots=${INKLING2_MAX_SEQS:-1}
case "$slots" in [1-4]) ;; *) echo "INKLING2_MAX_SEQS must be 1 through 4" >&2; exit 2 ;; esac
kv_type=${INKLING2_VLLM_KV_TYPE:-bfloat16}
runtime=${INKLING2_RUNTIME:-sha256:131dd9e11809de2f3136547618f130ee238f0b90cd5718567bcdd0dad5360e19}
render_node=$(readlink -f -- "${INKLING2_RENDER_NODE:-/dev/dri/by-path/pci-0000:03:00.0-render}")
vendor="$PWD/.work/trellis-serve"
work="$PWD/.work/vllm-inkling"
name=inkling2-vllm-exl3

deadline=$(TZ=Europe/Berlin date -d 'today 22:44' +%s)
seconds=$(( deadline - $(date +%s) ))
(( seconds > 120 ))
test -c "$render_node"
for lib in "${INKLING2_LIB:-/lab/.work/build/inkling2-vllm.so}" "${INKLING2_EXL3_MOE_LIB:-/lab/.work/build/inkling2-exl3-moe.so}"; do
    if [[ $lib == /lab/* && ! -s ${lib#/lab/} ]]; then
        echo "Missing build artifact: ${lib#/lab/}" >&2
        exit 1
    fi
done
if [[ ${INKLING2_VLLM_DSPARK:-0} == 1 ]]; then
    test -s models/exl3/mtp.safetensors || { echo "Missing models/exl3/mtp.safetensors" >&2; exit 1; }
fi
mkdir -p "$work"
exec 8>"$work/serve.lock"
flock -n 8 || { echo "An Inkling server already owns this checkout." >&2; exit 1; }
if systemctl --user is-active --quiet b70-qwen38-vllm.service || \
    [[ $(docker inspect --format '{{.State.Running}}' inkling2-sycl-baseline 2>/dev/null || true) == true ]]; then
    echo 'The B70 is occupied by another text model.' >&2
    exit 1
fi

check_vendor_patch() {
    local source_file=$1
    local patch_file="$PWD/patches/$2"
    shift 2
    local installed_sha
    installed_sha=$(sha256sum < "$vendor/$source_file" | cut -d' ' -f1)
    if git -C "$vendor" diff --quiet -- "$source_file"; then
        git -C "$vendor" apply "$patch_file"
    elif [[ " $* " == *" $installed_sha "* ]]; then
        # Replace only the exact previous released adapter. Preserve local edits.
        local staged
        staged=$(mktemp "$vendor/$source_file.update.XXXXXX")
        if git -C "$vendor" show "HEAD:$source_file" > "$staged" &&
            patch --silent --batch --forward --fuzz=0 "$staged" "$patch_file" &&
            chmod --reference="$vendor/$source_file" "$staged"; then
            mv -- "$staged" "$vendor/$source_file"
        else
            rm -f -- "$staged"
            return 1
        fi
    fi
    git -C "$vendor" diff -- "$source_file" | cmp - "$patch_file"
}
check_vendor_patch xpu/exl3xpu/inkling2/sgl_model.py trellis-inkling-external-runner.patch \
    9227ddc921e5dad0c598fb712d029534fb3ca6732cf0a98f38a38bcc8e5e2896 9188158ae9c55e91d9d17440fd5969bc5f6dade764cd5a4a1f0036426c3a5588 e30e35e03f57f17cffcc585f14ccab74839952db47c8effaea4d86d5fb6bfa70 \
    247ae4654c103fc199ca725c0c71b5ae91142625a03029fb30da30ed603d2f80
check_vendor_patch xpu/exl3xpu/inkling2/dense.py trellis-inkling-dense-w8a8-prefill.patch
check_vendor_patch xpu/exl3xpu/inkling2/vendor/inkling2_vision.py trellis-inkling-vision-qkv.patch

mkdir -p "$work/config" "$work/plugin/inkling2_b70-0.1.dist-info" .work/python
ln -sfn ../trellis-serve/xpu/exl3xpu .work/python/inkling_trellis
cat > "$work/plugin/inkling2_b70-0.1.dist-info/METADATA" <<'EOF'
Metadata-Version: 2.1
Name: inkling2-b70
Version: 0.1
EOF
cat > "$work/plugin/inkling2_b70-0.1.dist-info/entry_points.txt" <<'EOF'
[vllm.general_plugins]
inkling2_b70 = inkling_xpu.vllm_plugin:register
EOF
python3 - "$work/config/config.json" "${INKLING2_VISION:-0}" <<'PY'
import json
from pathlib import Path
import sys

checkpoint = json.loads(Path('models/exl3/config.json').read_text())
text = checkpoint['text_config']
config = {key: text[key] for key in ('hidden_size', 'num_attention_heads', 'num_key_value_heads',
                                     'head_dim', 'intermediate_size', 'rms_norm_eps')}
config.update(model_type='llama', architectures=['Inkling2B70ForCausalLM'], dtype='bfloat16',
              vocab_size=text['unpadded_vocab_size'], num_hidden_layers=8,
              max_position_embeddings=text['model_max_length'], eos_token_id=checkpoint['eos_token_id'],
              tie_word_embeddings=False)
if sys.argv[2] == '1':
    config['architectures'] = ['Inkling2B70ForConditionalGeneration']
Path(sys.argv[1]).write_text(json.dumps(config, indent=2) + '\n')
PY

cache_mounts=()
if [[ ${INKLING2_KERNEL_CACHE:-0} == 1 ]]; then
    mkdir -p "$work/compiler-cache"
    cache_mounts=(--mount "type=bind,source=$work/compiler-cache,target=/var/cache/inkling-kernels"
                  --env TRITON_CACHE_DIR=/var/cache/inkling-kernels/triton)
fi

spec_mounts=()
spec_args=()
if [[ ${INKLING2_VISION:-0} == 1 ]]; then
    spec_args+=(--limit-mm-per-prompt '{"image":64}')
fi
if [[ ${INKLING2_VLLM_DSPARK:-0} == 1 ]]; then
    base=/opt/venv/lib/python3.12/site-packages/vllm
    mkdir -p "$work/v2"
    files=(config/vllm.py v1/worker/gpu/spec_decode/__init__.py v1/worker/gpu/model_runner.py)
    adaptive=${INKLING2_DSPARK_ADAPTIVE:-0}
    if [[ $adaptive == 1 ]]; then
        if [[ ${INKLING2_RING_PREFIX:-0} != 1 || ${INKLING2_DSPARK_SAMPLING:-greedy} != probabilistic ]]; then
            echo "Adaptive DSpark requires synchronous prefix reuse and probabilistic proposals" >&2
            exit 2
        fi
        files+=(v1/worker/gpu/spec_decode/utils.py)
    fi
    docker run --rm --network none --entrypoint tar "$runtime" -C "$base" -cf - "${files[@]}" | tar -xf - -C "$work/v2"
    if [[ $adaptive == 1 ]]; then
        patch --silent --batch --forward --fuzz=0 -d "$work/v2" -p1 <patches/vllm-inkling-adaptive-draft.patch
        spec_mounts+=(--env INKLING2_DSPARK_ADAPTIVE=1)
    fi
    patch --silent --batch --forward --fuzz=0 -d "$work/v2" -p1 <patches/vllm-v2-inkling-dspark.patch
    for file in "${files[@]}"; do
        spec_mounts+=(--mount "type=bind,source=$work/v2/$file,target=$base/$file,readonly")
    done
    width=${INKLING2_DRAFT_MAX:-7}
    case "$width" in [1-7]) ;; *) exit 2;; esac
    draft_sampling=${INKLING2_DSPARK_SAMPLING:-greedy}
    case "$draft_sampling" in greedy|probabilistic) ;; *) exit 2;; esac
    verification=${INKLING2_DSPARK_VERIFICATION:-standard}
    case "$verification" in standard|block) ;; *) exit 2;; esac
    spec_args+=(--speculative-config "{\"method\":\"custom_class\",\"model\":\"inkling_xpu.dspark.InklingDSparkSpeculator\",\"num_speculative_tokens\":$width,\"draft_sample_method\":\"$draft_sampling\",\"rejection_sample_method\":\"$verification\"}")
fi

if [[ ${INKLING2_AGENT_TOOLS:-0} == 1 ]]; then
    base=/opt/venv/lib/python3.12/site-packages/vllm
    file=renderers/inkling.py
    mkdir -p "$work/text-renderer"
    docker run --rm --network none --entrypoint tar "$runtime" -C "$base" -cf - "$file" | tar -xf - -C "$work/text-renderer"
    patch --silent --batch --forward --fuzz=0 -d "$work/text-renderer" -p1 < patches/vllm-inkling-text-renderer.patch
    spec_mounts+=(--mount "type=bind,source=$work/text-renderer/$file,target=$base/$file,readonly")
    spec_args+=(--tokenizer-mode inkling --enable-auto-tool-choice --tool-call-parser inkling --reasoning-parser inkling
                --enable-prompt-tokens-details --enable-per-request-metrics)
    if [[ ${INKLING2_REASONING_BUDGET:-0} == 1 ]]; then
        budget_library=/lab/.work/build/inkling2-reasoning-budget.so
        test -s "${budget_library#/lab/}" || { echo 'Build the coding kernels with scripts/build-loop-control.sh.' >&2; exit 1; }
        budget_files=(v1/worker/gpu/sample/{sampler,thinking_budget}.py v1/worker/gpu/spec_decode/rejection_sampler.py)
        docker run --rm --network none --entrypoint tar "$runtime" -C "$base" -cf - "${budget_files[@]}" | tar -xf - -C "$work/text-renderer"
        patch --silent --batch --forward --fuzz=0 -d "$work/text-renderer" -p1 < patches/vllm-inkling-cumulative-thinking.patch
        for file in "${budget_files[@]}"; do
            spec_mounts+=(--mount "type=bind,source=$work/text-renderer/$file,target=$base/$file,readonly")
        done
        spec_mounts+=(--env "INKLING2_BUDGET_LIB=$budget_library")
        spec_args+=(--reasoning-config '{"reasoning_start_str":"<|content_thinking|>","reasoning_end_str":"<|end_message|>"}')
    fi
fi

prefix_args=(--no-enable-prefix-caching)
if [[ ${INKLING2_RING_PREFIX:-0} == 1 ]]; then
    base=/opt/venv/lib/python3.12/site-packages/vllm
    file=v1/core/kv_cache_manager.py
    mkdir -p "$work/ring-prefix"
    docker run --rm --network none --entrypoint tar "$runtime" -C "$base" -cf - "$file" | tar -xf - -C "$work/ring-prefix"
    patch --silent --batch --forward --fuzz=0 -d "$work/ring-prefix" -p1 < patches/vllm-single-slot-ring-prefix.patch
    spec_mounts+=(--mount "type=bind,source=$work/ring-prefix/$file,target=$base/$file,readonly")
    if (( slots > 1 )); then
        scheduler_files=(v1/core/sched/scheduler.py v1/core/sched/output.py)
        docker run --rm --network none --entrypoint tar "$runtime" -C "$base" -cf - "${scheduler_files[@]}" | tar -xf - -C "$work/ring-prefix"
        patch --silent --batch --forward --fuzz=0 -d "$work/ring-prefix" -p1 < patches/vllm-inkling-batch-prefix.patch
        for file in "${scheduler_files[@]}"; do
            spec_mounts+=(--mount "type=bind,source=$work/ring-prefix/$file,target=$base/$file,readonly")
        done
    fi
    prefix_args=(--enable-prefix-caching --no-async-scheduling)
fi

if (( slots > 1 )); then
    base=/opt/venv/lib/python3.12/site-packages/vllm
    runner=v1/worker/gpu/model_runner.py
    mkdir -p "$work/v2"
    if [[ ${INKLING2_VLLM_DSPARK:-0} != 1 ]]; then
        docker run --rm --network none --entrypoint tar "$runtime" -C "$base" -cf - "$runner" | tar -xf - -C "$work/v2"
        spec_mounts+=(--mount "type=bind,source=$work/v2/$runner,target=$base/$runner,readonly")
    fi
    patch --silent --batch --forward --fuzz=0 -d "$work/v2" -p1 < patches/vllm-inkling-batch-runner.patch
    if [[ ${INKLING2_RING_PREFIX:-0} != 1 ]]; then prefix_args+=(--no-async-scheduling); fi
fi

cid_dir=$(mktemp -d)
cleanup() {
    if [[ -s "$cid_dir/container" ]]; then
        docker stop --time 8 "$(<"$cid_dir/container")" >/dev/null 2>&1 || true
        unlink "$cid_dir/container"
    fi
    rmdir "$cid_dir"
}
trap cleanup EXIT

timeout --kill-after=12s "${seconds}s" docker run --rm --init --name "$name" \
    --cidfile "$cid_dir/container" --network host --device "$render_node" --memory 23g --cpus 8 \
    --workdir /tmp --shm-size 1g --ulimit core=0 \
    --mount "type=bind,source=$PWD,target=/lab,readonly" \
    "${spec_mounts[@]}" "${cache_mounts[@]}" \
    --env PYTHONPATH=/lab:/lab/.work/python:/lab/.work/vllm-inkling/plugin \
    --env "INKLING2_MAX_SEQS=$slots" --env INKLING2_TRACE_BATCH --env "INKLING2_VISION=${INKLING2_VISION:-0}" \
    --env PYTHONDONTWRITEBYTECODE=1 --env VLLM_PLUGINS=inkling2_b70 --env VLLM_NO_USAGE_STATS=1 \
    --env "INKLING2_EXL3_SHARED_INPUT=${INKLING2_EXL3_SHARED_INPUT:-1}" \
    --env "EXL3_MOE_LIB=${INKLING2_EXL3_MOE_LIB:-/lab/.work/build/inkling2-exl3-moe.so}" \
    --env "INKLING2_PREFILL_PREP=${INKLING2_PREFILL_PREP:-0}" \
    --env "INKLING2_PREFILL_DENSE_BF16=${INKLING2_PREFILL_DENSE_BF16:-0}" \
    --env "INKLING2_PREFILL_CONTROLLER=${INKLING2_PREFILL_CONTROLLER:-0}" \
    --env "INKLING2_PREFILL_BANK=${INKLING2_PREFILL_BANK:-0}" \
    --env "INKLING2_PREFILL_DEPTH=${INKLING2_PREFILL_DEPTH:-0}" \
    --env "INKLING2_PREFILL_TIGHT_PREFIX=${INKLING2_PREFILL_TIGHT_PREFIX:-0}" \
    --env "INKLING2_PREFILL_PREFIX_IO=${INKLING2_PREFILL_PREFIX_IO:-0}" \
    --env "INKLING2_VISION_FUSED_QKV=${INKLING2_VISION_FUSED_QKV:-0}" \
    --env "INKLING2_PREFILL_KV_CONV=${INKLING2_PREFILL_KV_CONV:-0}" \
    --env "INKLING2_PREFILL_OUTPUT_CONV=${INKLING2_PREFILL_OUTPUT_CONV:-0}" \
    --env "INKLING2_PREFILL_LOCAL_BIAS=${INKLING2_PREFILL_LOCAL_BIAS:-0}" \
    --env "INKLING2_PREFILL_GLUE=${INKLING2_PREFILL_GLUE:-0}" \
    --env "INKLING2_NATIVE_DEPTH_MIX=${INKLING2_NATIVE_DEPTH_MIX:-0}" \
    --env "INKLING2_QKV_LIB=${INKLING2_QKV_LIB:-}" \
    --env "INKLING2_QKV_SPLITS=${INKLING2_QKV_SPLITS:-2}" \
    --env "INKLING2_PREP_LIB=${INKLING2_PREP_LIB:-}" \
    --env "INKLING2_NATIVE_MERGE=${INKLING2_NATIVE_MERGE:-0}" \
    --env "INKLING2_TUNE_DENSE=${INKLING2_TUNE_DENSE:-0}" \
    --env "INKLING2_FIXED_ROUTER=${INKLING2_FIXED_ROUTER:-0}" \
    --env "INKLING2_GROUP_CONTROLLERS=${INKLING2_GROUP_CONTROLLERS:-0}" \
    --env "INKLING2_MIX_CROSS_ONCE=${INKLING2_MIX_CROSS_ONCE:-0}" \
    --env "INKLING2_LOCAL_XMX=${INKLING2_LOCAL_XMX:-0}" \
    --env "INKLING2_PACKED_KV=${INKLING2_PACKED_KV:-0}" \
    --env "INKLING2_PAGED_XMX=${INKLING2_PAGED_XMX:-0}" \
    --env "INKLING2_SYCL_ROUTER=${INKLING2_SYCL_ROUTER:-0}" \
    --env "INKLING2_DSPARK_LONG_SPLITS=${INKLING2_DSPARK_LONG_SPLITS:-0}" \
    --env "INKLING2_DSPARK_VERIFY_TUNING=${INKLING2_DSPARK_VERIFY_TUNING:-0}" \
    --env "INKLING2_DSPARK_BLOCK=${INKLING2_DSPARK_BLOCK:-0}" \
    --env "INKLING2_DSPARK_GRAPH=${INKLING2_DSPARK_GRAPH:-0}" \
    --env "INKLING2_COMPARE_DRAFT_GRAPH=${INKLING2_COMPARE_DRAFT_GRAPH:-0}" \
    --env "INKLING2_DRAFT_REFERENCE_MATH=${INKLING2_DRAFT_REFERENCE_MATH:-0}" \
    --env "INKLING2_DRAFT_SAMPLE_TOPK=${INKLING2_DRAFT_SAMPLE_TOPK:-0}" \
    --env "INKLING2_DRAFT_FUSED_TAPS=${INKLING2_DRAFT_FUSED_TAPS:-1}" \
    --env "INKLING2_DRAFT_FP8_GLOBAL=${INKLING2_DRAFT_FP8_GLOBAL:-0}" \
    --env "INKLING2_DRAFT_Q8_HEAD=${INKLING2_DRAFT_Q8_HEAD:-0}" \
    --env "INKLING2_DSPARK_SHORTLIST=${INKLING2_DSPARK_SHORTLIST:-0}" \
    --env "INKLING2_COMPARE_GRAPH=${INKLING2_COMPARE_GRAPH:-0}" \
    --env "INKLING2_EXL3_SMALL_DPAS=${INKLING2_EXL3_SMALL_DPAS:-0}" \
    --env "INKLING2_EXL3_PAIR_MAX=${INKLING2_EXL3_PAIR_MAX:-8}" \
    --env "INKLING2_RING_PREFIX=${INKLING2_RING_PREFIX:-0}" \
    --env "INKLING2_CONTROL_LIB=${INKLING2_CONTROL_LIB:-}" \
    --env "INKLING2_DECODE_GRAPH=${INKLING2_DECODE_GRAPH:-0}" \
    --env "INKLING2_LIB=${INKLING2_LIB:-/lab/.work/build/inkling2-vllm.so}" \
    --env "INKLING2_EXL3_PREFILL_MIN=${INKLING2_EXL3_PREFILL_MIN:-1025}" \
    --env "INKLING2_ONEDNN_PREFIX_BLOCK=${INKLING2_ONEDNN_PREFIX_BLOCK:-0}" \
    --env "INKLING2_PREFILL_W8A8=${INKLING2_PREFILL_W8A8:-0}" \
    --env "INKLING2_W8A8_LIBRARY=${INKLING2_W8A8_LIBRARY:-}" \
    --env HF_HUB_OFFLINE=1 --env OMP_NUM_THREADS=4 --env SYCL_CACHE_PERSISTENT=0 \
    --env ONEAPI_DEVICE_SELECTOR=level_zero:gpu --env "SYCL_UR_USE_LEVEL_ZERO_V2=${INKLING2_UR_LEVEL_ZERO_V2:-${INKLING2_DECODE_GRAPH:-0}}" \
    --entrypoint /opt/venv/bin/vllm "$runtime" serve /lab/models/exl3 \
    --hf-config-path /lab/.work/vllm-inkling/config --tokenizer /lab/models/exl3 \
    --served-model-name inkling2-mini-exl3-b70 --generation-config vllm --load-format safetensors \
    --host 127.0.0.1 --port "$port" --max-model-len "$context" --max-num-seqs "$slots" \
    --max-num-batched-tokens "$batch" --gpu-memory-utilization 0.90 \
    --dtype bfloat16 --kv-cache-dtype "$kv_type" "${prefix_args[@]}" --enforce-eager "${spec_args[@]}" "$@"
