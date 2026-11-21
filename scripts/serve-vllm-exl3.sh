#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
context=${1:?context length}
shift
port=${INKLING2_VLLM_PORT:-8093}
batch=${INKLING2_VLLM_BATCH:-128}
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
test -s .work/build/inkling2-vllm.so
test -s .work/build/inkling2-exl3-moe.so
if systemctl --user is-active --quiet b70-qwen38-vllm.service || \
    [[ $(docker inspect --format '{{.State.Running}}' inkling2-sycl-baseline 2>/dev/null || true) == true ]]; then
    echo 'The B70 is occupied by another text model.' >&2
    exit 1
fi

check_vendor_patch() {
    local source_file=$1
    local patch_file="$PWD/patches/$2"
    if git -C "$vendor" diff --quiet -- "$source_file"; then
        git -C "$vendor" apply "$patch_file"
    fi
    git -C "$vendor" diff -- "$source_file" | cmp - "$patch_file"
}
check_vendor_patch xpu/exl3xpu/inkling2/sgl_model.py trellis-inkling-external-runner.patch
check_vendor_patch xpu/exl3xpu/inkling2/dense.py trellis-inkling-dense-w8a8-prefill.patch

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
python3 - "$work/config/config.json" <<'PY'
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
Path(sys.argv[1]).write_text(json.dumps(config, indent=2) + '\n')
PY

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
    --env PYTHONPATH=/lab:/lab/.work/python:/lab/.work/vllm-inkling/plugin \
    --env PYTHONDONTWRITEBYTECODE=1 --env VLLM_PLUGINS=inkling2_b70 --env VLLM_NO_USAGE_STATS=1 \
    --env "EXL3_MOE_LIB=${INKLING2_EXL3_MOE_LIB:-/lab/.work/build/inkling2-exl3-moe.so}" \
    --env INKLING2_LIB=/lab/.work/build/inkling2-vllm.so \
    --env "INKLING2_EXL3_PREFILL_MIN=${INKLING2_EXL3_PREFILL_MIN:-1025}" \
    --env "INKLING2_ONEDNN_PREFIX_BLOCK=${INKLING2_ONEDNN_PREFIX_BLOCK:-0}" \
    --env "INKLING2_PREFILL_W8A8=${INKLING2_PREFILL_W8A8:-0}" \
    --env "INKLING2_W8A8_LIBRARY=${INKLING2_W8A8_LIBRARY:-}" \
    --env HF_HUB_OFFLINE=1 --env OMP_NUM_THREADS=4 --env SYCL_CACHE_PERSISTENT=0 \
    --env ONEAPI_DEVICE_SELECTOR=level_zero:gpu --env SYCL_UR_USE_LEVEL_ZERO_V2=0 \
    --entrypoint /opt/venv/bin/vllm "$runtime" serve /lab/models/exl3 \
    --hf-config-path /lab/.work/vllm-inkling/config --tokenizer /lab/models/exl3 \
    --served-model-name inkling2-mini-exl3-b70 --generation-config vllm --load-format safetensors \
    --host 127.0.0.1 --port "$port" --max-model-len "$context" --max-num-seqs 1 \
    --max-num-batched-tokens "$batch" --gpu-memory-utilization 0.90 \
    --dtype bfloat16 --kv-cache-dtype "$kv_type" --no-enable-prefix-caching --enforce-eager "$@"
