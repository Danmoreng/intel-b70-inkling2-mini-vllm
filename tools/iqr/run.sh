#!/usr/bin/env bash
# run.sh cpu|gpu <cpus> <memory> <script.py> [args] -- run a quantizer step in the runtime image (docs/IQR.md).
# IQR_MODEL (native checkpoint, read-only), IQR_DIR (work dir, read-write) and IQR_DENSE_SRC (EXL3 checkout,
# write_ckpt only) are absolute host paths, mounted at the same paths. Needs the Trellis checkout from check-sycl.sh.
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
mode=$1 cpus=$2 mem=$3; shift 3
runtime=${INKLING2_RUNTIME:-sha256:131dd9e11809de2f3136547618f130ee238f0b90cd5718567bcdd0dad5360e19}
: "${IQR_MODEL:?native checkpoint dir}" "${IQR_DIR:?work dir}"
mkdir -p "$IQR_DIR"
dev=() mounts=(--mount "type=bind,source=$IQR_MODEL,target=$IQR_MODEL,readonly")
[[ $mode == gpu ]] && dev=(--device "$(readlink -f -- "${INKLING2_RENDER_NODE:-/dev/dri/by-path/pci-0000:03:00.0-render}")")
[[ -n ${IQR_DENSE_SRC:-} ]] && mounts+=(--mount "type=bind,source=$IQR_DENSE_SRC,target=$IQR_DENSE_SRC,readonly")
exec docker run --rm --network none --cpus "$cpus" --memory "$mem" --shm-size 8g "${dev[@]}" "${mounts[@]}" \
    --user "$(id -u):$(id -g)" --mount "type=bind,source=$IQR_DIR,target=$IQR_DIR" \
    --mount "type=bind,source=$PWD,target=/lab,readonly" \
    --env PYTHONPATH=/lab/tools/iqr:/lab/.work/python:/lab/.work/trellis-serve/core/src --env PYTHONDONTWRITEBYTECODE=1 \
    --env IQR_MODEL --env IQR_DIR --env IQR_DENSE_SRC --env IQR_DEV --env IQR_TEST_T --env IQR_CALIB_NOTE \
    --env OMP_NUM_THREADS="$cpus" --env SYCL_CACHE_PERSISTENT=0 --env ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
    --entrypoint /opt/venv/bin/python "$runtime" -u "/lab/tools/iqr/$1" "${@:2}"
