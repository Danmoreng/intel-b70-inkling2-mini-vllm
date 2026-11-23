#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../../.."
out=.work/build
mkdir -p "$out"
builder=${INKLING2_BUILDER:-sha256:ae6950731b3c031f812a95c0eb615a572239426440bf67fd1b3c9c9cbfce9eca}
timeout --kill-after=8s 4m docker run --rm --network none --memory 6g --cpus 3 \
    --user "$(id -u):$(id -g)" --mount "type=bind,source=$PWD,target=/lab,readonly" \
    --mount "type=bind,source=$PWD/$out,target=/out" --entrypoint /bin/bash "$builder" -ec '
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
    torch_dir=$(python3 -c "import torch,pathlib; print(pathlib.Path(torch.__file__).parent)")
    icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg_g31 -options -ze-opt-large-register-file" \
        -O3 -std=c++20 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=1 \
        -I"$torch_dir/include" -I"$torch_dir/include/torch/csrc/api/include" \
        /lab/benchmarks/2026-11-23/attention/simt_m1.cpp -L"$torch_dir/lib" \
        -ltorch -ltorch_cpu -lc10 -lc10_xpu -ltorch_xpu -o /out/inkling2-bench-simt-m1.so
'
