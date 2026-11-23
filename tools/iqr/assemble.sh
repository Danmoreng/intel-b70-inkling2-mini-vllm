#!/bin/bash
# assemble.sh <ckpt_dir> <model_dir> -- build an IQR model dir: the EXL3 checkout's non-expert files ($IQR_DENSE_SRC) + the
# IQR expert shards, index and iqr_config.json from <ckpt_dir> (written by write_ckpt.py).
set -euo pipefail
src=${IQR_DENSE_SRC:?EXL3 checkout with the dense files} ck=$1 dst=$2
mkdir -p "$dst"
for f in "$src"/*; do
  b=$(basename "$f")
  [[ $b == exl3-layer-* || $b == model.safetensors.index.json || $b == quantization_config.json ]] && continue
  cp -a --reflink=auto "$f" "$dst/"
done
cp --reflink=auto "$ck"/iqr-layer-*.safetensors "$ck"/model.safetensors.index.json "$ck"/iqr_config.json "$dst/"
ls "$dst" | wc -l; du -sh "$dst"
