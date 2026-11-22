"""Compare complete XPU encoder latency and features on identical local screenshots."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from safetensors import safe_open
from inkling_trellis.inkling2.vendor.image_processing import Inkling2ImageProcessor
from inkling_trellis.inkling2.vendor.inkling2_vision import Inkling2VisionEncoder


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("images", nargs=2, type=Path)
    parser.add_argument("--repeats", type=int, default=16)
    args = parser.parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text())
    weights = {}
    for shard in sorted(args.checkpoint.glob("dense-*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name.startswith("model.visual."):
                    weights[name.removeprefix("model.visual.")] = reader.get_tensor(name)
    encoders = {}
    for variant in ("reference", "fused"):
        encoder = Inkling2VisionEncoder(config["vision_config"], config["text_config"]["hidden_size"])
        encoder.load_state_dict(weights, strict=True, assign=True)
        encoder.eval().to("xpu")
        if variant == "fused":
            for layer in encoder.backbone.layers:
                layer.self_attn.fuse_qkv()
        encoders[variant] = encoder
    images = []
    subjects = []
    for path in args.images:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
        subjects.append(dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=images[-1].size))
    quality_ok = True
    for indices in ([0], [1], [0, 1]):
        data = Inkling2ImageProcessor().preprocess([images[i] for i in indices])
        patches = data["vision_patches_bthwc"].to(device="xpu", dtype=torch.bfloat16)
        grid = data["vision_grid_thw"]
        output = {}
        samples = {variant: [] for variant in encoders}
        for repeat in range(args.repeats + 2):
            order = list(encoders) if repeat % 2 == 0 else list(reversed(encoders))
            for variant in order:
                torch.xpu.synchronize()
                started = time.perf_counter()
                result = encoders[variant](patches, grid)
                torch.xpu.synchronize()
                elapsed = (time.perf_counter() - started) * 1000
                if repeat >= 2:
                    samples[variant].append(elapsed)
                output[variant] = result
        ref, fused = (output[key].float() for key in ("reference", "fused"))
        assert torch.isfinite(ref).all() and torch.isfinite(fused).all()
        cosines = F.cosine_similarity(ref, fused, dim=-1)
        row = dict(images=[subjects[i] for i in indices], feature_shape=list(ref.shape),
                   exact=torch.equal(ref, fused), cosine_min=cosines.min().item(),
                   cosine_mean=cosines.mean().item(), max_absolute_error=(ref-fused).abs().max().item(),
                   relative_l2=((ref-fused).norm()/ref.norm()).item(),
                   median_ms={k:statistics.median(v) for k,v in samples.items()}, samples_ms=samples)
        print(json.dumps(row), flush=True)
        quality_ok &= row["cosine_min"] >= 0.999
    if not quality_ok:
        raise RuntimeError("Per-patch cosine similarity below 0.999")


if __name__ == "__main__":
    main()
