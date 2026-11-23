"""Load the packed 4-bit MCG experts into device memory without dequantizing them."""

import os

import torch
from inkling_trellis.moe_offload import ops, pack_expert, s64
from safetensors import safe_open


def read_experts(shard, layer, expert_ids):
    with safe_open(shard, framework="pt", device="cpu") as reader:
        for expert in expert_ids:
            projections = []
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"model.llm.layers.{layer}.mlp.experts.{expert}.{projection}"
                marker = reader.get_tensor(prefix + ".mcg")
                if marker.shape != torch.Size([]) or marker.dtype != torch.int32:
                    raise ValueError(f"Invalid MCG marker for {prefix}")
                projections.append({key: reader.get_tensor(f"{prefix}.{key}")
                                    for key in ("trellis", "suh", "svh")})
            yield projections


class InklingEXL3Experts:
    def __init__(self, checkpoint, layer, device, num_experts=128):
        self.num_experts = num_experts
        self.kernel = ops().moe_forward
        self.shared_kernel = None
        if os.environ.get("INKLING2_EXL3_SHARED_INPUT", "1") == "1":
            self.shared_kernel = getattr(ops(), "moe_shared_prefill", None)
        self.shared_scales = None
        blob_size = ops().blob_bytes(2304, 1152, 4)
        self.arena = torch.empty((num_experts, blob_size), dtype=torch.uint8, device=device)
        self.pointers = torch.arange(num_experts, dtype=torch.int64, device=device)
        self.pointers.mul_(blob_size).add_(s64(self.arena.data_ptr()))
        shard = checkpoint / f"exl3-layer-{layer:02d}.safetensors"
        for index, projections in enumerate(read_experts(shard, layer, range(num_experts))):
            if self.shared_kernel is not None:
                scales = torch.stack([proj["suh"].half().reshape(-1) for proj in projections[:2]])
                if index == 0:
                    self.shared_scales = scales
                elif not torch.equal(self.shared_scales, scales):
                    self.shared_kernel = None
                    self.shared_scales = None
            self.arena[index].copy_(pack_expert(*projections, 4))
        if self.shared_scales is not None:
            self.shared_scales = self.shared_scales.to(device)

    def __call__(self, hidden, indices, routing):
        inputs = hidden.to(torch.bfloat16).contiguous(), indices.int().contiguous(), routing.float().contiguous()
        if self.shared_scales is not None and hidden.shape[0] >= 2048 and indices.shape[1] == 8:
            return self.shared_kernel(*inputs, self.pointers, self.shared_scales)
        return self.kernel(*inputs, self.pointers, 1152, 4, self.num_experts)
