"""Load the packed 4-bit MCG experts into device memory without dequantizing them."""

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
        blob_size = ops().blob_bytes(2304, 1152, 4)
        self.arena = torch.empty((num_experts, blob_size), dtype=torch.uint8, device=device)
        self.pointers = torch.arange(num_experts, dtype=torch.int64, device=device)
        self.pointers.mul_(blob_size).add_(s64(self.arena.data_ptr()))
        shard = checkpoint / f"exl3-layer-{layer:02d}.safetensors"
        for index, projections in enumerate(read_experts(shard, layer, range(num_experts))):
            self.arena[index].copy_(pack_expert(*projections, 4))

    def __call__(self, hidden, indices, routing):
        return self.kernel(hidden.to(torch.bfloat16).contiguous(), indices.to(torch.int32).contiguous(),
                           routing.float().contiguous(), self.pointers, 1152, 4, self.num_experts)
