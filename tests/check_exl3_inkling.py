"""Compare the native SYCL MoE with a CPU reconstruction of real Inkling experts."""

import argparse
from pathlib import Path
from time import perf_counter

import torch
from inkling_trellis.moe_offload import ops, pack_expert
from trellis_core.reference import CB_MCG, weight_orig

from inkling_xpu.exl3_experts import read_experts


def reference(activation, indices, routing, weights):
    output = torch.zeros_like(activation, dtype=torch.float32)
    for expert, (gate, up, down) in enumerate(weights):
        rows, ranks = (indices == expert).nonzero(as_tuple=True)
        act = activation[rows].float()
        gate_logits = act @ gate
        gate_capped = 4 * torch.tanh(gate_logits / 4) * torch.sigmoid(gate_logits)
        up_capped = 25 * torch.tanh((act @ up) / 25)
        output.index_add_(0, rows, ((gate_capped * up_capped) @ down) * routing[rows, ranks, None])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, nargs="+", default=(1, 4, 8))
    parser.add_argument("--small-dpas", action="store_true", help="Use the eight-row XMX expert path for 4–8 rows")
    args = parser.parse_args()

    torch.set_num_threads(4)
    layer = args.layer
    expert_ids = (0, 3, 11, 16, 22, 31, 53, 64, 87, 99, 102, 127)
    shard = args.checkpoint / f"exl3-layer-{layer:02d}.safetensors"
    packed, weights = [], []
    for projections in read_experts(shard, layer, expert_ids):
        packed.append(pack_expert(*projections, 4))
        weights.append([weight_orig(**projection, K=4, cb=CB_MCG) for projection in projections])

    arena = torch.stack(packed).to("xpu")
    expert_ptrs = torch.arange(len(packed), device="xpu", dtype=torch.int64) * arena.stride(0) + arena.data_ptr()
    if args.small_dpas:
        ops().moe_set_tiny_dpas(4, 9, 18)
    kernel = ops().moe_forward
    generator = torch.Generator().manual_seed(23800 + layer)
    for dtype in (torch.float16, torch.bfloat16):
        for tokens in args.tokens:
            activation = torch.randn(tokens, 2304, generator=generator).to(dtype)
            scores = torch.randn(tokens, len(expert_ids), generator=generator)
            scores, indices = scores.topk(8)
            indices = indices.to(torch.int32)
            routing = scores.softmax(-1) * 8
            expected = reference(activation, indices, routing, weights)

            device_inputs = [tensor.to("xpu") for tensor in (activation, indices, routing)]
            call_args = (*device_inputs, expert_ptrs, 1152, 4, len(packed))
            output = kernel(*call_args).cpu().float()
            torch.xpu.synchronize()
            started = perf_counter()
            for _ in range(10):
                kernel(*call_args)
            torch.xpu.synchronize()
            elapsed_ms = (perf_counter() - started) * 100
            rel_l2 = ((output - expected).norm() / expected.norm()).item()
            print(f"layer={layer} dtype={dtype} tokens={tokens:3d} rel_l2={rel_l2:.5f} ms={elapsed_ms:.3f}", flush=True)
            assert rel_l2 < 0.02, rel_l2


if __name__ == "__main__":
    main()
