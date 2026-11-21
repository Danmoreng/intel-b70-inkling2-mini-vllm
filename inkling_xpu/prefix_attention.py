"""Reuse Qwen's native SDPA for Inkling keys outside the relative-bias band."""

from collections.abc import Callable

import torch


def hybrid_prefill(
    q: torch.Tensor,
    table: torch.Tensor,
    sink: torch.Tensor,
    first_position: int,
    total_tokens: int,
    prefix_block: int,
    read_kv: Callable[[int, int], tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    # Every query can see these older keys and none receives relative bias.
    far_end = max(0, first_position + 1 - table.shape[-1])
    prefix_end = far_end // prefix_block * prefix_block
    q_half = q.to(torch.float16).contiguous()
    k, v = read_kv(prefix_end, total_tokens)
    out = torch.empty_like(q_half, dtype=torch.float32)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    torch.ops.inkling2_xpu.prefill_attn(
        q_half, k.to(torch.float16).contiguous(), v.to(torch.float16).contiguous(), out,
        table.to(torch.bfloat16).contiguous(), sink.float().contiguous(),
        first_position, prefix_end, 0, 4, lse,
    )
    if not prefix_end:
        return out

    q_heads = q_half.transpose(0, 1).contiguous()
    part_out = torch.empty_like(q_heads)
    part_lse = torch.empty(q_heads.shape[:2], dtype=torch.float32, device=q.device)
    scale = q.shape[-1] ** -0.5
    for start in range(0, prefix_end, prefix_block):
        k, v = read_kv(start, start + prefix_block)
        k_heads = k.transpose(0, 1).to(torch.float16).contiguous()
        v_heads = v.transpose(0, 1).to(torch.float16).contiguous()
        torch.ops.exl3xpu_C.exl3_sdpa(q_heads, k_heads, v_heads, part_out, part_lse, False, scale)
        older_lse = part_lse.transpose(0, 1)
        normalizer = torch.logaddexp(lse, older_lse)
        out.mul_((lse - normalizer).exp().unsqueeze(-1))
        out.addcmul_(part_out.transpose(0, 1), (older_lse - normalizer).exp().unsqueeze(-1))
        lse = normalizer
    return out
