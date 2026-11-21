"""Access the XMX tile layout stored in vLLM's existing FP8 physical pages."""

import torch


def arena_view(key):
    pages, tokens, heads, width = key.shape
    if tokens % 16 or heads != 8 or width != 128 or key.stride(0) < tokens * heads * width * 2:
        raise ValueError("The packed Inkling cache requires complete, aligned physical KV pages")
    return key.view(torch.uint8).as_strided(
        (pages, heads, tokens, width * 2), (key.stride(0), tokens * 256, 256, 1)
    )


def store(key, rows, k, v):
    torch.ops.inkling2_xpu.packed_kv_store(
        arena_view(key),
        rows,
        k.to(key.dtype).view(torch.uint8).contiguous(),
        v.to(key.dtype).view(torch.uint8).contiguous(),
    )


def read(key, physical_pages, length):
    raw = arena_view(key).index_select(0, physical_pages)
    pages, heads, pg, _ = raw.shape
    # K stores dimension quads grouped by token. V stores token quads grouped by dimension.
    kt, vt = raw.view(pages, heads, pg // 16, 2, 8, 4, 16, 4).unbind(3)
    k = kt.permute(0, 2, 5, 1, 3, 4, 6).reshape(-1, heads, 128)
    v = vt.permute(0, 2, 4, 6, 1, 3, 5).reshape(-1, heads, 128)
    return k[:length].view(key.dtype), v[:length].view(key.dtype)
