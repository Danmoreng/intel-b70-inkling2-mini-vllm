"""Merge normalized key splits on Intel XPU without materializing a weighted output tensor."""

import os

import torch
import triton
import triton.language as tl


NATIVE_MERGE = os.environ.get("INKLING2_NATIVE_MERGE", "0") == "1"
LONG_VERIFY_SPLITS = os.environ.get("INKLING2_DSPARK_LONG_SPLITS", "0") == "1"
TUNE_VERIFY = (
    os.environ.get("INKLING2_DSPARK_VERIFY_TUNING", "0") == "1"
    and os.environ.get("INKLING2_PACKED_KV", "0") != "1"
)

# Tables start at two target rows (one anchor plus one proposal).
_GLOBAL_TO_64K = (192, 128, 128, 128, 256, 128, 128)
_GLOBAL_OVER_64K = (128, 128, 128, 128, 256, 256, 256)
_LOCAL_FULL_WINDOW = (96, 48, 48, 48, 96, 96, 96)



@triton.jit
def _merge(PART, LSE, OUT, heads: tl.constexpr, parts: tl.constexpr, BP: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    dim = tl.program_id(1) * BD + tl.arange(0, BD)
    p = tl.arange(0, BP)
    lse = tl.load(LSE + p * heads + row, mask=p < parts, other=-float("inf"))
    weights = tl.exp(lse - tl.max(lse, 0))
    block = tl.load(
        PART + ((p * heads + row) * 128)[:, None] + dim[None, :], mask=(p < parts)[:, None], other=0.0
    )
    result = tl.sum(block * weights[:, None], 0) / tl.sum(weights, 0)
    tl.store(OUT + row * 128 + dim, result)


def merge_splits(partial, lse):
    if NATIVE_MERGE:
        return torch.ops.inkling2_xpu.merge_attention(partial, lse)
    parts, rows, heads, dim = partial.shape
    if dim != 128:
        raise ValueError("Inkling split attention requires head dimension 128")
    result = partial.new_empty((rows, heads, dim))
    _merge[(rows * heads, 8)](
        partial, lse, result, rows * heads, parts, triton.next_power_of_2(parts), 16, num_warps=8
    )
    return result


def paged_splits(tokens, rows=1):
    # B70 single-row XMX decode benefits from more independent key tiles on
    # short prompts and fewer partial results to merge at very long lengths.
    if TUNE_VERIFY and 2 <= rows <= 8 and 50_000 <= tokens < 150_000:
        table = _GLOBAL_TO_64K if tokens <= 65_536 else _GLOBAL_OVER_64K
        return table[rows - 2]
    if LONG_VERIFY_SPLITS and rows == 4 and tokens >= 196608:
        return 128
    if rows == 1 and (tokens <= 8192 or tokens >= 131072):
        tile_count = (max(1, tokens) + 15) // 16
        partitions = 1 << (tile_count - 1).bit_length()
        return min(192, max(8, partitions))
    return min(512, max(16, 1 << (max(1, (tokens - 1) // 256).bit_length())))


def paged_xmx(q, key_cache, value_cache, pages, length, pos, rel, proj, sink, parts, packed=False):
    rows, heads, width = q.shape
    partial = q.new_empty((parts, rows, heads, width), dtype=torch.float32)
    lse = q.new_empty((parts, rows, heads), dtype=torch.float32)
    torch.ops.inkling2_xpu.paged_xmx(
        q.half().contiguous(),
        key_cache,
        value_cache,
        pages,
        length,
        pos.int().contiguous(),
        (rel.float() @ proj.float()).contiguous(),
        sink.contiguous(),
        partial,
        lse,
        *([True] if packed else []),
    )
    return merge_splits(partial, lse)


def local_splits(tokens, rows):
    if TUNE_VERIFY and tokens >= 4096 and 2 <= rows <= 8:
        return _LOCAL_FULL_WINDOW[rows - 2]
    return 16


def local_xmx(q, k, v, ring_positions, length, pos, rel, proj, sink, *, parts=16):
    rows, heads, width = q.shape
    partial = q.new_empty((parts, rows, heads, width), dtype=torch.float32)
    lse = q.new_empty((parts, rows, heads), dtype=torch.float32)
    torch.ops.inkling2_xpu.local_xmx(
        q.half().contiguous(),
        k,
        v,
        ring_positions,
        length,
        pos.int().contiguous(),
        (rel.float() @ proj.float()).contiguous(),
        sink.contiguous(),
        partial,
        lse,
    )
    return merge_splits(partial, lse)


def batch_xmx(q, k, v, pages, lengths, positions, rel, proj, sink, parts, layout, *, local, packed=False):
    offsets, slots, max_rows = layout
    rows, heads, width = q.shape
    partial = q.new_empty((parts, rows, heads, width), dtype=torch.float32)
    lse = q.new_empty((parts, rows, heads), dtype=torch.float32)
    torch.ops.inkling2_xpu.batch_xmx(
        q.half().contiguous(), k, v, pages, lengths, positions.int().contiguous(),
        (rel.float() @ proj.float()).contiguous(), sink.contiguous(), offsets, slots,
        partial, lse, local, packed, max_rows,
    )
    return merge_splits(partial, lse)
