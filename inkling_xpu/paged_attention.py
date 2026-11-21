"""Inkling decode attention over vLLM's logical [block, token, head, dim] KV layout."""

import torch


def decode_paged(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    relative_query: torch.Tensor,
    relative_projection: torch.Tensor,
    sink: torch.Tensor,
    *,
    queries_per_request: int = 1,
    window: int = 0,
    num_splits: int = 8,
    key_scale: float = 1.0,
    value_scale: float = 1.0,
) -> torch.Tensor:
    """Read populated vLLM cache blocks, with uniform query counts and Inkling's bias/sink."""
    tokens, heads, width = query.shape
    if block_table.ndim != 2:
        raise ValueError("Expected a two-dimensional vLLM block table")
    requests = block_table.shape[0]
    if queries_per_request not in (1, 2, 4, 8) or tokens != requests * queries_per_request:
        raise ValueError("Expected 1, 2, 4 or 8 query rows per request, uniform across the batch")
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("Expected matching logical [block, token, head, dim] KV caches")
    if seq_lens.shape != (requests,) or positions.shape != (tokens,):
        raise ValueError("Invalid vLLM block table, sequence lengths or query positions")
    if relative_query.shape != (tokens, heads, 16) or sink.shape != (tokens, heads):
        raise ValueError("Invalid Inkling relative query or attention sink shape")
    if relative_projection.ndim != 2 or relative_projection.shape[0] != 16:
        raise ValueError("Expected a relative projection with 16 input channels")
    if num_splits < 1 or window < 0:
        raise ValueError("Expected a positive split count and a nonnegative sliding window")
    device_inputs = (key_cache, value_cache, block_table, seq_lens, positions,
                     relative_query, relative_projection, sink)
    if query.device.type != "xpu" or any(t.device != query.device for t in device_inputs):
        raise ValueError("Inkling decode requires all tensors on the same XPU device")
    if tokens == 0:
        return query.new_empty(query.shape, dtype=torch.float32)

    slots = torch.arange(requests, dtype=torch.int32, device=query.device)
    maxima = query.new_empty((tokens, heads, num_splits), dtype=torch.float32)
    masses = torch.empty_like(maxima)
    partial = query.new_empty((tokens, heads, num_splits, width), dtype=torch.float32)
    torch.ops.inkling2_xpu.decode_attn(
        query.to(torch.bfloat16).contiguous(),
        key_cache,
        value_cache,
        slots,
        seq_lens.to(torch.int32).contiguous(),
        block_table.to(torch.int32).contiguous(),
        None,
        positions.to(torch.int32).contiguous(),
        relative_query.float().contiguous(),
        relative_projection.float().contiguous(),
        sink.float(),
        partial,
        maxima,
        masses,
        queries_per_request,
        0,
        window,
        key_scale,
        value_scale,
    )
    weights = torch.exp(maxima - maxima.amax(dim=-1, keepdim=True))
    return (partial * weights.unsqueeze(-1)).sum(dim=2) / (masses * weights).sum(dim=2).unsqueeze(-1)
