"""Bounded single-B70 correctness probes. No model download or server startup."""

import argparse
from itertools import product

import torch

from exl3xpu.inkling2.attn_triton import (
    attention_decode_sycl,
    attention_prefill_esimd,
    reference_attention,
)
from exl3xpu.inkling2.moe_triton import Inkling2ExpertsSYCL
from exl3xpu.inkling2.ref import mxfp8_dequant, nvfp4_dequant
from inkling_xpu.paged_attention import decode_paged
from inkling_xpu.prefix_attention import hybrid_prefill


def check_paged(layout, dtype, rows, block_size, window):
    heads, kv_heads, width = 24, 8, 128
    page_ids = torch.tensor([[4, 0, 5], [3, 1, 4]], dtype=torch.int32)
    lengths = torch.tensor([3 * block_size - 1, 2 * block_size + 3], dtype=torch.int32)
    positions = (lengths[:, None] - rows + torch.arange(rows)).flatten().int()
    count = positions.numel()
    shape = (6, block_size, kv_heads, width)
    keys = torch.randn(shape).to(dtype)
    values = torch.randn(shape).to(dtype)
    q = torch.randn(count, heads, width).bfloat16()
    relative = torch.randn(count, heads, 16) / 4
    projection = torch.randn(16, 2 * block_size) / 4
    sink = torch.randn(count, heads)
    key_scale, value_scale = (0.75, 1.25) if keys.element_size() == 1 else (1.0, 1.0)

    def device_cache(tensor):
        if layout == "HND":
            return tensor.transpose(1, 2).contiguous().to("xpu").transpose(1, 2)
        return tensor.to("xpu")

    actual = decode_paged(
        q.to("xpu"), device_cache(keys), device_cache(values), page_ids.to("xpu"),
        lengths.to("xpu"), positions.to("xpu"), relative.to("xpu"), projection.to("xpu"),
        sink.to("xpu"), queries_per_request=rows, window=window, num_splits=4,
        key_scale=key_scale, value_scale=value_scale,
    ).cpu()
    table = relative @ projection
    expected = []
    for request, length in enumerate(lengths.tolist()):
        span = slice(request * rows, (request + 1) * rows)
        pages = page_ids[request].long()
        k = keys.float()[pages].flatten(0, 1)[:length] * key_scale
        v = values.float()[pages].flatten(0, 1)[:length] * value_scale
        expected.append(reference_attention(q[span], k, v, torch.arange(length),
                                            positions[span], table[span], sink[span], window=window))
    expected = torch.cat(expected)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=5e-3)
    error = (actual - expected).abs().max().item()
    print(f"paged {layout} {dtype} q={rows} page={block_size} window={window}: max_abs={error:.3g}", flush=True)
    if layout == "NHD" and dtype == torch.bfloat16 and rows == 4 and not window:
        token_table = page_ids[..., None] * block_size + torch.arange(block_size, dtype=torch.int32)
        upstream = attention_decode_sycl(
            q.to("xpu"), keys.flatten(0, 1).to("xpu"), values.flatten(0, 1).to("xpu"),
            torch.arange(2, dtype=torch.int32, device="xpu"), positions.to("xpu"),
            relative.to("xpu"), projection.to("xpu"), sink.to("xpu"), n_q=rows,
            paged=True, r2t=token_table.flatten(1).to("xpu"), kv_len=lengths.to("xpu"), nsplit=4,
        ).cpu()
        torch.testing.assert_close(actual, upstream, atol=1e-6, rtol=1e-6)
        print("Legacy flat token-table path matches the block-table path", flush=True)


def check_prefill(window):
    query_pos, key_pos = 237, 0
    query = torch.randn(97, 24, 128).half()
    keys = torch.randn(query_pos + len(query), 8, 128).half()
    values = torch.randn_like(keys)
    table = (torch.randn(len(query), 24, 64) / 4).bfloat16()
    sink = torch.randn(len(query), 24)
    expected = reference_attention(
        query, keys, values, torch.arange(len(keys)), query_pos + torch.arange(len(query)),
        table, sink, window=window,
    )
    actual = attention_prefill_esimd(
        query.to("xpu"), keys.to("xpu"), values.to("xpu"), query_pos, key_pos,
        table.to("xpu"), sink.to("xpu"), window=window, qtw=4,
    ).cpu()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=5e-3)
    print(f"DPAS prefill window={window}: max_abs={(actual - expected).abs().max().item():.3g}", flush=True)


def check_hybrid_prefill():
    torch.ops.load_library("/opt/exl3xpu/exl3xpu/_C.so")
    query_pos, count, extent = 9230, 97, 1024
    end = query_pos + count
    q = torch.randn(count, 24, 128, device="xpu").half()
    k = torch.randn(end, 8, 128, device="xpu").half()
    v = torch.randn_like(k)
    table = (torch.randn(count, 24, extent, device="xpu") / 2).bfloat16()
    sink = torch.randn(count, 24, device="xpu") + 7
    actual = hybrid_prefill(q, table, sink, query_pos, end, 4096,
                            lambda start, stop: (k[start:stop], v[start:stop]))
    baseline = attention_prefill_esimd(q, k, v, query_pos, 0, table, sink, qtw=4)
    error = ((actual - baseline).norm() / baseline.norm()).item()
    assert error < 0.003, f"Hybrid relative error: {error:.3g}"
    rows = torch.tensor([0, 1, count - 1])
    expected = reference_attention(q.cpu()[rows], k.cpu(), v.cpu(), torch.arange(end),
                                   query_pos + rows, table.cpu()[rows], sink.cpu()[rows])
    torch.testing.assert_close(actual.cpu()[rows], expected, atol=2e-4, rtol=5e-3)
    print(f"oneDNN hybrid, two prefix blocks and biased tail: relative_l2={error:.3g}", flush=True)


def check_experts(nvfp4):
    experts, hidden, intermediate = 8, 2304, 1152

    def weights(n, k):
        if nvfp4:
            packed = torch.randint(256, (experts, n, k // 2), dtype=torch.uint8)
            scales = torch.randint(0x30, 0x40, (experts, n, k // 16), dtype=torch.uint8)
            return packed, scales.view(torch.float8_e4m3fn), torch.rand(experts) * 0.01 + 0.003
        packed = torch.randint(0x28, 0x38, (experts, n, k), dtype=torch.uint8)
        packed ^= torch.randint(0, 2, packed.shape, dtype=torch.uint8) << 7
        scales = torch.randint(116, 120, (experts, n, k // 32), dtype=torch.uint8)
        return packed.view(torch.float8_e4m3fn), scales, None

    w13, s13, a13 = weights(2 * intermediate, hidden)
    w2, s2, a2 = weights(hidden, intermediate)
    xpu_weights = [t.to("xpu") if t is not None else None for t in (w13, s13, w2, s2, a13, a2)]
    model = Inkling2ExpertsSYCL(*xpu_weights)
    x = torch.randn(1, hidden).bfloat16()
    expert_ids = torch.randperm(experts).reshape(1, -1)
    routing = torch.rand(1, experts).softmax(dim=-1) * 8
    actual = model.decode(x.to("xpu"), expert_ids.to("xpu"), routing.to("xpu")).cpu()
    expected = torch.zeros((1, hidden))
    for slot, expert in enumerate(expert_ids[0].tolist()):
        if nvfp4:
            gate_up = nvfp4_dequant(w13[expert], s13[expert], a13[expert], dtype=torch.float32)
            down = nvfp4_dequant(w2[expert], s2[expert], a2[expert], dtype=torch.float32)
        else:
            gate_up = mxfp8_dequant(w13[expert], s13[expert], dtype=torch.float32)
            down = mxfp8_dequant(w2[expert], s2[expert], dtype=torch.float32)
        logits = x.float() @ gate_up.T
        gate, up = logits[:, 0::2], logits[:, 1::2]
        activated = (4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * 25 * torch.tanh(up / 25))
        expected += (activated.bfloat16().float() @ down.T) * routing[0, slot]
    error = ((actual - expected).norm() / expected.norm()).item()
    assert error < 0.015, f"MoE relative error: {error:.3g}"
    print(f"{'NVFP4' if nvfp4 else 'MXFP8'} MoE H={hidden} I={intermediate} top8: relative_l2={error:.3g}", flush=True)


def check_fused_bias():
    from inkling_xpu.prefill_kernels import local_bias, pack_local_projection

    for rows, start, residual in ((17, 0, False), (251, 79, True), (257, 30720, False)):
        history = min(start, 1023)
        q = torch.randn(rows, 24, 128, device="xpu").half()
        k = torch.randn(rows + history, 8, 128, device="xpu").half()
        v = torch.randn_like(k)
        relative = torch.randn(rows, 24, 16, device="xpu")
        projection = torch.randn(16, 1024, device="xpu") * 0.1
        if not residual:
            projection = projection.half().float()
        packed = pack_local_projection(projection)
        assert packed.size(0) == 1 + int(residual)
        sink = torch.randn(rows, 24, device="xpu")
        expected = attention_prefill_esimd(q, k, v, start, start - history, local_bias(relative, projection),
                                          sink, window=1024, qtw=4)
        actual = torch.empty_like(expected)
        torch.ops.inkling2_xpu.prefill_fused_bias(q, k, v, actual, relative, packed, sink,
                                                  start, start - history, 1024, 4)
        rel_l2 = ((actual - expected).norm() / expected.norm()).item()
        assert rel_l2 < 3e-4, (rows, start, rel_l2)
        print(f"Fused local bias, {rows} rows at {start}, residual={residual}, relative_l2={rel_l2:.5g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    parser.add_argument("--fused-bias", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    device = torch.xpu.get_device_properties(0)
    print(f"{torch.__version__} on {device.name}", flush=True)
    torch.ops.load_library(args.library)
    for layout, dtype, rows, window in product(
        ("NHD", "HND"), (torch.bfloat16, torch.float8_e4m3fn), (1, 4), (0, 32),
    ):
        check_paged(layout, dtype, rows, 16 if rows == 1 else 64, window)
    for nvfp4 in (True, False):
        check_experts(nvfp4)
    for window in (0, 128):
        check_prefill(window)
    check_hybrid_prefill()
    if args.fused_bias:
        check_fused_bias()
    torch.xpu.synchronize()
    print("PASS: 16 block-table cases, 1 legacy table case, 3 prefill cases and both MoE formats", flush=True)


if __name__ == "__main__":
    main()
