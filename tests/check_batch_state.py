"""Check ragged convolutions and DSpark sampling against independent request slots."""

import os
import sys

import torch


def check_convolutions():
    device = torch.device("xpu")
    layout = [(2, 37, 7), (0, 3, 2), (3, 77, 1), (1, 15, 8)]
    slots = torch.tensor([s for s, _, n in layout for _ in range(n)], device=device, dtype=torch.int)
    offsets = [0]
    for _, _, n in layout:
        offsets.append(offsets[-1] + n)
    firsts = torch.tensor([o for o, (_, _, n) in zip(offsets, layout) for _ in range(n)],
                          device=device, dtype=torch.int)
    pos = torch.tensor([p + j for _, p, n in layout for j in range(n)], device=device)
    total = pos.numel()
    tags = torch.full((4, 36), -1, dtype=torch.long, device=device)
    for slot, start, n in layout:
        recent = torch.arange(max(0, start - 4), start + n, device=device)
        tags[slot, recent % 36] = recent
    def rand(*shape):
        return torch.randn(*shape, device=device)

    qs, rs, kw, vw = rand(24, 128), rand(24, 16), rand(1024, 4), rand(1024, 4)
    for local in (False, True):
        dims = 1024 if local else 1056
        q, k, v = rand(total, 6552), rand(total, dims), rand(total, dims)
        ring = rand(4, 36, 6656)
        reference_ring = ring.clone()
        caches = [torch.zeros(4 * 1088, 8, 128, device=device, dtype=torch.bfloat16) for _ in range(2)]
        reference_caches = [b.clone() for b in caches]
        def prep(q, k, v, p, history, valid, kv, *layout_args):
            return torch.ops.inkling_prep.forward(q, k, v, p, qs, rs, kw, vw, history, valid,
                                                  *(kv if local else [None, None]),
                                                  22.62741699796952, 8., .1, 8192., *layout_args)
        actual = prep(q, k, v, pos, ring, tags, caches, slots, firsts)
        for offset, (slot, start, n) in zip(offsets, layout):
            rows = slice(offset, offset + n)
            history = reference_ring[slot:slot + 1]
            kv = [buf[slot * 1088:(slot + 1) * 1088] for buf in reference_caches]
            ref = prep(q[rows], k[rows], v[rows], pos[rows], history, tags[slot:slot + 1], kv)
            for a, b in zip(actual, ref, strict=True):
                if b.numel():
                    torch.testing.assert_close(a[rows], b, rtol=0, atol=0)
        torch.testing.assert_close(ring, reference_ring, atol=0, rtol=0)
        for a, b in zip(caches, reference_caches, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    # The output convolution interleaves slots and crosses the physical ring wrap.
    x, w, history = rand(total, 24), rand(24, 4), rand(4, 36, 64)
    offset = 9
    expected_history = history.clone()
    expected = torch.empty_like(x)
    for first, (slot, p, n) in zip(offsets, layout):
        for j in range(n):
            row, position = first + j, p + j
            acc = x[row] * w[:, 3]
            for tap in range(3):
                old = position + tap - 3
                if old >= 0:
                    acc = torch.addcmul(acc, expected_history[slot, old % 36, offset:offset + 24], w[:, tap])
            expected[row] = (x[row] + acc) * (5. ** -.5)
            expected_history[slot, position % 36, offset:offset + 24] = x[row]
    actual = torch.ops.inkling_prep.output_conv(x, w, history, tags, pos, slots, firsts, offset)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(history, expected_history, rtol=0, atol=0)


def check_sampling():
    from inkling_xpu.dspark import DraftSampling

    device = torch.device("xpu")
    os.environ.update(INKLING2_DSPARK_ADAPTIVE="1", INKLING2_DRAFT_SAMPLE_TOPK="20")
    logits = torch.full((4, 7, 256), -float("inf"), device=device)
    batch = DraftSampling(7, logits, device, False)
    batch.seed.copy_(torch.tensor([177, 39, 1111, 8]))
    batch.temperature.copy_(torch.tensor([0.6, 1., 0., 0.9]))
    batch.slot = torch.tensor([2, 0, 3], device=device)
    single = []
    for slot in (2, 0, 3):
        s = DraftSampling(7, torch.full_like(logits[:1], -float("inf")), device, False)
        s.seed.copy_(batch.seed[slot:slot + 1])
        s.temperature.copy_(batch.temperature[slot:slot + 1])
        single.append(s)
    for step in range(7):
        positions = torch.tensor([8192, 99, 716], device=device) + step
        values = torch.randn(3, 256, device=device)
        chosen = batch.sample(values, None, positions, step)
        for row, s in enumerate(single):
            one = s.sample(values[row:row + 1], None, positions[row:row + 1], step)
            torch.testing.assert_close(chosen[row:row + 1], one, rtol=0, atol=0)
            torch.testing.assert_close(batch.confidence[step][row:row + 1], s.confidence[step])
            torch.testing.assert_close(logits[batch.slot[row], step], s.logits[0, step], atol=0, rtol=0)
    assert logits[1].isneginf().all(), "Inactive worker slot was overwritten"


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.ops.load_library(sys.argv[1])
    check_convolutions()
    check_sampling()
    print("Ragged K/V, output convolutions and all seven per-slot proposal distributions passed.")
