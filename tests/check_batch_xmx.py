"""Compare ragged C2/C4 launches with the single-sequence XMX kernels on the same KV pages."""

import sys

import torch

device = torch.device("xpu")
ops = torch.ops.inkling2_xpu


def ints(values):
    return torch.tensor(values, device=device, dtype=torch.int32)


def check():
    torch.manual_seed(20261123)
    for counts, slots in [([1, 1], [3, 1]), ([7, 2, 1, 8], [2, 0, 3, 1])]:
        starts = [1039, 47, 1084, 1361][:len(counts)]
        offsets = [0]
        for n in counts:
            offsets.append(offsets[-1] + n)
        total = offsets[-1]
        lens = ints([p + n for p, n in zip(starts, counts)])
        positions = ints([p + j for p, n in zip(starts, counts) for j in range(n)])
        q = torch.randn(total, 24, 128, device=device).half()
        bias = torch.randn(total, 24, 64, device=device)
        sink = torch.randn(total, 24, device=device)
        for local in (False, True):
            ring = 1088
            if local:
                sizes = (4 * ring, 8, 128)
                k, v = [torch.randn(sizes, device=device).bfloat16() for _ in range(2)]
                pages = torch.full((4, ring), -1, dtype=torch.int32, device=device)
                for start, n, slot in zip(starts, counts, slots):
                    pos = torch.arange(max(0, start + n - ring), start + n, device=device, dtype=torch.int32)
                    pages[slot, pos.long() % ring] = pos
            else:
                columns = (max(starts) + 8 + 15) // 16
                sizes = (len(counts) * columns, 16, 8, 128)
                k, v = [torch.randn(sizes, device=device).to(torch.float8_e4m3fn) for _ in range(2)]
                pages = torch.randperm(sizes[0], device=device).int().reshape(len(counts), columns)
            out = torch.empty(16, total, 24, 128, device=device)
            lse = out.new_empty((16, total, 24))
            ops.batch_xmx(q, k, v, pages, lens, positions, bias, sink, ints(offsets), ints(slots),
                          out, lse, local, False, max(counts))
            for i, (n, slot) in enumerate(zip(counts, slots)):
                rows = slice(offsets[i], offsets[i + 1])
                one = out.new_empty((16, n, 24, 128))
                norm = lse.new_empty((16, n, 24))
                if local:
                    kv = slice(slot * ring, (slot + 1) * ring)
                    ops.local_xmx(q[rows], k[kv], v[kv], pages[slot:slot + 1], lens[i:i + 1],
                                  positions[rows], bias[rows], sink[rows], one, norm)
                else:
                    ops.paged_xmx(q[rows], k, v, pages[i:i + 1], lens[i:i + 1], positions[rows],
                                  bias[rows], sink[rows], one, norm)
                torch.testing.assert_close(out[:, rows], one, rtol=0, atol=0)
                torch.testing.assert_close(lse[:, rows], norm, rtol=0, atol=0)
            print(f"OK widths={counts} slots={slots} local={local}", flush=True)


if __name__ == "__main__":
    torch.ops.load_library(sys.argv[1])
    check()
