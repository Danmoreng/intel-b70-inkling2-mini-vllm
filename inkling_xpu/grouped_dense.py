"""Project the three attention inputs in one native matrix-engine dispatch."""

import torch
import torch.nn.functional as F


class GroupedQKV:
    def __init__(self, projections, splits):
        weights = []
        scales = []
        self.widths = []
        self.bounds = [0]
        self.splits = splits
        for projection in projections:
            if projection.alpha is not None or projection.K != 2304:
                raise ValueError("The grouped QKV kernel requires group-32 INT8 target projections")
            n, k = projection.N, projection.K
            pad = -n % 16
            q = F.pad(projection.q[0], (0, 0, 0, pad))
            s = F.pad(projection.s[0], (0, 0, 0, pad))
            weights.append(q.reshape(-1, 16, k // 32, 8, 4).permute(0, 2, 3, 1, 4))
            scales.append(s.reshape(-1, 16, k // 32).transpose(1, 2))
            self.widths.append(n)
            self.bounds.append(self.bounds[-1] + n + pad)
        self.weights = torch.cat(weights).contiguous()
        self.scales = torch.cat(scales).contiguous()

    def __call__(self, query, key, value):
        if query.shape[0] > 8:
            outputs = [self.project(q, k_in, v_in) for q, k_in, v_in in
                       zip(query.split(8), key.split(8), value.split(8), strict=True)]
            out = torch.cat(outputs)
        else:
            out = self.project(query, key, value)
        return [out.narrow(1, offset, width) for offset, width in zip(self.bounds, self.widths)]

    def project(self, query, key, value):
        _, k, v, n = self.bounds
        return torch.ops.inkling_qkv.dense(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            self.weights,
            self.scales,
            n,
            k,
            v,
            self.splits,
            True,
        )
