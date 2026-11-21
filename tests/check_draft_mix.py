"""Compare the native DSpark tap mixture and RMS norms with the original tensor math."""
import os

import torch


def main():
    torch.ops.load_library(os.environ["INKLING2_CONTROL_LIB"])
    torch.manual_seed(416)
    for rows in (1, 2, 3, 4, 8):
        taps = [torch.randn(rows, 2304, device="xpu") for _ in range(3)]
        taps.append(torch.randn(rows, 2304, device="xpu", dtype=torch.bfloat16))
        weights = torch.randn(4, 4, device="xpu").softmax(-1).bfloat16().float()
        norms = torch.randn(4, 2304, device="xpu").bfloat16().float()
        sources = torch.stack(taps, dim=1).float()
        expected = []
        for mix, scale in zip(weights, norms):
            x = (sources * mix[:, None]).sum(1)
            expected.append((x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * scale).bfloat16())
        reference = torch.stack(expected).float()
        actual = torch.ops.inkling2_xpu.draft_mix_norm(*taps, weights, norms).float()
        relative_error = ((actual - reference).norm() / reference.norm()).item()
        print(f"DSpark tap mixing, {rows} rows, relative L2 {relative_error:.6g}")
        assert relative_error < 0.0005


if __name__ == "__main__":
    main()
