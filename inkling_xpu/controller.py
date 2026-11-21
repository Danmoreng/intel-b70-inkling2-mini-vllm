"""The cross-pass bank stays fixed for all twelve layers of a repeat."""

import torch

from inkling_trellis.inkling2.fused import rms_norm


def pack_controller_weights(model, layer):
    down = [model.lin[f"layers.{layer}.controllers.{name}.down_weight"] for name in ("attn", "k", "v", "mlp")]
    up = [model.fused[layer, "gf" + name][0] for name in ("attn", "k", "v", "mlp")]
    return (
        torch.cat([w.q for w in down]),
        torch.cat([w.s for w in down]),
        torch.stack([w.q.reshape(2, model.D, 64).transpose(1, 2) for w in up]),
        torch.stack([w.s.reshape(2, model.D, 2).transpose(1, 2) for w in up]),
    )


def pack_cross_queries(main, kv):
    """Pack queries in the consumption order of the four controllers."""
    layers = main.size(0) // 2
    return torch.stack([main[::2], kv[::2], kv[1::2], main[1::2]], dim=1).reshape(layers * 4, -1)


def mix_bank(buf, bank, last, queries):
    raw = torch.stack([*(buf[index] for index in bank), last])
    states = rms_norm(raw.flatten(0, 1)).reshape_as(raw)
    result = torch.ops.inkling2_xpu.bank_mix(states, queries)
    return result.reshape(-1, 4, *last.shape)
