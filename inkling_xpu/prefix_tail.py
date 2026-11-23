"""Exact outputs and cache rows for the final tokens of a saved text prefill."""

from dataclasses import dataclass

import torch


@dataclass
class PrefillTail:
    token_ids: torch.Tensor
    hidden: torch.Tensor
    target_kv: list[torch.Tensor]
    draft_kv: list[torch.Tensor]

    @classmethod
    def capture(cls, ids, hidden, slots, page_size, end, target_caches, draft_caches):
        n = ids.numel()
        pages, offsets = slots // page_size, slots % page_size
        # Retain values independently of vLLM's block allocator. FP8 and BF16
        # are copied byte-for-byte, including for noncontiguous page assignments.
        target_kv = [cache.view(torch.uint8)[pages, :, offsets] for cache in target_caches]
        draft_kv = [cache[:, end - n:end].clone() for cache in draft_caches]
        return cls(ids.clone(), hidden.clone(), target_kv, draft_kv)

    def restore(self, ids, slots, page_size, end, target_caches, draft_caches):
        n = ids.numel()
        if n > self.token_ids.numel() or not torch.equal(ids, self.token_ids[-n:]):
            return None
        pages, offsets = slots // page_size, slots % page_size
        for cache, rows in zip(target_caches, self.target_kv, strict=True):
            cache.view(torch.uint8)[pages, :, offsets] = rows[-n:]
        for cache, rows in zip(draft_caches, self.draft_kv, strict=True):
            cache[:, end - n:end].copy_(rows[:, -n:])
        return self.hidden[-n:]
