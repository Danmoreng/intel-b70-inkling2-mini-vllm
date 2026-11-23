"""Exercise captured hidden/KV tail replay independently of vLLM or an accelerator."""
from unittest import TestCase

import torch

from inkling_xpu.prefix_tail import PrefillTail


class TestPrefillTail(TestCase):
    def test_exact_suffix_after_page_reassignment(self):
        for dtype in (torch.uint8, torch.bfloat16):
            # Logical page order differs between prefill and resend. Each head has padding.
            raw = torch.arange(6 * 3 * 8 * 10).reshape(6, 3, 8, 10).to(dtype)
            cache = raw[:, :, :, :6]
            ids = torch.arange(6)
            slots = torch.tensor([22, 23, 0, 1, 2, 3])
            hidden = torch.arange(24).reshape(6, 4)
            draft = torch.arange(2 * 140).reshape(2, 140, 1).to(dtype)
            pages, offsets = slots // 8, slots % 8
            original = cache[pages, :, offsets].clone()
            original_draft = draft[:, 104:110].clone()
            expected_hidden = hidden.clone()
            saved = PrefillTail.capture(ids, hidden, slots, 8, 110, [cache], [draft])
            hidden.zero_()
            cache.zero_()
            draft.zero_()
            ids.fill_(-1)
            dst = torch.tensor([35, 36, 37])
            args = (dst, 8, 110, [cache], [draft])
            self.assertIsNone(saved.restore(torch.tensor([3, 4, 6]), *args))
            self.assertIsNone(saved.restore(torch.arange(7), *args))
            self.assertFalse(cache.count_nonzero())
            self.assertFalse(draft.count_nonzero())
            got = saved.restore(torch.tensor([3, 4, 5]), *args)
            self.assertTrue(torch.equal(got, expected_hidden[-3:]))
            want = torch.zeros_like(cache)
            want[dst // 8, :, dst % 8] = original[-3:]
            self.assertTrue(torch.equal(cache, want))
            self.assertTrue(torch.equal(draft[:, 107:110], original_draft[:, -3:]))
            self.assertFalse(draft[:, :107].count_nonzero())
            self.assertFalse(draft[:, 110:].count_nonzero())
