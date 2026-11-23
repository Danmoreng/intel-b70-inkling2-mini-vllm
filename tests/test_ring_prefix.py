"""Scheduler slot ownership and history boundaries without a GPU or vLLM dependency."""

from types import SimpleNamespace
from unittest import TestCase

from inkling_xpu.ring_prefix import RingPrefix, RingSlotPool


class TestRingPrefix(TestCase):
    def setUp(self):
        self.cache = RingPrefix(16)
        self.hashes = list(range(1, 202))
        self.cache.scheduled("a", 2048, 2048)
        for end in range(2052, 2177, 4):
            self.cache.scheduled("a", end, 4)
        self.cache.finish("a", 2174, self.hashes)

    def test_committed_only(self):
        self.assertEqual(self.cache.limit(self.hashes, 3000, 3001), 2160)
        self.assertEqual(self.cache.limit(self.hashes, 2145, 3001), 2144)

    def test_rollback(self):
        self.assertEqual(self.cache.limit(self.hashes[:128] + [-1] * 20, 3000, 3001), 2048)
        self.assertEqual(self.cache.limit(self.hashes, 2015, 3001), 0)
        self.assertFalse(self.cache.covers(2128))
        self.assertTrue(self.cache.covers(2016))

    def test_changed_namespace(self):
        self.assertEqual(self.cache.limit([-1] * 200, 3000, 3001), 0)
        self.cache.scheduled("b", 32, 32)
        self.assertEqual(self.cache.limit(self.hashes, 3000, 3001), 0)
        self.cache.finish("a", 2048, self.hashes)
        self.assertEqual(self.cache.limit(self.hashes, 3000, 3001), 0)

    def test_aborted_request(self):
        self.cache.finish("a", 0, self.hashes)
        self.assertEqual(self.cache.limit(self.hashes, 3000, 3001), 0)

    def test_repeated_resends_across_page_boundaries(self):
        for remainder in range(16):
            prompt = 74688 + remainder
            hashes = list(range((prompt + 1040) // 16))
            cache = RingPrefix(16)
            for owner in range(4):
                hit = cache.limit(hashes, prompt - 1, prompt)
                if owner:
                    self.assertTrue(prompt - cache.slack <= hit <= prompt - cache.min_snapshot_rows)
                else:
                    self.assertEqual(hit, 0)
                cache.scheduled(owner, prompt, prompt - hit)
                for end in range(prompt + 8, prompt + 1025, 8):
                    cache.scheduled(owner, end, 8)
                cache.finish(owner, prompt + 1024, hashes)

    def test_short_branch_recomputes_divergent_tail(self):
        # Only the shared blocks can be reused when an identical saved prefix forks.
        old = self.hashes[:127] + [-1] * 32
        self.assertEqual(self.cache.limit(old, 2048, 2049), 2032)
        self.assertEqual(self.cache.limit(self.hashes, 2048, 2112), 2048)
        self.assertEqual(self.cache.limit(self.hashes, 7, 8), 0)

    def test_tiny_final_chunk_after_an_aligned_checkpoint(self):
        hashes = list(range(4700))
        for rest in range(1, 9):
            cache = RingPrefix(16)
            cache.scheduled("cold", 73728, 6144)
            cache.scheduled("cold", 73728 + rest, rest)
            for i, owner in enumerate(("cold", "warm1", "warm2")):
                cache.scheduled(owner, 73800, 8)
                cache.finish(owner, 73800, hashes)
                self.assertEqual(cache.limit(hashes, 73728 + rest - 1, 73728 + rest), 73728)
                self.assertEqual(cache.limit([-1] * len(hashes), 73735, 73728 + rest), 0)
                cache.scheduled(f"warm{i + 1}", 73728 + rest, rest)


class TestRingSlotPool(TestCase):
    def test_active_slot_exclusion_and_salted_reuse(self):
        pool = RingSlotPool(16, 2)
        a = SimpleNamespace(request_id="a", num_tokens=1033, block_hashes=list(range(65)))
        b = SimpleNamespace(request_id="b", num_tokens=1033, block_hashes=list(range(65, 130)))
        for req in (a, b):
            pool.scheduled(req, 1033, 1033, 0)
        self.assertNotEqual(pool.active["a"], pool.active["b"])
        self.assertEqual(pool.limit(a.block_hashes, 1032, 1033), 0)
        slot = pool.active["a"]
        pool.finish("a", 1033, a.block_hashes)
        hit = pool.limit(a.block_hashes, 1032, 1033)
        self.assertEqual(hit, 1024)
        self.assertEqual(pool.limit(b.block_hashes, 1032, 1033), 0)
        a.request_id = "resend"
        pool.scheduled(a, 1033, 9, hit)
        self.assertEqual(pool.active["resend"], slot)
        pool.finish("resend", 0, a.block_hashes)
        self.assertEqual(pool.limit(a.block_hashes, 1032, 1033), 0)
        pool.finish("b", 1033, b.block_hashes)
        self.assertEqual(pool.limit(b.block_hashes, 1032, 1033), 1024)
        pool.reset()
        self.assertEqual(pool.limit(b.block_hashes, 1032, 1033), 0)
