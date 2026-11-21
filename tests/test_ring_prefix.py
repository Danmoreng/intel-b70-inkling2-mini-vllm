"""Single-sequence scheduler history boundaries without a GPU or vLLM dependency."""

from unittest import TestCase

from inkling_xpu.ring_prefix import RingPrefix


class TestRingPrefix(TestCase):
    def setUp(self):
        self.cache = RingPrefix(16)
        self.hashes = list(range(1, 202))
        self.cache.scheduled("a", 2048, 2048)
        for end in range(2052, 2177, 4):
            self.cache.scheduled("a", end, 4)
        self.cache.finish("a", 2174, self.hashes)

    def test_committed_only(self):
        self.assertEqual(self.cache.limit(self.hashes, 3000), 2160)
        self.assertEqual(self.cache.limit(self.hashes, 2145), 2144)

    def test_rollback(self):
        self.assertEqual(self.cache.limit(self.hashes[:128] + [-1] * 20, 3000), 2048)
        self.assertEqual(self.cache.limit(self.hashes, 2015), 0)
        self.assertFalse(self.cache.covers(2128))
        self.assertTrue(self.cache.covers(2016))

    def test_changed_namespace(self):
        self.assertEqual(self.cache.limit([-1] * 200, 3000), 0)
        self.cache.scheduled("b", 32, 32)
        self.assertEqual(self.cache.limit(self.hashes, 3000), 0)
        self.cache.finish("a", 2048, self.hashes)
        self.assertEqual(self.cache.limit(self.hashes, 3000), 0)

    def test_aborted_request(self):
        self.cache.finish("a", 0, self.hashes)
        self.assertEqual(self.cache.limit(self.hashes, 3000), 0)
