"""Guard vLLM prefix hits with the single slot's resident or saved ring history."""


class RingPrefix:
    # Both the convolution and local-attention rings retain this many extra
    # positions beyond their receptive fields. One frontier represents live
    # decode state, the other a snapshot from the last sizeable prompt chunk.
    slack = 32

    def __init__(self, hash_tokens):
        self.hash_tokens = hash_tokens
        self.owner = None
        self.high = 0
        self.snapshot = 0
        self.hashes = ()

    def scheduled(self, owner, end, rows):
        if self.owner != owner:
            self.high = self.snapshot = 0
        self.owner = owner
        self.high = max(self.high, end)
        if rows > 8:
            self.snapshot = end
        self.hashes = ()

    def finish(self, owner, committed, hashes):
        if owner == self.owner:
            self.hashes = tuple(hashes[: min(committed, self.high) // self.hash_tokens])

    def covers(self, length):
        return length > 0 and any(
            end - self.slack <= length <= end for end in (self.high, self.snapshot) if end > 0
        )

    def limit(self, hashes, allowed):
        """Block hashes include the full prefix and the caller's cache salt."""
        available = min(len(self.hashes), len(hashes), allowed // self.hash_tokens)
        for end in (self.high, self.snapshot):
            latest = min(available, end // self.hash_tokens)
            for index in range(latest, 0, -1):
                if index * self.hash_tokens < end - self.slack:
                    break
                if self.hashes[index - 1] == hashes[index - 1]:
                    return index * self.hash_tokens
        return 0
