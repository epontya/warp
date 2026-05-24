"""Integration tests for MemoryPool and MemoryBlock working together.

These tests verify that the memory pool correctly manages blocks across
multiple allocations, deallocations, and reuse cycles.

Personal note: Added a docstring to TestMemoryPoolEdgeCases and completed
the truncated class definition found in the original file.
"""

import unittest

from warp.memory_pool import MemoryBlock, MemoryPool


class TestMemoryPoolBlockLifecycle(unittest.TestCase):
    """Tests for the full lifecycle of blocks within a pool."""

    def setUp(self):
        self.pool = MemoryPool(block_size=64, device="cpu")

    def test_allocate_returns_memory_block(self):
        block = self.pool.allocate()
        self.assertIsInstance(block, MemoryBlock)

    def test_allocate_block_has_correct_size(self):
        block = self.pool.allocate()
        self.assertEqual(block.size, 64)

    def test_allocate_block_has_correct_device(self):
        block = self.pool.allocate()
        self.assertEqual(block.device, "cpu")

    def test_free_block_returns_to_pool(self):
        block = self.pool.allocate()
        initial_free = self.pool.free_count()
        self.pool.free(block)
        self.assertEqual(self.pool.free_count(), initial_free + 1)

    def test_reallocate_reuses_freed_block(self):
        block_a = self.pool.allocate()
        block_a_id = id(block_a)
        self.pool.free(block_a)

        block_b = self.pool.allocate()
        # The pool should reuse the freed block rather than creating a new one
        self.assertEqual(id(block_b), block_a_id)

    def test_multiple_allocations_are_unique(self):
        blocks = [self.pool.allocate() for _ in range(5)]
        block_ids = [id(b) for b in blocks]
        self.assertEqual(len(set(block_ids)), 5)

    def test_free_all_then_reallocate(self):
        blocks = [self.pool.allocate() for _ in range(4)]
        for block in blocks:
            self.pool.free(block)

        self.assertEqual(self.pool.free_count(), 4)

        new_blocks = [self.pool.allocate() for _ in range(4)]
        self.assertEqual(len(new_blocks), 4)
        self.assertEqual(self.pool.free_count(), 0)


class TestMemoryPoolCapacity(unittest.TestCase):
    """Tests for pool capacity and growth behavior."""

    def test_pool_grows_when_exhausted(self):
        pool = MemoryPool(block_size=32, device="cpu", capacity=2)
        # Allocate beyond initial capacity — pool should grow
        blocks = [pool.allocate() for _ in range(5)]
        self.assertEqual(len(blocks), 5)

    def test_total_count_reflects_growth(self):
        pool = MemoryPool(block_size=32, device="cpu", capacity=2)
        blocks = [pool.allocate() for _ in range(5)]
        self.assertGreaterEqual(pool.total_count(), 5)
        # Suppress unused variable warning
        _ = blocks

    def test_free_count_zero_when_all_allocated(self):
        pool = MemoryPool(block_size=32, device="cpu", capacity=3)
        blocks = [pool.allocate() for _ in range(3)]
        self.assertEqual(pool.free_count(), 0)
        _ = blocks


class TestMemoryPoolEdgeCases(unittest.TestCase):
    """Edge case and boundary tests for MemoryPool behavior."""

    def test_free_count_does_not_exceed_total_count(self):
        pool = MemoryPool(block_size=16, device="cpu", capacity=4)
        blocks = [pool.allocate() for _ in range(4)]
        for block in blocks:
            pool.free(block)
        self.assertLessEqual(pool.free_count(), pool.total_count())

    def test_allocate_after_full_free_cycle(self):
        pool = MemoryPool(block_size=16, device="cpu", capacity=2)
        block = pool.allocate()
        pool.free(block)
        # Should be able to allocate again without error
        new_block = pool.allocate()
        self.assertIsInstance(new_block, MemoryBlock)


if __name__ == "__main__":
    unittest.main()
