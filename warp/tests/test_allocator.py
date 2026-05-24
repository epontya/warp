"""Unit tests for the Allocator class in warp/allocator.py."""

import unittest
from unittest.mock import MagicMock, patch, call

from warp.allocator import Allocator
from warp.memory_pool import MemoryBlock, MemoryPool


class TestAllocatorInit(unittest.TestCase):
    """Tests for Allocator.__init__."""

    def test_init_creates_pool(self):
        """Allocator should create an internal MemoryPool on init."""
        allocator = Allocator(device="cpu")
        self.assertIsInstance(allocator.pool, MemoryPool)

    def test_init_stores_device(self):
        """Allocator should store the device string."""
        allocator = Allocator(device="cuda:0")
        self.assertEqual(allocator.device, "cuda:0")

    def test_init_default_device(self):
        """Allocator should default to 'cpu' if no device is specified."""
        allocator = Allocator()
        self.assertEqual(allocator.device, "cpu")


class TestAllocatorAlloc(unittest.TestCase):
    """Tests for Allocator.alloc."""

    def setUp(self):
        self.allocator = Allocator(device="cpu")

    def test_alloc_returns_memory_block(self):
        """alloc() should return a MemoryBlock instance."""
        block = self.allocator.alloc(64)
        self.assertIsInstance(block, MemoryBlock)

    def test_alloc_block_has_requested_size(self):
        """Returned MemoryBlock should have the requested size."""
        block = self.allocator.alloc(128)
        self.assertEqual(block.size, 128)

    def test_alloc_block_has_correct_device(self):
        """Returned MemoryBlock should carry the allocator's device."""
        allocator = Allocator(device="cuda:1")
        block = allocator.alloc(32)
        self.assertEqual(block.device, "cuda:1")

    def test_alloc_zero_size_raises(self):
        """Allocating zero bytes should raise a ValueError."""
        with self.assertRaises(ValueError):
            self.allocator.alloc(0)

    def test_alloc_negative_size_raises(self):
        """Allocating a negative number of bytes should raise a ValueError."""
        with self.assertRaises(ValueError):
            self.allocator.alloc(-1)

    def test_alloc_adds_block_to_pool(self):
        """alloc() should register the block with the internal pool."""
        block = self.allocator.alloc(64)
        self.assertIn(block, self.allocator.pool.blocks)


class TestAllocatorFree(unittest.TestCase):
    """Tests for Allocator.free."""

    def setUp(self):
        self.allocator = Allocator(device="cpu")

    def test_free_removes_block_from_pool(self):
        """free() should remove the block from the pool."""
        block = self.allocator.alloc(64)
        self.allocator.free(block)
        self.assertNotIn(block, self.allocator.pool.blocks)

    def test_free_unknown_block_raises(self):
        """Freeing a block not owned by this allocator should raise a KeyError."""
        other_allocator = Allocator(device="cpu")
        foreign_block = other_allocator.alloc(64)
        with self.assertRaises(KeyError):
            self.allocator.free(foreign_block)


class TestAllocatorFreeAll(unittest.TestCase):
    """Tests for Allocator.free_all."""

    def setUp(self):
        self.allocator = Allocator(device="cpu")

    def test_free_all_clears_pool(self):
        """free_all() should leave the pool empty."""
        self.allocator.alloc(32)
        self.allocator.alloc(64)
        self.allocator.alloc(128)
        self.allocator.free_all()
        self.assertEqual(len(self.allocator.pool.blocks), 0)

    def test_free_all_on_empty_pool_is_safe(self):
        """Calling free_all() on an already-empty pool should not raise."""
        try:
            self.allocator.free_all()
        except Exception as exc:  # pragma: no cover
            self.fail(f"free_all() raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
