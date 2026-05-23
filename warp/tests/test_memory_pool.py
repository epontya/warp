# Copyright (c) 2024 NVIDIA Corporation. All rights reserved.
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.

"""Unit tests for warp.memory_pool."""

import threading
import unittest
from unittest.mock import MagicMock

from warp.memory_pool import MemoryBlock, MemoryPool


class TestMemoryBlock(unittest.TestCase):
    def test_repr(self):
        block = MemoryBlock(ptr=0xDEADBEEF, size=1024, device="cuda:0")
        r = repr(block)
        self.assertIn("0xdeadbeef", r)
        self.assertIn("1024", r)

    def test_repr_includes_device(self):
        # Personal note: also verify device string appears in repr
        block = MemoryBlock(ptr=0x1000, size=512, device="cuda:1")
        r = repr(block)
        self.assertIn("cuda:1", r)


class TestMemoryPool(unittest.TestCase):
    def _make_pool(self, tolerance: int = 0) -> MemoryPool:
        return MemoryPool(device="cpu", tolerance=tolerance)

    def _counter_allocator(self):
        """Returns a simple allocator that hands out sequential fake pointers."""
        state = {"next": 0x1000}

        def alloc(size: int) -> int:
            ptr = state["next"]
            state["next"] += size
            return ptr

        return alloc

    # ------------------------------------------------------------------

    def test_alloc_calls_allocator_on_first_use(self):
        pool = self._make_pool()
        alloc_fn = self._counter_allocator()
        ptr = pool.alloc(256, alloc_fn)
        self.assertEqual(ptr, 0x1000)
        self.assertEqual(pool.stats["active_blocks"], 1)
        self.assertEqual(pool.stats["total_allocated_bytes"], 256)

    def test_free_then_realloc_reuses_block(self):
        pool = self._make_pool()
        alloc_fn = MagicMock(return_value=0xAAAA)
        ptr1 = pool.alloc(512, alloc_fn)
        pool.free(ptr1)
        ptr2 = pool.alloc(512, alloc_fn)
        self.assertEqual(ptr1, ptr2)
        alloc_fn.assert_called_once()  # underlying allocator called only once

    def test_free_unknown_pointer_raises(self):
        pool = self._make_pool()
        with self.assertRaises(KeyError):
            pool.free(0xBAD)

    def test_tolerance_allows_slightly_larger_block(self):
        pool = self._make_pool(tolerance=64)
        alloc_fn = MagicMock(return_value=0x2000)
        ptr1 = pool.alloc(500, alloc_fn)  # allocates a 500-byte block
        pool.free(ptr1)
        # Request 480 bytes — within tolerance of 500
        ptr2 = pool.alloc(480, alloc_fn)
        self.assertEqual(ptr1, ptr2)
        alloc_fn.assert_called_once()

    def test_tolerance_rejects_block_outside_range(self):
        pool = self._make_pool(tolerance=8)
        counter = self._counter_allocator()
        ptr1 = pool.alloc(500, counter)
        pool.free(ptr1)
        # Request 400 bytes — outside tolerance of 8
        ptr2 = pool.alloc(400, counter)
        self.assertNotEqual(ptr1, ptr2)

    def test_release_all_calls_free_fn(self):
        pool = self._make_pool()
        counter = self._counter_allocator()
        ptr = pool.alloc(128, counter)
        pool.free(ptr)
        free_fn = MagicMock
