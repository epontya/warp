# Copyright (c) 2024 NVIDIA Corporation. All rights reserved.
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.

"""Memory pool management for efficient allocation and reuse of device memory."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Dict, List, Optional, Tuple


class MemoryBlock:
    """Represents a contiguous block of allocated memory."""

    def __init__(self, ptr: int, size: int, device: str):
        self.ptr = ptr
        self.size = size
        self.device = device
        self.in_use = False

    def __repr__(self) -> str:
        return f"MemoryBlock(ptr={self.ptr:#x}, size={self.size}, device={self.device!r}, in_use={self.in_use})"


class MemoryPool:
    """Thread-safe memory pool that caches freed allocations for reuse.

    Blocks are bucketed by size to allow O(1) lookup of a suitable free block.
    A block is considered suitable if its size is within `tolerance` bytes of
    the requested size.

    Note: I bumped the default tolerance to 256 bytes so that allocations
    differing by small amounts (e.g. off-by-one padding) can still reuse
    cached blocks rather than triggering a fresh allocation every time.
    """

    def __init__(self, device: str, tolerance: int = 256):
        self.device = device
        self.tolerance = tolerance
        self._free: Dict[int, List[MemoryBlock]] = defaultdict(list)
        self._active: Dict[int, MemoryBlock] = {}
        self._lock = Lock()
        self._total_allocated: int = 0
        self._total_freed: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def alloc(self, size: int, allocator_fn) -> int:
        """Return a pointer to *size* bytes, reusing a cached block if possible."""
        with self._lock:
            block = self._pop_free(size)
            if block is None:
                ptr = allocator_fn(size)
                block = MemoryBlock(ptr=ptr, size=size, device=self.device)
                self._total_allocated += size
            block.in_use = True
            self._active[block.ptr] = block
            return block.ptr

    def free(self, ptr: int, free_fn=None) -> None:
        """Return the block identified by *ptr* to the pool."""
        with self._lock:
            block = self._active.pop(ptr, None)
            if block is None:
                raise KeyError(f"Pointer {ptr:#x} is not managed by this pool")
            block.in_use = False
            self._free[block.size].append(block)
            self._total_freed += block.size

    def release_all(self, free_fn) -> None:
        """Release every cached (free) block back to the underlying allocator."""
        with self._lock:
            for blocks in self._free.values():
                for block in blocks:
                    free_fn(block.ptr)
            self._free.clear()

    @property
    def stats(self) -> dict:
        with self._lock:
            cached = sum(len(v) for v in self._free.values())
            return {
                "device": self.device,
                "total_allocated": self._total_allocated,
                "total_freed": self._total_freed,
                "active_blocks": len(self._active),
                "cached_blocks": cached,
            }
