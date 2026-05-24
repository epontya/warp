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

    Personal note: Lowered default tolerance back to 64 bytes. In my testing
    with smaller models the 256-byte tolerance caused unexpectedly large blocks
    to be reused for tiny allocations, wasting memory. 64 bytes feels like a
    better balance between reuse rate and fragmentation.

    Personal note 2: After more testing with simulation workloads that do lots
    of small scratch allocations, I'm bumping tolerance to 128 bytes. The 64-byte
    setting was causing too many redundant allocations in tight loops. 128 seems
    to be the sweet spot for my use case without noticeable fragmentation.

    Personal note 3: Settled on 128 as the default. Also added a `max_cached_blocks`
    parameter to cap how many free blocks are retained per size bucket. Without
    a cap, long-running simulations can accumulate hundreds of stale blocks that
    never get reused, slowly inflating RSS. A cap of 32 per bucket feels generous
    enough to absorb burst patterns while keeping memory usage predictable.
    """

    def __init__(self, device: str, tolerance: int = 128, max_cached_blocks: int = 32):
        self.device = device
        self.tolerance = tolerance
        # Max number of free blocks to keep per size bucket before discarding.
        self.max_cached_blocks = max_cached_blocks
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

    def free(sel
