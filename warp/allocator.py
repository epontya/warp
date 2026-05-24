"""Allocator module for warp memory management.

Provides a high-level allocator interface that uses MemoryPool internally
to manage device memory blocks with optional tracking support.
"""

from __future__ import annotations

from typing import Optional

from warp.memory_pool import MemoryBlock, MemoryPool


class Allocator:
    """Allocates memory blocks from a device-specific memory pool.

    Wraps a MemoryPool to provide a simple alloc/free interface.
    Optionally integrates with a ScopedMemoryTracker for leak detection.

    Args:
        device: The device identifier string (e.g. ``"cpu"``, ``"cuda:0"``).
        pool: An existing MemoryPool to use. If None, a new pool is created.
    """

    def __init__(self, device: str, pool: Optional[MemoryPool] = None) -> None:
        self.device = device
        self._pool = pool if pool is not None else MemoryPool(device)
        self._active_blocks: dict[int, MemoryBlock] = {}

    def alloc(self, size: int) -> MemoryBlock:
        """Allocate a memory block of the given size.

        Args:
            size: Number of bytes to allocate. Must be greater than zero.

        Returns:
            A :class:`~warp.memory_pool.MemoryBlock` representing the
            allocated region.

        Raises:
            ValueError: If *size* is not positive.
        """
        if size <= 0:
            raise ValueError(f"Allocation size must be positive, got {size}")

        block = self._pool.allocate(size)
        self._active_blocks[id(block)] = block
        return block

    def free(self, block: MemoryBlock) -> None:
        """Return a previously allocated block to the pool.

        Args:
            block: The block to free. Must have been allocated by this
                allocator instance.

        Raises:
            KeyError: If *block* was not allocated by this allocator.
        """
        key = id(block)
        if key not in self._active_blocks:
            raise KeyError(
                f"Block {block!r} was not allocated by this allocator or has already been freed."
            )
        del self._active_blocks[key]
        self._pool.free(block)

    def free_all(self) -> int:
        """Free all currently active blocks tracked by this allocator.

        Useful for bulk cleanup without needing to track individual blocks.

        Returns:
            The number of blocks that were freed.
        """
        # Snapshot the keys so we're not modifying the dict while iterating
        blocks = list(self._active_blocks.values())
        for block in blocks:
            self.free(block)
        return len(blocks)

    @property
    def active_count(self) -> int:
        """Number of blocks currently allocated and not yet freed."""
        return len(self._active_blocks)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Allocator(device={self.device!r}, active_blocks={self.active_count})"
        )


class ScopedMemoryTracker:
    """Context manager that tracks allocations made within a scope.

    Any blocks allocated through the managed :class:`Allocator` while the
    context is active are freed automatically on exit, making it easy to
    write leak-free resource scopes.

    Example::

        allocator = Allocator("cpu")
        with ScopedMemoryTracker(allocator) as tracker:
            block = allocator.alloc(1024)
            # block is freed automatically when the context exits

    Args:
        allocator:
