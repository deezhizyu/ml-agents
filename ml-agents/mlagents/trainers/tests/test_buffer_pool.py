"""Tests for AgentBufferPool."""

import pytest
import threading
from mlagents.trainers.buffer import (
    AgentBuffer,
    AgentBufferPool,
    BufferKey,
    get_global_buffer_pool,
    set_global_buffer_pool,
)
import numpy as np


class TestAgentBufferPool:
    """Tests for the AgentBufferPool class."""

    def test_acquire_creates_new_buffer_when_empty(self):
        """Test that acquire() creates a new buffer when pool is empty."""
        pool = AgentBufferPool(pool_size=4)
        buffer = pool.acquire()
        assert buffer is not None
        assert isinstance(buffer, AgentBuffer)
        assert pool.pool_size == 0

    def test_release_adds_buffer_to_pool(self):
        """Test that release() adds buffer back to pool."""
        pool = AgentBufferPool(pool_size=4)
        buffer = pool.acquire()
        pool.release(buffer)
        assert pool.pool_size == 1

    def test_acquire_reuses_released_buffer(self):
        """Test that acquire() reuses a released buffer."""
        pool = AgentBufferPool(pool_size=4)
        buffer1 = pool.acquire()
        pool.release(buffer1)
        buffer2 = pool.acquire()
        # Should be the same object reused
        assert buffer1 is buffer2

    def test_released_buffer_is_reset(self):
        """Test that released buffers are reset before reuse."""
        pool = AgentBufferPool(pool_size=4)
        buffer = pool.acquire()
        # Add some data
        buffer[BufferKey.CONTINUOUS_ACTION].append(np.array([1.0, 2.0]))
        assert buffer.num_experiences == 1

        pool.release(buffer)
        buffer2 = pool.acquire()
        # Buffer should be reset
        assert buffer2.num_experiences == 0

    def test_pool_respects_max_size(self):
        """Test that pool doesn't exceed max size."""
        pool = AgentBufferPool(pool_size=2)
        buffers = [pool.acquire() for _ in range(5)]

        for buf in buffers:
            pool.release(buf)

        # Pool should only hold 2 buffers
        assert pool.pool_size == 2

    def test_release_handles_none(self):
        """Test that release() handles None gracefully."""
        pool = AgentBufferPool(pool_size=4)
        pool.release(None)  # Should not raise
        assert pool.pool_size == 0

    def test_clear_empties_pool(self):
        """Test that clear() removes all buffers from pool."""
        pool = AgentBufferPool(pool_size=4)
        buffers = [pool.acquire() for _ in range(3)]
        for buf in buffers:
            pool.release(buf)
        assert pool.pool_size == 3

        pool.clear()
        assert pool.pool_size == 0

    def test_stats_tracking(self):
        """Test that pool tracks usage statistics."""
        pool = AgentBufferPool(pool_size=4)

        # Acquire 3 new buffers
        buffers = [pool.acquire() for _ in range(3)]
        stats = pool.stats
        assert stats["acquired_count"] == 3
        assert stats["created_count"] == 3
        assert stats["reuse_rate"] == 0.0

        # Release and reacquire
        for buf in buffers:
            pool.release(buf)

        _ = [pool.acquire() for _ in range(3)]
        stats = pool.stats
        assert stats["acquired_count"] == 6
        assert stats["created_count"] == 3
        assert stats["reuse_rate"] == 0.5  # 3 reused out of 6

    def test_thread_safety(self):
        """Test that pool is thread-safe."""
        pool = AgentBufferPool(pool_size=100)
        num_threads = 10
        iterations_per_thread = 100
        errors = []

        def worker():
            try:
                for _ in range(iterations_per_thread):
                    buffer = pool.acquire()
                    # Simulate some work
                    buffer[BufferKey.CONTINUOUS_ACTION].append(np.array([1.0]))
                    pool.release(buffer)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"


class TestGlobalBufferPool:
    """Tests for the global buffer pool functions."""

    def setup_method(self):
        """Reset global pool before each test."""
        set_global_buffer_pool(None)

    def teardown_method(self):
        """Reset global pool after each test."""
        set_global_buffer_pool(None)

    def test_get_global_pool_creates_singleton(self):
        """Test that get_global_buffer_pool creates a singleton."""
        pool1 = get_global_buffer_pool()
        pool2 = get_global_buffer_pool()
        assert pool1 is pool2

    def test_set_global_pool(self):
        """Test that set_global_buffer_pool replaces the global pool."""
        custom_pool = AgentBufferPool(pool_size=8)
        set_global_buffer_pool(custom_pool)
        assert get_global_buffer_pool() is custom_pool

    def test_set_global_pool_to_none_resets(self):
        """Test that setting pool to None allows new pool creation."""
        pool1 = get_global_buffer_pool()
        set_global_buffer_pool(None)
        pool2 = get_global_buffer_pool()
        assert pool1 is not pool2
