"""
Tests for profiling utilities
"""

import pytest
import time
import numpy as np
from mlagents.trainers.utils.profiling import (
    PerformanceMonitor,
    profile_block,
    profile_function,
    get_global_monitor,
    reset_global_monitor,
    PSUTIL_AVAILABLE,
)


class TestPerformanceMonitor:
    """Test PerformanceMonitor class"""

    def test_monitor_creation(self):
        """Test that monitor can be created"""
        monitor = PerformanceMonitor()
        assert monitor is not None
        assert isinstance(monitor.metrics, dict)
        assert len(monitor.metrics) > 0  # Should have initialized metric keys

    def test_record_metric(self):
        """Test recording metrics"""
        monitor = PerformanceMonitor()

        monitor.record_metric("test_metric", 1.5)
        monitor.record_metric("test_metric", 2.0)

        assert "test_metric" in monitor.metrics
        assert len(monitor.metrics["test_metric"]) == 2
        assert monitor.metrics["test_metric"][0] == 1.5
        assert monitor.metrics["test_metric"][1] == 2.0

    def test_get_summary(self):
        """Test summary statistics"""
        monitor = PerformanceMonitor()

        # Add some test data
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for val in values:
            monitor.record_metric("test_metric", val)

        summary = monitor.get_summary()

        assert "test_metric" in summary
        stats = summary["test_metric"]

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "p50" in stats
        assert "p95" in stats
        assert "p99" in stats

        assert stats["mean"] == pytest.approx(3.0)
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["p50"] == pytest.approx(3.0)

    def test_summary_with_empty_metrics(self):
        """Test summary with no data"""
        monitor = PerformanceMonitor()
        summary = monitor.get_summary()

        assert isinstance(summary, dict)
        assert len(summary) == 0

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_check_memory_usage(self):
        """Test memory usage monitoring"""
        monitor = PerformanceMonitor()
        memory_mb = monitor.check_memory_usage()

        assert isinstance(memory_mb, float)
        assert memory_mb > 0

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_check_cpu_usage(self):
        """Test CPU usage monitoring"""
        monitor = PerformanceMonitor()
        cpu_percent = monitor.check_cpu_usage()

        assert isinstance(cpu_percent, float)
        assert cpu_percent >= 0


class TestProfileBlock:
    """Test profile_block context manager"""

    def test_profile_block_basic(self):
        """Test basic profiling of code block"""
        monitor = PerformanceMonitor()

        with profile_block(monitor, "test_operation"):
            time.sleep(0.01)  # Sleep for 10ms

        assert "test_operation" in monitor.metrics
        assert len(monitor.metrics["test_operation"]) == 1

        # Should be approximately 10ms (allow some variance)
        elapsed = monitor.metrics["test_operation"][0]
        assert elapsed >= 0.01
        assert elapsed < 0.1  # Should be much less than 100ms

    def test_profile_block_multiple_calls(self):
        """Test profiling same block multiple times"""
        monitor = PerformanceMonitor()

        for _ in range(3):
            with profile_block(monitor, "repeated_operation"):
                time.sleep(0.005)

        assert len(monitor.metrics["repeated_operation"]) == 3

        # All measurements should be similar
        times = monitor.metrics["repeated_operation"]
        for t in times:
            assert t >= 0.005
            assert t < 0.05

    def test_profile_block_with_exception(self):
        """Test that profiling still records time even with exception"""
        monitor = PerformanceMonitor()

        try:
            with profile_block(monitor, "failing_operation"):
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Time should still be recorded
        assert "failing_operation" in monitor.metrics
        assert len(monitor.metrics["failing_operation"]) == 1


class TestProfileFunction:
    """Test profile_function decorator"""

    def test_profile_function_decorator(self):
        """Test function profiling decorator"""

        @profile_function()
        def slow_function():
            time.sleep(0.01)
            return 42

        result = slow_function()

        assert result == 42

    def test_profile_function_with_custom_name(self):
        """Test decorator with custom metric name"""

        @profile_function(metric_name="custom_metric")
        def test_function():
            return "test"

        result = test_function()
        assert result == "test"

    def test_profile_function_preserves_signature(self):
        """Test that decorator preserves function signature"""

        @profile_function()
        def function_with_args(x, y, z=10):
            return x + y + z

        result = function_with_args(1, 2, z=3)
        assert result == 6

        result = function_with_args(1, 2)
        assert result == 13


class TestGlobalMonitor:
    """Test global monitor functionality"""

    def test_get_global_monitor(self):
        """Test getting global monitor instance"""
        reset_global_monitor()

        monitor1 = get_global_monitor()
        monitor2 = get_global_monitor()

        # Should return same instance
        assert monitor1 is monitor2

    def test_reset_global_monitor(self):
        """Test resetting global monitor"""
        monitor1 = get_global_monitor()
        monitor1.record_metric("test", 1.0)

        reset_global_monitor()
        monitor2 = get_global_monitor()

        # Should be a new instance
        assert monitor2 is not monitor1
        # Metrics dict is initialized with keys but empty lists
        for metric_list in monitor2.metrics.values():
            assert len(metric_list) == 0


class TestIntegration:
    """Integration tests for profiling"""

    def test_profile_numpy_operations(self):
        """Test profiling numpy operations"""
        monitor = PerformanceMonitor()

        with profile_block(monitor, "numpy_matmul"):
            a = np.random.randn(100, 100)
            b = np.random.randn(100, 100)
            c = np.matmul(a, b)

        assert "numpy_matmul" in monitor.metrics
        assert len(monitor.metrics["numpy_matmul"]) == 1

    def test_multiple_metrics(self):
        """Test tracking multiple different metrics"""
        monitor = PerformanceMonitor()

        with profile_block(monitor, "operation_a"):
            time.sleep(0.01)

        with profile_block(monitor, "operation_b"):
            time.sleep(0.02)

        summary = monitor.get_summary()

        assert "operation_a" in summary
        assert "operation_b" in summary

        # operation_b should take longer
        assert summary["operation_b"]["mean"] > summary["operation_a"]["mean"]

    def test_nested_profiling(self):
        """Test nested profiling blocks"""
        monitor = PerformanceMonitor()

        with profile_block(monitor, "outer"):
            time.sleep(0.01)
            with profile_block(monitor, "inner"):
                time.sleep(0.01)
            time.sleep(0.01)

        assert "outer" in monitor.metrics
        assert "inner" in monitor.metrics

        # Outer should take longer than inner
        outer_time = monitor.metrics["outer"][0]
        inner_time = monitor.metrics["inner"][0]
        assert outer_time > inner_time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
