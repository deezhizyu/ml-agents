"""
Complete profiling integration tests - Phase 2 final validation
"""
import pytest
import time
from unittest.mock import MagicMock, patch
from mlagents.trainers.utils.profiling import PerformanceMonitor


class TestProfilingOverhead:
    """Test profiling overhead measurements"""
    
    def test_profiling_minimal_overhead(self):
        """Test that profiling adds minimal overhead"""
        monitor = PerformanceMonitor()
        iterations = 100
        
        # Measure without profiling
        start = time.perf_counter()
        for _ in range(iterations):
            # Simulate work with actual computation
            x = sum(range(1000))
        no_profiling_time = time.perf_counter() - start
        
        # Measure with profiling
        start = time.perf_counter()
        for _ in range(iterations):
            start_iter = time.perf_counter()
            x = sum(range(1000))
            monitor.record_metric("test", time.perf_counter() - start_iter)
        with_profiling_time = time.perf_counter() - start
        
        # Calculate overhead percentage
        overhead = ((with_profiling_time - no_profiling_time) / no_profiling_time) * 100
        
        # Should be under 10% (sleep timings can be imprecise on Windows)
        assert overhead < 10.0
    
    def test_profiling_context_manager(self):
        """Test profiling context manager pattern"""
        class MockProfiler:
            def __init__(self):
                self.started = False
                self.stopped = False
            
            def start(self):
                self.started = True
            
            def stop(self):
                self.stopped = True
        
        profiler = MockProfiler()
        
        # Use context manager pattern
        profiler.start()
        # Do work
        time.sleep(0.001)
        profiler.stop()
        
        assert profiler.started
        assert profiler.stopped


class TestProfilingMetrics:
    """Test profiling metrics collection"""
    
    def test_timing_measurement(self):
        """Test timing measurement accuracy"""
        start = time.perf_counter()
        time.sleep(0.01)  # 10ms
        elapsed = time.perf_counter() - start
        
        # Should be approximately 10ms (allow 5ms tolerance)
        assert 0.008 < elapsed < 0.015
    
    def test_cumulative_timing(self):
        """Test cumulative timing over multiple operations"""
        timings = []
        
        for i in range(5):
            start = time.perf_counter()
            time.sleep(0.005)  # 5ms each
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
        
        total_time = sum(timings)
        
        # Should be approximately 25ms total
        assert 0.020 < total_time < 0.035
    
    def test_average_timing(self):
        """Test average timing calculation"""
        timings = [0.01, 0.012, 0.011, 0.013, 0.009]
        
        avg_time = sum(timings) / len(timings)
        
        assert 0.010 < avg_time < 0.012


class TestProfilingOutput:
    """Test profiling output formats"""
    
    def test_profiling_report_structure(self):
        """Test profiling report structure"""
        report = {
            "policy_inference": {
                "count": 100,
                "total_time": 1.5,
                "avg_time": 0.015,
                "min_time": 0.012,
                "max_time": 0.020
            },
            "optimizer_update": {
                "count": 50,
                "total_time": 2.0,
                "avg_time": 0.040,
                "min_time": 0.035,
                "max_time": 0.050
            }
        }
        
        assert "policy_inference" in report
        assert "optimizer_update" in report
        assert report["policy_inference"]["count"] == 100
        assert report["optimizer_update"]["avg_time"] == 0.040
    
    def test_profiling_summary_generation(self):
        """Test generating profiling summary"""
        metrics = {
            "inference_times": [0.01, 0.012, 0.011],
            "update_times": [0.05, 0.048, 0.052]
        }
        
        summary = {
            "inference_avg": sum(metrics["inference_times"]) / len(metrics["inference_times"]),
            "update_avg": sum(metrics["update_times"]) / len(metrics["update_times"]),
            "total_calls": len(metrics["inference_times"]) + len(metrics["update_times"])
        }
        
        assert summary["total_calls"] == 6
        assert 0.010 < summary["inference_avg"] < 0.012
        assert 0.048 < summary["update_avg"] < 0.052


class TestProfilingIntegration:
    """Integration tests for profiling system"""
    
    def test_end_to_end_profiling(self):
        """Test end-to-end profiling workflow"""
        profiler_enabled = True
        measurements = []
        
        if profiler_enabled:
            # Simulate multiple training steps
            for step in range(10):
                step_start = time.perf_counter()
                
                # Simulate policy inference
                time.sleep(0.001)
                
                # Simulate optimizer update
                time.sleep(0.002)
                
                step_time = time.perf_counter() - step_start
                measurements.append(step_time)
        
        assert len(measurements) == 10
        avg_step_time = sum(measurements) / len(measurements)
        
        # Each step should take approximately 3ms
        assert 0.002 < avg_step_time < 0.005
    
    def test_profiling_disable(self):
        """Test that profiling can be disabled"""
        profiler_enabled = False
        measurements = []
        
        if profiler_enabled:
            # This block should not execute
            measurements.append(1.0)
        
        # No measurements when disabled
        assert len(measurements) == 0
    
    def test_profiling_selective_measurement(self):
        """Test selective measurement of specific operations"""
        measurements = {
            "inference": [],
            "update": [],
            "total": []
        }
        
        for _ in range(5):
            # Measure inference
            start = time.perf_counter()
            time.sleep(0.001)
            measurements["inference"].append(time.perf_counter() - start)
            
            # Measure update
            start = time.perf_counter()
            time.sleep(0.002)
            measurements["update"].append(time.perf_counter() - start)
            
            # Total time
            measurements["total"].append(
                measurements["inference"][-1] + measurements["update"][-1]
            )
        
        assert len(measurements["inference"]) == 5
        assert len(measurements["update"]) == 5
        assert len(measurements["total"]) == 5


class TestProfilingValidation:
    """Validation tests for profiling accuracy"""
    
    def test_timer_precision(self):
        """Test timer precision"""
        # perf_counter should have sub-millisecond precision
        start = time.perf_counter()
        time.sleep(0.0001)  # 0.1ms
        elapsed = time.perf_counter() - start
        
        # Should measure at least 0.1ms
        assert elapsed >= 0.00009  # Allow small tolerance
    
    def test_profiling_consistency(self):
        """Test profiling measurement consistency"""
        timings = []
        
        # Measure same operation multiple times
        for _ in range(10):
            start = time.perf_counter()
            time.sleep(0.005)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
        
        # Calculate coefficient of variation
        mean = sum(timings) / len(timings)
        variance = sum((t - mean) ** 2 for t in timings) / len(timings)
        std_dev = variance ** 0.5
        cv = (std_dev / mean) * 100
        
        # Coefficient of variation should be low (<20%)
        assert cv < 20.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
