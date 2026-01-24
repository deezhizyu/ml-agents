"""
Tests for CLI benchmark tool - Phase 3
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from mlagents.trainers.cli_benchmark import (
    BenchmarkResult,
    MLAgentsBenchmark,
)


class TestBenchmarkResult:
    """Test BenchmarkResult class"""

    def test_result_creation(self):
        """Test creating a benchmark result"""
        result = BenchmarkResult("Test Benchmark")

        assert result.name == "Test Benchmark"
        assert len(result.timings) == 0
        assert len(result.throughput) == 0
        assert len(result.memory_usage) == 0

    def test_add_timing(self):
        """Test adding timing measurements"""
        result = BenchmarkResult("Test")

        result.add_timing(0.1)
        result.add_timing(0.2)
        result.add_timing(0.15)

        assert len(result.timings) == 3
        assert result.timings[0] == 0.1

    def test_add_throughput(self):
        """Test adding throughput measurements"""
        result = BenchmarkResult("Test")

        result.add_throughput(100.0)
        result.add_throughput(200.0)

        assert len(result.throughput) == 2
        assert result.throughput[0] == 100.0

    def test_add_memory(self):
        """Test adding memory measurements"""
        result = BenchmarkResult("Test")

        result.add_memory(1024)
        result.add_memory(2048)

        assert len(result.memory_usage) == 2
        assert result.memory_usage[0] == 1024

    def test_mean_timing(self):
        """Test calculating mean timing"""
        result = BenchmarkResult("Test")

        result.add_timing(0.1)
        result.add_timing(0.2)
        result.add_timing(0.3)

        mean = result.mean_timing
        assert mean == pytest.approx(0.2, rel=0.01)

    def test_std_timing(self):
        """Test calculating timing standard deviation"""
        result = BenchmarkResult("Test")

        result.add_timing(0.1)
        result.add_timing(0.2)
        result.add_timing(0.3)

        std = result.std_timing
        assert std > 0  # Should have some variation

    def test_mean_throughput(self):
        """Test calculating mean throughput"""
        result = BenchmarkResult("Test")

        result.add_throughput(100.0)
        result.add_throughput(150.0)
        result.add_throughput(200.0)

        mean = result.mean_throughput
        assert mean == pytest.approx(150.0, rel=0.01)

    def test_empty_result_metrics(self):
        """Test metrics on empty result"""
        result = BenchmarkResult("Test")

        assert result.mean_timing == 0.0
        assert result.std_timing == 0.0
        assert result.mean_throughput == 0.0


class TestMLAgentsBenchmark:
    """Test MLAgentsBenchmark class"""

    def test_benchmark_creation(self):
        """Test creating benchmark tool"""
        benchmark = MLAgentsBenchmark()

        assert benchmark is not None
        assert len(benchmark.results) == 0

    def test_benchmark_buffer_operations(self):
        """Test benchmarking buffer operations"""
        benchmark = MLAgentsBenchmark()

        result = benchmark.benchmark_buffer_operations(
            buffer_size=100, num_iterations=2
        )

        assert result is not None
        assert result.name == "Buffer Operations"
        assert len(result.timings) == 2
        assert "buffer_ops" in benchmark.results

    def test_run_quick_benchmark(self):
        """Test running quick benchmark suite"""
        benchmark = MLAgentsBenchmark()

        results = benchmark.run_quick_benchmark()

        assert results is not None
        assert len(results) > 0
        assert "buffer_ops" in results

    def test_print_results(self):
        """Test printing results"""
        benchmark = MLAgentsBenchmark()

        # Add some results
        result = BenchmarkResult("Test")
        result.add_timing(0.1)
        result.add_timing(0.2)
        benchmark.results["test"] = result

        # Should not raise exception
        benchmark.print_results()

    def test_compare_results(self):
        """Test comparing two results"""
        benchmark = MLAgentsBenchmark()

        # Baseline: slower
        baseline = BenchmarkResult("Baseline")
        baseline.add_timing(0.2)
        baseline.add_timing(0.3)

        # Optimized: faster
        optimized = BenchmarkResult("Optimized")
        optimized.add_timing(0.1)
        optimized.add_timing(0.15)

        speedup, comparison = benchmark.compare_results(baseline, optimized)

        # Optimized should be faster (speedup > 1)
        assert speedup > 1.0
        assert "faster" in comparison.lower()

    def test_compare_results_slower(self):
        """Test comparing results where optimized is slower"""
        benchmark = MLAgentsBenchmark()

        # Baseline: faster
        baseline = BenchmarkResult("Baseline")
        baseline.add_timing(0.1)

        # Optimized: slower
        optimized = BenchmarkResult("Optimized")
        optimized.add_timing(0.2)

        speedup, comparison = benchmark.compare_results(baseline, optimized)

        # Should be slower (speedup < 1)
        assert speedup < 1.0
        assert "slower" in comparison.lower()


class TestBenchmarkIntegration:
    """Integration tests for benchmark tool"""

    def test_full_benchmark_workflow(self):
        """Test complete benchmark workflow"""
        benchmark = MLAgentsBenchmark()

        # Run quick benchmark
        results = benchmark.run_quick_benchmark()

        # Should have results
        assert len(results) > 0

        # Results should have timings
        for name, result in results.items():
            assert len(result.timings) > 0
            assert result.mean_timing >= 0

    def test_benchmark_statistics(self):
        """Test that benchmark produces valid statistics"""
        benchmark = MLAgentsBenchmark()

        result = benchmark.benchmark_buffer_operations(buffer_size=50, num_iterations=5)

        # Should have valid statistics
        assert result.mean_timing > 0
        assert result.std_timing >= 0
        assert len(result.timings) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
