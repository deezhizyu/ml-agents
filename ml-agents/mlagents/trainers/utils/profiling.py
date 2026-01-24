"""
Performance profiling utilities for ML-Agents training
"""
import time
import functools
import os
from typing import Callable, Any, Dict, Optional
from contextlib import contextmanager
from mlagents_envs import logging_util

# Optional psutil for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging_util.get_logger(__name__)

# Track if we've warned about missing psutil
_psutil_warning_shown = False


class PerformanceMonitor:
    """Monitor performance metrics during training"""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {
            "step_time": [],
            "inference_time": [],
            "env_step_time": [],
            "update_time": [],
            "memory_mb": [],
            "cpu_percent": [],
        }
        self.process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None
    
    def record_metric(self, name: str, value: float):
        """Record a performance metric"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics"""
        import numpy as np
        
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)),
                    "p99": float(np.percentile(values, 99)),
                }
        return summary
    
    def log_summary(self):
        """Log performance summary"""
        summary = self.get_summary()
        logger.info("=" * 60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        for metric, stats in summary.items():
            logger.info(f"{metric}:")
            logger.info(f"  Mean: {stats['mean']:.4f}")
            logger.info(f"  Std:  {stats['std']:.4f}")
            logger.info(f"  P50:  {stats['p50']:.4f}")
            logger.info(f"  P95:  {stats['p95']:.4f}")
            logger.info(f"  P99:  {stats['p99']:.4f}")
        logger.info("=" * 60)
    
    def check_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        global _psutil_warning_shown
        if self.process is not None:
            return self.process.memory_info().rss / 1024 / 1024
        elif not PSUTIL_AVAILABLE and not _psutil_warning_shown:
            logger.warning(
                "Memory profiling requested but psutil is not installed. "
                "Install with: pip install psutil"
            )
            _psutil_warning_shown = True
        return 0.0
    
    def check_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        if self.process is not None:
            return self.process.cpu_percent(interval=0.1)
        return 0.0


@contextmanager
def profile_block(monitor: PerformanceMonitor, metric_name: str):
    """Context manager to profile a code block"""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        monitor.record_metric(metric_name, elapsed)


def profile_function(metric_name: Optional[str] = None):
    """Decorator to profile function execution time"""
    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            
            # Log slow functions (>100ms)
            if elapsed > 0.1:
                logger.debug(f"[PERF] {name} took {elapsed:.3f}s")
            
            return result
        return wrapper
    return decorator


class BatchedInference:
    """Helper for batched inference to reduce overhead"""
    
    def __init__(self, policy, max_batch_size: int = 128):
        self.policy = policy
        self.max_batch_size = max_batch_size
        self.pending_observations = []
        self.pending_agent_ids = []
    
    def add_observation(self, agent_id: str, observation):
        """Add observation to batch"""
        self.pending_observations.append(observation)
        self.pending_agent_ids.append(agent_id)
    
    def should_flush(self) -> bool:
        """Check if batch should be processed"""
        return len(self.pending_observations) >= self.max_batch_size
    
    def flush(self):
        """Process all pending observations"""
        if not self.pending_observations:
            return {}
        
        # Batch process all observations
        results = {}
        # Implementation would call policy.evaluate in batch
        
        # Clear pending
        self.pending_observations = []
        self.pending_agent_ids = []
        
        return results


def enable_profiling(enable: bool = True):
    """Enable or disable profiling globally"""
    global _PROFILING_ENABLED
    _PROFILING_ENABLED = enable


def is_profiling_enabled() -> bool:
    """Check if profiling is enabled"""
    return globals().get("_PROFILING_ENABLED", False)


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_global_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def reset_global_monitor():
    """Reset global performance monitor"""
    global _global_monitor
    _global_monitor = PerformanceMonitor()
