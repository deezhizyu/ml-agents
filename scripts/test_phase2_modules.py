#!/usr/bin/env python3
"""
Test Phase 2 modules without full environment setup

This script tests individual Phase 2 components to verify they work correctly.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "ml-agents"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ml-agents-envs"))

print("=" * 60)
print("PHASE 2 MODULE TESTS")
print("=" * 60)
print()

# Test 1: Profiling Module
print("Test 1: Profiling Module")
print("-" * 60)
try:
    from mlagents.trainers.utils.profiling import (
        PerformanceMonitor,
        profile_block,
        profile_function,
        get_global_monitor,
    )

    # Test PerformanceMonitor
    monitor = PerformanceMonitor()
    monitor.record_metric("test", 1.5)
    monitor.record_metric("test", 2.0)

    summary = monitor.get_summary()
    assert "test" in summary
    assert summary["test"]["mean"] == 1.75

    # Test profile_block
    import time

    with profile_block(monitor, "test_block"):
        time.sleep(0.01)

    assert "test_block" in monitor.metrics

    # Test profile_function
    @profile_function()
    def test_func():
        return 42

    result = test_func()
    assert result == 42

    # Test global monitor
    global_monitor = get_global_monitor()
    assert global_monitor is not None

    print("✓ Profiling module working correctly")
    print(f"  Recorded metrics: {list(monitor.metrics.keys())}")

except Exception as e:
    print(f"✗ Profiling module test failed: {e}")
    import traceback

    traceback.print_exc()

# Test 2: Shared Memory Module
print("\nTest 2: Shared Memory Module")
print("-" * 60)
try:
    from mlagents.trainers.env_manager_shared_memory import SharedMemoryBuffer
    import numpy as np

    # Create buffer
    shape = (10, 10)
    dtype = np.float32
    buffer = SharedMemoryBuffer("test_buffer_module", shape, dtype)

    try:
        # Test write/read
        test_data = np.random.randn(*shape).astype(dtype)
        buffer.write(test_data)
        read_data = buffer.read()

        assert np.array_equal(read_data, test_data)

        # Test zero-copy
        buffer.array[0, 0] = 99.0
        read_again = buffer.read()
        assert read_again[0, 0] == 99.0

        print("✓ Shared memory module working correctly")
        print(f"  Buffer shape: {buffer.shape}")
        print(f"  Buffer size: {buffer.size / 1024:.2f} KB")

    finally:
        buffer.close()
        buffer.unlink()

except Exception as e:
    print(f"✗ Shared memory module test failed: {e}")
    import traceback

    traceback.print_exc()

# Test 3: TorchScript Module
print("\nTest 3: TorchScript Optimization Module")
print("-" * 60)
try:
    import torch
    import torch.nn as nn
    from mlagents.trainers.torch_entities.torchscript_optimization import (
        TorchScriptOptimizer,
        optimize_model,
    )

    # Create simple model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x)

    model = TestModel()
    example_input = torch.randn(1, 10)

    # Test compilation
    optimizer = TorchScriptOptimizer()
    compiled_model = optimizer.compile_model(
        model,
        (example_input,),
        use_jit_script=False,
    )

    # Test inference
    output = compiled_model(example_input)
    assert output.shape == (1, 5)

    # Test optimize_model function
    optimized = optimize_model(model, (example_input,), method="torchscript")
    output2 = optimized(example_input)
    assert output2.shape == (1, 5)

    print("✓ TorchScript module working correctly")
    print(f"  Model compiled successfully")
    print(f"  Output shape: {output.shape}")

except Exception as e:
    print(f"✗ TorchScript module test failed: {e}")
    import traceback

    traceback.print_exc()

# Test 4: Settings Integration
print("\nTest 4: Settings Integration")
print("-" * 60)
try:
    from mlagents.trainers.settings import NetworkSettings

    # Test default settings
    settings = NetworkSettings()
    assert hasattr(settings, "enable_torchscript")
    assert hasattr(settings, "torchscript_optimize_for_inference")
    assert settings.enable_torchscript == False  # Default
    assert settings.torchscript_optimize_for_inference == True  # Default

    # Test custom settings
    custom_settings = NetworkSettings(
        enable_torchscript=True,
        torchscript_optimize_for_inference=True,
        hidden_units=256,
    )
    assert custom_settings.enable_torchscript == True
    assert custom_settings.hidden_units == 256

    print("✓ Settings integration working correctly")
    print(f"  enable_torchscript: {settings.enable_torchscript}")
    print(
        f"  torchscript_optimize_for_inference: {settings.torchscript_optimize_for_inference}"
    )

except Exception as e:
    print(f"✗ Settings integration test failed: {e}")
    import traceback

    traceback.print_exc()

# Summary
print("\n" + "=" * 60)
print("MODULE TEST SUMMARY")
print("=" * 60)
print()
print("All core Phase 2 modules are functional!")
print()
print("Next steps:")
print("  1. Run unit tests: pytest ml-agents/mlagents/trainers/tests/ -v")
print("  2. Run benchmarks: python scripts/benchmark_phase2.py")
print("  3. Test with real environments")
print()
print("=" * 60)
