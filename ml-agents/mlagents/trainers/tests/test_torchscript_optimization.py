"""
Tests for TorchScript optimization utilities
"""

import pytest
import torch
import torch.nn as nn
from mlagents.trainers.torch_entities.torchscript_optimization import (
    TorchScriptOptimizer,
    optimize_model,
)


class SimpleModel(nn.Module):
    """Simple model for testing"""

    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ModelWithControlFlow(nn.Module):
    """Model with control flow for testing torch.jit.script"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        if x.sum() > 0:
            return self.fc(x)
        else:
            return x


class TestTorchScriptOptimizer:
    """Test TorchScriptOptimizer class"""

    def test_compile_model_with_trace(self):
        """Test compiling model with torch.jit.trace"""
        model = SimpleModel()
        example_input = torch.randn(1, 10)

        optimizer = TorchScriptOptimizer()
        compiled_model = optimizer.compile_model(
            model,
            (example_input,),
            use_jit_script=False,
            optimize_for_inference=False,
        )

        # Should return a model
        assert compiled_model is not None

        # Should be able to run inference
        output = compiled_model(example_input)
        assert output.shape == (1, 5)

    def test_compile_model_with_script(self):
        """Test compiling model with torch.jit.script"""
        model = ModelWithControlFlow()
        example_input = torch.randn(1, 10)

        optimizer = TorchScriptOptimizer()
        compiled_model = optimizer.compile_model(
            model,
            (example_input,),
            use_jit_script=True,
            optimize_for_inference=False,
        )

        # Should return a model
        assert compiled_model is not None

        # Should handle control flow
        output_pos = compiled_model(torch.ones(1, 10))
        output_neg = compiled_model(-torch.ones(1, 10))

        assert output_pos.shape == (1, 10)
        assert output_neg.shape == (1, 10)

    def test_optimize_for_inference(self):
        """Test inference optimization"""
        model = SimpleModel()
        example_input = torch.randn(1, 10)

        optimizer = TorchScriptOptimizer()
        compiled_model = optimizer.compile_model(
            model,
            (example_input,),
            use_jit_script=False,
            optimize_for_inference=True,
        )

        # Should be able to run inference
        output = compiled_model(example_input)
        assert output.shape == (1, 5)

    def test_compiled_model_output_matches(self):
        """Test that compiled model produces same output as original"""
        model = SimpleModel()
        model.eval()

        example_input = torch.randn(1, 10)

        # Get output from original model
        with torch.no_grad():
            original_output = model(example_input)

        # Compile model
        optimizer = TorchScriptOptimizer()
        compiled_model = optimizer.compile_model(
            model,
            (example_input,),
            use_jit_script=False,
        )

        # Get output from compiled model
        with torch.no_grad():
            compiled_output = compiled_model(example_input)

        # Outputs should match
        torch.testing.assert_close(
            compiled_output, original_output, rtol=1e-5, atol=1e-5
        )

    def test_benchmark_model(self):
        """Test benchmarking functionality"""
        original_model = SimpleModel()
        original_model.eval()

        example_input = torch.randn(1, 10)

        # Compile model
        optimizer = TorchScriptOptimizer()
        optimized_model = optimizer.compile_model(
            original_model,
            (example_input,),
            use_jit_script=False,
        )

        # Benchmark
        results = optimizer.benchmark_model(
            original_model,
            optimized_model,
            (example_input,),
            num_iterations=100,
        )

        # Check results structure
        assert "original_time" in results
        assert "optimized_time" in results
        assert "speedup" in results
        assert "original_fps" in results
        assert "optimized_fps" in results

        # Times should be positive
        assert results["original_time"] > 0
        assert results["optimized_time"] > 0

        # FPS should be positive
        assert results["original_fps"] > 0
        assert results["optimized_fps"] > 0

    def test_save_and_load_scripted_model(self, tmp_path):
        """Test saving and loading TorchScript model"""
        model = SimpleModel()
        example_input = torch.randn(1, 10)

        optimizer = TorchScriptOptimizer()
        compiled_model = optimizer.compile_model(
            model,
            (example_input,),
            use_jit_script=False,
        )

        # Save model
        save_path = tmp_path / "model.pt"
        optimizer.save_scripted_model(compiled_model, str(save_path))

        assert save_path.exists()

        # Load model
        loaded_model = optimizer.load_scripted_model(str(save_path))

        # Should be able to run inference
        with torch.no_grad():
            output = loaded_model(example_input)

        assert output.shape == (1, 5)


class TestOptimizeModelFunction:
    """Test the optimize_model convenience function"""

    def test_optimize_model_torchscript(self):
        """Test optimize_model with TorchScript"""
        model = SimpleModel()
        example_input = torch.randn(1, 10)

        optimized_model = optimize_model(
            model,
            (example_input,),
            method="torchscript",
        )

        # Should return a model
        assert optimized_model is not None

        # Should be able to run inference
        output = optimized_model(example_input)
        assert output.shape == (1, 5)

    def test_optimize_model_invalid_method(self):
        """Test that invalid method raises error"""
        model = SimpleModel()
        example_input = torch.randn(1, 10)

        with pytest.raises(ValueError, match="Unknown optimization method"):
            optimize_model(
                model,
                (example_input,),
                method="invalid_method",
            )


class TestIntegration:
    """Integration tests for TorchScript optimization"""

    def test_batch_inference(self):
        """Test optimized model with batch inputs"""
        model = SimpleModel()
        model.eval()

        # Compile with single example
        single_input = torch.randn(1, 10)
        optimizer = TorchScriptOptimizer()
        compiled_model = optimizer.compile_model(
            model,
            (single_input,),
            use_jit_script=False,
        )

        # Test with batch
        batch_input = torch.randn(32, 10)
        with torch.no_grad():
            original_output = model(batch_input)
            compiled_output = compiled_model(batch_input)

        torch.testing.assert_close(
            compiled_output, original_output, rtol=1e-5, atol=1e-5
        )

    def test_multiple_forward_passes(self):
        """Test that compiled model works for multiple forward passes"""
        model = SimpleModel()
        example_input = torch.randn(1, 10)

        optimizer = TorchScriptOptimizer()
        compiled_model = optimizer.compile_model(
            model,
            (example_input,),
            use_jit_script=False,
        )

        # Run multiple times
        for _ in range(10):
            test_input = torch.randn(1, 10)
            output = compiled_model(test_input)
            assert output.shape == (1, 5)

    def test_model_with_batch_norm(self):
        """Test compiling model with batch normalization"""

        class ModelWithBatchNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.bn = nn.BatchNorm1d(20)
                self.fc2 = nn.Linear(20, 5)

            def forward(self, x):
                x = self.fc1(x)
                x = self.bn(x)
                x = self.fc2(x)
                return x

        model = ModelWithBatchNorm()
        model.eval()  # Important: set to eval mode for batch norm

        example_input = torch.randn(4, 10)  # Batch of 4

        optimizer = TorchScriptOptimizer()
        compiled_model = optimizer.compile_model(
            model,
            (example_input,),
            use_jit_script=False,
        )

        # Should work in eval mode
        with torch.no_grad():
            output = compiled_model(example_input)

        assert output.shape == (4, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
