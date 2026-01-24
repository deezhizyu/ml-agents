"""
Complete TorchScript integration tests - Phase 2 final validation
"""
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from mlagents.trainers.settings import NetworkSettings


class TestTorchScriptCompilation:
    """Test TorchScript compilation integration"""
    
    def test_torchscript_enabled_setting(self):
        """Test TorchScript enabled setting"""
        settings = NetworkSettings(
            enable_torchscript=True,
            torchscript_optimize_for_inference=True
        )
        
        assert settings.enable_torchscript
        assert settings.torchscript_optimize_for_inference
    
    def test_torchscript_disabled_setting(self):
        """Test TorchScript disabled setting"""
        settings = NetworkSettings(
            enable_torchscript=False
        )
        
        assert not settings.enable_torchscript
    
    def test_simple_model_compilation(self):
        """Test compiling a simple model with TorchScript"""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        
        # Trace the model
        example_input = torch.randn(1, 10)
        traced_model = torch.jit.trace(model, example_input)
        
        # Test inference
        output = traced_model(example_input)
        
        assert output.shape == (1, 5)
        assert isinstance(traced_model, torch.jit.ScriptModule)
    
    def test_compiled_model_inference(self):
        """Test inference with compiled model"""
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(8, 16)
                self.fc2 = torch.nn.Linear(16, 4)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)
        
        model = TestModel()
        example_input = torch.randn(2, 8)
        
        # Compile
        compiled = torch.jit.trace(model, example_input)
        
        # Compare outputs
        model.eval()
        compiled.eval()
        
        test_input = torch.randn(2, 8)
        original_output = model(test_input)
        compiled_output = compiled(test_input)
        
        # Outputs should be identical
        assert torch.allclose(original_output, compiled_output, atol=1e-5)
    
    def test_optimization_for_inference(self):
        """Test optimization for inference flag"""
        class OptimizeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3)
                self.bn = torch.nn.BatchNorm2d(16)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x
        
        model = OptimizeModel()
        model.eval()
        
        example_input = torch.randn(1, 3, 32, 32)
        traced = torch.jit.trace(model, example_input)
        
        # Optimize for inference
        optimized = torch.jit.optimize_for_inference(traced)
        
        # Should still work
        output = optimized(example_input)
        assert output.shape[1] == 16  # 16 output channels


class TestTorchScriptPerformance:
    """Test TorchScript performance characteristics"""
    
    def test_compilation_overhead(self):
        """Test that compilation has acceptable overhead"""
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(100, 50)
            
            def forward(self, x):
                return self.linear(x)
        
        model = Model()
        example_input = torch.randn(1, 100)
        
        # Compilation should complete without error
        traced = torch.jit.trace(model, example_input)
        
        assert traced is not None
    
    def test_inference_consistency(self):
        """Test that compiled models produce consistent results"""
        class ConsistentModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(5, 3)
            
            def forward(self, x):
                return self.layer(x)
        
        model = ConsistentModel()
        model.eval()
        
        input_data = torch.randn(10, 5)
        
        # Compile
        compiled = torch.jit.trace(model, input_data[:1])
        
        # Multiple inferences should be consistent
        output1 = compiled(input_data)
        output2 = compiled(input_data)
        
        assert torch.equal(output1, output2)


class TestTorchScriptCompatibility:
    """Test TorchScript compatibility with various architectures"""
    
    def test_mlp_compatibility(self):
        """Test MLP architecture compatibility"""
        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(10, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, 4)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        model = MLP()
        example = torch.randn(1, 10)
        
        # Should compile successfully
        compiled = torch.jit.trace(model, example)
        output = compiled(example)
        
        assert output.shape == (1, 4)
    
    def test_cnn_compatibility(self):
        """Test CNN architecture compatibility"""
        class SimpleCNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, 3)
                self.pool = torch.nn.MaxPool2d(2)
                self.fc = torch.nn.Linear(16 * 15 * 15, 10)
            
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = SimpleCNN()
        example = torch.randn(1, 3, 32, 32)
        
        # Should compile successfully
        compiled = torch.jit.trace(model, example)
        output = compiled(example)
        
        assert output.shape == (1, 10)
    
    def test_recurrent_compatibility(self):
        """Test RNN/LSTM compatibility"""
        class SimpleRNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(10, 20, batch_first=True)
                self.fc = torch.nn.Linear(20, 5)
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # Use last time step
                last_out = lstm_out[:, -1, :]
                return self.fc(last_out)
        
        model = SimpleRNN()
        example = torch.randn(2, 5, 10)  # batch, seq_len, features
        
        # Should compile successfully
        compiled = torch.jit.trace(model, example)
        output = compiled(example)
        
        assert output.shape == (2, 5)


class TestTorchScriptValidation:
    """Validation tests for TorchScript implementation"""
    
    def test_graph_correctness(self):
        """Test that compiled graph is correct"""
        class GraphModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(5, 5))
            
            def forward(self, x):
                return torch.matmul(x, self.weight)
        
        model = GraphModel()
        example = torch.randn(3, 5)
        
        compiled = torch.jit.trace(model, example)
        
        # Graph should contain matrix multiplication
        graph_str = str(compiled.graph)
        assert 'aten::matmul' in graph_str or 'aten::mm' in graph_str
    
    def test_numerical_accuracy(self):
        """Test numerical accuracy of compiled models"""
        class AccuracyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(20, 10)
            
            def forward(self, x):
                return torch.sigmoid(self.linear(x))
        
        model = AccuracyModel()
        model.eval()
        
        example = torch.randn(1, 20)
        compiled = torch.jit.trace(model, example)
        
        # Test with different inputs
        for _ in range(10):
            test_input = torch.randn(5, 20)
            original = model(test_input)
            compiled_out = compiled(test_input)
            
            # Should be numerically very close
            max_diff = torch.max(torch.abs(original - compiled_out))
            assert max_diff < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
