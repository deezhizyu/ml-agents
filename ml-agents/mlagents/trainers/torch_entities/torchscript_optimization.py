"""
TorchScript optimization for faster inference

This module provides utilities to compile PyTorch models with TorchScript
for 2-3x faster inference during training and deployment.

Usage:
    from mlagents.trainers.torch_entities.torchscript_optimization import optimize_model

    optimized_model = optimize_model(model, example_inputs)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List
from mlagents_envs import logging_util
from mlagents.torch_utils import torch as mlagents_torch

logger = logging_util.get_logger(__name__)


class TorchScriptOptimizer:
    """Optimize models with TorchScript for faster inference"""

    @staticmethod
    def compile_model(
        model: nn.Module,
        example_inputs: Tuple[torch.Tensor, ...],
        use_jit_script: bool = True,
        optimize_for_inference: bool = True,
    ) -> nn.Module:
        """
        Compile model with TorchScript

        :param model: PyTorch model to optimize
        :param example_inputs: Example inputs for tracing
        :param use_jit_script: Use torch.jit.script (better for control flow)
        :param optimize_for_inference: Apply inference optimizations
        :return: Optimized model
        """
        model.eval()

        try:
            if use_jit_script:
                # Script compilation (better for models with control flow)
                logger.info("Compiling model with torch.jit.script...")
                scripted_model = torch.jit.script(model)
            else:
                # Trace compilation (faster for models without control flow)
                logger.info("Compiling model with torch.jit.trace...")
                scripted_model = torch.jit.trace(model, example_inputs)

            if optimize_for_inference:
                # Apply optimization pass
                logger.info("Applying inference optimizations...")
                scripted_model = torch.jit.optimize_for_inference(scripted_model)

            _compilation_stats["attempts"] += 1
            _compilation_stats["successes"] += 1
            logger.info("Model successfully compiled with TorchScript")
            return scripted_model

        except Exception as e:
            _compilation_stats["attempts"] += 1
            _compilation_stats["failures"] += 1

            # Calculate failure rate
            failure_rate = (
                _compilation_stats["failures"] / _compilation_stats["attempts"]
            )

            logger.warning(f"Failed to compile model with TorchScript: {e}")
            logger.warning("Falling back to original model")
            logger.warning(
                f"TorchScript compilation stats: {_compilation_stats['successes']}/{_compilation_stats['attempts']} "
                f"successful ({failure_rate:.1%} failure rate)"
            )

            # Alert if failure rate is high
            if _compilation_stats["attempts"] >= 3 and failure_rate > 0.5:
                logger.error(
                    f"HIGH TorchScript failure rate detected ({failure_rate:.1%})! "
                    "Performance optimizations are not being applied. Check model compatibility."
                )

            return model

    @staticmethod
    def benchmark_model(
        original_model: nn.Module,
        optimized_model: nn.Module,
        example_inputs: Tuple[torch.Tensor, ...],
        num_iterations: int = 1000,
    ) -> Dict[str, float]:
        """
        Benchmark original vs optimized model

        :param original_model: Original PyTorch model
        :param optimized_model: TorchScript optimized model
        :param example_inputs: Example inputs for benchmarking
        :param num_iterations: Number of inference iterations
        :return: Dictionary with timing results
        """
        import time

        # Warm up
        with torch.no_grad():
            for _ in range(10):
                original_model(*example_inputs)
                optimized_model(*example_inputs)

        # Benchmark original model
        original_model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                original_model(*example_inputs)
        original_time = time.perf_counter() - start

        # Benchmark optimized model
        optimized_model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                optimized_model(*example_inputs)
        optimized_time = time.perf_counter() - start

        speedup = original_time / optimized_time

        results = {
            "original_time": original_time,
            "optimized_time": optimized_time,
            "speedup": speedup,
            "original_fps": num_iterations / original_time,
            "optimized_fps": num_iterations / optimized_time,
        }

        logger.info("=" * 60)
        logger.info("TorchScript Benchmark Results")
        logger.info("=" * 60)
        logger.info(
            f"Original model: {original_time:.3f}s ({results['original_fps']:.1f} inferences/sec)"
        )
        logger.info(
            f"Optimized model: {optimized_time:.3f}s ({results['optimized_fps']:.1f} inferences/sec)"
        )
        logger.info(f"Speedup: {speedup:.2f}x")
        logger.info("=" * 60)

        return results

    @staticmethod
    def save_scripted_model(model: nn.Module, path: str):
        """
        Save TorchScript model

        :param model: TorchScript compiled model
        :param path: Path to save model
        """
        torch.jit.save(model, path)
        logger.info(f"TorchScript model saved to: {path}")

    @staticmethod
    def load_scripted_model(path: str, device: Optional[str] = None) -> nn.Module:
        """
        Load TorchScript model

        :param path: Path to TorchScript model
        :param device: Device to load model on
        :return: Loaded model
        """
        if device is None:
            device = "cpu"

        model = torch.jit.load(path, map_location=device)
        logger.info(f"TorchScript model loaded from: {path}")
        return model


class ONNXExporter:
    """Export models to ONNX format for deployment"""

    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        example_inputs: Tuple[torch.Tensor, ...],
        output_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        opset_version: int = 11,
    ):
        """
        Export model to ONNX format

        :param model: PyTorch model to export
        :param example_inputs: Example inputs for export
        :param output_path: Path to save ONNX model
        :param input_names: Names for input tensors
        :param output_names: Names for output tensors
        :param opset_version: ONNX opset version
        """
        model.eval()

        if input_names is None:
            input_names = [f"input_{i}" for i in range(len(example_inputs))]

        if output_names is None:
            output_names = ["output"]

        try:
            torch.onnx.export(
                model,
                example_inputs,
                output_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=opset_version,
                export_params=True,
                do_constant_folding=True,
            )
            logger.info(f"Model exported to ONNX: {output_path}")

        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")
            raise

    @staticmethod
    def verify_onnx_model(onnx_path: str, example_inputs: Tuple[torch.Tensor, ...]):
        """
        Verify ONNX model can be loaded and runs correctly

        :param onnx_path: Path to ONNX model
        :param example_inputs: Example inputs for verification
        """
        try:
            import onnx
            import onnxruntime as ort

            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            # Create inference session
            ort_session = ort.InferenceSession(onnx_path)

            # Run inference
            input_names = [input.name for input in ort_session.get_inputs()]
            ort_inputs = {
                name: inp.cpu().numpy()
                for name, inp in zip(input_names, example_inputs)
            }
            ort_outputs = ort_session.run(None, ort_inputs)

            logger.info("ONNX model verification successful")
            logger.info(f"  Inputs: {input_names}")
            logger.info(f"  Outputs: {len(ort_outputs)} tensors")

        except ImportError:
            logger.warning("onnx or onnxruntime not installed, skipping verification")
        except Exception as e:
            logger.error(f"ONNX verification failed: {e}")
            raise


def optimize_model(
    model: nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    method: str = "torchscript",
    **kwargs,
) -> nn.Module:
    """
    Optimize model for faster inference

    :param model: PyTorch model to optimize
    :param example_inputs: Example inputs for optimization
    :param method: Optimization method ("torchscript" or "onnx")
    :param kwargs: Additional arguments for optimization
    :return: Optimized model
    """
    optimizer = TorchScriptOptimizer()

    if method == "torchscript":
        return optimizer.compile_model(model, example_inputs, **kwargs)
    elif method == "onnx":
        exporter = ONNXExporter()
        output_path = kwargs.get("output_path", "model.onnx")
        exporter.export_to_onnx(model, example_inputs, output_path)
        return model
    else:
        raise ValueError(f"Unknown optimization method: {method}")


# Integration with TorchPolicy
class OptimizedInferenceMixin:
    """
    Mixin for TorchPolicy to enable optimized inference

    Add to TorchPolicy:
        class TorchPolicy(OptimizedInferenceMixin, Policy):
            ...
    """

    def enable_torchscript_optimization(self, example_inputs: Tuple[torch.Tensor, ...]):
        """
        Enable TorchScript optimization for this policy

        :param example_inputs: Example inputs matching expected observation format
        """
        if not hasattr(self, "actor"):
            logger.warning("Policy has no actor, skipping optimization")
            return

        logger.info("Optimizing policy with TorchScript...")
        self.actor = TorchScriptOptimizer.compile_model(
            self.actor,
            example_inputs,
            use_jit_script=True,
            optimize_for_inference=True,
        )
        logger.info("Policy optimization complete")

    def benchmark_inference(self, example_inputs: Tuple[torch.Tensor, ...]):
        """
        Benchmark inference performance

        :param example_inputs: Example inputs for benchmarking
        """
        if not hasattr(self, "actor"):
            logger.warning("Policy has no actor, skipping benchmark")
            return

        # Create unoptimized copy for comparison
        import copy

        original_actor = copy.deepcopy(self.actor)

        # Optimize current actor
        self.enable_torchscript_optimization(example_inputs)

        # Benchmark
        results = TorchScriptOptimizer.benchmark_model(
            original_actor,
            self.actor,
            example_inputs,
        )

        return results
