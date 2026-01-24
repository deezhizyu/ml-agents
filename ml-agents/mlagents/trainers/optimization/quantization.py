"""
Model quantization utilities for ML-Agents

Provides INT8 and FP16 quantization for faster inference and smaller model sizes.
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np

from mlagents.torch_utils import torch
from mlagents_envs import logging_util

logger = logging_util.get_logger(__name__)


def quantize_model_int8(
    model: torch.nn.Module,
    validation_data: Optional[torch.Tensor] = None
) -> torch.nn.Module:
    """
    Quantize model to INT8 using dynamic quantization

    :param model: PyTorch model to quantize
    :param validation_data: Optional data to validate quantization quality
    :return: Quantized model
    """
    logger.info("Quantizing model to INT8...")

    # Store original outputs for validation
    original_outputs = None
    if validation_data is not None:
        model.eval()
        with torch.no_grad():
            original_outputs = model(validation_data)

    # Dynamic quantization (Linear layers)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # Validate quantization quality
    if validation_data is not None and original_outputs is not None:
        quantized_model.eval()
        with torch.no_grad():
            quantized_outputs = quantized_model(validation_data)

        # Compute MSE
        mse = torch.mean((original_outputs - quantized_outputs) ** 2).item()
        max_diff = torch.max(torch.abs(original_outputs - quantized_outputs)).item()

        logger.info(f"Quantization quality:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  Max difference: {max_diff:.6f}")

        if mse > 0.01:
            logger.warning(f"High quantization error (MSE={mse:.6f}) - may affect performance")

    logger.info("INT8 quantization complete")
    return quantized_model


def quantize_model_fp16(model: torch.nn.Module) -> torch.nn.Module:
    """
    Quantize model to FP16 (half precision)

    :param model: PyTorch model to quantize
    :return: FP16 model
    """
    logger.info("Converting model to FP16...")

    # Convert to half precision
    fp16_model = model.half()

    logger.info("FP16 conversion complete")
    return fp16_model


def quantize_and_save(
    model_path: str,
    output_path: str,
    quantization_type: str = 'int8',
    validation_data_path: Optional[str] = None
):
    """
    Quantize a saved model and save the result

    :param model_path: Path to model to quantize
    :param output_path: Path to save quantized model
    :param quantization_type: 'int8' or 'fp16'
    :param validation_data_path: Optional path to validation data (numpy file)
    """
    logger.info(f"Loading model from {model_path}")

    # Load model with weights_only=True for security (prevents arbitrary code execution)
    try:
        model = torch.load(model_path, weights_only=True)
    except Exception as e:
        # Fallback for legacy model formats (with security warning)
        logger.warning(
            f"Failed to load {model_path} with weights_only=True ({e}). "
            "Falling back to weights_only=False. "
            "SECURITY WARNING: This is a potential security risk. "
            "Re-save model with TorchScript for secure loading."
        )
        model = torch.load(model_path, weights_only=False)

    # Load validation data if provided
    validation_data = None
    if validation_data_path:
        logger.info(f"Loading validation data from {validation_data_path}")
        val_np = np.load(validation_data_path)
        validation_data = torch.tensor(val_np, dtype=torch.float32)

    # Quantize
    if quantization_type == 'int8':
        quantized_model = quantize_model_int8(model, validation_data)
    elif quantization_type == 'fp16':
        quantized_model = quantize_model_fp16(model)
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")

    # Save quantized model
    logger.info(f"Saving quantized model to {output_path}")

    # Use TorchScript for better compatibility
    try:
        scripted_model = torch.jit.script(quantized_model)
        scripted_model.save(output_path)
        logger.info("Saved as TorchScript model")
    except Exception as e:
        logger.warning(f"Could not script model ({e}), saving as state dict")
        torch.save(quantized_model.state_dict(), output_path)

    # Report model sizes
    original_size = Path(model_path).stat().st_size / 1e6  # MB
    quantized_size = Path(output_path).stat().st_size / 1e6  # MB
    reduction = (1 - quantized_size / original_size) * 100

    logger.info(f"Model size reduction:")
    logger.info(f"  Original: {original_size:.2f} MB")
    logger.info(f"  Quantized: {quantized_size:.2f} MB")
    logger.info(f"  Reduction: {reduction:.1f}%")


def compare_inference_speed(
    original_model: torch.nn.Module,
    quantized_model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    num_iterations: int = 1000
) -> Dict[str, float]:
    """
    Compare inference speed between original and quantized models

    :param original_model: Original model
    :param quantized_model: Quantized model
    :param input_shape: Shape of input observations
    :param num_iterations: Number of inference iterations
    :return: Performance comparison metrics
    """
    # Generate random input
    test_input = torch.randn(input_shape)

    original_model.eval()
    quantized_model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = original_model(test_input)
            _ = quantized_model(test_input)

    # Benchmark original
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = original_model(test_input)
    original_time = time.perf_counter() - start

    # Benchmark quantized
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = quantized_model(test_input)
    quantized_time = time.perf_counter() - start

    speedup = original_time / quantized_time

    results = {
        'original_time_sec': original_time,
        'quantized_time_sec': quantized_time,
        'original_time_per_inference_ms': (original_time / num_iterations) * 1000,
        'quantized_time_per_inference_ms': (quantized_time / num_iterations) * 1000,
        'speedup': speedup
    }

    logger.info("Inference speed comparison:")
    logger.info(f"  Original: {results['original_time_per_inference_ms']:.3f} ms/inference")
    logger.info(f"  Quantized: {results['quantized_time_per_inference_ms']:.3f} ms/inference")
    logger.info(f"  Speedup: {speedup:.2f}x")

    return results


# CLI interface
def main():
    """Command-line interface for model quantization"""
    parser = argparse.ArgumentParser(
        description="Quantize ML-Agents models for faster inference"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model to quantize (.pt file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for quantized model (default: <model>_quantized.pt)"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=['int8', 'fp16'],
        default='int8',
        help="Quantization type (default: int8)"
    )
    parser.add_argument(
        "--validation-data",
        type=str,
        default=None,
        help="Path to validation data (.npy file) for quality check"
    )

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        model_path = Path(args.model_path)
        args.output = str(model_path.parent / f"{model_path.stem}_quantized{model_path.suffix}")

    # Quantize
    quantize_and_save(
        model_path=args.model_path,
        output_path=args.output,
        quantization_type=args.type,
        validation_data_path=args.validation_data
    )

    logger.info(f"Quantization complete! Quantized model saved to: {args.output}")


if __name__ == "__main__":
    main()
