"""Model optimization utilities for ML-Agents"""

from mlagents.trainers.optimization.quantization import (
    quantize_model_int8,
    quantize_model_fp16,
    quantize_model
)

__all__ = ["quantize_model_int8", "quantize_model_fp16", "quantize_model"]
