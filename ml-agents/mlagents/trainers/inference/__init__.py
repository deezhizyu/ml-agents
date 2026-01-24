"""Inference utilities for ML-Agents"""

from mlagents.trainers.inference.async_batch_inference import (
    AsyncBatchInference,
    MultiModelAsyncInference
)

__all__ = ["AsyncBatchInference", "MultiModelAsyncInference"]
