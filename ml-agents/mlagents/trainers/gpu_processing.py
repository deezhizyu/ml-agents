"""
GPU-accelerated observation processing utilities

Provides GPU-optimized preprocessing for observations to reduce CPU overhead
and enable faster training with large batch sizes.
"""

import time
import numpy as np
from typing import Union, Optional, Dict, Tuple
from mlagents.torch_utils import torch
from mlagents_envs import logging_util

logger = logging_util.get_logger(__name__)


class GPUObservationProcessor:
    """
    GPU-accelerated observation processing

    Moves observation preprocessing to GPU for improved performance with large batches.
    Supports normalization, batching, and efficient CPU-GPU transfer.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize GPU observation processor

        :param device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Running statistics for normalization
        self.running_mean: Optional[torch.Tensor] = None
        self.running_std: Optional[torch.Tensor] = None
        self.count = 0

        # Normalization parameters
        self.epsilon = 1e-8
        self.momentum = 0.99

        logger.info(f"GPUObservationProcessor initialized on device: {self.device}")

    def update_statistics(self, observations: "torch.Tensor") -> None:
        """
        Update running mean and std for normalization

        :param observations: Batch of observations (batch_size, obs_dim)
        """
        batch_mean = observations.mean(dim=0)
        # unbiased=False avoids NaN when a batch has exactly 1 sample (n-1 == 0)
        batch_std = observations.std(dim=0, unbiased=False)

        if self.running_mean is None:
            # Initialize
            self.running_mean = batch_mean
            self.running_std = batch_std
        else:
            # Exponential moving average
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )
            self.running_std = (
                self.momentum * self.running_std + (1 - self.momentum) * batch_std
            )

        self.count += observations.shape[0]

    def normalize(
        self, observations: "torch.Tensor", update_stats: bool = True
    ) -> "torch.Tensor":
        """
        Normalize observations using running statistics

        :param observations: Observations to normalize (batch_size, obs_dim)
        :param update_stats: Whether to update running statistics
        :return: Normalized observations
        """
        if update_stats:
            self.update_statistics(observations)

        if self.running_mean is None:
            # No statistics yet, return as-is
            return observations

        # Normalize: (obs - mean) / (std + epsilon)
        normalized = (observations - self.running_mean) / (
            self.running_std + self.epsilon
        )

        return normalized

    def process_batch(
        self,
        obs_batch: Union[np.ndarray, "torch.Tensor"],
        normalize: bool = True,
        update_stats: bool = True,
    ) -> "torch.Tensor":
        """
        Process a batch of observations on GPU

        :param obs_batch: Numpy array or tensor of observations
        :param normalize: Whether to apply normalization
        :param update_stats: Whether to update running statistics
        :return: Processed observations on GPU
        """
        # Convert to tensor if numpy
        if isinstance(obs_batch, np.ndarray):
            # Single CPU->GPU transfer for entire batch
            gpu_obs = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        elif obs_batch.device != self.device:
            # Move to target device
            gpu_obs = obs_batch.to(self.device)
        else:
            gpu_obs = obs_batch

        # Apply normalization if requested
        if normalize:
            gpu_obs = self.normalize(gpu_obs, update_stats=update_stats)

        return gpu_obs

    def process_single(
        self, observation: Union[np.ndarray, "torch.Tensor"], normalize: bool = True
    ) -> "torch.Tensor":
        """
        Process a single observation on GPU

        Note: Prefer process_batch for better performance (amortize transfer cost)

        :param observation: Single observation
        :param normalize: Whether to apply normalization
        :return: Processed observation on GPU
        """
        # Add batch dimension
        obs_batch = (
            observation[np.newaxis, :]
            if isinstance(observation, np.ndarray)
            else observation.unsqueeze(0)
        )

        # Process as batch
        processed = self.process_batch(
            obs_batch, normalize=normalize, update_stats=False
        )

        # Remove batch dimension
        return processed.squeeze(0)

    def save_statistics(self, path: str) -> None:
        """
        Save running statistics to file

        :param path: Path to save statistics
        """
        if self.running_mean is None:
            logger.warning("No statistics to save (processor not used yet)")
            return

        torch.save(
            {
                "running_mean": self.running_mean.cpu(),
                "running_std": self.running_std.cpu(),
                "count": self.count,
                "device": str(self.device),
            },
            path,
        )

        logger.info(f"Saved GPU processor statistics to {path}")

    def load_statistics(self, path: str) -> None:
        """
        Load running statistics from file

        :param path: Path to load statistics from
        """
        checkpoint = torch.load(path, weights_only=True)

        self.running_mean = checkpoint["running_mean"].to(self.device)
        self.running_std = checkpoint["running_std"].to(self.device)
        self.count = checkpoint["count"]

        logger.info(f"Loaded GPU processor statistics from {path} (count={self.count})")

    def reset_statistics(self) -> None:
        """Reset running statistics"""
        self.running_mean = None
        self.running_std = None
        self.count = 0
        logger.info("Reset GPU processor statistics")


class BatchedGPUInference:
    """
    GPU-optimized batched inference for policies

    Keeps tensors on GPU throughout inference pipeline to minimize CPU-GPU transfers
    """

    def __init__(self, policy, device: Optional[str] = None):
        """
        Initialize batched GPU inference

        :param policy: Policy network to use for inference
        :param device: Device to run inference on
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Move policy to device
        self.policy = policy.to(self.device)
        self.policy.eval()  # Inference mode

        logger.info(f"BatchedGPUInference initialized on {self.device}")

    @torch.no_grad()
    def infer_actions(self, observations: "torch.Tensor") -> "torch.Tensor":
        """
        Infer actions for batch of observations

        :param observations: Batch of observations already on GPU
        :return: Actions tensor on GPU
        """
        # Observations should already be on GPU from processor
        if observations.device != self.device:
            logger.warning(f"Observations not on {self.device}, transferring...")
            observations = observations.to(self.device)

        # Inference (stays on GPU)
        with torch.no_grad():
            actions = self.policy(observations)

        return actions

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage

        :return: Dictionary with memory statistics in GB
        """
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "device": str(self.device),  # type: ignore[dict-item]
            }
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "device": "cpu"}  # type: ignore[dict-item]


# Utility functions
def move_to_gpu(data: Union[np.ndarray, "torch.Tensor"], device: str = "cuda") -> "torch.Tensor":
    """
    Move data to GPU efficiently

    :param data: Numpy array or tensor
    :param device: Target device
    :return: Tensor on GPU
    """
    if isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=torch.float32, device=device)
    return data.to(device)


def benchmark_gpu_processing(
    processor: GPUObservationProcessor,
    obs_shape: Tuple[int, ...],
    batch_size: int = 32,
    num_iterations: int = 1000,
) -> Dict[str, float]:
    """
    Benchmark GPU observation processing performance

    :param processor: GPUObservationProcessor instance
    :param obs_shape: Shape of observations
    :param batch_size: Batch size to test
    :param num_iterations: Number of iterations
    :return: Performance metrics
    """
    # Generate random observations
    obs_batch = np.random.randn(batch_size, *obs_shape).astype(np.float32)

    # Warmup
    for _ in range(10):
        processor.process_batch(obs_batch)

    # Benchmark
    start = time.perf_counter()

    for _ in range(num_iterations):
        processed = processor.process_batch(obs_batch)

    elapsed = time.perf_counter() - start

    obs_per_sec = (batch_size * num_iterations) / elapsed

    return {
        "total_time_sec": elapsed,
        "time_per_batch_ms": (elapsed / num_iterations) * 1000,
        "observations_per_sec": obs_per_sec,
        "batch_size": batch_size,
        "num_iterations": num_iterations,
    }
