"""
Async batching inference server for ML-Agents

Provides low-latency inference through dynamic batching and async processing
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional, List, Tuple, Any, Dict, Dict
from collections import deque
import numpy as np

from mlagents.torch_utils import torch
from mlagents_envs import logging_util

logger = logging_util.get_logger(__name__)


class AsyncBatchInference:
    """
    Asynchronous batching inference server

    Collects inference requests and processes them in batches to maximize throughput
    while maintaining low latency through adaptive batching
    """

    def __init__(
        self,
        model: torch.nn.Module,
        max_batch_size: int = 32,
        max_latency_ms: float = 10.0,
        device: Optional[str] = None
    ):
        """
        Initialize async batch inference server

        :param model: PyTorch model for inference
        :param max_batch_size: Maximum batch size (process when reached)
        :param max_latency_ms: Maximum latency before processing partial batch
        :param device: Device to run inference on
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

        self.max_batch_size = max_batch_size
        self.max_latency_sec = max_latency_ms / 1000.0

        # Request queue: (observation, future)
        self.request_queue: asyncio.Queue = asyncio.Queue()

        # Statistics
        self.total_requests = 0
        self.total_batches = 0
        self.total_latency = 0.0

        # Start batch processing loop
        self._batch_task = None
        self._running = False

        logger.info(
            f"AsyncBatchInference initialized: "
            f"max_batch={max_batch_size}, max_latency={max_latency_ms}ms, device={self.device}"
        )

    async def start(self):
        """Start the batch processing loop"""
        if self._running:
            logger.warning("Batch server already running")
            return

        self._running = True
        self._batch_task = asyncio.create_task(self._batch_loop())
        logger.info("Async batch inference server started")

    async def stop(self):
        """Stop the batch processing loop"""
        if not self._running:
            return

        self._running = False

        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        logger.info("Async batch inference server stopped")

    async def infer(self, observation: np.ndarray) -> np.ndarray:
        """
        Infer action for observation (async)

        :param observation: Observation array
        :return: Action array
        """
        if not self._running:
            await self.start()

        # Create future for result
        future = asyncio.Future()

        # Add to queue
        await self.request_queue.put((observation, future, time.perf_counter()))

        self.total_requests += 1

        # Wait for result
        result = await future

        return result

    async def _batch_loop(self):
        """Main batch processing loop"""
        logger.info("Batch processing loop started")

        try:
            while self._running:
                # Collect batch
                batch_obs = []
                batch_futures = []
                batch_start_times = []

                deadline = time.perf_counter() + self.max_latency_sec

                # Collect requests until batch full or deadline reached
                while len(batch_obs) < self.max_batch_size:
                    timeout = max(0, deadline - time.perf_counter())

                    if timeout <= 0:
                        break

                    try:
                        obs, future, start_time = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=timeout
                        )
                        batch_obs.append(obs)
                        batch_futures.append(future)
                        batch_start_times.append(start_time)

                    except asyncio.TimeoutError:
                        break

                # Process batch if not empty
                if batch_obs:
                    await self._process_batch(batch_obs, batch_futures, batch_start_times)

                # Small sleep to avoid busy waiting when queue is empty
                if not batch_obs:
                    await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            logger.info("Batch loop cancelled")
        except Exception as e:
            logger.error(f"Batch loop error: {e}", exc_info=True)

    async def _process_batch(
        self,
        observations: List[np.ndarray],
        futures: List[asyncio.Future],
        start_times: List[float]
    ):
        """
        Process a batch of inference requests

        :param observations: List of observations
        :param futures: List of futures to resolve
        :param start_times: Request start times for latency tracking
        """
        try:
            # Stack observations into batch
            obs_batch = np.stack(observations)

            # Convert to tensor and move to device
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)

            # Inference
            with torch.no_grad():
                action_tensor = self.model(obs_tensor)

            # Convert back to numpy
            actions = action_tensor.cpu().numpy()

            # Resolve futures
            for i, future in enumerate(futures):
                future.set_result(actions[i])

                # Track latency
                latency = time.perf_counter() - start_times[i]
                self.total_latency += latency

            self.total_batches += 1

        except Exception as e:
            logger.error(f"Batch processing error: {e}", exc_info=True)

            # Resolve futures with error
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    def get_statistics(self) -> Dict[str, float]:
        """
        Get inference statistics

        :return: Dictionary of performance metrics
        """
        if self.total_requests == 0:
            return {}

        avg_latency_ms = (self.total_latency / self.total_requests) * 1000
        avg_batch_size = self.total_requests / self.total_batches if self.total_batches > 0 else 0

        return {
            'total_requests': self.total_requests,
            'total_batches': self.total_batches,
            'avg_latency_ms': avg_latency_ms,
            'avg_batch_size': avg_batch_size,
            'requests_per_second': self.total_requests / self.total_latency if self.total_latency > 0 else 0
        }

    def reset_statistics(self):
        """Reset inference statistics"""
        self.total_requests = 0
        self.total_batches = 0
        self.total_latency = 0.0


class MultiModelAsyncInference:
    """
    Manages multiple models with async batching

    Supports hot-swapping models and load balancing across model versions
    """

    def __init__(self):
        """Initialize multi-model inference manager"""
        self.models: Dict[str, AsyncBatchInference] = {}
        logger.info("MultiModelAsyncInference initialized")

    async def register_model(
        self,
        name: str,
        model: torch.nn.Module,
        max_batch_size: int = 32,
        max_latency_ms: float = 10.0
    ):
        """
        Register a model for inference

        :param name: Model name/identifier
        :param model: PyTorch model
        :param max_batch_size: Max batch size for this model
        :param max_latency_ms: Max latency for this model
        """
        inference_server = AsyncBatchInference(
            model=model,
            max_batch_size=max_batch_size,
            max_latency_ms=max_latency_ms
        )

        await inference_server.start()

        self.models[name] = inference_server

        logger.info(f"Registered model: {name}")

    async def infer(self, model_name: str, observation: np.ndarray) -> np.ndarray:
        """
        Infer action using specified model

        :param model_name: Name of model to use
        :param observation: Observation array
        :return: Action array
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered")

        return await self.models[model_name].infer(observation)

    async def hot_swap(self, name: str, new_model: torch.nn.Module):
        """
        Hot-swap a model without downtime

        :param name: Model name to replace
        :param new_model: New model instance
        """
        logger.info(f"Hot-swapping model: {name}")

        # Create new inference server
        new_server = AsyncBatchInference(
            model=new_model,
            max_batch_size=self.models[name].max_batch_size,
            max_latency_ms=self.models[name].max_latency_sec * 1000
        )

        await new_server.start()

        # Stop old server
        old_server = self.models[name]
        await old_server.stop()

        # Replace
        self.models[name] = new_server

        logger.info(f"Model {name} hot-swapped successfully")

    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all models"""
        return {
            name: server.get_statistics()
            for name, server in self.models.items()
        }

    async def shutdown(self):
        """Shutdown all models"""
        logger.info("Shutting down all inference servers...")

        for name, server in self.models.items():
            await server.stop()

        self.models.clear()

        logger.info("All inference servers shutdown")


# Example usage
async def example_usage():
    """Example of using async batch inference"""

    # Create model (example)
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 3)
    )

    # Create inference server
    server = AsyncBatchInference(
        model=model,
        max_batch_size=16,
        max_latency_ms=5.0
    )

    await server.start()

    # Make concurrent requests
    tasks = []
    for i in range(100):
        obs = np.random.randn(10).astype(np.float32)
        task = server.infer(obs)
        tasks.append(task)

    # Wait for all results
    results = await asyncio.gather(*tasks)

    # Get statistics
    stats = server.get_statistics()
    print(f"Processed {stats['total_requests']} requests in {stats['total_batches']} batches")
    print(f"Average latency: {stats['avg_latency_ms']:.2f} ms")
    print(f"Average batch size: {stats['avg_batch_size']:.1f}")

    await server.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())
