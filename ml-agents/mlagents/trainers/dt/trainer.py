"""
Decision Transformer trainer for ML-Agents

Trains Decision Transformer on offline trajectory data
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List
import numpy as np
from mlagents.torch_utils import torch, default_device
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mlagents.trainers.trainer.rl_trainer import RLTrainer
from mlagents.trainers.dt.decision_transformer import DecisionTransformer
from mlagents.trainers.dt.trajectory_dataset import TrajectoryDataset, collate_fn

from mlagents_envs.logging_util import get_logger
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.policy.torch_policy import TorchPolicy

logger = get_logger(__name__)


class DecisionTransformerTrainer(RLTrainer):
    """
    Trainer for Decision Transformer

    Trains on offline data (demonstrations or logged trajectories)
    Can condition on desired return for zero-shot generalization
    """

    def __init__(
        self,
        behavior_name: str,
        reward_buff_cap: int,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
    ):
        """
        Initialize Decision Transformer trainer

        :param behavior_name: Behavior name for this trainer
        :param reward_buff_cap: Reward buffer capacity
        :param trainer_settings: Trainer configuration settings
        :param training: Whether in training mode
        :param load: Whether to load existing model
        :param seed: Random seed
        :param artifact_path: Path for saving artifacts
        """
        super().__init__(
            behavior_name,
            reward_buff_cap,
            trainer_settings,
            training,
            load,
            seed,
            artifact_path,
        )

        # Decision Transformer specific parameters
        dt_params = trainer_settings.hyperparameters

        self.hidden_dim = getattr(dt_params, "hidden_dim", 128)
        self.num_layers = getattr(dt_params, "num_layers", 3)
        self.num_heads = getattr(dt_params, "num_heads", 1)
        self.max_len = getattr(dt_params, "max_len", 20)
        self.batch_size = getattr(dt_params, "batch_size", 64)
        self.learning_rate = getattr(dt_params, "learning_rate", 1e-4)

        # Dataset
        self.dataset: TrajectoryDataset | None = None
        self.dataloader: DataLoader | None = None

        logger.info(f"DecisionTransformerTrainer initialized for {behavior_name}")

    def _create_model(self, behavior_spec: BehaviorSpec) -> None:
        """Create Decision Transformer model"""
        # Get dimensions from behavior spec
        state_dim = behavior_spec.observation_specs[0].shape[0]  # Vector obs
        action_dim = behavior_spec.action_spec.continuous_size

        # Create Decision Transformer
        self.model = DecisionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            action_tanh=True,
        ).to(default_device())

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100000  # Total training steps
        )

        logger.info("Decision Transformer model created")

    def load_dataset(
        self,
        states: list[np.ndarray],
        actions: list[np.ndarray],
        rewards: list[np.ndarray],
        terminals: list[np.ndarray],
    ):
        """
        Load offline dataset for training

        :param states: List of state trajectories
        :param actions: List of action trajectories
        :param rewards: List of reward trajectories
        :param terminals: List of terminal flags
        """
        self.dataset = TrajectoryDataset(
            states=states,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            max_len=self.max_len,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # Avoid multiprocessing issues
        )

        logger.info(f"Dataset loaded: {len(self.dataset)} trajectories")

    def _update_policy(self) -> bool:  # type: ignore[override]
        """
        Update Decision Transformer on offline data

        :return: Dictionary of training metrics
        """
        if self.dataloader is None:
            logger.warning("No dataset loaded - skipping update")
            return {}

        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for batch in self.dataloader:
            # Move batch to device
            states = batch["states"].to(default_device())
            actions = batch["actions"].to(default_device())
            returns_to_go = batch["returns_to_go"].to(default_device())
            timesteps = batch["timesteps"].to(default_device())

            # Forward pass
            action_preds = self.model(
                states=states,
                actions=actions,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
            )

            # Loss: MSE between predicted and actual actions
            loss = F.mse_loss(action_preds, actions)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.25)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Update learning rate
        self.scheduler.step()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Log metrics
        self.stats_reporter.add_stat("Losses/Decision Transformer Loss", avg_loss)
        self.stats_reporter.add_stat(
            "Policy/Learning Rate", self.scheduler.get_last_lr()[0]
        )
        return True

    def advance(self) -> None:  # type: ignore[override]
        """
        Advance the trainer by one step

        :return: Training metrics
        """
        if not self.should_still_train:
            return

        # Update policy
        self._update_policy()

        # Update step count is handled by parent class

    def create_policy(
        self,
        parsed_behavior_id: Any,
        behavior_spec: Any,
    ) -> TorchPolicy:
        """
        Create policy for Decision Transformer

        :param parsed_behavior_id: Parsed behavior ID
        :param behavior_spec: Behavior specification
        :return: TorchPolicy instance
        """
        # Create model if not exists
        if not hasattr(self, "model"):
            self._create_model(behavior_spec)

        # Create policy wrapper
        # Note: Decision Transformer has different inference interface
        # This is a simplified wrapper - full implementation would need
        # to maintain context (previous states, actions, returns)

        logger.info("Decision Transformer policy created")

        # Return standard TorchPolicy for compatibility
        # Actual implementation would need custom policy class
        return super().create_policy(parsed_behavior_id, behavior_spec)

    def _is_ready_update(self) -> bool:
        """
        Check if ready to update

        For offline learning, always ready if dataset is loaded
        """
        return self.dataset is not None

    def _process_trajectory(self, trajectory) -> None:
        """
        Process trajectory for offline learning

        Decision Transformer uses pre-collected data, so this is a no-op
        """
        pass

    def add_policy(self, parsed_behavior_id: Any, policy: Any) -> None:  # type: ignore[override]
        """Add policy to trainer"""
        self.policies[parsed_behavior_id] = policy
        logger.info(f"Policy added for {parsed_behavior_id}")
