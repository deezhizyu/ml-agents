"""
Optimizer utility classes for common patterns across PPO, SAC, and POCA optimizers.

These utilities extract common initialization and scheduling patterns to reduce
code duplication and improve maintainability.
"""

from typing import Tuple, Optional
from mlagents.torch_utils import torch, is_amp_enabled, create_optimizer
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.settings import ScheduleType


class OptimizerSetupHelper:
    """
    Helper class for common optimizer setup patterns.
    
    Provides static methods for creating optimizers with AMP support
    and managing related training infrastructure.
    """

    @staticmethod
    def create_with_amp(
        params, learning_rate: float
    ) -> Tuple[torch.optim.Optimizer, bool, Optional[torch.amp.GradScaler]]:
        """
        Create an optimizer with automatic mixed precision (AMP) support if available.

        :param params: Model parameters to optimize
        :param learning_rate: Initial learning rate
        :return: Tuple of (optimizer, use_amp flag, grad_scaler or None)
        """
        optimizer = create_optimizer(params, lr=learning_rate)

        use_amp = is_amp_enabled()
        grad_scaler = torch.amp.GradScaler() if use_amp else None

        return optimizer, use_amp, grad_scaler

    @staticmethod
    def create_lr_schedule(
        schedule_type: ScheduleType,
        initial_lr: float,
        max_steps: int,
        min_lr: float = 1e-10,
    ) -> ModelUtils.DecayedValue:
        """
        Create a learning rate decay schedule.

        :param schedule_type: Type of schedule (LINEAR, CONSTANT, etc.)
        :param initial_lr: Initial learning rate
        :param max_steps: Maximum training steps
        :param min_lr: Minimum learning rate floor
        :return: DecayedValue instance for learning rate scheduling
        """
        return ModelUtils.DecayedValue(schedule_type, initial_lr, min_lr, max_steps)


class HyperparameterScheduler:
    """
    Manages multiple scheduled hyperparameters for on-policy algorithms.
    
    Simplifies the management of epsilon, beta, and learning rate schedules
    commonly used in PPO and POCA.
    """

    def __init__(self):
        self._schedules = {}

    def add_schedule(
        self, name: str, schedule_type: ScheduleType, initial: float, min_val: float, max_steps: int
    ) -> None:
        """
        Add a hyperparameter schedule.

        :param name: Name of the hyperparameter (e.g., "epsilon", "beta", "learning_rate")
        :param schedule_type: Type of schedule (LINEAR, CONSTANT, etc.)
        :param initial: Initial value
        :param min_val: Minimum value floor
        :param max_steps: Maximum training steps
        """
        self._schedules[name] = ModelUtils.DecayedValue(
            schedule_type, initial, min_val, max_steps
        )

    def get_value(self, name: str, steps: int) -> float:
        """
        Get the current value for a hyperparameter at the given step.

        :param name: Name of the hyperparameter
        :param steps: Current training step
        :return: Current hyperparameter value
        :raises KeyError: If hyperparameter name not found
        """
        if name not in self._schedules:
            raise KeyError(f"Hyperparameter '{name}' not found in scheduler")
        return self._schedules[name].get_value(steps)

    def get_all_values(self, steps: int) -> dict:
        """
        Get all current hyperparameter values at the given step.

        :param steps: Current training step
        :return: Dictionary mapping hyperparameter names to their current values
        """
        return {name: schedule.get_value(steps) for name, schedule in self._schedules.items()}

    def has_schedule(self, name: str) -> bool:
        """
        Check if a hyperparameter schedule exists.

        :param name: Name of the hyperparameter
        :return: True if schedule exists, False otherwise
        """
        return name in self._schedules


class OnPolicyScheduleHelper:
    """
    Helper for setting up common on-policy algorithm schedules (PPO, POCA).
    
    Provides a convenient way to set up learning rate, epsilon, and beta schedules
    with sensible defaults.
    """

    @staticmethod
    def create_standard_schedules(
        learning_rate: float,
        lr_schedule: ScheduleType,
        epsilon: float,
        epsilon_schedule: ScheduleType,
        beta: float,
        beta_schedule: ScheduleType,
        max_steps: int,
    ) -> HyperparameterScheduler:
        """
        Create standard on-policy algorithm schedules.

        :param learning_rate: Initial learning rate
        :param lr_schedule: Learning rate schedule type
        :param epsilon: Initial epsilon (PPO clipping parameter)
        :param epsilon_schedule: Epsilon schedule type
        :param beta: Initial beta (entropy coefficient)
        :param beta_schedule: Beta schedule type
        :param max_steps: Maximum training steps
        :return: Configured HyperparameterScheduler
        """
        scheduler = HyperparameterScheduler()

        # Add learning rate schedule
        scheduler.add_schedule("learning_rate", lr_schedule, learning_rate, 1e-10, max_steps)

        # Add epsilon schedule
        scheduler.add_schedule("epsilon", epsilon_schedule, epsilon, 0.1, max_steps)

        # Add beta schedule
        scheduler.add_schedule("beta", beta_schedule, beta, 1e-5, max_steps)

        return scheduler
