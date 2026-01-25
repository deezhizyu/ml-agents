"""
Pytest configuration for ml-agents trainer tests.

This module ensures that trainer plugins are registered before any tests run,
which is required for tests that use TrainerSettings with trainer types like 'sac' or 'ppo'.
"""
import pytest

# Register trainer plugins at module load time so all tests have access to trainer types
from mlagents.plugins.trainer_type import register_trainer_plugins
register_trainer_plugins()
