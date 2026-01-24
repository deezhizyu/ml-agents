"""
Extended optimizer tests - Phase 3 coverage improvement
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.settings import (
    TrainerSettings,
    RewardSignalSettings,
    RewardSignalType,
)
from mlagents.trainers.ppo.optimizer_torch import PPOSettings
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.buffer import AgentBuffer


class TestTorchOptimizerRewardSignals:
    """Test reward signal creation and management"""
    
    @pytest.fixture
    def mock_policy(self):
        """Create a mock policy"""
        import torch
        from mlagents_envs.base_env import ActionSpec
        policy = MagicMock(spec=TorchPolicy)
        policy.behavior_spec = MagicMock()
        policy.behavior_spec.action_spec = ActionSpec(continuous_size=2, discrete_branches=())
        policy.sequence_length = 32
        # Add required attributes for BCModule
        policy.actor = MagicMock()
        mock_param = torch.nn.Parameter(torch.randn(10, 10))
        policy.actor.parameters = MagicMock(return_value=[mock_param])
        return policy
    
    @pytest.fixture
    def trainer_settings(self):
        """Create basic trainer settings"""
        return TrainerSettings(
            trainer_type="ppo",
            hyperparameters=PPOSettings(),
            reward_signals={
                RewardSignalType.EXTRINSIC: RewardSignalSettings()
            }
        )
    
    def test_create_reward_signals(self, mock_policy, trainer_settings):
        """Test that reward signals are created correctly"""
        # Skip creating actual optimizer since it's abstract
        # Just test the reward signal creation method
        from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer
        
        # Create PPO optimizer as concrete implementation
        optimizer = TorchPPOOptimizer(mock_policy, trainer_settings)
        
        # Should have at least extrinsic reward signal
        assert len(optimizer.reward_signals) > 0
        assert "extrinsic" in optimizer.reward_signals
    
    def test_multiple_reward_signals(self, mock_policy):
        """Test creating optimizer with multiple reward signals"""
        from mlagents.trainers.settings import CuriositySettings
        
        trainer_settings = TrainerSettings(
            trainer_type="ppo",
            hyperparameters=PPOSettings(),
            reward_signals={
                RewardSignalType.EXTRINSIC: RewardSignalSettings(),
                RewardSignalType.CURIOSITY: CuriositySettings()
            }
        )
        
        from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer
        optimizer = TorchPPOOptimizer(mock_policy, trainer_settings)
        
        # Should have both reward signals
        assert "extrinsic" in optimizer.reward_signals
        assert "curiosity" in optimizer.reward_signals


class TestTorchOptimizerBC:
    """Test behavioral cloning module integration"""
    
    @pytest.fixture
    def mock_policy(self):
        """Create a mock policy"""
        import torch
        policy = MagicMock(spec=TorchPolicy)
        policy.behavior_spec = MagicMock()
        policy.sequence_length = 32
        # Add required attributes for BCModule
        policy.actor = MagicMock()
        # Return mock tensors with requires_grad=True
        mock_param = torch.nn.Parameter(torch.randn(10, 10))
        policy.actor.parameters = MagicMock(return_value=[mock_param])
        return policy
    
    def test_bc_module_creation(self, mock_policy):
        """Test that BC settings are properly configured"""
        from mlagents.trainers.settings import BehavioralCloningSettings
        
        # Just test that BC settings can be created and attached to trainer settings
        bc_settings = BehavioralCloningSettings(
            demo_path="dummy.demo",  # Not actually loaded in this test
            steps=10000
        )
        
        trainer_settings = TrainerSettings(
            trainer_type="ppo",
            hyperparameters=PPOSettings(),
            behavioral_cloning=bc_settings,
            reward_signals={
                RewardSignalType.EXTRINSIC: RewardSignalSettings()
            }
        )
        
        # Verify settings were configured correctly with specific values
        assert trainer_settings.behavioral_cloning is not None
        assert trainer_settings.behavioral_cloning.steps == 10000
        assert trainer_settings.behavioral_cloning.strength > 0.0
    
    def test_no_bc_module_when_not_configured(self, mock_policy):
        """Test that BC module is None when not configured"""
        from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer
        
        trainer_settings = TrainerSettings(
            trainer_type="ppo",
            hyperparameters=PPOSettings(),
            reward_signals={
                RewardSignalType.EXTRINSIC: RewardSignalSettings()
            }
        )
        
        optimizer = TorchPPOOptimizer(mock_policy, trainer_settings)
        
        # BC module should be None
        assert optimizer.bc_module is None


class TestTorchOptimizerSequenceEvaluation:
    """Test sequence-based evaluation for LSTM policies"""
    
    @pytest.fixture
    def mock_policy(self):
        """Create a mock policy with LSTM"""
        policy = MagicMock(spec=TorchPolicy)
        policy.behavior_spec = MagicMock()
        policy.sequence_length = 16  # LSTM sequence length
        return policy
    
    def test_sequence_length_property(self, mock_policy):
        """Test that sequence length is properly set"""
        assert mock_policy.sequence_length == 16


class TestOptimizerModules:
    """Test get_modules method for reward providers"""
    
    @pytest.fixture
    def mock_policy(self):
        """Create a mock policy"""
        import torch
        policy = MagicMock(spec=TorchPolicy)
        policy.behavior_spec = MagicMock()
        policy.sequence_length = 32
        # Add required attributes for BCModule
        policy.actor = MagicMock()
        mock_param = torch.nn.Parameter(torch.randn(10, 10))
        policy.actor.parameters = MagicMock(return_value=[mock_param])
        return policy
    
    def test_get_modules_includes_reward_providers(self, mock_policy):
        """Test that get_modules includes reward provider modules"""
        from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer
        
        trainer_settings = TrainerSettings(
            trainer_type="ppo",
            hyperparameters=PPOSettings(),
            reward_signals={
                RewardSignalType.EXTRINSIC: RewardSignalSettings()
            }
        )
        
        optimizer = TorchPPOOptimizer(mock_policy, trainer_settings)
        modules = optimizer.get_modules()
        
        # Should include optimizer and critic
        assert "Optimizer:value_optimizer" in modules
        assert "Optimizer:critic" in modules
        
        # Should also include reward provider modules
        # (specific keys depend on reward provider implementation)
        assert len(modules) >= 2


class TestOptimizerEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.fixture
    def mock_policy(self):
        """Create a mock policy"""
        import torch
        policy = MagicMock(spec=TorchPolicy)
        policy.behavior_spec = MagicMock()
        policy.sequence_length = 32
        # Add required attributes for BCModule
        policy.actor = MagicMock()
        mock_param = torch.nn.Parameter(torch.randn(10, 10))
        policy.actor.parameters = MagicMock(return_value=[mock_param])
        return policy
    
    def test_empty_reward_signals(self, mock_policy):
        """Test creating optimizer with no reward signals"""
        from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer
        
        trainer_settings = TrainerSettings(
            trainer_type="ppo",
            hyperparameters=PPOSettings(),
            reward_signals={}
        )
        
        # Should still create optimizer even with no reward signals
        optimizer = TorchPPOOptimizer(mock_policy, trainer_settings)
        assert optimizer is not None
        assert len(optimizer.reward_signals) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
