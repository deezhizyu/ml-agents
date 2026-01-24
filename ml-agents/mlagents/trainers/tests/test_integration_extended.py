"""
Extended integration tests - Phase 3
"""
import pytest
import numpy as np
from mlagents.trainers.settings import (
    TrainerSettings,
    NetworkSettings,
    RewardSignalSettings,
    RewardSignalType,
)
from mlagents.trainers.ppo.optimizer_torch import PPOSettings


class TestConfigurationIntegration:
    """Test complete training configurations"""
    
    def test_complete_ppo_config_with_torchscript(self):
        """Test complete PPO configuration with TorchScript enabled"""
        network_settings = NetworkSettings(
            hidden_units=256,
            num_layers=3,
            normalize=False,
            enable_torchscript=True,
            torchscript_optimize_for_inference=True
        )
        
        hyperparameters = PPOSettings(
            batch_size=1024,
            buffer_size=10240,
            learning_rate=3.0e-4,
            beta=5.0e-3,
            epsilon=0.2,
            lambd=0.95,
            num_epoch=3
        )
        
        trainer_settings = TrainerSettings(
            trainer_type="ppo",
            hyperparameters=hyperparameters,
            network_settings=network_settings,
            max_steps=1000000,
            time_horizon=64,
            summary_freq=10000,
            reward_signals={
                RewardSignalType.EXTRINSIC: RewardSignalSettings()
            }
        )
        
        # Verify all settings are properly configured
        assert trainer_settings.trainer_type == "ppo"
        assert trainer_settings.network_settings.enable_torchscript == True
        assert trainer_settings.hyperparameters.batch_size == 1024
        assert trainer_settings.max_steps == 1000000
        
        # Verify network settings
        assert trainer_settings.network_settings.hidden_units == 256
        assert trainer_settings.network_settings.num_layers == 3
        
        # Verify hyperparameters
        assert trainer_settings.hyperparameters.learning_rate == 3.0e-4
        assert trainer_settings.hyperparameters.num_epoch == 3
    
    def test_multi_reward_signal_configuration(self):
        """Test configuration with multiple reward signals"""
        trainer_settings = TrainerSettings(
            trainer_type="ppo",
            hyperparameters=PPOSettings(),
            reward_signals={
                RewardSignalType.EXTRINSIC: RewardSignalSettings(
                    gamma=0.99,
                    strength=1.0
                ),
                RewardSignalType.CURIOSITY: RewardSignalSettings(
                    gamma=0.99,
                    strength=0.01
                ),
                RewardSignalType.GAIL: RewardSignalSettings(
                    gamma=0.99,
                    strength=0.01
                )
            }
        )
        
        # Should have all three reward signals
        assert len(trainer_settings.reward_signals) == 3
        assert RewardSignalType.EXTRINSIC in trainer_settings.reward_signals
        assert RewardSignalType.CURIOSITY in trainer_settings.reward_signals
        assert RewardSignalType.GAIL in trainer_settings.reward_signals
    
    def test_recurrent_network_configuration(self):
        """Test configuration with recurrent network"""
        memory_settings = NetworkSettings.MemorySettings(
            sequence_length=64,
            memory_size=128
        )
        
        network_settings = NetworkSettings(
            hidden_units=128,
            num_layers=2,
            memory=memory_settings,
            enable_torchscript=True
        )
        
        trainer_settings = TrainerSettings(
            trainer_type="ppo",
            hyperparameters=PPOSettings(),
            network_settings=network_settings,
            max_steps=500000
        )
        
        # Verify memory settings with specific expected values
        assert trainer_settings.network_settings.memory is not None
        assert trainer_settings.network_settings.memory.sequence_length == 64
        assert trainer_settings.network_settings.memory.memory_size == 128
        # Verify memory is properly configured as MemorySettings instance
        from mlagents.trainers.settings import MemorySettings
        assert isinstance(trainer_settings.network_settings.memory, MemorySettings)


class TestPhase2Integration:
    """Integration tests for Phase 2 features"""
    
    def test_torchscript_optimization_config(self):
        """Test TorchScript optimization configuration"""
        network_settings = NetworkSettings(
            enable_torchscript=True,
            torchscript_optimize_for_inference=True
        )
        
        assert network_settings.enable_torchscript == True
        assert network_settings.torchscript_optimize_for_inference == True
    
    def test_profiling_compatible_config(self):
        """Test configuration compatible with profiling"""
        # Configuration that works well with profiling
        trainer_settings = TrainerSettings(
            trainer_type="ppo",
            hyperparameters=PPOSettings(
                batch_size=512,  # Smaller batch for profiling
                buffer_size=5120
            ),
            network_settings=NetworkSettings(
                hidden_units=128,  # Smaller network
                num_layers=2
            ),
            max_steps=100000,
            summary_freq=1000  # Frequent summaries
        )
        
        assert trainer_settings.hyperparameters.batch_size == 512
        assert trainer_settings.network_settings.hidden_units == 128


class TestConfigurationValidation:
    """Test configuration validation"""
    
    def test_valid_learning_rate_range(self):
        """Test valid learning rate values"""
        for lr in [1e-5, 3e-4, 1e-3, 1e-2]:
            settings = PPOSettings(learning_rate=lr)
            assert settings.learning_rate == lr
    
    def test_valid_batch_sizes(self):
        """Test valid batch size values"""
        for batch_size in [64, 128, 256, 512, 1024, 2048]:
            settings = PPOSettings(batch_size=batch_size)
            assert settings.batch_size == batch_size
    
    def test_valid_network_sizes(self):
        """Test valid network architecture sizes"""
        for hidden_units in [32, 64, 128, 256, 512]:
            for num_layers in [1, 2, 3, 4]:
                settings = NetworkSettings(
                    hidden_units=hidden_units,
                    num_layers=num_layers
                )
                assert settings.hidden_units == hidden_units
                assert settings.num_layers == num_layers


class TestCurriculumIntegration:
    """Integration tests for curriculum learning (Phase 3)"""
    
    def test_curriculum_with_trainer_settings(self):
        """Test curriculum learning integration with trainer settings"""
        from mlagents.trainers.settings import Lesson, CompletionCriteriaSettings, ConstantSettings
        
        # Create curriculum lessons
        lessons = [
            Lesson(
                name="Easy",
                value=ConstantSettings(value=1.0),
                completion_criteria=CompletionCriteriaSettings(
                    behavior="TestBehavior",
                    measure=CompletionCriteriaSettings.MeasureType.REWARD,
                    threshold=0.5,
                    min_lesson_length=100
                )
            ),
            Lesson(
                name="Hard",
                value=ConstantSettings(value=10.0),
                completion_criteria=None
            )
        ]
        
        # Should be able to create lessons
        assert lessons is not None
        assert len(lessons) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
