"""
Extended tests for settings module - Phase 3 coverage improvement
"""

import pytest
from mlagents.trainers.settings import (
    NetworkSettings,
    TrainerSettings,
    ScheduleType,
    EncoderType,
)
from mlagents.trainers.ppo.optimizer_torch import PPOSettings


class TestNetworkSettingsTorchScript:
    """Test TorchScript settings in NetworkSettings"""

    def test_default_torchscript_settings(self):
        """Test that TorchScript settings have correct defaults"""
        settings = NetworkSettings()

        assert hasattr(settings, "enable_torchscript")
        assert hasattr(settings, "torchscript_optimize_for_inference")
        assert not settings.enable_torchscript
        assert settings.torchscript_optimize_for_inference

    def test_enable_torchscript(self):
        """Test enabling TorchScript"""
        settings = NetworkSettings(enable_torchscript=True)

        assert settings.enable_torchscript
        assert settings.torchscript_optimize_for_inference

    def test_disable_torchscript_optimization(self):
        """Test disabling TorchScript inference optimization"""
        settings = NetworkSettings(
            enable_torchscript=True, torchscript_optimize_for_inference=False
        )

        assert settings.enable_torchscript
        assert not settings.torchscript_optimize_for_inference

    def test_torchscript_with_other_settings(self):
        """Test TorchScript settings combined with other network settings"""
        settings = NetworkSettings(
            hidden_units=256, num_layers=3, enable_torchscript=True, normalize=True
        )

        assert settings.hidden_units == 256
        assert settings.num_layers == 3
        assert settings.enable_torchscript
        assert settings.normalize


class TestNetworkSettingsValidation:
    """Test NetworkSettings validation"""

    def test_valid_hidden_units(self):
        """Test that hidden_units accepts valid values"""
        for units in [32, 64, 128, 256, 512]:
            settings = NetworkSettings(hidden_units=units)
            assert settings.hidden_units == units

    def test_valid_num_layers(self):
        """Test that num_layers accepts valid values"""
        for layers in [1, 2, 3, 4, 5]:
            settings = NetworkSettings(num_layers=layers)
            assert settings.num_layers == layers

    def test_encoder_types(self):
        """Test all encoder types"""
        for encoder_type in EncoderType:
            settings = NetworkSettings(vis_encode_type=encoder_type)
            assert settings.vis_encode_type == encoder_type

    def test_memory_settings(self):
        """Test memory settings for recurrent networks"""
        memory_settings = NetworkSettings.MemorySettings(
            sequence_length=128, memory_size=256
        )

        assert memory_settings.sequence_length == 128
        assert memory_settings.memory_size == 256

        settings = NetworkSettings(memory=memory_settings)
        assert settings.memory is not None
        assert settings.memory.sequence_length == 128
        assert settings.memory.memory_size == 256


class TestPPOSettingsExtended:
    """Extended PPO settings tests"""

    def test_ppo_defaults(self):
        """Test PPO default hyperparameters"""
        settings = PPOSettings()

        assert settings.beta == 5.0e-3
        assert settings.epsilon == 0.2
        assert settings.lambd == 0.95
        assert settings.num_epoch == 3

    def test_ppo_custom_values(self):
        """Test PPO with custom hyperparameters"""
        settings = PPOSettings(beta=0.01, epsilon=0.3, lambd=0.99, num_epoch=5)

        assert settings.beta == 0.01
        assert settings.epsilon == 0.3
        assert settings.lambd == 0.99
        assert settings.num_epoch == 5

    def test_ppo_learning_rate_schedule(self):
        """Test PPO with different learning rate schedules"""
        for schedule in ScheduleType:
            settings = PPOSettings(learning_rate_schedule=schedule)
            assert settings.learning_rate_schedule == schedule


class TestTrainerSettingsExtended:
    """Extended trainer settings tests"""

    def test_trainer_settings_with_ppo(self):
        """Test creating trainer settings with PPO"""
        ppo_settings = PPOSettings()
        trainer_settings = TrainerSettings(
            trainer_type="ppo", hyperparameters=ppo_settings, max_steps=1000000
        )

        assert trainer_settings.trainer_type == "ppo"
        assert trainer_settings.max_steps == 1000000
        assert isinstance(trainer_settings.hyperparameters, PPOSettings)

    def test_trainer_settings_with_network(self):
        """Test trainer settings with network configuration"""
        network_settings = NetworkSettings(
            hidden_units=256, num_layers=3, enable_torchscript=True
        )

        ppo_settings = PPOSettings()

        trainer_settings = TrainerSettings(
            trainer_type="ppo",
            hyperparameters=ppo_settings,
            network_settings=network_settings,
            max_steps=500000,
        )

        assert trainer_settings.network_settings.hidden_units == 256
        assert trainer_settings.network_settings.enable_torchscript


class TestSettingsEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_zero_hidden_units(self):
        """Test that zero hidden units can be set (though not practical)"""
        settings = NetworkSettings(hidden_units=0)
        assert settings.hidden_units == 0

    def test_zero_layers(self):
        """Test that zero layers can be set (though not practical)"""
        settings = NetworkSettings(num_layers=0)
        assert settings.num_layers == 0

    def test_very_large_hidden_units(self):
        """Test very large hidden unit values"""
        settings = NetworkSettings(hidden_units=4096)
        assert settings.hidden_units == 4096

    def test_very_large_num_layers(self):
        """Test many layers"""
        settings = NetworkSettings(num_layers=10)
        assert settings.num_layers == 10


class TestSettingsIntegration:
    """Integration tests for settings"""

    def test_settings_serialization(self):
        """Test that settings can be converted to dict"""
        settings = NetworkSettings(hidden_units=256, enable_torchscript=True)

        settings_dict = (
            settings.as_dict() if hasattr(settings, "as_dict") else vars(settings)
        )

        assert "hidden_units" in str(settings_dict)
        assert "enable_torchscript" in str(settings_dict)

    def test_complete_training_configuration(self):
        """Test a complete training configuration"""
        network_settings = NetworkSettings(
            hidden_units=128,
            num_layers=2,
            normalize=False,
            enable_torchscript=True,
            torchscript_optimize_for_inference=True,
        )

        ppo_settings = PPOSettings(beta=0.001, epsilon=0.2, lambd=0.99, num_epoch=3)

        trainer_settings = TrainerSettings(
            trainer_type="ppo",
            hyperparameters=ppo_settings,
            network_settings=network_settings,
            max_steps=1000000,
            summary_freq=10000,
        )

        # Verify all settings are correctly configured
        assert trainer_settings.network_settings.enable_torchscript
        assert trainer_settings.hyperparameters.beta == 0.001
        assert trainer_settings.max_steps == 1000000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
