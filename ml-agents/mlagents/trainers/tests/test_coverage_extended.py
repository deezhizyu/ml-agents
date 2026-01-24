"""
Extended coverage tests for previously untested modules - Phase 3
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from mlagents.trainers.buffer import AgentBuffer, BufferKey, ObservationKeyPrefix
from mlagents.trainers.trajectory import Trajectory, AgentExperience


class TestAgentBufferExtended:
    """Extended tests for AgentBuffer"""
    
    def test_buffer_creation(self):
        """Test creating an agent buffer"""
        buffer = AgentBuffer()
        assert buffer is not None
        assert buffer.num_experiences == 0
    
    def test_buffer_append_data(self):
        """Test appending data to buffer"""
        buffer = AgentBuffer()
        
        # Add some data
        buffer[BufferKey.CONTINUOUS_ACTION].append(np.array([1.0, 2.0, 3.0]))
        buffer[BufferKey.CONTINUOUS_ACTION].append(np.array([4.0, 5.0, 6.0]))
        
        assert len(buffer[BufferKey.CONTINUOUS_ACTION]) == 2
    
    def test_buffer_key_access(self):
        """Test accessing buffer keys"""
        buffer = AgentBuffer()
        
        # Create observation key
        obs_key = ObservationKeyPrefix.OBSERVATION, 0
        buffer[obs_key].append(np.array([[1.0, 2.0], [3.0, 4.0]]))
        
        assert obs_key in buffer
        assert len(buffer[obs_key]) == 1
    
    def test_buffer_truncation(self):
        """Test buffer truncation"""
        buffer = AgentBuffer()
        
        # Add many experiences
        for i in range(100):
            buffer[BufferKey.CONTINUOUS_ACTION].append(np.array([float(i)]))
        
        # Truncate to 50
        buffer.truncate(50)
        
        # Should have 50 experiences
        assert len(buffer[BufferKey.CONTINUOUS_ACTION]) == 50
    
    def test_buffer_reset(self):
        """Test buffer reset"""
        buffer = AgentBuffer()
        
        # Add data
        buffer[BufferKey.CONTINUOUS_ACTION].append(np.array([1.0]))
        buffer[BufferKey.DISCRETE_ACTION].append(np.array([1]))
        
        # Reset
        buffer.reset_agent()
        
        # Should be empty
        assert buffer.num_experiences == 0


class TestTrajectoryExtended:
    """Extended tests for Trajectory"""
    
    def test_trajectory_creation(self):
        """Test creating a trajectory"""
        # Create mock agent experiences
        experiences = []
        for i in range(10):
            exp = AgentExperience(
                obs=[np.array([1.0, 2.0])],
                reward=float(i),
                done=False,
                action=np.array([0.5]),
                action_probs=np.array([0.5]),
                action_mask=None,
                prev_action=np.array([0.0]),
                interrupted=False,
                memory=None,
                group_status=[],
                group_reward=[]
            )
            experiences.append(exp)
        
        trajectory = Trajectory(
            steps=experiences,
            next_obs=[np.array([1.0, 2.0])],
            agent_id="test_agent",
            behavior_id="test_behavior",
            next_group_obs=[]
        )
        
        assert trajectory is not None
        assert len(trajectory.steps) == 10
        assert trajectory.agent_id == "test_agent"


class TestRewardSignalExtended:
    """Extended tests for reward signals"""
    
    def test_extrinsic_reward_signal(self):
        """Test extrinsic reward signal"""
        from mlagents.trainers.settings import RewardSignalSettings, RewardSignalType
        
        settings = RewardSignalSettings(
            gamma=0.99,
            strength=1.0
        )
        
        assert settings.gamma == 0.99
        assert settings.strength == 1.0
    
    def test_curiosity_reward_signal(self):
        """Test curiosity reward signal settings"""
        from mlagents.trainers.settings import CuriositySettings
        
        settings = CuriositySettings(
            gamma=0.99,
            strength=0.01,
            encoding_size=256
        )
        
        assert settings.gamma == 0.99
        assert settings.strength == 0.01
        assert settings.encoding_size == 256
        assert settings.encoding_size == 256


class TestEnvironmentParametersExtended:
    """Extended tests for environment parameters"""
    
    def test_parameter_randomization(self):
        """Test parameter randomization settings"""
        from mlagents.trainers.settings import UniformSettings
        
        settings = UniformSettings(
            min_value=0.0,
            max_value=1.0
        )
        
        assert settings.min_value == 0.0
        assert settings.max_value == 1.0
    
    def test_gaussian_sampler(self):
        """Test Gaussian sampler settings"""
        from mlagents.trainers.settings import GaussianSettings
        
        settings = GaussianSettings(
            mean=0.5,
            st_dev=0.1
        )
        
        assert settings.mean == 0.5
        assert settings.st_dev == 0.1
    
    def test_constant_sampler(self):
        """Test constant sampler settings"""
        from mlagents.trainers.settings import ConstantSettings
        
        settings = ConstantSettings(
            value=1.0
        )
        
        assert settings.value == 1.0


class TestScheduleTypeExtended:
    """Extended tests for schedule types"""
    
    def test_linear_schedule(self):
        """Test linear schedule type"""
        from mlagents.trainers.settings import ScheduleType
        
        schedule = ScheduleType.LINEAR
        assert schedule.value == "linear"
    
    def test_constant_schedule(self):
        """Test constant schedule type"""
        from mlagents.trainers.settings import ScheduleType
        
        schedule = ScheduleType.CONSTANT
        assert schedule.value == "constant"


class TestEncoderTypeExtended:
    """Extended tests for encoder types"""
    
    def test_simple_encoder(self):
        """Test simple encoder type"""
        from mlagents.trainers.settings import EncoderType
        
        encoder = EncoderType.SIMPLE
        assert encoder.value == "simple"
    
    def test_nature_cnn_encoder(self):
        """Test Nature CNN encoder type"""
        from mlagents.trainers.settings import EncoderType
        
        encoder = EncoderType.NATURE_CNN
        assert encoder.value == "nature_cnn"
    
    def test_resnet_encoder(self):
        """Test ResNet encoder type"""
        from mlagents.trainers.settings import EncoderType
        
        encoder = EncoderType.RESNET
        assert encoder.value == "resnet"


class TestTrainerTypeExtended:
    """Extended tests for trainer types"""
    
    def test_ppo_trainer_type(self):
        """Test PPO trainer type"""
        from mlagents.trainers.settings import TrainerSettings
        from mlagents.trainers.ppo.optimizer_torch import PPOSettings
        
        ppo_hyper = PPOSettings()
        settings = TrainerSettings(trainer_type="ppo", hyperparameters=ppo_hyper)
        assert settings.trainer_type == "ppo"
    
    def test_sac_trainer_type(self):
        """Test SAC trainer type"""
        from mlagents.trainers.settings import TrainerSettings
        from mlagents.trainers.sac.optimizer_torch import SACSettings
        
        sac_hyper = SACSettings()
        settings = TrainerSettings(trainer_type="sac", hyperparameters=sac_hyper)
        assert settings.trainer_type == "sac"
    
    def test_poca_trainer_type(self):
        """Test POCA trainer type"""
        from mlagents.trainers.settings import TrainerSettings
        from mlagents.trainers.poca.optimizer_torch import POCASettings
        
        poca_hyper = POCASettings()
        settings = TrainerSettings(trainer_type="poca", hyperparameters=poca_hyper)
        assert settings.trainer_type == "poca"


class TestBehaviorSpecExtended:
    """Extended tests for behavior specifications"""
    
    def test_behavior_spec_creation(self):
        """Test creating behavior spec"""
        from mlagents_envs.base_env import BehaviorSpec, ActionSpec
        from mlagents_envs.base_env import ObservationSpec, DimensionProperty, ObservationType
        
        obs_spec = ObservationSpec(
            name="obs",
            shape=(3,),
            observation_type=ObservationType.DEFAULT,
            dimension_property=(DimensionProperty.UNSPECIFIED,)
        )
        
        action_spec = ActionSpec(
            continuous_size=2,
            discrete_branches=()
        )
        
        spec = BehaviorSpec(
            observation_specs=[obs_spec],
            action_spec=action_spec
        )
        
        assert spec is not None
        assert len(spec.observation_specs) == 1
        assert spec.action_spec.continuous_size == 2


class TestCheckpointSettingsExtended:
    """Extended tests for checkpoint settings"""
    
    def test_checkpoint_interval(self):
        """Test checkpoint interval settings"""
        from mlagents.trainers.settings import CheckpointSettings
        
        settings = CheckpointSettings(
            run_id="test_run",
            initialize_from=None,
            load_model=False,
            resume=False,
            force=False,
            train_model=True,
            inference=False,
            results_dir="results"
        )
        
        assert settings.run_id == "test_run"
        assert settings.train_model == True
        assert settings.inference == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
