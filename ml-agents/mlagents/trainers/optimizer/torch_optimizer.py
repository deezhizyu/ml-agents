from typing import Dict, Optional, Tuple, List
from mlagents.torch_utils import torch, default_device
import numpy as np
from collections import defaultdict

from mlagents.trainers.buffer import AgentBuffer, AgentBufferField
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.torch_entities.components.bc.module import BCModule
from mlagents.trainers.torch_entities.components.reward_providers import (
    create_reward_provider,
)

from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer import Optimizer
from mlagents.trainers.settings import (
    TrainerSettings,
    RewardSignalSettings,
    RewardSignalType,
)
from mlagents.trainers.torch_entities.utils import ModelUtils


class TorchOptimizer(Optimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__()
        self.policy = policy
        self.trainer_settings = trainer_settings
        self.update_dict: Dict[str, torch.Tensor] = {}
        self.value_heads: Dict[str, torch.Tensor] = {}
        self.memory_in: torch.Tensor = None
        self.memory_out: torch.Tensor = None
        self.m_size: int = 0
        self.global_step = torch.tensor(0)
        self.bc_module: Optional[BCModule] = None
        self.create_reward_signals(trainer_settings.reward_signals)
        self.critic_memory_dict: Dict[str, torch.Tensor] = {}
        if trainer_settings.behavioral_cloning is not None:
            self.bc_module = BCModule(
                self.policy,
                trainer_settings.behavioral_cloning,
                policy_learning_rate=trainer_settings.hyperparameters.learning_rate,
                default_batch_size=trainer_settings.hyperparameters.batch_size,
                default_num_epoch=3,
            )

    @property
    def critic(self):
        raise NotImplementedError

    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        pass

    def create_reward_signals(
        self, reward_signal_configs: Dict[RewardSignalType, RewardSignalSettings]
    ) -> None:
        """
        Create reward signals
        :param reward_signal_configs: Reward signal config.
        """
        for reward_signal, settings in reward_signal_configs.items():
            # Name reward signals by string in case we have duplicates later
            self.reward_signals[reward_signal.value] = create_reward_provider(
                reward_signal, self.policy.behavior_spec, settings
            )

    def _evaluate_by_sequence(
        self, tensor_obs: List[torch.Tensor], initial_memory: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], AgentBufferField, torch.Tensor]:
        """
        Evaluate a trajectory sequence-by-sequence, assembling the result. This enables us to get the
        intermediate memories for the critic.
        :param tensor_obs: A List of tensors of shape (trajectory_len, <obs_dim>) that are the agent's
            observations for this trajectory.
        :param initial_memory: The memory that preceeds this trajectory. Of shape (1,1,<mem_size>), i.e.
            what is returned as the output of a MemoryModules.
        :return: A Tuple of the value estimates as a Dict of [name, tensor], an AgentBufferField of the initial
            memories to be used during value function update, and the final memory at the end of the trajectory.
        """
        num_experiences = tensor_obs[0].shape[0]
        all_next_memories = AgentBufferField()
        # When using LSTM, we need to divide the trajectory into sequences of equal length. Sometimes,
        # that division isn't even, and we must pad the leftover sequence.
        # When it is added to the buffer, the last sequence will be padded. So if seq_len = 3 and
        # trajectory is of length 10, the last sequence is [obs,pad,pad] once it is added to the buffer.
        # Compute the number of elements in this sequence that will end up being padded.
        leftover_seq_len = num_experiences % self.policy.sequence_length

        all_values: Dict[str, List[np.ndarray]] = defaultdict(list)
        _mem = initial_memory
        # Evaluate other trajectories, carrying over _mem after each
        # trajectory
        for seq_num in range(num_experiences // self.policy.sequence_length):
            seq_obs = []
            for _ in range(self.policy.sequence_length):
                all_next_memories.append(ModelUtils.to_numpy(_mem.squeeze()))
            start = seq_num * self.policy.sequence_length
            end = (seq_num + 1) * self.policy.sequence_length

            for _obs in tensor_obs:
                seq_obs.append(_obs[start:end])
            values, _mem = self.critic.critic_pass(
                seq_obs, _mem, sequence_length=self.policy.sequence_length
            )
            for signal_name, _val in values.items():
                all_values[signal_name].append(_val)

        # Compute values for the potentially truncated last sequence. Note that this
        # sequence isn't padded yet, but will be.
        seq_obs = []

        if leftover_seq_len > 0:
            for _obs in tensor_obs:
                last_seq_obs = _obs[-leftover_seq_len:]
                seq_obs.append(last_seq_obs)

            # For the last sequence, the initial memory should be the one at the
            # end of this trajectory.
            for _ in range(leftover_seq_len):
                all_next_memories.append(ModelUtils.to_numpy(_mem.squeeze()))

            last_values, _mem = self.critic.critic_pass(
                seq_obs, _mem, sequence_length=leftover_seq_len
            )
            for signal_name, _val in last_values.items():
                all_values[signal_name].append(_val)

        # Create one tensor per reward signal
        all_value_tensors = {
            signal_name: torch.cat(value_list, dim=0)
            for signal_name, value_list in all_values.items()
        }
        next_mem = _mem
        return all_value_tensors, all_next_memories, next_mem

    def update_reward_signals(self, batch: AgentBuffer) -> Dict[str, float]:
        update_stats: Dict[str, float] = {}
        for reward_provider in self.reward_signals.values():
            update_stats.update(reward_provider.update(batch))
        return update_stats

    def get_modules(self) -> Dict[str, any]:
        """
        Get all modules for this optimizer including reward provider modules.
        Subclasses should override to include optimizer and critic modules.
        :return: Dictionary mapping module names to modules
        """
        modules = {}
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules

    def get_trajectory_value_estimates(
        self,
        batch: AgentBuffer,
        next_obs: List[np.ndarray],
        done: bool,
        agent_id: str = "",
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Optional[AgentBufferField]]:
        """
        Get value estimates and memories for a trajectory, in batch form.
        :param batch: An AgentBuffer that consists of a trajectory.
        :param next_obs: the next observation (after the trajectory). Used for bootstrapping
            if this is not a terminal trajectory.
        :param done: Set true if this is a terminal trajectory.
        :param agent_id: Agent ID of the agent that this trajectory belongs to.
        :returns: A Tuple of the Value Estimates as a Dict of [name, np.ndarray(trajectory_len)],
            the final value estimate as a Dict of [name, float], and optionally (if using memories)
            an AgentBufferField of initial critic memories to be used during update.
        """
        n_obs = len(self.policy.behavior_spec.observation_specs)

        if agent_id in self.critic_memory_dict:
            memory = self.critic_memory_dict[agent_id]
        else:
            memory = (
                torch.zeros((1, 1, self.critic.memory_size), device=default_device())
                if self.policy.use_recurrent
                else None
            )

        # Convert to tensors
        current_obs = [
            ModelUtils.list_to_tensor(obs) for obs in ObsUtil.from_buffer(batch, n_obs)
        ]
        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]

        next_obs = [obs.unsqueeze(0) for obs in next_obs]

        # If we're using LSTM, we want to get all the intermediate memories.
        all_next_memories: Optional[AgentBufferField] = None

        # To prevent memory leak and improve performance, evaluate with no_grad.
        with torch.no_grad():
            if self.policy.use_recurrent:
                (
                    value_estimates,
                    all_next_memories,
                    next_memory,
                ) = self._evaluate_by_sequence(current_obs, memory)
            else:
                value_estimates, next_memory = self.critic.critic_pass(
                    current_obs, memory, sequence_length=batch.num_experiences
                )

        # Store the memory for the next trajectory. This should NOT have a gradient.
        self.critic_memory_dict[agent_id] = next_memory

        next_value_estimate, _ = self.critic.critic_pass(
            next_obs, next_memory, sequence_length=1
        )

        for name, estimate in value_estimates.items():
            value_estimates[name] = ModelUtils.to_numpy(estimate)
            next_value_estimate[name] = ModelUtils.to_numpy(next_value_estimate[name])

        if done:
            for k in next_value_estimate:
                if not self.reward_signals[k].ignore_done:
                    next_value_estimate[k] = 0.0
            if agent_id in self.critic_memory_dict:
                self.critic_memory_dict.pop(agent_id)
        return value_estimates, next_value_estimate, all_next_memories

    def get_batched_trajectory_value_estimates(
        self,
        batches: List[AgentBuffer],
        next_obs_list: List[List[np.ndarray]],
        dones: List[bool],
        agent_ids: List[str],
    ) -> List[Tuple[Dict[str, np.ndarray], Dict[str, float], Optional[AgentBufferField]]]:
        """
        Batched version of get_trajectory_value_estimates: runs one critic
        forward pass covering every trajectory in `batches` concatenated
        together (plus one more for all their bootstrap "next obs" values),
        instead of one pair of forward passes per trajectory. Each
        trajectory previously paid the same fixed per-call overhead
        (tensor construction, no_grad context, dispatch) for a forward
        pass over just its own few timesteps; batching amortizes that
        overhead across every trajectory that finished around the same
        time.

        Recurrent policies carry a separate hidden state per agent across
        calls, which this does not attempt to batch - falls back to the
        per-trajectory path in that case, where it's handled already.
        """
        if self.policy.use_recurrent or not batches:
            return [
                self.get_trajectory_value_estimates(
                    batch, next_obs, done, agent_id
                )
                for batch, next_obs, done, agent_id in zip(
                    batches, next_obs_list, dones, agent_ids
                )
            ]

        n_obs = len(self.policy.behavior_spec.observation_specs)
        lengths = [batch.num_experiences for batch in batches]

        per_traj_current_obs = [
            [ModelUtils.list_to_tensor(obs) for obs in ObsUtil.from_buffer(batch, n_obs)]
            for batch in batches
        ]
        current_obs = [
            torch.cat([traj_obs[obs_idx] for traj_obs in per_traj_current_obs], dim=0)
            for obs_idx in range(n_obs)
        ]

        per_traj_next_obs = [
            [ModelUtils.list_to_tensor(obs) for obs in next_obs]
            for next_obs in next_obs_list
        ]
        next_obs_batched = [
            torch.stack([traj_obs[obs_idx] for traj_obs in per_traj_next_obs], dim=0)
            for obs_idx in range(n_obs)
        ]

        with torch.no_grad():
            value_estimates, _ = self.critic.critic_pass(
                current_obs, None, sequence_length=sum(lengths)
            )
            next_value_estimates, _ = self.critic.critic_pass(
                next_obs_batched, None, sequence_length=len(batches)
            )

        for name, estimate in value_estimates.items():
            value_estimates[name] = ModelUtils.to_numpy(estimate)
        for name, estimate in next_value_estimates.items():
            next_value_estimates[name] = ModelUtils.to_numpy(estimate)

        results: List[
            Tuple[Dict[str, np.ndarray], Dict[str, float], Optional[AgentBufferField]]
        ] = []
        offset = 0
        for i, length in enumerate(lengths):
            traj_value_estimates = {
                name: v[offset : offset + length] for name, v in value_estimates.items()
            }
            traj_next_value_estimate = {
                name: v[i] for name, v in next_value_estimates.items()
            }
            if dones[i]:
                for k in traj_next_value_estimate:
                    if not self.reward_signals[k].ignore_done:
                        traj_next_value_estimate[k] = 0.0
            results.append((traj_value_estimates, traj_next_value_estimate, None))
            offset += length

        return results
