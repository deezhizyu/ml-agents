"""
Dataset utilities for Decision Transformer training

Handles trajectory data collection, preprocessing, and batching
"""

import numpy as np
from typing import Dict, List, Optional
from mlagents.torch_utils import torch
from mlagents_envs import logging_util

logger = logging_util.get_logger(__name__)


class TrajectoryDataset:
    """
    Dataset of trajectories for Decision Transformer training

    Stores (state, action, reward) sequences and computes returns-to-go
    """

    def __init__(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[np.ndarray],
        terminals: List[np.ndarray],
        max_len: int = 20,
        discount: float = 0.99
    ):
        """
        Initialize trajectory dataset

        :param states: List of state trajectories
        :param actions: List of action trajectories
        :param rewards: List of reward trajectories
        :param terminals: List of terminal flags
        :param max_len: Maximum sequence length for training
        :param discount: Discount factor for returns-to-go
        """
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.max_len = max_len
        self.discount = discount

        # Compute returns-to-go for each trajectory
        self.returns_to_go = self._compute_returns_to_go()

        logger.info(
            f"TrajectoryDataset initialized: "
            f"{len(states)} trajectories, max_len={max_len}"
        )

    def _compute_returns_to_go(self) -> List[np.ndarray]:
        """Compute discounted returns-to-go for each trajectory"""
        returns_to_go = []

        for rewards in self.rewards:
            traj_len = len(rewards)
            rtg = np.zeros(traj_len, dtype=np.float32)

            # Compute backward
            rtg[-1] = rewards[-1]
            for t in range(traj_len - 2, -1, -1):
                rtg[t] = rewards[t] + self.discount * rtg[t + 1]

            returns_to_go.append(rtg)

        return returns_to_go

    def __len__(self) -> int:
        """Number of trajectories"""
        return len(self.states)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get a trajectory segment

        :param idx: Trajectory index
        :return: Dictionary with states, actions, returns_to_go, timesteps
        """
        # Get full trajectory
        states = self.states[idx]
        actions = self.actions[idx]
        rtg = self.returns_to_go[idx]

        traj_len = len(states)

        # Sample random starting point
        if traj_len > self.max_len:
            start_idx = np.random.randint(0, traj_len - self.max_len)
            end_idx = start_idx + self.max_len
        else:
            start_idx = 0
            end_idx = traj_len

        # Extract segment
        seg_states = states[start_idx:end_idx]
        seg_actions = actions[start_idx:end_idx]
        seg_rtg = rtg[start_idx:end_idx]

        # Timesteps (relative to trajectory start)
        timesteps = np.arange(start_idx, end_idx)

        # Pad if necessary
        seg_len = end_idx - start_idx
        if seg_len < self.max_len:
            # Pad with zeros
            pad_len = self.max_len - seg_len

            seg_states = np.concatenate([
                seg_states,
                np.zeros((pad_len, *seg_states.shape[1:]), dtype=seg_states.dtype)
            ])
            seg_actions = np.concatenate([
                seg_actions,
                np.zeros((pad_len, *seg_actions.shape[1:]), dtype=seg_actions.dtype)
            ])
            seg_rtg = np.concatenate([
                seg_rtg,
                np.zeros(pad_len, dtype=seg_rtg.dtype)
            ])
            timesteps = np.concatenate([
                timesteps,
                np.zeros(pad_len, dtype=timesteps.dtype)
            ])

        return {
            'states': seg_states.astype(np.float32),
            'actions': seg_actions.astype(np.float32),
            'returns_to_go': seg_rtg.astype(np.float32),
            'timesteps': timesteps.astype(np.int64),
            'attention_mask': np.ones(self.max_len if seg_len == self.max_len else seg_len, dtype=np.bool_)
        }


def collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader

    :param batch: List of trajectory segments
    :return: Batched tensors
    """
    states = torch.tensor(np.stack([b['states'] for b in batch]), dtype=torch.float32)
    actions = torch.tensor(np.stack([b['actions'] for b in batch]), dtype=torch.float32)
    returns_to_go = torch.tensor(np.stack([b['returns_to_go'] for b in batch]), dtype=torch.float32)
    timesteps = torch.tensor(np.stack([b['timesteps'] for b in batch]), dtype=torch.long)
    attention_mask = torch.tensor(np.stack([b['attention_mask'] for b in batch]), dtype=torch.bool)

    # Add dimension for returns_to_go (needs to be seq_len, 1)
    returns_to_go = returns_to_go.unsqueeze(-1)

    return {
        'states': states,
        'actions': actions,
        'returns_to_go': returns_to_go,
        'timesteps': timesteps,
        'attention_mask': attention_mask
    }


def load_trajectories_from_demonstrations(demo_file: str) -> TrajectoryDataset:
    """
    Load trajectories from ML-Agents demonstration file

    :param demo_file: Path to .demo file
    :return: TrajectoryDataset instance
    """
    from mlagents.trainers.demo_loader import load_demonstration

    # Load demonstration data
    behavior_spec, info_action_pair, _ = load_demonstration(demo_file)

    # Extract trajectories (group by episode)
    trajectories = {
        'states': [],
        'actions': [],
        'rewards': [],
        'terminals': []
    }

    current_traj = {
        'states': [],
        'actions': [],
        'rewards': [],
        'terminals': []
    }

    for brain_info, action_info in info_action_pair:
        # Assuming single agent for simplicity
        if len(brain_info.agents) > 0:
            obs = brain_info.visual_observations[0] if brain_info.visual_observations else brain_info.vector_observations[0]
            action = action_info.continuous_actions[0] if len(action_info.continuous_actions) > 0 else action_info.discrete_actions[0]

            current_traj['states'].append(obs)
            current_traj['actions'].append(action)
            current_traj['rewards'].append(brain_info.rewards[0] if len(brain_info.rewards) > 0 else 0.0)

            # Check if episode ended
            if len(brain_info.local_done) > 0 and brain_info.local_done[0]:
                # Episode complete
                current_traj['terminals'].append(True)

                # Save trajectory
                trajectories['states'].append(np.array(current_traj['states']))
                trajectories['actions'].append(np.array(current_traj['actions']))
                trajectories['rewards'].append(np.array(current_traj['rewards']))
                trajectories['terminals'].append(np.array(current_traj['terminals']))

                # Reset for next episode
                current_traj = {'states': [], 'actions': [], 'rewards': [], 'terminals': []}
            else:
                current_traj['terminals'].append(False)

    logger.info(f"Loaded {len(trajectories['states'])} trajectories from {demo_file}")

    return TrajectoryDataset(
        trajectories['states'],
        trajectories['actions'],
        trajectories['rewards'],
        trajectories['terminals']
    )


def load_trajectories_from_buffer(buffer_file: str) -> TrajectoryDataset:
    """
    Load trajectories from saved AgentBuffer

    :param buffer_file: Path to saved buffer (.pkl file)
    :return: TrajectoryDataset instance
    """
    import pickle

    with open(buffer_file, 'rb') as f:
        buffer = pickle.load(f)

    # Extract trajectories from buffer
    # This is a simplified implementation - actual implementation
    # would need to handle buffer structure properly

    logger.info(f"Loaded trajectories from buffer: {buffer_file}")

    # TODO: Implement proper buffer parsing
    raise NotImplementedError("Buffer loading not yet implemented")
