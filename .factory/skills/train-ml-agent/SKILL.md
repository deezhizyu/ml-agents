---
name: Train ML Agent
description: Train a Unity ML-Agents reinforcement learning agent using PPO, SAC, or MA-POCA algorithms
---

# Train ML Agent Skill

Use this skill to train reinforcement learning agents in Unity ML-Agents environments.

## When to Use

- Starting a new training run for an ML agent
- Training agents in Unity environments (3DBall, Hallway, etc.)
- Experimenting with different hyperparameters
- Training cooperative or competitive multi-agent scenarios

## Prerequisites

- ML-Agents packages installed (`pip install -e ./ml-agents-envs ./ml-agents`)
- Unity environment built or available
- Training configuration file prepared (YAML)

## Basic Training

```bash
# Train with a config file
mlagents-learn config/ppo/3DBall.yaml --run-id=MyTraining_01

# Resume training from checkpoint
mlagents-learn config/ppo/3DBall.yaml --run-id=MyTraining_01 --resume

# Train with custom environment
mlagents-learn config/ppo/3DBall.yaml --run-id=MyTraining --env=./builds/3DBall
```

## Training Configuration

Key hyperparameters in YAML config:

```yaml
behaviors:
  MyBehavior:
    trainer_type: ppo  # or sac, poca
    hyperparameters:
      learning_rate: 3.0e-4
      batch_size: 1024
      buffer_size: 10240
      beta: 5.0e-3
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
    max_steps: 500000
    time_horizon: 64
    summary_freq: 10000
```

## Algorithms

- **PPO**: General purpose, stable, good for most tasks
- **SAC**: Continuous control, off-policy, sample efficient
- **MA-POCA**: Multi-agent cooperative scenarios

## Monitoring

```bash
# Monitor training with TensorBoard
tensorboard --logdir=results

# Key metrics to watch:
# - Environment/Cumulative Reward (should increase)
# - Losses/Policy Loss (should stabilize)
# - Policy/Learning Rate (decreases if scheduled)
```

## Common Issues

- **Not learning**: Check reward signals, adjust learning rate
- **Unstable training**: Reduce learning rate, increase batch size
- **Too slow**: Enable GPU, increase parallel environments
- **Connection timeout**: Check Unity build path, verify port 5005

## Output

Training produces:
- `results/<run-id>/` - TensorBoard logs and checkpoints
- `results/<run-id>/*.onnx` - Exported ONNX models for Unity
- `results/<run-id>/configuration.yaml` - Config snapshot

## Examples

```bash
# PPO with custom hyperparameters
mlagents-learn config/ppo/3DBall.yaml --run-id=HighLR \
  --hyperparameters=learning_rate=5.0e-4

# Multi-environment training
mlagents-learn config/poca/SoccerTwos.yaml --run-id=Soccer \
  --num-envs=4

# Curriculum learning
mlagents-learn config/ppo/Pyramids.yaml --run-id=Curriculum \
  --env-args --scene=Pyramids
```

## Related Skills

- `debug-training` - Troubleshoot training issues
- `export-models` - Export trained models to Unity
- `optimize-performance` - Improve training speed
