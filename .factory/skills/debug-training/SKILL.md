---
name: Debug Training Issues
description: Diagnose and fix common ML-Agents training problems including convergence issues, connection failures, and performance bottlenecks
---

# Debug Training Issues Skill

Systematically diagnose and resolve ML-Agents training problems.

## When to Use

- Agent not learning or reward staying flat
- Unity environment connection failures
- Training running but no progress
- GPU not being utilized
- Out of memory errors

## Quick Diagnostics

### 1. Check Training is Running

```bash
# Monitor GPU usage
nvidia-smi

# Check TensorBoard
tensorboard --logdir=results/<run-id>

# Look for:
# - Environment/Cumulative Reward updates
# - Policy/Learning Rate changes
# - Losses/Policy Loss values
```

### 2. Verify Environment Connection

```bash
# Check if Unity environment is accessible
ls -la ./builds/  # or your env path

# Test port availability
netstat -an | grep 5005  # Linux/Mac
netstat -an | findstr 5005  # Windows

# Try with explicit path
mlagents-learn config.yaml --run-id=test --env=./builds/MyEnv/MyEnv.x86_64
```

### 3. Check Reward Signal

```python
# In Unity Agent script:
public override void OnEpisodeBegin()
{
    Debug.Log("Episode started");
}

public override void CollectObservations(VectorSensor sensor)
{
    Debug.Log($"Observations: {sensor.ObservationSize()}");
}

public void OnActionReceived(ActionBuffers actions)
{
    Debug.Log($"Reward this step: {GetCumulativeReward()}");
}
```

## Common Problems & Solutions

### Problem: Agent Not Learning

**Symptoms:**
- Flat cumulative reward
- Random behavior
- No improvement over time

**Diagnosis:**
```bash
# Check TensorBoard metrics
tensorboard --logdir=results

# Verify observations are being collected
# Check Unity logs for reward values
```

**Solutions:**
1. Adjust reward shaping (add intermediate rewards)
2. Normalize observations to [-1, 1] or [0, 1]
3. Reduce learning rate (try 1e-4 instead of 3e-4)
4. Increase max_steps (try 2M instead of 500K)
5. Check reward scale (too small or too large)

### Problem: Connection Timeout

**Symptoms:**
```
UnityTimeOutException: The Unity environment took too long to respond
```

**Solutions:**
```bash
# 1. Check Unity build exists
ls ./builds/MyEnv/

# 2. Increase timeout
# In Python:
from mlagents_envs.environment import UnityEnvironment
env = UnityEnvironment(file_name="path", timeout_wait=120)

# 3. Check firewall
# 4. Verify no other process using port 5005
```

### Problem: GPU Not Used

**Symptoms:**
- Training slow
- GPU utilization 0%
- CPU maxed out

**Solutions:**
```bash
# 1. Verify CUDA available
python -c "import torch; print(torch.cuda.is_available())"

# 2. Check PyTorch version
pip show torch

# 3. Reinstall with CUDA support
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 4. Set environment variable
export CUDA_VISIBLE_DEVICES=0
```

### Problem: Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```yaml
# In config.yaml, reduce:
hyperparameters:
  batch_size: 512  # Down from 1024
  buffer_size: 5120  # Down from 10240

# Also reduce:
num_envs: 1  # Train with single environment first
```

### Problem: NaN Loss

**Symptoms:**
- Policy loss shows NaN in TensorBoard
- Training crashes or produces invalid models

**Solutions:**
1. Reduce learning rate significantly (1e-5)
2. Check for NaN in observations
3. Enable gradient clipping in config
4. Normalize inputs properly
5. Check reward scale (avoid extremely large rewards)

## Debug Mode

Enable verbose logging:

```bash
# Run with debug flag
mlagents-learn config.yaml --run-id=debug --debug

# Set Python logging level
export MLAGENTS_LOG_LEVEL=DEBUG
mlagents-learn config.yaml --run-id=debug
```

## Profiling

```bash
# Profile Python training loop
python -m cProfile -o training.prof \
  -m mlagents.trainers.learn \
  config.yaml --run-id=profile

# Analyze profile
python -m pstats training.prof
```

## Testing with Example Environments

Verify setup works with known-good environment:

```bash
# Train on 3DBall (should converge quickly)
mlagents-learn config/ppo/3DBall.yaml --run-id=baseline_test

# Watch for reward increasing within 10K steps
tensorboard --logdir=results
```

If baseline works but your environment doesn't, issue is in your custom environment.

## Logs to Check

1. **TensorBoard logs**: `results/<run-id>/`
2. **Unity Player logs**:
   - Windows: `%USERPROFILE%\AppData\LocalLow\CompanyName\ProductName\Player.log`
   - Mac: `~/Library/Logs/Company Name/Product Name/Player.log`
   - Linux: `~/.config/unity3d/CompanyName/ProductName/Player.log`
3. **Python stdout/stderr**: Terminal output

## Related Resources

- Runbook: `docs/runbooks/training-convergence.md`
- Runbook: `docs/runbooks/gpu-memory.md`
- Unity Discussions: https://discussions.unity.com/tag/ml-agents
- GitHub Issues: https://github.com/Unity-Technologies/ml-agents/issues

## Related Skills

- `train-ml-agent` - Basic training workflow
- `optimize-performance` - Speed up training
- `export-models` - Export after successful training
