# Training Convergence Issues

## Symptoms

- Agent reward remains flat or near zero
- Agent exhibits random behavior without improvement
- Training metrics show no learning progress
- Episode length remains constant at max_steps
- TensorBoard shows no improvement in cumulative reward

## Diagnosis Steps

### 1. Verify Reward Signal

```bash
# Check TensorBoard for reward trends
tensorboard --logdir=results/<run-id>
```

Look for:
- Is reward actually being received? (check Environment/Cumulative Reward)
- Are rewards too sparse? (long periods with zero reward)
- Are rewards too dense? (reward at every step)

### 2. Check Observation Space

```python
# In your Unity Agent code, verify observations
def CollectObservations(VectorSensor sensor)
{
    // Ensure observations are:
    // 1. Normalized (typically -1 to 1 or 0 to 1)
    // 2. Relevant to the task
    // 3. Not constant values
    // 4. Not NaN or infinite
}
```

### 3. Review Hyperparameters

Check your training config (`config/ppo/*.yaml`):
- `learning_rate`: Too high (>1e-3) or too low (<1e-5)?
- `batch_size`: Should be 64-2048 for most tasks
- `buffer_size`: Should be 10-100x batch_size
- `max_steps`: Sufficient for learning? (typically 500K-5M)

### 4. Analyze Behavior

Watch the agent in Unity Editor:
- Is the agent receiving observations correctly?
- Are actions being applied as expected?
- Is the environment resetting properly?

## Resolution

### Solution 1: Adjust Reward Shaping

**If rewards are too sparse:**

```yaml
# Add reward bonuses for progress
# In your Unity Agent:
AddReward(0.01f);  // Small reward for staying alive
AddReward(progressTowardGoal * 0.1f);  // Incremental reward
```

**If rewards are too dense:**

```yaml
# Reduce reward frequency
# Give larger rewards less often
```

### Solution 2: Tune Learning Rate

```yaml
# In your config file:
hyperparameters:
  learning_rate: 3.0e-4  # Start here for most tasks
  learning_rate_schedule: linear  # Decay over time
```

Try increasing if learning is too slow, decreasing if unstable.

### Solution 3: Increase Training Steps

```yaml
behaviors:
  YourBehavior:
    max_steps: 2000000  # Increase from default
    time_horizon: 64
    summary_freq: 10000
```

### Solution 4: Normalize Observations

```python
# Use Unity's normalization
sensor.AddObservation(normalizedValue);

# Where normalizedValue is in range [-1, 1] or [0, 1]
float normalizedValue = (rawValue - minValue) / (maxValue - minValue);
```

### Solution 5: Adjust Network Architecture

```yaml
network_settings:
  hidden_units: 128  # Try 256 or 512 for complex tasks
  num_layers: 2      # Try 3 for complex observations
  normalize: true    # Enable input normalization
```

## Verification

After applying fixes:

1. **Monitor TensorBoard** for 50K-100K steps:
   ```bash
   tensorboard --logdir=results
   ```

2. **Check these metrics:**
   - Environment/Cumulative Reward: Should trend upward
   - Losses/Policy Loss: Should stabilize
   - Policy/Learning Rate: Should be stable or declining smoothly

3. **Watch agent behavior** in Unity:
   - Actions should become more purposeful
   - Agent should show signs of learning patterns

## Prevention

### Best Practices

1. **Start Simple:**
   - Use simple environments first
   - Verify learning before adding complexity
   - Use example configs as starting points

2. **Monitor Early:**
   - Check TensorBoard after first 10K steps
   - Don't wait for full training to diagnose issues

3. **Use Curriculum Learning:**
   ```yaml
   environment_parameters:
     difficulty:
       curriculum:
         - name: easy
           completion_criteria:
             measure: reward
             behavior: YourBehavior
             threshold: 0.5
           value: 1.0
         - name: hard
           value: 2.0
   ```

4. **Enable Proper Logging:**
   ```yaml
   behaviors:
     YourBehavior:
       summary_freq: 10000  # Not too frequent
       checkpoint_interval: 50000
   ```

## Related Issues

- [GitHub Issue #XXXX: Agent not learning](https://github.com/Unity-Technologies/ml-agents/issues)
- [Reward Design Best Practices](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest/index.html?subfolder=/manual/Reward-System.html)
- [Hyperparameter Tuning Guide](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest/index.html?subfolder=/manual/Training-Configuration-File.html)

## Advanced Troubleshooting

### Debug Mode

Enable verbose logging:

```bash
mlagents-learn config.yaml --run-id=debug --debug
```

### Profile Training

Check if training is actually running:

```python
# Monitor GPU utilization
watch -n 1 nvidia-smi

# Monitor CPU usage
top
```

### Compare with Baseline

Use a known-working configuration:

```bash
# Train on example environment
mlagents-learn config/ppo/3DBall.yaml --run-id=baseline
```

If example works but yours doesn't, the issue is likely in your environment setup.
