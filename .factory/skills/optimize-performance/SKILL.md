---
name: Optimize Performance
description: Improve ML-Agents training speed, reduce memory usage, and optimize inference performance for faster iteration and deployment
---

# Optimize Performance Skill

Optimize ML-Agents training and inference for maximum efficiency.

## When to Use

- Training is too slow
- Running out of GPU memory
- Need faster iteration cycles
- Deploying to resource-constrained devices
- Scaling to many parallel environments

## Quick Wins

### 1. Enable GPU Training

```bash
# Verify GPU available
python -c "import torch; print(torch.cuda.is_available())"

# Should print: True

# Training automatically uses GPU if available
mlagents-learn config/ppo/3DBall.yaml --run-id=gpu_test
```

### 2. Increase Parallel Environments

```bash
# Train with multiple environments in parallel
mlagents-learn config/ppo/3DBall.yaml --run-id=parallel --num-envs=8

# More environments = faster data collection
# But requires more CPU/memory
```

### 3. Adjust Time Horizon

```yaml
# In config.yaml:
behaviors:
  MyBehavior:
    time_horizon: 128  # Increase from 64
    # Larger horizon = less frequent model updates
    # Faster training but potentially less stable
```

## Training Optimization

### Hyperparameter Tuning

```yaml
behaviors:
  MyBehavior:
    hyperparameters:
      # Larger batches = better GPU utilization
      batch_size: 2048      # Up from 1024
      buffer_size: 20480    # 10x batch_size
      
      # Reduce epochs for faster updates
      num_epoch: 3          # Down from 5
      
      # Larger learning rate = faster convergence (if stable)
      learning_rate: 5.0e-4  # Up from 3.0e-4
```

### Network Architecture

```yaml
network_settings:
  # Smaller networks train faster
  hidden_units: 64       # Down from 128 or 256
  num_layers: 2          # Down from 3
  
  # But may reduce learning capacity
  # Balance speed vs performance
```

### Summary Frequency

```yaml
behaviors:
  MyBehavior:
    summary_freq: 50000   # Up from 10000
    # Less frequent summaries = faster training
    # But less granular monitoring
```

## Memory Optimization

### GPU Memory

```yaml
# Reduce memory usage:
hyperparameters:
  batch_size: 512       # Down from 1024
  buffer_size: 5120     # 10x batch_size

# Train fewer parallel environments
num_envs: 2  # Down from 4 or 8
```

### Buffer Management

```yaml
behaviors:
  MyBehavior:
    max_steps: 1000000
    
    # Smaller buffer = less memory
    hyperparameters:
      buffer_size: 2048  # Minimum: batch_size * 2
```

## Inference Optimization

### Model Compression

```yaml
# Create smaller inference model:
network_settings:
  hidden_units: 32      # Minimal network
  num_layers: 2
  normalize: true       # Keep normalization
```

### Inference Settings in Unity

```csharp
// In Unity Behavior Parameters:
// - Model: Your .onnx model
// - Inference Device: GPU (if available)
// - Behavior Type: Inference Only

// Optimize decision frequency
public class OptimizedAgent : Agent
{
    public override void OnEpisodeBegin()
    {
        // Request decisions less frequently
        DecisionRequester.DecisionPeriod = 5;  // Every 5 steps
        // vs default 1 (every step)
    }
}
```

## Profiling

### Python Profiling

```bash
# Profile training loop
python -m cProfile -o profile.stats \
  -m mlagents.trainers.learn \
  config.yaml --run-id=profile

# View results
python -c "import pstats; p=pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

### GPU Profiling

```bash
# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Look for:
# - GPU utilization > 80% (good)
# - Memory usage (shouldn't hit limit)
# - Temperature (should be reasonable)
```

### TensorBoard Profiling

```python
# Enable profiling in custom training (advanced)
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Training code here
    pass

prof.export_chrome_trace("trace.json")
# View in chrome://tracing
```

## Performance Benchmarks

### Baseline Performance

```bash
# Test training speed
time mlagents-learn config/ppo/3DBall.yaml \
  --run-id=speed_test \
  --env=./builds/3DBall \
  --max-steps=10000

# Note time taken
# Compare after optimizations
```

### Measure FPS

```bash
# Check environment frames per second
# Look for "Steps per second" in training output

# Target: > 10K steps/sec for simple environments
# Complex environments: > 1K steps/sec
```

## Optimization Checklist

- [ ] GPU enabled and utilized
- [ ] Parallel environments configured (4-8 typical)
- [ ] Batch size optimized for GPU (1024-4096)
- [ ] Network architecture appropriate for task
- [ ] Time horizon balanced (64-128)
- [ ] Summary frequency reduced for production
- [ ] Memory usage within limits
- [ ] Decision frequency optimized in Unity
- [ ] Model size appropriate for deployment

## Common Bottlenecks

### CPU Bottleneck

**Symptoms:**
- Low GPU utilization
- High CPU usage
- Slow environment steps

**Solutions:**
- Increase `num_envs` (more parallel environments)
- Simplify environment logic
- Use compiled/optimized Unity builds

### GPU Bottleneck

**Symptoms:**
- 100% GPU utilization
- Slow training despite fast environment

**Solutions:**
- Reduce batch_size or network size
- Enable mixed precision training (advanced)
- Upgrade GPU if possible

### Memory Bottleneck

**Symptoms:**
- Training crashes with OOM
- System freezes
- Slow disk swapping

**Solutions:**
- Reduce buffer_size and batch_size
- Train fewer parallel environments
- Close other applications
- Use smaller network architecture

## Advanced Optimizations

### Mixed Precision Training

```python
# In custom training code (advanced)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    # Forward pass in FP16
    loss = compute_loss()

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Distributed Training

```bash
# Train across multiple GPUs (advanced)
# Requires custom setup
torchrun --nproc_per_node=4 \
  -m mlagents.trainers.learn \
  config.yaml --run-id=distributed
```

## Monitoring Performance

```bash
# Real-time monitoring
tensorboard --logdir=results --reload_interval=5

# Key metrics:
# - Environment/Step: Steps per second
# - Losses/Policy Loss: Should stabilize quickly
# - Policy/Learning Rate: Verify schedule
```

## Expected Performance

| Environment | Steps/sec | GPU Usage | Time to Train |
|-------------|-----------|-----------|---------------|
| 3DBall      | 20K-50K   | 20-40%    | 2-5 min       |
| GridWorld   | 15K-30K   | 30-50%    | 5-10 min      |
| Hallway     | 10K-20K   | 40-60%    | 10-20 min     |
| SoccerTwos  | 5K-10K    | 60-80%    | 30-60 min     |

*With GPU, 4-8 parallel environments*

## Related Skills

- `train-ml-agent` - Apply optimizations during training
- `debug-training` - Diagnose performance issues
- `export-models` - Create optimized inference models
