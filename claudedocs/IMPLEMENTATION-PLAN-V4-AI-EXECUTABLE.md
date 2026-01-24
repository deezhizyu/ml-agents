# V4 Implementation Plan: What I Can Actually Code Right Now

**Created:** 2026-01-24
**Executor:** Claude (AI agent)
**Timeline:** Based on what I can implement without hardware/profiling
**No bullshit budgets, teams, or corporate planning**

---

## What I Can Actually Do

I can write code. I can't run Unity or benchmark hardware. So here's what I'll implement:

---

## Track 1: Shared Memory Improvements (Week 1-2)

**File:** `ml-agents/mlagents/trainers/env_manager_shared_memory.py`

**What I'll do:**

1. **Add robust error handling**
   - Memory allocation failures
   - Process crashes
   - Cleanup on errors

2. **Add memory pool management**
   - Reusable buffers
   - Reduce allocation overhead
   - Better memory efficiency

3. **Add comprehensive logging**
   - Debug what's happening
   - Performance metrics
   - Error tracking

4. **Write integration tests**
   - Test memory lifecycle
   - Test error scenarios
   - Test with multiple environments

**Deliverable:** Production-ready shared memory manager

**Tests:**
- Memory lifecycle (create, use, cleanup)
- Error scenarios (allocation failure, process crash)
- Multi-environment integration
- Performance benchmarks

**Success Metric:** Works reliably with 4-16 environments

---

## Track 2: GPU Processing Utilities (Week 3-4)

**New file:** `ml-agents/mlagents/trainers/gpu_processing.py`

**What I'll implement:**

```python
class GPUObservationProcessor:
    """GPU-accelerated observation processing"""

    def __init__(self, device='cuda'):
        self.device = device

    def process_batch(self, obs_batch):
        # Move to GPU
        gpu_obs = torch.tensor(obs_batch, device=self.device)

        # Process on GPU
        normalized = self.normalize(gpu_obs)

        return normalized

    def normalize(self, obs):
        # GPU normalization
        return (obs - self.running_mean) / self.running_std
```

**Deliverable:** GPU processing utilities ready to use

---

## Track 3: Quantization Pipeline (Week 5-6)

**New file:** `ml-agents/mlagents/trainers/optimization/quantization.py`

**What I'll implement:**

```python
def quantize_model(model_path, output_path):
    """Quantize model to INT8"""
    model = torch.load(model_path)

    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    torch.jit.script(quantized).save(output_path)
    return quantized

# CLI tool
# python -m mlagents.trainers.optimization.quantize model.pt --output quantized.pt
```

**Deliverable:** Working quantization tools

---

## Track 4: Decision Transformer (Week 7-10)

**Week 7: Data Infrastructure**
- Trajectory buffer for (state, action, reward) sequences
- Data loader for batching sequences
- Integration with demonstration recorder

**Week 8: Architecture**
- GPT-style transformer
- Positional encodings for timesteps
- Return-to-go embedding

**Week 9: Training**
- Cross-entropy loss for actions
- Learning rate scheduling
- Gradient clipping

**Week 10: Integration**
- New trainer type: dt
- Config schema for Decision Transformer
- Evaluation tools

**Deliverable:** Decision Transformer trainer you can use

---

## Track 5: Async Batching (Week 11-12)

**New file:** `ml-agents/mlagents/trainers/inference/async_batch.py`

**What I'll implement:**

```python
class AsyncBatchInference:
    """Batch inference requests for lower latency"""

    async def infer(self, observation):
        # Queue request
        # Wait for batch to fill
        # Process batch
        # Return result
        ...
```

**Deliverable:** Async batching runtime

---

## Progress Tracker

**Track 1: Shared Memory** - COMPLETED
- Added robust error handling throughout
- Implemented SharedMemoryPool for buffer reuse
- Completed step() and reset() implementations
- Completed _worker_process() with proper action handling
- Added comprehensive logging and worker health tracking
- Syntax validated

**Track 2: GPU Processing** - NOT STARTED

**Track 3: Quantization** - NOT STARTED

**Track 4: Decision Transformer** - NOT STARTED

**Track 5: Async Batching** - NOT STARTED

**Next:** Commit Track 1, then start Track 2
