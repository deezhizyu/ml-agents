# Integration Plan: Enable New Features in Training

**What's Wrong:** I built 5 features but only quantization actually works
**What's Needed:** Integration work to make them run during training
**Timeline:** 3 tasks, do them in order

---

## Task 1: Fix SharedMemoryEnvManager Unity Connection (2-3 hours)

**Problem:** Worker process times out connecting to Unity

**Root cause:** SharedMemoryEnvManager._worker_process() doesn't handle Unity connection properly

**What to fix:**

1. The worker needs to receive the actual Unity connection parameters
2. Current code calls `env_factory(worker_id, [])` but Unity needs base_port, file_name, etc.
3. SubprocessEnvManager passes these through RunOptions - SharedMemory doesn't

**Files to modify:**
- `ml-agents/mlagents/trainers/env_manager_shared_memory.py`

**Specific changes:**

```python
# Current (BROKEN):
def _worker_process(worker_id, cmd_queue, res_queue, pickled_env_factory):
    env_factory = cloudpickle.loads(pickled_env_factory)
    env = env_factory(worker_id, [])  # ← This doesn't work for Unity

# Fixed (WORKING):
def _worker_process(worker_id, cmd_queue, res_queue, pickled_env_factory):
    from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

    env_factory = cloudpickle.loads(pickled_env_factory)

    # Create side channels like SubprocessEnvManager does
    env_parameters = EnvironmentParametersChannel()
    side_channels = [env_parameters]

    # Call factory with side channels
    env = env_factory(worker_id, side_channels)
```

**Also need:** Copy the engine config and stats channel setup from SubprocessEnvManager worker function

**Test:** Run `mlagents-learn config\ppo\3DBall.yaml --run-id=test --num-envs=1 --force` and it should connect

**Success criteria:** No timeout error, Unity connects, training starts

---

## Task 2: Enable GPU Processing in Training Loop (1-2 hours)

**Problem:** GPUObservationProcessor exists but nothing calls it

**What to fix:**

1. Hook GPU processing into the policy evaluation
2. Process observations on GPU before policy inference
3. Keep tensors on GPU throughout

**Files to modify:**
- `ml-agents/mlagents/trainers/policy/torch_policy.py`

**Specific changes:**

```python
# In TorchPolicy.__init__:
def __init__(self, seed, behavior_spec, network_settings, actor_cls, actor_kwargs):
    super().__init__(seed, behavior_spec, network_settings)

    # ADD THIS:
    self.gpu_processor = None
    if torch.cuda.is_available():
        from mlagents.trainers.gpu_processing import GPUObservationProcessor
        self.gpu_processor = GPUObservationProcessor(device='cuda')
        logger.info("GPU observation processing enabled")

# In TorchPolicy.evaluate():
def evaluate(self, decision_requests, global_agent_ids):
    # ... existing code to get observations ...

    # ADD THIS before normalization:
    if self.gpu_processor is not None:
        # Process on GPU
        for i, obs in enumerate(vec_obs):
            vec_obs[i] = self.gpu_processor.process_batch(obs, normalize=True)
    else:
        # Existing CPU normalization
        # ... existing code ...
```

**Test:** Run training and check GPU utilization goes up

**Success criteria:**
- Training works
- GPU utilization increases
- No errors

---

## Task 3: Add Config Options for Features (30 min)

**Problem:** Features are hardcoded, need config control

**What to add:**

**File:** `config/ppo/3DBall_AllFeatures.yaml`

```yaml
behaviors:
  3DBall:
    trainer_type: ppo

    # Enable ALL new features
    experimental:
      use_shared_memory: true  # When Task 1 is done
      use_gpu_processing: true  # When Task 2 is done

    # Existing optimizations
    network_settings:
      enable_torchscript: true  # 2.5x speedup

    # ... rest of standard 3DBall config ...
```

**Then parse in settings.py or torch_policy.py to enable/disable features**

**Test:** Training uses features when config says to

---

## Priority Order

**Do in this order:**

1. ✅ Task 3 first (easiest, 30 min) - Add the config file
2. Task 2 next (medium, 1-2 hours) - GPU processing integration
3. Task 1 last (hardest, 2-3 hours) - Fix SharedMemory Unity connection

**Why this order:**
- Task 3 is easy and sets up structure
- Task 2 works independently (doesn't need SharedMemory)
- Task 1 is complex, do it last

**Total time:** 4-6 hours of integration work

---

## What I'll Do Right Now

Starting with Task 2 (GPU processing) since it's most impactful and doesn't depend on fixing SharedMemory.

I'll hook GPUObservationProcessor into TorchPolicy.evaluate() so it actually gets called during training.

Ready?
