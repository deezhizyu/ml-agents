# ML-Agents Improvement Project

## Vision

Improve Unity ML-Agents to be faster, more scalable, and easier to use. Use GPU acceleration insights from MuJoCo MJX to optimize training performance.

## Completed Work

### Phase 1: Foundation (DONE)
- [x] Python 3.11 support
- [x] Fixed deprecated `pkg_resources` -> `importlib.metadata`
- [x] Fixed deprecated `distutils.version.LooseVersion` -> `packaging.version.Version`
- [x] Fixed CPUTensorData resource leak in TensorProxy.cs
- [x] Created MJX GPU benchmark for performance comparison
- [x] Pushed to GitHub

---

## Improvement Plan

### KNOWN PROBLEMS (from research)

1. **Performance Bottlenecks**
   - Sequential `RequestDecision()` calls per agent cause cumulative delays
   - High CPU usage even in simple scenes
   - FPS drops with many agents (>512 agents kills performance)
   - Inference pipeline inefficiencies (300ms+ frame times reported)

2. **Scalability Issues**
   - No efficient batching for multi-agent decisions
   - Decision-making is the main bottleneck
   - Users forced to use workarounds (centralized agents, subprocesses)

3. **Training Speed**
   - Unity simulation bottlenecks training (<100 steps/sec in complex envs)
   - MJX achieves 10-100x faster training with GPU-accelerated physics
   - ML-Agents Python training underutilizes GPU

4. **Usability**
   - Complex Python dependencies that break on updates
   - Steep learning curve
   - Poor debugging/interpretability

---

## Phase 2: Performance Optimization

### 2.1 Decision Batching (HIGH PRIORITY)
- [x] Profile `RequestDecision()` to understand the bottleneck
- [ ] Implement batched decision requests for multiple agents
- [ ] Reduce per-agent overhead in the decision pipeline

#### Profiling Results

**Current Decision Flow:**
1. `Agent.RequestDecision()` → sets `m_RequestDecision = true`
2. `Academy.EnvironmentStep()` triggers events: `AgentPreStep` → `AgentSendState` → `DecideAction` → `AgentAct`
3. Each agent's `SendInfo()` → `SendInfoToBrain()` runs sequentially
4. `SendInfoToBrain()` calls: `UpdateSensors()` → `CollectObservations()` → `WriteActionMask()` → `m_Brain.RequestDecision()`
5. Policies batch and process (gRPC to Python OR Sentis inference)

**Identified Bottlenecks:**

| Bottleneck | Impact | Description |
|------------|--------|-------------|
| Sequential Agent Processing | HIGH | Each agent's `SendInfo()` runs sequentially through Unity's event system. With 512+ agents, cumulative delay is significant. |
| Synchronous gRPC Communication | HIGH (Training) | `RpcCommunicator.Exchange()` is blocking - Unity waits for Python response. |
| Tensor Generation | MEDIUM (Inference) | `TensorGenerator.GenerateTensors()` iterates through all agents sequentially. |
| Dictionary Action Lookups | LOW | `Dictionary<int, ActionBuffers>` hash lookups per agent add overhead. |

**Key Files:**
- `com.unity.ml-agents/Runtime/Agent.cs` - `RequestDecision()`, `SendInfoToBrain()`
- `com.unity.ml-agents/Runtime/Academy.cs` - `EnvironmentStep()`, event orchestration
- `com.unity.ml-agents/Runtime/Policies/RemotePolicy.cs` - Training communication
- `com.unity.ml-agents/Runtime/Policies/SentisPolicy.cs` - Inference
- `com.unity.ml-agents/Runtime/Inference/ModelRunner.cs` - Batched inference
- `com.unity.ml-agents/Runtime/Communicator/RpcCommunicator.cs` - gRPC to Python

**Proposed Optimizations:**

*Quick Wins:*
- Replace dictionary with array-based action storage indexed by batch position
- Pre-allocate observation buffers for reuse
- Use NativeArray<float> to reduce GC pressure

*Parallel Processing:*
- Parallelize sensor updates using Unity Job System
- Batch observation collection with agents writing to shared buffers
- Implement async gRPC communication

*Architecture Changes:*
- Implement centralized decision batching at the Academy level
- Add sensor data sharing for identical agent configurations
- Create action buffer pooling system

#### Implementation Status

**Completed Optimizations:**

1. **Array-based Batch Storage (ModelRunner.cs)**
   - Added `ActionBuffers[]` array for batch processing
   - Pre-allocated with 512 capacity (configurable)
   - Avoids dictionary lookups in hot path
   - Arrays grow dynamically as needed

2. **Dictionary Capacity Hints**
   - `m_LastActionsReceived`: initialized with 512 capacity
   - `m_OrderedAgentsRequestingDecisions`: initialized with 512 capacity
   - `m_Infos`: initialized with 512 capacity
   - Reduces dictionary resizing during gameplay

3. **TensorUtils Optimization (TensorProxy.cs)**
   - Added shape array caching to avoid allocations
   - Optimized `FillTensorBatch` with 2D fast path
   - Cached tensor references to avoid repeated casts

4. **Generator Optimizations (GeneratorImpl.cs)**
   - Cached tensor references outside loops
   - Added fast path for null action masks
   - Pre-calculated loop bounds
   - Added profiler markers for all generators

5. **Applier Optimizations (ApplierImpl.cs)**
   - Changed `ContainsKey` to `TryGetValue` (single lookup instead of two)
   - Cached tensor references outside loops
   - Added profiler markers for all appliers

6. **Profiler Integration**
   - Added `Profiler.BeginSample`/`EndSample` throughout hot paths
   - Enables precise measurement in Unity Profiler
   - Markers: `TensorGenerator.GenerateTensors`, `TensorApplier.ApplyTensors`, etc.

7. **BatchedObservationManager (New File)**
   - Object pooling for float arrays and lists
   - Reduces GC pressure during training
   - Statistics tracking for pool efficiency

**Files Modified:**
- `com.unity.ml-agents/Runtime/Inference/ModelRunner.cs`
- `com.unity.ml-agents/Runtime/Inference/TensorProxy.cs`
- `com.unity.ml-agents/Runtime/Inference/TensorGenerator.cs`
- `com.unity.ml-agents/Runtime/Inference/TensorApplier.cs`
- `com.unity.ml-agents/Runtime/Inference/GeneratorImpl.cs`
- `com.unity.ml-agents/Runtime/Inference/ApplierImpl.cs`
- `com.unity.ml-agents/Runtime/Inference/BatchedObservationManager.cs` (new)

### 2.2 Inference Optimization
- [ ] Profile inference pipeline (Sentis/Barracuda)
- [ ] Optimize tensor operations
- [ ] Better GPU utilization for inference

### 2.3 Training Loop Optimization
- [ ] Profile Python training code
- [ ] Optimize observation batching and transfer
- [ ] Reduce Unity <-> Python communication overhead

---

## Phase 3: Scalability Improvements

### 3.1 Multi-Agent Performance
- [ ] Parallel decision processing
- [ ] Shared policy optimization for similar agents
- [ ] Efficient memory management for large agent counts

### 3.2 Distributed Training
- [ ] Investigate Ray/RLlib integration
- [ ] Multi-environment parallel training improvements
- [ ] GPU cluster support

---

## Phase 4: Algorithm Improvements

### 4.1 PPO Enhancements
- [ ] Implement PPO-clip optimizations
- [ ] Better advantage estimation
- [ ] Adaptive learning rate scheduling

### 4.2 New Algorithms
- [ ] Evaluate IMPALA for faster training
- [ ] Consider SAC improvements
- [ ] Investigate model-based RL options

---

## Phase 5: Usability

### 5.1 Dependency Cleanup
- [ ] Simplify Python package dependencies
- [ ] Better version compatibility
- [ ] Easier installation process

### 5.2 Debugging Tools
- [ ] Better reward visualization
- [ ] Training progress insights
- [ ] Agent behavior analysis tools

---

## Benchmark Reference

MJX GPU benchmark (`mjx_benchmark.py`) establishes baseline:
- Run in WSL2: `source ~/mjx-env/bin/activate && python3 mjx_benchmark.py`
- Compare ML-Agents training speed against MJX results
- Target: Close the performance gap

---

## Git Setup

- origin: https://github.com/quanticsoul4772/ml-agents (your fork)
- upstream: https://github.com/Unity-Technologies/ml-agents
