# ML-Agents Performance Optimization Plan

This document outlines the implementation plan for performance optimizations in the ML-Agents fork.

## Overview

The optimizations are organized into three phases:
- **Phase 1**: Quick wins (Low effort, immediate impact)
- **Phase 2**: Medium effort optimizations
- **Phase 3**: Architectural improvements (High effort, significant impact)

---

## Phase 1: Quick Wins (1-2 days)

### 1.1 TryGetValue in ModelRunner.GetAction()
**File:** `com.unity.ml-agents/Runtime/Inference/ModelRunner.cs`

**Current Code:**
```csharp
public ActionBuffers GetAction(int agentId)
{
    if (m_LastActionsReceived.ContainsKey(agentId))
    {
        return m_LastActionsReceived[agentId];
    }
    return ActionBuffers.Empty;
}
```

**Optimized Code:**
```csharp
public ActionBuffers GetAction(int agentId)
{
    if (m_LastActionsReceived.TryGetValue(agentId, out var action))
    {
        return action;
    }
    return ActionBuffers.Empty;
}
```

**Impact:** Reduces dictionary lookups from 2 to 1 per agent per step.

---

### 1.2 TryGetValue in RpcCommunicator.GetActions()
**File:** `com.unity.ml-agents/Runtime/Communicator/RpcCommunicator.cs`

**Current Code:**
```csharp
public ActionBuffers GetActions(string behaviorName, int agentId)
{
    if (m_LastActionsReceived.ContainsKey(behaviorName))
    {
        if (m_LastActionsReceived[behaviorName].ContainsKey(agentId))
        {
            return m_LastActionsReceived[behaviorName][agentId];
        }
    }
    return ActionBuffers.Empty;
}
```

**Optimized Code:**
```csharp
public ActionBuffers GetActions(string behaviorName, int agentId)
{
    if (m_LastActionsReceived.TryGetValue(behaviorName, out var agentActions))
    {
        if (agentActions.TryGetValue(agentId, out var action))
        {
            return action;
        }
    }
    return ActionBuffers.Empty;
}
```

**Impact:** Reduces dictionary lookups from 4 to 2 per agent per step during training.

---

### 1.3 Pool NativeArrays in RayPerceptionSensor
**File:** `com.unity.ml-agents/Runtime/Sensors/RayPerceptionSensor.cs`

**Changes:**
1. Add cached NativeArrays as class members
2. Only reallocate when ray count changes
3. Dispose in sensor cleanup

**Current Code (in PerceiveBatchedRays):**
```csharp
var results = new NativeArray<RaycastHit>(numRays, Allocator.TempJob);
var raycastCommands = new NativeArray<RaycastCommand>(...);
// ... use arrays ...
results.Dispose();
raycastCommands.Dispose();
```

**Optimized Approach:**
```csharp
// Class members
NativeArray<RaycastHit> m_CachedResults;
NativeArray<RaycastCommand> m_CachedRaycastCommands;
NativeArray<SpherecastCommand> m_CachedSpherecastCommands;
int m_CachedRayCount = -1;

// In PerceiveBatchedRays - reuse if size matches
if (m_CachedRayCount != numRays)
{
    DisposeCachedArrays();
    m_CachedResults = new NativeArray<RaycastHit>(numRays, Allocator.Persistent);
    // ... allocate others ...
    m_CachedRayCount = numRays;
}
```

**Impact:** Eliminates allocation per frame for ray sensors.

---

### 1.4 Replace Busy Polling in subprocess_env_manager.py
**File:** `ml-agents/mlagents/trainers/subprocess_env_manager.py`

**Current Code:**
```python
while len(worker_steps) < 1:
    try:
        while True:
            step = self.step_queue.get_nowait()
            # ...
    except EmptyQueueException:
        pass  # Busy spin
```

**Optimized Code:**
```python
while len(worker_steps) < 1:
    try:
        step = self.step_queue.get(timeout=0.001)  # 1ms timeout
        # Process step...
    except EmptyQueueException:
        continue  # Small sleep built into timeout
```

**Impact:** Reduces CPU usage during training by ~10-20%.

---

## Phase 2: Medium Effort Optimizations (3-5 days)

### 2.1 Parallelize Sensor Updates with Job System
**File:** `com.unity.ml-agents/Runtime/Agent.cs`

**Current Code:**
```csharp
void UpdateSensors()
{
    foreach (var sensor in sensors)
    {
        sensor.Update();
    }
}
```

**Proposed Architecture:**
1. Create `ISensorJobified` interface for sensors that support job-based updates
2. Implement job scheduling for compatible sensors
3. Fall back to sequential for sensors that don't support jobs

```csharp
void UpdateSensors()
{
    // Schedule all jobified sensors
    var handles = new List<JobHandle>();
    foreach (var sensor in sensors)
    {
        if (sensor is ISensorJobified jobSensor)
        {
            handles.Add(jobSensor.ScheduleUpdate());
        }
    }
    
    // Update non-jobified sensors while jobs run
    foreach (var sensor in sensors)
    {
        if (!(sensor is ISensorJobified))
        {
            sensor.Update();
        }
    }
    
    // Complete all jobs
    JobHandle.CompleteAll(handles.ToArray());
}
```

**Impact:** Significant speedup with multiple sensors per agent.

---

### 2.2 Pool AgentBuffer Objects in Python
**File:** `ml-agents/mlagents/trainers/buffer.py`

**Changes:**
1. Create `AgentBufferPool` class
2. Implement `acquire()` and `release()` methods
3. Pre-allocate buffers based on batch size

```python
class AgentBufferPool:
    def __init__(self, pool_size: int = 16):
        self._pool: List[AgentBuffer] = []
        self._pool_size = pool_size
        
    def acquire(self) -> AgentBuffer:
        if self._pool:
            buffer = self._pool.pop()
            buffer.reset_agent()
            return buffer
        return AgentBuffer()
    
    def release(self, buffer: AgentBuffer) -> None:
        if len(self._pool) < self._pool_size:
            buffer.reset_agent()
            self._pool.append(buffer)
```

**Impact:** Reduces Python object allocation overhead during training.

---

### 2.3 Additional ContainsKey → TryGetValue Conversions
**Files:** Multiple files in `com.unity.ml-agents/Runtime/`

Search for remaining `ContainsKey` patterns and convert to `TryGetValue`:
- `RpcCommunicator.PutObservations()`
- `RpcCommunicator.SendBatchedMessageHelper()`
- Any other hot paths

---

## Phase 3: Architectural Improvements (1-2 weeks)

### 3.1 Replace String Behavior Names with Integer IDs
**Scope:** Multiple files across C# and Python

**Changes:**
1. Assign integer IDs to behavior names at registration time
2. Use integer IDs in all internal dictionaries and communication
3. Maintain string→int mapping only at boundaries

**Benefits:**
- Faster dictionary lookups (int hash vs string hash)
- Reduced memory allocation (no string interning issues)
- Smaller serialization payloads

---

### 3.2 Implement Async gRPC Communication
**File:** `com.unity.ml-agents/Runtime/Communicator/RpcCommunicator.cs`

**Current:** Synchronous blocking call to Python
**Proposed:** Async communication with double-buffering

```csharp
// Concept
async Task<UnityInputProto> ExchangeAsync(UnityOutputProto output)
{
    var task = m_Client.ExchangeAsync(WrapMessage(output, 200));
    // Unity continues simulation while waiting
    return await task;
}
```

**Challenges:**
- Unity's single-threaded nature
- Need to buffer observations while waiting
- Synchronization between Unity main thread and async completion

---

### 3.3 Centralized Decision Batching at Academy Level
**Files:** `Agent.cs`, `Academy.cs`

**Current:** Each agent manages its own decision request
**Proposed:** Academy collects all requests and batches them

**Benefits:**
- Single point of optimization
- Better batch formation
- Reduced per-agent overhead

---

### 3.4 Sensor Data Sharing for Identical Agents
**Concept:** When multiple agents have identical sensor configurations and positions, share computed data.

**Use Cases:**
- Grid-based environments with identical observation spaces
- Team-based games with shared observations

---

## Implementation Checklist

### Phase 1
- [ ] 1.1 TryGetValue in ModelRunner.GetAction()
- [ ] 1.2 TryGetValue in RpcCommunicator.GetActions()
- [ ] 1.3 Pool NativeArrays in RayPerceptionSensor
- [ ] 1.4 Replace busy polling in subprocess_env_manager.py
- [ ] Run Unity C# tests to verify changes
- [ ] Run Python tests to verify changes
- [ ] Commit Phase 1 changes

### Phase 2
- [x] 2.1 Design ISensorJobified interface
- [x] 2.1 Implement parallel sensor updates
- [x] 2.2 Implement AgentBufferPool
- [x] 2.3 Convert remaining ContainsKey patterns
- [ ] Run full test suite
- [ ] Benchmark performance improvements
- [ ] Commit Phase 2 changes

### Phase 3
- [ ] 3.1 Design integer ID system for behavior names
- [ ] 3.1 Implement C# side changes
- [ ] 3.1 Implement Python side changes
- [ ] 3.2 Research async gRPC patterns for Unity
- [ ] 3.2 Implement async communication (if feasible)
- [ ] 3.3 Design centralized batching architecture
- [ ] 3.3 Implement Academy-level batching
- [ ] Full integration testing
- [ ] Performance benchmarking
- [ ] Documentation updates

---

## Benchmarking Plan

To measure the impact of optimizations:

1. **Baseline Measurement**
   - Run 3DBall environment with 512 agents
   - Measure FPS, CPU usage, memory allocation
   - Record inference time per step

2. **After Each Phase**
   - Re-run same benchmark
   - Compare metrics to baseline
   - Document improvements

3. **Tools**
   - Unity Profiler (with new profiler markers)
   - Python cProfile for training
   - Custom timing scripts

---

## Risk Assessment

| Optimization | Risk Level | Mitigation |
|-------------|------------|------------|
| TryGetValue changes | Low | Simple refactor, easy to test |
| NativeArray pooling | Medium | Must handle disposal correctly |
| Busy polling fix | Low | Well-understood pattern |
| Parallel sensors | Medium | May have sensor-specific issues |
| AgentBuffer pooling | Medium | Must ensure clean state on reuse |
| Integer IDs | High | Affects API, needs careful design |
| Async gRPC | High | Complex Unity threading issues |

---

## Getting Started

To begin implementation:

```bash
# Create feature branch
git checkout -b perf/phase-1-quick-wins

# Implement Phase 1 changes
# ... make changes ...

# Run tests
# Unity: Window → General → Test Runner → EditMode
# Python: wsl -d Ubuntu -- bash -c "cd /mnt/c/Development/Projects/ml-agents && source .venv-wsl/bin/activate && pytest ml-agents-envs/tests/ -v"

# Commit and push
git add -A
git commit -m "Performance: Phase 1 quick wins"
git push origin perf/phase-1-quick-wins
```

---

## References

- [Unity Job System](https://docs.unity3d.com/Manual/JobSystem.html)
- [NativeArray Best Practices](https://docs.unity3d.com/Manual/JobSystemNativeContainer.html)
- [gRPC Async Patterns](https://grpc.io/docs/languages/csharp/async/)
- [Python multiprocessing Queue](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue)
