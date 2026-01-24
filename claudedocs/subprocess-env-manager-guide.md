# Subprocess Environment Manager - Developer Guide

**Module:** `ml-agents/mlagents/trainers/subprocess_env_manager.py`
**Purpose:** Parallel Unity environment execution for distributed training
**Last Updated:** 2026-01-23

---

## Overview

The subprocess environment manager enables parallel training by running multiple Unity environment instances in separate processes. This architecture allows ML-Agents to scale training across multiple CPU cores, significantly improving throughput for sample collection.

### Key Benefits

- **Parallel Execution:** Run multiple environments simultaneously
- **Process Isolation:** Each environment runs in isolated subprocess
- **Shared Memory Option:** Alternative implementation available for reduced serialization overhead
- **Fault Tolerance:** Worker failures don't crash main training process

---

## Architecture

### Component Hierarchy

```
Main Training Process (TrainerController)
    └── SubprocessEnvManager
        ├── UnityEnvWorker 1 (subprocess)
        │   └── UnityEnvironment instance
        ├── UnityEnvWorker 2 (subprocess)
        │   └── UnityEnvironment instance
        └── UnityEnvWorker N (subprocess)
            └── UnityEnvironment instance
```

### Communication Flow

```
Main Process                Worker Process
    |                            |
    |--- EnvironmentRequest ---->|
    |    (STEP, RESET, etc.)     |
    |                            |
    |                      [Execute Command]
    |                            |
    |<-- EnvironmentResponse ----|
    |    (observations, rewards) |
```

### Process Lifecycle

1. **Initialization:** Main process spawns N worker subprocesses
2. **Environment Creation:** Each worker creates Unity environment instance
3. **Command Loop:** Workers listen for commands via inter-process communication
4. **Step Execution:** Workers execute environment steps and return results
5. **Shutdown:** Graceful cleanup of environments and subprocess termination

---

## Worker Function Design

### Architecture Pattern: Command Dispatcher

The worker function implements a command dispatcher pattern with extracted handler functions for single responsibility and reduced complexity.

**Design Goals:**
- Cyclomatic complexity < 10 per function
- Single responsibility principle
- Easy to test individual command handlers
- Clear separation of concerns

### Worker Function Structure

```python
def worker(parent_conn, step_queue, pickled_env_factory, worker_id, run_options, log_level):
    """
    Main worker function that dispatches commands to specialized handlers.

    Complexity: 11 (reduced from 21)
    Responsibilities: Command dispatching + Error handling
    """

    # 1. Initialize environment and channels
    env, env_parameters, stats_channel, training_analytics = (
        _initialize_worker_environment(...)
    )

    # 2. Main command loop
    while True:
        req = parent_conn.recv()

        if req.cmd == EnvironmentCommand.STEP:
            _handle_step_command(...)
        elif req.cmd == EnvironmentCommand.RESET:
            _handle_reset_command(...)
        elif req.cmd == EnvironmentCommand.ENVIRONMENT_PARAMETERS:
            _handle_environment_parameters(...)
        elif req.cmd == EnvironmentCommand.TRAINING_STARTED:
            _handle_training_started(...)
        elif req.cmd == EnvironmentCommand.CLOSE:
            break

    # 3. Error handling and cleanup
```

### Helper Functions

#### _initialize_worker_environment()

**Purpose:** Set up environment and communication channels
**Complexity:** 5
**Returns:** Tuple of (env, env_parameters, stats_channel, training_analytics)

**Responsibilities:**
- Deserialize environment factory function
- Create side channels for communication
- Configure engine settings (resolution, time scale, etc.)
- Initialize training analytics (worker 0 only)
- Create Unity environment instance

**Usage:**
```python
env, env_params, stats, analytics = _initialize_worker_environment(
    pickled_factory="<serialized factory>",
    worker_id=0,
    run_options=RunOptions(...),
    log_level=logging.INFO
)
```

#### _handle_step_command()

**Purpose:** Execute environment step and collect results
**Complexity:** 4

**Execution Flow:**
1. Extract action info from request payload
2. Set actions for all behaviors with active agents
3. Execute environment step
4. Collect observations and rewards for all behaviors
5. Gather timing statistics and environment metrics
6. Send results asynchronously via step queue

**Usage:**
```python
_handle_step_command(
    env=unity_env,
    req=EnvironmentRequest(cmd=STEP, payload=action_info),
    worker_id=0,
    step_queue=queue,
    stats_channel=stats
)
```

#### _handle_reset_command()

**Purpose:** Reset environment to initial state
**Complexity:** 2

**Execution Flow:**
1. Call env.reset() to reset Unity environment
2. Collect initial observations for all behaviors
3. Send results synchronously via parent connection

**Usage:**
```python
_handle_reset_command(
    env=unity_env,
    parent_conn=connection,
    worker_id=0
)
```

#### _handle_environment_parameters()

**Purpose:** Apply parameter randomization for curriculum learning
**Complexity:** 3

**Execution Flow:**
1. Iterate through parameter settings in request
2. Apply ParameterRandomizationSettings to environment channel
3. Unity environment receives updated parameters

**Usage:**
```python
_handle_environment_parameters(
    req=EnvironmentRequest(cmd=ENV_PARAMS, payload=param_dict),
    env_parameters=env_params_channel
)
```

#### _handle_training_started()

**Purpose:** Notify Unity of training start for analytics
**Complexity:** 2

**Execution Flow:**
1. Check if analytics channel is available (worker 0 only)
2. Extract behavior name and trainer config from request
3. Send training_started notification to Unity

**Usage:**
```python
_handle_training_started(
    req=EnvironmentRequest(cmd=TRAINING_STARTED, payload=(name, config)),
    training_analytics_channel=analytics
)
```

---

## Inter-Process Communication

### Communication Primitives

**Parent Connection (Pipe):**
- Bidirectional communication between main process and worker
- Used for command/response synchronous communication
- Types: `EnvironmentRequest` -> `EnvironmentResponse`

**Step Queue:**
- Unidirectional queue from worker to main process
- Used for asynchronous step result delivery
- Reduces blocking time in main training loop

### Message Types

**EnvironmentCommand Enum:**
```python
STEP                # Execute environment step
RESET               # Reset environment
BEHAVIOR_SPECS      # Request behavior specifications
ENVIRONMENT_PARAMETERS  # Apply parameter randomization
TRAINING_STARTED    # Notify training start
CLOSE               # Shutdown worker
ENV_EXITED          # Worker exception notification
CLOSED              # Worker cleanup complete
```

**Request/Response Pattern:**
```python
# Request structure
EnvironmentRequest(
    cmd: EnvironmentCommand,
    payload: Any  # Command-specific data
)

# Response structure
EnvironmentResponse(
    cmd: EnvironmentCommand,
    worker_id: int,
    payload: Any  # Command-specific results
)
```

---

## Security Considerations

### Cloudpickle Usage

**Context:** Environment factory functions are serialized using cloudpickle for inter-process communication.

**Security Assessment:**
- **Risk Level:** LOW (documented and justified)
- **Trust Boundary:** Factory functions defined in trusted code only
- **Network Exposure:** None (local subprocess communication only)
- **User Input:** Never passed to pickle serialization

**Design Rationale:**

Cloudpickle was chosen over alternatives because:

1. **Flexibility:** Supports lambda functions and closures
2. **Performance:** Minimal serialization overhead
3. **Simplicity:** No need for importable factory functions
4. **Isolation:** Each worker gets independent factory execution

**Alternative Approaches Considered:**

- **multiprocessing.spawn:** Requires importable functions (less flexible)
- **Shared Memory:** Available via `env_manager_shared_memory.py` (more complex)
- **JSON Configuration:** Cannot serialize function objects

**Mitigation Measures:**

- Factory functions defined in `subprocess_env_manager.py` only
- No user-provided code paths to pickle serialization
- Subprocess workers operate in controlled training environment
- No network communication of pickled data

### Best Practices

1. **Never deserialize untrusted pickle data**
2. **Keep factory functions in trusted code modules**
3. **Document security context for future reviewers**
4. **Consider shared memory alternative for high-security environments**

---

## Performance Considerations

### Optimization Strategies

**Process Pool Sizing:**
- Optimal: Number of CPU cores - 1 (leave one for main process)
- Too many: Context switching overhead increases
- Too few: Underutilizes available CPU resources

**Memory Management:**
- Each subprocess requires full Unity environment memory
- Monitor total memory usage: N_workers * per_env_memory
- Consider environment complexity when scaling worker count

**Communication Overhead:**
- Step queue reduces blocking in main process
- Pipe communication is synchronous (minimize round-trips)
- Batch operations when possible to reduce IPC overhead

**Timing Considerations:**
- Timer statistics collected per worker independently
- Timers reset after each step to avoid memory growth
- Main process aggregates timing data across workers

### Performance Monitoring

**Key Metrics:**
```python
# Environment steps per second
steps_per_second = total_steps / elapsed_time

# Worker utilization
utilization = active_time / total_time

# IPC overhead
ipc_overhead = (communication_time / total_time) * 100
```

**Profiling Tools:**
- Python cProfile for subprocess profiling
- Unity Profiler for environment performance
- Process monitoring tools (htop, Task Manager)

---

## Error Handling

### Exception Categories

**Expected Exceptions (Graceful Shutdown):**
- `KeyboardInterrupt` - User cancellation
- `UnityCommunicationException` - Connection lost
- `UnityTimeOutException` - Environment timeout
- `UnityEnvironmentException` - Unity-side error
- `UnityCommunicatorStoppedException` - Clean shutdown

**Unexpected Exceptions (Logged and Reported):**
- All other exceptions logged with stack trace
- Reported via ENV_EXITED message
- Main process can decide whether to retry or abort

### Recovery Strategies

**Worker Failure:**
1. Worker sends ENV_EXITED message
2. Main process marks worker as failed
3. Option 1: Continue with remaining workers
4. Option 2: Restart failed worker
5. Option 3: Abort training (critical failure)

**Communication Failure:**
- BrokenPipeError caught and logged
- Worker marked as disconnected
- Main process can spawn replacement worker

**Clean Shutdown:**
1. Send CLOSE command to all workers
2. Wait for CLOSED response (with timeout)
3. Force terminate unresponsive workers
4. Clean up IPC resources (pipes, queues)

---

## Testing Strategies

### Unit Testing

**Helper Function Tests:**
```python
def test_handle_step_command():
    """Test step command handler with mock environment."""
    mock_env = MagicMock(spec=UnityEnvironment)
    mock_queue = Queue()
    mock_stats = MagicMock(spec=StatsSideChannel)

    req = EnvironmentRequest(
        cmd=EnvironmentCommand.STEP,
        payload=action_info
    )

    _handle_step_command(mock_env, req, 0, mock_queue, mock_stats)

    mock_env.set_actions.assert_called()
    mock_env.step.assert_called_once()
    assert not mock_queue.empty()
```

### Integration Testing

**Full Worker Lifecycle:**
```python
def test_worker_lifecycle():
    """Test complete worker initialization and command processing."""
    # Create pipes and queues
    parent_conn, child_conn = Pipe()
    step_queue = Queue()

    # Start worker in subprocess
    worker_process = Process(
        target=worker,
        args=(child_conn, step_queue, pickled_factory, 0, run_options, logging.INFO)
    )
    worker_process.start()

    # Send commands and verify responses
    parent_conn.send(EnvironmentRequest(EnvironmentCommand.BEHAVIOR_SPECS))
    response = parent_conn.recv()
    assert response.cmd == EnvironmentCommand.BEHAVIOR_SPECS

    # Clean shutdown
    parent_conn.send(EnvironmentRequest(EnvironmentCommand.CLOSE))
    worker_process.join(timeout=10)
    assert worker_process.exitcode == 0
```

### Performance Testing

**Throughput Measurement:**
```python
def benchmark_worker_throughput():
    """Measure steps per second for worker process."""
    start_time = time.time()
    num_steps = 1000

    for _ in range(num_steps):
        parent_conn.send(EnvironmentRequest(
            EnvironmentCommand.STEP,
            payload=action_info
        ))
        response = step_queue.get()

    elapsed = time.time() - start_time
    steps_per_second = num_steps / elapsed
    print(f"Throughput: {steps_per_second:.2f} steps/sec")
```

---

## Maintenance Guidelines

### Code Modification Rules

**When Adding New Commands:**

1. Add enum value to `EnvironmentCommand`
2. Create `_handle_<command>_command()` helper function
3. Add elif branch in worker command loop
4. Document command behavior in this guide
5. Add unit tests for command handler
6. Update integration tests

**When Modifying Helper Functions:**

1. Preserve function signature compatibility
2. Maintain complexity below threshold (< 10)
3. Update docstrings with parameter changes
4. Run full test suite before committing
5. Update this guide with behavior changes

**When Changing Communication Protocol:**

1. Ensure backward compatibility with existing workers
2. Update request/response type definitions
3. Version protocol changes appropriately
4. Update serialization/deserialization code
5. Test with multiple worker configurations

### Complexity Monitoring

**Target Metrics:**
- Worker function: < 15 complexity
- Helper functions: < 10 complexity each
- Total lines per function: < 50

**Refactoring Triggers:**
- Any function exceeds complexity threshold
- Helper function grows beyond single responsibility
- Code duplication across command handlers
- New command requires significant logic addition

**Recommended Tools:**
```bash
# Measure complexity
python -c "import ast; ..." subprocess_env_manager.py

# Run linting
flake8 subprocess_env_manager.py

# Type checking
mypy subprocess_env_manager.py
```

---

## Troubleshooting

### Common Issues

**Problem: Worker hangs during initialization**

Symptoms: Worker process starts but never sends BEHAVIOR_SPECS
Root Cause: Unity environment fails to start or connect
Solution: Check Unity build path, firewall settings, port availability

**Problem: Intermittent communication failures**

Symptoms: BrokenPipeError or EOFError during training
Root Cause: Worker process crashes or connection lost
Solution: Check worker logs, increase timeout, verify resource availability

**Problem: Memory growth over time**

Symptoms: Worker memory usage increases continuously
Root Cause: Timer statistics not being reset, memory leak in Unity
Solution: Verify reset_timers() called after each step, check Unity profiler

**Problem: Slow training throughput**

Symptoms: Steps per second lower than expected
Root Cause: Too many workers, communication overhead, Unity performance
Solution: Profile worker utilization, reduce worker count, optimize Unity scene

### Debug Logging

Enable detailed logging:
```python
logging_util.set_log_level(logging.DEBUG)
```

Key log messages to watch:
- "UnityEnvironment worker N: environment stopping" - Normal shutdown
- "UnityEnvironment worker N: environment raised an unexpected exception" - Worker crash
- "UnityEnvironment worker N closing" - Cleanup starting
- "UnityEnvironment worker N done" - Cleanup complete

---

## References

### Related Documentation

- [CLAUDE.md](../CLAUDE.md) - Project guidance and architecture overview
- [codebase-analysis-report.md](./codebase-analysis-report.md) - Comprehensive analysis including complexity metrics
- [improvement-summary-report.md](./improvement-summary-report.md) - Refactoring details and metrics

### Related Code Files

- `env_manager.py` - Base EnvManager interface
- `env_manager_shared_memory.py` - Shared memory alternative implementation
- `simple_env_manager.py` - Single-process environment manager
- `settings.py` - RunOptions and configuration classes

### External Resources

- [Python multiprocessing documentation](https://docs.python.org/3/library/multiprocessing.html)
- [Unity ML-Agents documentation](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest)
- [Cloudpickle GitHub](https://github.com/cloudpipe/cloudpickle)

---

**Document Version:** 1.0
**Last Updated:** 2026-01-23
**Maintainer:** ML-Agents Development Team
