# Architecture Improvements Recommendations

**Date:** 2026-01-24  
**Scope:** Phase 4 - Architectural improvements for ML-Agents codebase  
**Purpose:** Document architectural patterns, identify refactoring opportunities, and propose improvements

---

## Overview

This document analyzes the ML-Agents architecture and identifies opportunities for improvement through:
1. **Pattern Extraction** - Common code patterns that can be abstracted
2. **God Object Refactoring** - Large classes with too many responsibilities
3. **Module Boundary Improvements** - Better separation of concerns

---

## 1. Optimizer Common Patterns

### Current State

Three main optimizer implementations share significant common patterns:
- `ppo/optimizer_torch.py` (PPO - Proximal Policy Optimization)
- `sac/optimizer_torch.py` (SAC - Soft Actor-Critic)
- `poca/optimizer_torch.py` (POCA - POsthumous Credit Assignment)

### Common Patterns Identified

#### Pattern A: Optimizer Initialization
All three optimizers share this initialization pattern:

```python
# In PPO, SAC, and POCA:
self.optimizer = create_optimizer(params, lr=learning_rate)
self._use_amp = is_amp_enabled()
if self._use_amp:
    self._grad_scaler = torch.amp.GradScaler()
```

**Duplication:** 3 instances, ~10 lines each  
**Recommendation:** Extract to base class `TorchOptimizer`

#### Pattern B: Learning Rate Decay
All on-policy algorithms (PPO, POCA) use learning rate schedules:

```python
# In PPO and POCA:
self.decay_learning_rate = ModelUtils.DecayedValue(
    self.hyperparameters.learning_rate_schedule,
    self.hyperparameters.learning_rate,
    1e-10,
    self.trainer_settings.max_steps,
)
```

**Duplication:** 2 instances  
**Recommendation:** Extract to `OnPolicyOptimizer` base class

#### Pattern C: Hyperparameter Scheduling
PPO and POCA both use multiple scheduled hyperparameters:

```python
# In PPO:
self.decay_epsilon = ModelUtils.DecayedValue(...)
self.decay_beta = ModelUtils.DecayedValue(...)

# In POCA:
self.decay_epsilon = ModelUtils.DecayedValue(...)
self.decay_beta = ModelUtils.DecayedValue(...)
```

**Duplication:** Nearly identical implementations  
**Recommendation:** Create `ScheduledHyperparameters` helper class

#### Pattern D: Network Compilation
All optimizers use `maybe_compile()` for TorchScript optimization:

```python
# In all optimizers:
self._critic = maybe_compile(self._critic)
```

**Current State:** ✅ Already well-abstracted via utility function  
**No action needed**

### Proposed Refactoring

#### Option 1: Extract Common Base Classes

**Create hierarchy:**
```
TorchOptimizer (base)
├── OnPolicyOptimizer
│   ├── TorchPPOOptimizer
│   └── TorchPOCAOptimizer
└── OffPolicyOptimizer
    └── TorchSACOptimizer
```

**Implementation:**

```python
# ml-agents/mlagents/trainers/optimizer/torch_optimizer.py

class TorchOptimizer:
    """Base class for all PyTorch optimizers."""
    
    def _setup_optimizer(self, params, learning_rate):
        """Common optimizer setup with AMP support."""
        self.optimizer = create_optimizer(params, lr=learning_rate)
        self._use_amp = is_amp_enabled()
        if self._use_amp:
            self._grad_scaler = torch.amp.GradScaler()
    
    def _backward_pass(self, loss):
        """Execute backward pass with optional AMP scaling."""
        if self._use_amp:
            self._grad_scaler.scale(loss).backward()
            self._grad_scaler.step(self.optimizer)
            self._grad_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()


class OnPolicyOptimizer(TorchOptimizer):
    """Base class for on-policy algorithms (PPO, POCA)."""
    
    def _setup_learning_rate_schedule(self, schedule_type, initial_lr, max_steps):
        """Setup learning rate decay schedule."""
        self.decay_learning_rate = ModelUtils.DecayedValue(
            schedule_type, initial_lr, 1e-10, max_steps
        )
    
    def _setup_hyperparameter_schedules(self, settings, max_steps):
        """Setup common hyperparameter schedules (epsilon, beta)."""
        self.decay_epsilon = ModelUtils.DecayedValue(
            settings.epsilon_schedule, settings.epsilon, 0.1, max_steps
        )
        self.decay_beta = ModelUtils.DecayedValue(
            settings.beta_schedule, settings.beta, 1e-5, max_steps
        )
```

**Benefits:**
- Reduces code duplication by ~50 lines
- Makes optimizer behavior more consistent
- Easier to add new optimizers
- Centralized AMP logic

**Risks:**
- Requires careful testing to ensure behavior unchanged
- May complicate debugging initially
- Breaking change if external plugins exist

**Effort Estimate:** 12-16 hours (implementation + testing)

#### Option 2: Extract Helper Classes

**Create utility classes without changing hierarchy:**

```python
# ml-agents/mlagents/trainers/optimizer/optimizer_utils.py

class OptimizerSetupHelper:
    """Helper for common optimizer setup patterns."""
    
    @staticmethod
    def create_with_amp(params, learning_rate):
        """Create optimizer with AMP support if available."""
        optimizer = create_optimizer(params, lr=learning_rate)
        
        use_amp = is_amp_enabled()
        grad_scaler = torch.amp.GradScaler() if use_amp else None
        
        return optimizer, use_amp, grad_scaler
    
    @staticmethod
    def create_hyperparameter_schedule(schedule_type, initial_value, 
                                      min_value, max_steps):
        """Create a hyperparameter decay schedule."""
        return ModelUtils.DecayedValue(
            schedule_type, initial_value, min_value, max_steps
        )


class HyperparameterScheduler:
    """Manages multiple scheduled hyperparameters."""
    
    def __init__(self, settings, max_steps):
        self.schedules = {}
        
        # Setup common on-policy schedules
        if hasattr(settings, 'learning_rate_schedule'):
            self.schedules['learning_rate'] = ModelUtils.DecayedValue(...)
        if hasattr(settings, 'epsilon_schedule'):
            self.schedules['epsilon'] = ModelUtils.DecayedValue(...)
        if hasattr(settings, 'beta_schedule'):
            self.schedules['beta'] = ModelUtils.DecayedValue(...)
    
    def get_value(self, name, steps):
        """Get current value for a scheduled hyperparameter."""
        return self.schedules[name].get_value(steps)
    
    def get_all_values(self, steps):
        """Get all current hyperparameter values."""
        return {name: sched.get_value(steps) 
                for name, sched in self.schedules.items()}
```

**Benefits:**
- No breaking changes to existing code
- Reduces duplication without major refactoring
- Easier to adopt incrementally
- Lower risk

**Risks:**
- Doesn't fully address architectural issues
- Still some duplication remaining

**Effort Estimate:** 4-6 hours

#### Recommendation: **Option 2** (Extract Helper Classes)

**Rationale:**
- Lower risk - no changes to inheritance hierarchy
- Faster to implement - can be done incrementally
- Easier to test - helpers can be unit tested independently
- Non-breaking - existing code continues to work

---

## 2. God Object Refactoring

### Issue: `trainer_controller.py` (~700 lines)

**Responsibilities (too many):**
1. Training loop coordination
2. Environment management
3. Checkpoint management
4. Statistics aggregation
5. Ghost training orchestration
6. Thread management
7. Signal handling
8. Configuration management

**Current Structure:**
```
TrainerController (700 lines)
├── __init__() - Setup (80 lines)
├── _get_measure_vals() - Statistics (30 lines)
├── _save_model() - Checkpointing (40 lines)
├── _save_models_when_interrupted() - Signal handling (20 lines)
├── _export_graph() - Model export (30 lines)
├── _create_model_path() - Path management (15 lines)
├── start_learning() - Main training loop (200+ lines)
└── ... 10+ more methods
```

**Recommendation: Extract into Multiple Classes**

#### Proposed Refactoring:

```python
# ml-agents/mlagents/trainers/training_session.py

class TrainingSession:
    """Manages a single training session lifecycle."""
    
    def __init__(self, run_id: str, config: RunConfiguration):
        self.run_id = run_id
        self.config = config
        self.checkpoint_manager = CheckpointManager(run_id)
        self.stats_aggregator = TrainingStatsAggregator()
        self.env_manager = EnvironmentManager(config)
    
    def start(self):
        """Start the training session."""
        ...


# ml-agents/mlagents/trainers/checkpoint_manager.py

class CheckpointManager:
    """Manages model checkpointing and restoration."""
    
    def __init__(self, run_id: str, checkpoint_settings: CheckpointSettings):
        self.run_id = run_id
        self.settings = checkpoint_settings
        self.model_path = self._create_model_path()
    
    def save_checkpoint(self, trainers, global_step):
        """Save a training checkpoint."""
        ...
    
    def load_checkpoint(self, run_id):
        """Load a training checkpoint."""
        ...
    
    def export_for_inference(self, trainers):
        """Export models for inference."""
        ...


# ml-agents/mlagents/trainers/stats_aggregator.py

class TrainingStatsAggregator:
    """Aggregates and reports training statistics."""
    
    def __init__(self):
        self.stats_reporters = []
        self.stats_history = defaultdict(list)
    
    def add_reporter(self, reporter: StatsReporter):
        """Add a statistics reporter."""
        self.stats_reporters.append(reporter)
    
    def record(self, category: str, key: str, value: float, step: int):
        """Record a statistics value."""
        ...
    
    def flush(self):
        """Flush statistics to all reporters."""
        ...


# ml-agents/mlagents/trainers/environment_manager.py

class EnvironmentManager:
    """Manages Unity environment lifecycle and interactions."""
    
    def __init__(self, config: EnvironmentConfiguration):
        self.config = config
        self.env_factory = self._create_env_factory()
        self.envs = []
    
    def create_environments(self, num_envs: int):
        """Create multiple parallel environments."""
        ...
    
    def reset_all(self):
        """Reset all environments."""
        ...
    
    def close_all(self):
        """Close all environments."""
        ...
```

**Refactored `TrainerController`:**

```python
# ml-agents/mlagents/trainers/trainer_controller.py (now ~300 lines)

class TrainerController:
    """Coordinates training across multiple trainers."""
    
    def __init__(self, session: TrainingSession):
        self.session = session
        self.trainers = {}
        self.training_loop = TrainingLoop(self.trainers, session)
    
    def start_learning(self):
        """Start the training process."""
        return self.training_loop.run()
```

**Benefits:**
- Each class has single responsibility
- Easier to test individual components
- Easier to understand and modify
- Better code reusability
- Reduced file size (~700 → ~300 lines)

**Risks:**
- Large refactoring effort
- Risk of introducing bugs
- Breaking change for anyone importing TrainerController directly
- Requires extensive integration testing

**Effort Estimate:** 40-60 hours (refactoring + comprehensive testing)

**Recommendation:** **Defer to v5.0** - Too risky for incremental update

**Alternative:** Document the responsibilities and add TODO comments for future refactoring:

```python
# trainer_controller.py

# TODO (v5.0): This class has too many responsibilities. Consider extracting:
# - CheckpointManager: Model saving/loading logic
# - StatsAggregator: Statistics collection and reporting
# - EnvironmentManager: Environment lifecycle management
# - TrainingLoop: Main training loop coordination
# See ARCHITECTURE-IMPROVEMENTS.md for detailed refactoring plan.

class TrainerController:
    ...
```

**Effort Estimate:** 1 hour (documentation only)

---

### Issue: `settings.py` (~1000 lines)

**Responsibilities:**
1. All configuration dataclasses
2. Configuration validation
3. Default value management
4. Serialization/deserialization

**Current Structure:**
- 20+ dataclass definitions
- Nested configuration hierarchy
- Mixed concerns (settings + validation)

**Recommendation: Split into Multiple Modules**

#### Proposed Structure:

```
settings/
├── __init__.py              # Public API
├── trainer_settings.py      # TrainerSettings, PPOSettings, SACSettings, etc.
├── network_settings.py      # NetworkSettings, EncoderSettings
├── reward_settings.py       # RewardSignalSettings and subclasses
├── environment_settings.py  # EnvironmentSettings, EngineSettings
├── run_settings.py          # RunOptions, CheckpointSettings
└── validation.py            # Configuration validation logic
```

**Example:**

```python
# ml-agents/mlagents/trainers/settings/trainer_settings.py

@attr.s(auto_attribs=True)
class TrainerSettings:
    """Settings for a trainer."""
    trainer_type: TrainerType = TrainerType.PPO
    hyperparameters: HyperparamSettings = attr.ib(factory=PPOSettings)
    network_settings: NetworkSettings = attr.ib(factory=NetworkSettings)
    ...


# ml-agents/mlagents/trainers/settings/__init__.py

# Public API - import all settings for backward compatibility
from .trainer_settings import TrainerSettings, TrainerType
from .network_settings import NetworkSettings, EncoderType
from .reward_settings import RewardSignalSettings, RewardSignalType
...

__all__ = [
    'TrainerSettings', 'TrainerType',
    'NetworkSettings', 'EncoderType',
    ...
]
```

**Benefits:**
- Easier to navigate (smaller files)
- Better organization by domain
- Maintains backward compatibility via __init__.py
- Easier to test individual settings groups
- Reduces merge conflicts

**Risks:**
- Large refactoring (20+ imports to update)
- Must maintain backward compatibility
- Import path changes

**Effort Estimate:** 16-20 hours

**Recommendation:** **Implement in v4.1** - Medium priority, good ROI

**Alternative (Low Effort):** Add section comments and organize imports:

```python
# settings.py

# ===================================================================
# SECTION 1: TRAINER SETTINGS
# ===================================================================

@attr.s(auto_attribs=True)
class TrainerSettings:
    ...

# ===================================================================
# SECTION 2: NETWORK SETTINGS
# ===================================================================

@attr.s(auto_attribs=True)
class NetworkSettings:
    ...

# ... etc
```

**Effort Estimate:** 1 hour

---

## 3. Module Boundary Improvements

### Current Issues

1. **Circular Dependencies** - ✅ Already resolved in Phase 3
2. **Unclear Module Responsibilities** - Some modules do too much
3. **Deep Import Paths** - `from mlagents.trainers.torch_entities.components.reward_providers.base_reward_provider import BaseRewardProvider`

### Recommendations

#### A. Simplify Import Paths

**Current:**
```python
from mlagents.trainers.torch_entities.components.reward_providers.curiosity_reward_provider import CuriosityRewardProvider
```

**Proposed:**
```python
from mlagents.trainers.reward_providers import CuriosityRewardProvider
```

**Implementation:**
```python
# ml-agents/mlagents/trainers/reward_providers/__init__.py

from .curiosity_reward_provider import CuriosityRewardProvider
from .rnd_reward_provider import RNDRewardProvider
from .gail_reward_provider import GAILRewardProvider
from .extrinsic_reward_provider import ExtrinsicRewardProvider

__all__ = [
    'CuriosityRewardProvider',
    'RNDRewardProvider',
    'GAILRewardProvider',
    'ExtrinsicRewardProvider',
]
```

**Benefits:**
- Shorter, cleaner imports
- Easier to remember and type
- More Pythonic (like popular libraries)

**Effort Estimate:** 4-6 hours

#### B. Create Facade Module for Common Imports

**Problem:** Users need to import from many places

**Current:**
```python
# User code
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.learn import run_training
```

**Proposed:**
```python
# User code
from mlagents import (
    UnityEnvironment,
    EnvironmentParametersChannel,
    TrainerSettings,
    run_training
)
```

**Implementation:**
```python
# ml-agents/mlagents/__init__.py

# Re-export commonly used classes for easier imports
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel
)
from mlagents.trainers.settings import TrainerSettings, RunOptions
from mlagents.trainers.learn import run_training

__all__ = [
    'UnityEnvironment',
    'EnvironmentParametersChannel',
    'TrainerSettings',
    'RunOptions',
    'run_training',
]
```

**Benefits:**
- Improved developer experience
- Matches patterns in popular libraries (pytorch, tensorflow)
- Easier for beginners

**Effort Estimate:** 2 hours

---

## 4. Performance Optimizations (Architecture-Level)

### Global State Issues

**Issue:** `_compilation_stats` dictionary in `torchscript_optimization.py`

**Current:**
```python
# Global mutable state (not thread-safe)
_compilation_stats = {
    "attempts": 0,
    "successes": 0,
    "failures": 0
}
```

**Proposed:**
```python
# ml-agents/mlagents/trainers/torch_entities/compilation_stats.py

class CompilationStats:
    """Thread-safe compilation statistics tracker."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._stats = {
            "attempts": 0,
            "successes": 0,
            "failures": 0
        }
    
    def record_attempt(self):
        with self._lock:
            self._stats["attempts"] += 1
    
    def record_success(self):
        with self._lock:
            self._stats["successes"] += 1
    
    def record_failure(self):
        with self._lock:
            self._stats["failures"] += 1
    
    def get_stats(self):
        with self._lock:
            return self._stats.copy()
    
    def get_success_rate(self):
        with self._lock:
            if self._stats["attempts"] == 0:
                return 0.0
            return self._stats["successes"] / self._stats["attempts"]


# Singleton instance
_compilation_stats = CompilationStats()


def get_compilation_stats():
    """Get the global compilation stats tracker."""
    return _compilation_stats
```

**Benefits:**
- Thread-safe
- Encapsulated state
- Better testability
- Easier to add metrics

**Effort Estimate:** 2 hours

---

## Summary and Recommendations

### Immediate Actions (Phase 4 - Next 8 hours)

1. ✅ **Document Architecture Issues** (DONE - this document)
2. ⏳ **Extract Optimizer Helper Classes** (4-6 hours)
   - Create `OptimizerSetupHelper`
   - Create `HyperparameterScheduler`
   - Update PPO, SAC, POCA to use helpers
3. ⏳ **Encapsulate Compilation Stats** (2 hours)
   - Replace global dict with CompilationStats class
   - Update torchscript_optimization.py

### Short-Term (v4.1 - Next 2-4 weeks)

4. **settings.py Refactoring** (16-20 hours)
   - Split into settings/ module
   - Maintain backward compatibility
   - Update documentation

5. **Simplify Import Paths** (4-6 hours)
   - Create facade modules
   - Add __init__.py exports
   - Update examples

### Long-Term (v5.0 - Next major version)

6. **TrainerController Refactoring** (40-60 hours)
   - Extract CheckpointManager
   - Extract StatsAggregator
   - Extract EnvironmentManager
   - Extract TrainingLoop
   - Comprehensive testing

7. **Optimizer Base Class Refactoring** (12-16 hours)
   - Create OnPolicyOptimizer base
   - Create OffPolicyOptimizer base
   - Migrate existing optimizers
   - Update plugin system

### Priority Matrix

| Task | Effort | Impact | Risk | Priority | Timeframe |
|------|--------|--------|------|----------|-----------|
| Document Architecture | 2h | Medium | Low | ✅ Complete | Phase 4 |
| Extract Optimizer Helpers | 6h | Medium | Low | HIGH | Phase 4 |
| Encapsulate Global State | 2h | Low | Low | MEDIUM | Phase 4 |
| Simplify Import Paths | 6h | High | Low | HIGH | v4.1 |
| Split settings.py | 20h | Medium | Medium | MEDIUM | v4.1 |
| Refactor TrainerController | 60h | High | High | LOW | v5.0 |
| Optimizer Inheritance | 16h | Medium | Medium | LOW | v5.0 |

---

## Conclusion

The ML-Agents architecture is **generally well-designed** with a few areas for improvement:

### Strengths:
- ✅ Clear separation between envs and trainers
- ✅ Good use of composition over inheritance
- ✅ Plugin system for extensibility
- ✅ Modern PyTorch best practices (AMP, compilation)

### Areas for Improvement:
- ⚠️ Some code duplication in optimizers (addressable via helpers)
- ⚠️ Large "god objects" (trainer_controller, settings)
- ⚠️ Global mutable state (compilation_stats)
- ⚠️ Deep import paths (can be simplified)

### Overall Assessment: **GOOD**

The architecture issues identified are **not critical** and can be addressed incrementally:
- **Phase 4 (now):** Extract helpers, document patterns (8 hours)
- **v4.1 (next):** Import simplification, settings split (20-30 hours)
- **v5.0 (future):** Major refactorings (60-80 hours)

**Total Technical Debt (Architecture):** ~110 hours
**Phase 4 Deliverable:** ~8 hours of high-priority improvements

---

**Document Status:** ✅ Complete  
**Next Steps:** Implement Phase 4 recommendations
