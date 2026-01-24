# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unity ML-Agents Toolkit enhanced fork with Python 3.10-3.12 support, Unity 6 compatibility, performance optimizations (2.5x TorchScript speedup), and comprehensive security hardening. This is a dual-language codebase: Python (training/environment) and C# (Unity runtime).

## Development Setup

**One-command setup (recommended):**
```bash
./setup-dev.sh              # Linux/macOS
.\setup-dev.ps1             # Windows PowerShell
source wsl-setup.sh         # WSL
```

**Manual setup:**
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -e ./ml-agents-envs
pip install -e ./ml-agents
pip install -r test_requirements.txt
pre-commit install
```

## Build and Test Commands

### Python Tests
```bash
# Run all tests (excluding slow integration tests)
pytest --cov=ml-agents --cov=ml-agents-envs -m "not slow"

# Run with parallelization (8 workers)
pytest --cov=ml-agents --cov=ml-agents-envs -m "not slow" -n 8

# Run slow integration/training tests
pytest -m "slow"

# Generate HTML coverage report
pytest --cov=ml-agents --cov=ml-agents-envs --cov-report=html -m "not slow"

# List all tests without running
pytest --collect-only
```

**Coverage requirement:** Minimum 60% (enforced in setup.cfg)

### Code Quality
```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Specific hooks
pre-commit run black --all-files    # Format
pre-commit run mypy --all-files     # Type checking
pre-commit run flake8 --all-files   # Linting
```

### Training Commands
```bash
# Basic training
mlagents-learn config/ppo/3DBall.yaml --run-id=3DBall_01

# Resume training
mlagents-learn config/ppo/3DBall.yaml --run-id=3DBall_01 --resume

# Train with GPU optimizations (TorchScript + AMP)
mlagents-learn config/ppo/3DBall_MaxGPU.yaml --run-id=3DBall_GPU

# Monitor with TensorBoard
tensorboard --logdir=results
```

### Unity C# Tests
Run from Unity Editor: Window → General → Test Runner → EditMode/PlayMode

### Configuration Tools
```bash
# Upgrade deprecated config fields
python -m mlagents.trainers.upgrade_config old_config.yaml --dry-run
python -m mlagents.trainers.upgrade_config old_config.yaml  # apply changes

# Load model from HuggingFace with revision pinning (security best practice)
mlagents-load-from-hf --repo-id username/model --revision main --local-dir ./models
```

## Architecture Overview

### Python Package Structure (`ml-agents/`)
**Core training algorithms:**
- `mlagents/trainers/ppo/` - Proximal Policy Optimization
- `mlagents/trainers/sac/` - Soft Actor-Critic
- `mlagents/trainers/poca/` - Multi-Agent Posthumous Credit Assignment
- `mlagents/trainers/optimizer/` - Optimizer utilities (PPO, SAC, POCA helpers)

**Key modules:**
- `mlagents/trainers/trainer_controller.py` - Orchestrates training lifecycle
- `mlagents/trainers/agent_processor.py` - Processes agent experiences
- `mlagents/trainers/buffer.py` - Experience replay buffer
- `mlagents/torch_utils/` - PyTorch utilities with AMP support and TorchScript compilation
- `mlagents/trainers/upgrade_config.py` - Migration tool for deprecated config fields

**Plugin system:** Uses entry points for extensibility (see setup.py)

### Environment Interface (`ml-agents-envs/`)
- `mlagents_envs/environment.py` - Core `UnityEnvironment` class for Python↔Unity communication
- `mlagents_envs/rpc/` - gRPC communication protocol
- `mlagents_envs/registry/` - Binary download utilities (security hardened)

### Unity C# Package (`com.unity.ml-agents/Runtime/`)
**Inference pipeline (optimized for performance):**
- `Inference/ModelRunner.cs` - Batched inference with pre-allocated collections (512 capacity), array-based batch storage
- `Inference/TensorProxy.cs` - Tensor management (fixed resource leak)
- `Policies/SentisPolicy.cs` - Local inference using Unity Sentis
- `Policies/RemotePolicy.cs` - Training communication with Python

**Core agent system:**
- `Agent.cs` - Base class for all agents (OnEpisodeBegin, CollectObservations, OnActionReceived)
- `Academy.cs` - Environment coordinator, event orchestration
- `DecisionRequester.cs` - Controls decision request timing
- `Sensors/` - Observation collection (Visual, Ray, Vector)
- `Actuators/` - Action execution interfaces

**Key optimization notes:**
- Pre-allocated collections to reduce GC pressure
- Array-based batch storage for faster action lookups (avoids dictionary overhead)
- `TryGetValue` pattern instead of `ContainsKey` throughout
- Cached tensor references outside loops
- Unity Profiler markers for performance measurement

### Example Environments (`Project/Assets/ML-Agents/Examples/`)
17+ environments including: 3DBall, GridWorld, Hallway, Pyramids, PushBlock, Walker, Crawler, Soccer, etc.

**Unity 6 migration:** All examples migrated from legacy `Input.GetKey()` to new Input System (`Keyboard.current`)

### Configuration Files (`config/`)
- `ppo/` - PPO training configs (includes `*_MaxGPU.yaml` variants with TorchScript + AMP)
- `sac/` - SAC training configs
- `poca/` - Multi-agent POCA configs
- `imitation/` - Imitation learning configs

## Naming Conventions

**Python:**
- Modules: `lowercase_with_underscores`
- Classes: `PascalCase`
- Functions/methods: `lowercase_with_underscores`
- Constants: `UPPER_CASE_WITH_UNDERSCORES`
- Private: `_leading_underscore`

**Unity C#:**
- Classes: `PascalCase`
- Public methods: `PascalCase`
- Private fields: `m_camelCase` (with `m_` prefix)
- Public properties: `PascalCase`
- Interfaces: `IPascalCase` (with `I` prefix)

**Config YAML:**
- Files: `kebab-case.yaml`
- Keys: `snake_case`

## Important Rules

### Banned Module Imports (setup.cfg)
- ❌ `import tensorflow` → Use `mlagents.tf_utils`
- ❌ `import logging` → Use `mlagents_envs.logging_util`
- ❌ `import torch` → Use `mlagents.torch_utils` (handles GPU detection)

### Code Quality Standards
- Line length: 88 (code), 120 (docstrings/comments)
- Formatter: Black
- Type hints required for public APIs
- Minimum test coverage: 60%

### Performance Considerations
**When modifying Unity inference pipeline:**
- Maintain pre-allocated collection capacity (512) in ModelRunner.cs
- Use array-based storage, not dictionaries for hot paths
- Cache tensor references outside loops
- Add Unity Profiler markers for new bottlenecks
- Test with 512+ agents to verify scalability

**When modifying Python training:**
- TorchScript compilation enabled by `compile_model: true` in config
- AMP training enabled by `use_amp: true` in trainer_config
- Verify GPU utilization with `nvidia-smi` during training

### Security Hardening
This fork has comprehensive security hardening:
- MD5 hashes use `usedforsecurity=False` (non-cryptographic use only)
- File permissions set to 0o700 for downloaded binaries
- URL scheme validation prevents file:// attacks
- PyTorch loads use `weights_only=True` to prevent arbitrary code execution
- HuggingFace downloads require revision pinning

When modifying security-sensitive code, maintain these protections.

### Testing Requirements
- Mark slow tests with `@pytest.mark.slow` (training/integration tests)
- All new features need tests (coverage must stay ≥60%)
- Run `pytest -m "not slow"` for quick validation
- Run `pytest -m "slow"` for full integration testing

## Git Workflow

**Feature branches required:**
```bash
git checkout -b feature/your-feature
# Make changes
git commit -m "Descriptive message"
git push origin feature/your-feature
```

**Never work directly on main/master.** Create feature branches for all work.

## Key Files to Check Before Changes

**Before modifying Python training:**
- `ml-agents/mlagents/trainers/trainer_controller.py` - Main orchestration
- `ml-agents/mlagents/torch_utils/torch.py` - GPU detection, TorchScript compilation
- Configuration schemas in `ml-agents/mlagents/trainers/settings.py`

**Before modifying Unity inference:**
- `com.unity.ml-agents/Runtime/Inference/ModelRunner.cs` - Batched inference
- `com.unity.ml-agents/Runtime/Agent.cs` - Agent decision flow
- `com.unity.ml-agents/Runtime/Academy.cs` - Environment stepping

**Before modifying configs:**
- `ml-agents/mlagents/trainers/upgrade_config.py` - Migration tool
- Existing configs in `config/` for pattern reference

## Common Patterns

### Adding New Training Algorithm
1. Create trainer class in `ml-agents/mlagents/trainers/`
2. Register in `mlagents.plugins.trainer_type` (setup.py entry point)
3. Add configuration schema to settings.py
4. Write tests in `ml-agents/mlagents/trainers/tests/`

### Adding New Unity Agent
1. Create C# class inheriting from `Unity.MLAgents.Agent`
2. Override: `OnEpisodeBegin()`, `CollectObservations()`, `OnActionReceived()`
3. Attach `BehaviorParameters` component with observation/action specs
4. Create training config in `config/`

### Optimizer Pattern (for PPO/SAC/POCA)
See `ml-agents/mlagents/trainers/optimizer/` for reusable optimizer utilities:
- `fused_optimizer.py` - Fused optimizer creation
- `ppo_optimizer.py` - PPO-specific helpers
- `sac_optimizer.py` - SAC-specific helpers

## Troubleshooting

**Import errors:** Ensure packages installed with `pip install -e ./ml-agents-envs ./ml-agents`

**Unity connection errors:**
- Check Unity environment built for your platform
- Verify port 5005 not in use
- Ensure Unity project uses ML-Agents package 4.0.0

**Test failures:**
- Install test dependencies: `pip install -r test_requirements.txt`
- Check Python version: 3.10.1 - 3.12.8 required

**GPU training issues:**
- Verify CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check torch version: `torch>=2.1.1` required for CUDA support

## Additional Resources

- AGENTS.md - Comprehensive development guide with all commands
- PROJECT-NOTES.md - Improvement roadmap and performance optimization notes
- README.md - Fork improvements and feature overview
- Official Unity ML-Agents docs: https://docs.unity3d.com/Packages/com.unity.ml-agents@latest
