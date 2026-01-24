# ML-Agents Toolkit - Agent Development Guide

This guide provides essential information for AI agents and developers working with the Unity ML-Agents Toolkit.

## Quick Start

### Prerequisites
- **Python**: 3.10.1 to 3.11.9
- **Unity**: 6000.0 or later
- **Operating System**: Windows, macOS, or Linux

### One-Command Setup (Recommended)

```bash
# Linux/macOS
./setup-dev.sh

# Windows PowerShell
.\setup-dev.ps1
```

This script will:
- Create and activate a virtual environment
- Install ml-agents and ml-agents-envs packages
- Install test dependencies
- Set up pre-commit hooks
- Create .env file from template
- Verify installation

### Manual Installation

If you prefer manual setup:

```bash
# Clone the repository
git clone https://github.com/Unity-Technologies/ml-agents.git
cd ml-agents

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -e ./ml-agents-envs
pip install -e ./ml-agents

# Install test dependencies
pip install -r test_requirements.txt

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Build Commands

### Python Packages

**Install packages in editable mode:**
```bash
pip install -e ./ml-agents-envs
pip install -e ./ml-agents
```

**Build distribution packages:**
```bash
cd ml-agents
python setup.py sdist
python setup.py bdist_wheel
```

**Install test dependencies:**
```bash
pip install -r test_requirements.txt
```

## Test Commands

### Run Tests

**Run all tests (excluding slow tests):**
```bash
pytest --cov=ml-agents --cov=ml-agents-envs -m "not slow"
```

**Run tests in parallel (8 workers):**
```bash
pytest --cov=ml-agents --cov=ml-agents-envs -m "not slow" -n 8
```

**Run tests with coverage report:**
```bash
pytest --cov=ml-agents --cov=ml-agents-envs --cov-report=html -m "not slow"
```

**Run slow tests (integration/training tests):**
```bash
pytest -m "slow"
```

**List all tests without running:**
```bash
pytest --collect-only
```

### Coverage Requirements
- Minimum coverage threshold: **60%** (enforced in setup.cfg)
- Coverage reports generated in `htmlcov/` directory

## Development Workflow

### Code Quality Tools

**Run all pre-commit hooks:**
```bash
pre-commit run --all-files
```

**Run specific hooks:**
```bash
pre-commit run black --all-files      # Format with black
pre-commit run mypy --all-files       # Type checking
pre-commit run flake8 --all-files     # Linting
```

**Install pre-commit hooks:**
```bash
pre-commit install
```

### Code Style
- **Formatter**: Black (max line length: 88 for code, 120 for docstrings)
- **Linter**: flake8 with plugins (comprehensions, tidy-imports, bugbear)
- **Type Checker**: mypy with `--disallow-incomplete-defs`
- **Import Sorting**: pyupgrade (Python 3.6+)

### Naming Conventions

Follow these naming standards consistently across the codebase:

**Python:**
- **Modules/Packages**: `lowercase_with_underscores` (e.g., `trainer_util.py`, `policy_network.py`)
- **Classes**: `PascalCase` (e.g., `PolicyNetwork`, `PPOTrainer`, `UnityEnvironment`)
- **Functions/Methods**: `lowercase_with_underscores` (e.g., `train_model()`, `get_action()`)
- **Constants**: `UPPER_CASE_WITH_UNDERSCORES` (e.g., `MAX_STEPS`, `DEFAULT_PORT`)
- **Private methods**: `_leading_underscore` (e.g., `_internal_method()`)
- **Protected methods**: `_single_underscore` (e.g., `_helper_function()`)

**Unity C#:**
- **Classes**: `PascalCase` (e.g., `Agent`, `BehaviorParameters`)
- **Public methods**: `PascalCase` (e.g., `CollectObservations()`, `OnActionReceived()`)
- **Private fields**: `m_camelCase` with `m_` prefix (e.g., `m_rewardScale`, `m_maxSteps`)
- **Public properties**: `PascalCase` (e.g., `MaxStep`, `BehaviorName`)
- **Interfaces**: `IPascalCase` with `I` prefix (e.g., `ISensor`, `IActuator`)

**Configuration Files:**
- **YAML files**: `kebab-case.yaml` (e.g., `ppo-config.yaml`, `3d-ball.yaml`)
- **YAML keys**: `snake_case` (e.g., `learning_rate`, `batch_size`, `max_steps`)

### Banned Modules
Per setup.cfg, do not directly import:
- `tensorflow` → use `mlagents.tf_utils` instead
- `logging` → use `mlagents_envs.logging_util` instead
- `torch` → use `mlagents.torch_utils` instead (handles GPU detection)

## Training Agents

### Basic Training

**Train an agent:**
```bash
mlagents-learn config/ppo/3DBall.yaml --run-id=3DBall_01
```

**Resume training:**
```bash
mlagents-learn config/ppo/3DBall.yaml --run-id=3DBall_01 --resume
```

**Train with custom hyperparameters:**
```bash
mlagents-learn config/ppo/3DBall.yaml --run-id=3DBall_custom --env=./envs/3DBall
```

### Monitor Training

Training results are saved in `results/<run-id>/`:
- **Checkpoints**: Model checkpoints saved periodically
- **TensorBoard logs**: View with `tensorboard --logdir=results`
- **Configuration**: Copy of training config for reproducibility

### HuggingFace Integration

**Push trained model to HuggingFace:**
```bash
mlagents-push-to-hf --run-id=3DBall_01 --repo-id=username/model-name
```

**Load model from HuggingFace:**
```bash
mlagents-load-from-hf --repo-id=username/model-name
```

## Unity Project

### Example Environments Location
- `Project/Assets/ML-Agents/Examples/` - 17+ example environments
- Key examples: 3DBall, GridWorld, Hallway, Pyramids, PushBlock, Reacher

### Running Unity Environment with Python

```python
from mlagents_envs.environment import UnityEnvironment

# Create environment
env = UnityEnvironment(file_name="path/to/unity/build")

# Reset environment
env.reset()

# Step environment
behavior_name = list(env.behavior_specs)[0]
decision_steps, terminal_steps = env.get_steps(behavior_name)

env.close()
```

## Project Structure

```
ml-agents/
├── ml-agents/              # Main training package
│   ├── mlagents/
│   │   ├── trainers/      # Training algorithms (PPO, SAC, MA-POCA)
│   │   ├── plugins/       # Plugin system
│   │   └── utils/         # Utilities (HF integration, etc.)
│   └── setup.py
├── ml-agents-envs/        # Unity environment interface
│   ├── mlagents_envs/
│   │   ├── environment.py # Core UnityEnvironment class
│   │   ├── logging_util.py # Logging utilities
│   │   └── rpc/          # gRPC communication
│   └── setup.py
├── com.unity.ml-agents/   # Unity package
├── Project/               # Unity example project
├── config/                # Training configurations
└── test_requirements.txt  # Test dependencies
```

## Key Configuration Files

- **pytest.ini**: Test configuration (markers for slow tests)
- **setup.cfg**: Package config (coverage thresholds, flake8 rules)
- **.pre-commit-config.yaml**: Pre-commit hooks configuration
- **.gitignore**: Excludes venv/, results/, models/, build artifacts

## Common Tasks

### Add New Training Algorithm
1. Create trainer class in `ml-agents/mlagents/trainers/`
2. Register in `mlagents.plugins.trainer_type`
3. Add configuration schema
4. Write unit tests in `ml-agents/mlagents/trainers/tests/`

### Add New Environment
1. Create Unity scene in `Project/Assets/ML-Agents/Examples/`
2. Implement Agent class inheriting from `Unity.MLAgents.Agent`
3. Define observations, actions, and rewards
4. Create training config in `config/`

## Environment Variables

Currently, no environment variables are required for basic usage. For CI/CD:
- `GITHUB_REF`: Used in release verification
- `TEST_ENFORCE_BUFFER_KEY_TYPES`: Set to 1 in CI for strict type checking

## Troubleshooting

### Common Issues

**Import errors:**
- Ensure packages installed with `pip install -e ./ml-agents-envs` and `pip install -e ./ml-agents`

**Tests fail with ModuleNotFoundError:**
- Install test dependencies: `pip install -r test_requirements.txt`
- Install package in editable mode

**Unity connection errors:**
- Check Unity environment is built for your platform
- Verify port 5005 is not in use

## Resources

- **Documentation**: https://docs.unity3d.com/Packages/com.unity.ml-agents@latest
- **Unity Discussions**: https://discussions.unity.com/tag/ml-agents
- **GitHub Issues**: https://github.com/Unity-Technologies/ml-agents/issues
- **Discord**: https://discord.com/channels/489222168727519232/1202574086115557446

## Version Information

- **Python Package Version**: Defined in `mlagents.trainers.__version__`
- **Unity Package Version**: 4.0.0
- **Supported Python**: 3.10.1 to 3.11.9
- **Supported Unity**: 6000.0+
