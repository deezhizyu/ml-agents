# Unity ML-Agents Toolkit (Enhanced Fork)

[![docs badge](https://img.shields.io/badge/docs-reference-blue.svg)](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest)
[![license badge](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE.md)
[![Python 3.10-3.11](https://img.shields.io/badge/python-3.10--3.11-blue.svg)](https://www.python.org/)
[![Unity 6](https://img.shields.io/badge/Unity-6000.0+-black.svg)](https://unity.com/)

This is an enhanced fork of [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) with performance optimizations, Python 3.11 support, and Unity 6 compatibility.

**Based on:** Release 23 / Unity Package 4.0.0

## Fork Improvements

### Performance Optimizations

**Inference Pipeline:**
- Pre-allocated collections with 512 capacity in `ModelRunner.cs` (reduces GC pressure)
- Array-based batch storage for faster action lookups (avoids dictionary overhead)
- Changed `ContainsKey` to `TryGetValue` throughout (single lookup instead of two)
- Cached tensor references outside loops to avoid repeated casts
- New `BatchedObservationManager` for object pooling

**Profiler Integration:**
- Added `Profiler.BeginSample`/`EndSample` markers throughout the inference pipeline
- Enables precise measurement of bottlenecks in Unity Profiler
- Markers in: `ModelRunner`, `TensorGenerator`, `TensorApplier`, `GeneratorImpl`, `ApplierImpl`

**Files Modified:**
- `com.unity.ml-agents/Runtime/Inference/ModelRunner.cs`
- `com.unity.ml-agents/Runtime/Inference/TensorProxy.cs`
- `com.unity.ml-agents/Runtime/Inference/TensorGenerator.cs`
- `com.unity.ml-agents/Runtime/Inference/TensorApplier.cs`
- `com.unity.ml-agents/Runtime/Inference/GeneratorImpl.cs`
- `com.unity.ml-agents/Runtime/Inference/ApplierImpl.cs`
- `com.unity.ml-agents/Runtime/Inference/BatchedObservationManager.cs` (new)

### Python 3.11 Support

- Updated deprecated `pkg_resources` → `importlib.metadata`
- Fixed `distutils.version.LooseVersion` → `packaging.version.Version`
- Fixed pytest hooks for modern pytest/pytest-xdist compatibility
- Applied black formatting fixes

**Supported Python versions:** 3.10.1 - 3.11.9

### Unity 6 Compatibility

- **Input System Migration:** All 17+ example environments migrated from legacy `Input.GetKey()` to new Input System (`Keyboard.current`)
- Fixed package manifest for Unity 6 (removed non-existent modules)
- Tested with Unity 6000.0.40f1

**Example Scripts Updated:**
- `Ball3DAgent.cs`, `BasicActuatorComponent.cs`, `PushAgentEscape.cs`
- `FoodCollectorAgent.cs`, `GridAgent.cs`, `HallwayAgent.cs`
- `PushAgentBasic.cs`, `PushAgentCollab.cs`, `PyramidAgent.cs`
- `AgentSoccer.cs`, `SorterAgent.cs`, `WallJumpAgent.cs`
- `AdjustTrainingTimescale.cs`, `FlyCamera.cs`

### Bug Fixes

- Fixed CPUTensorData resource leak in `TensorProxy.cs`
- Fixed unused variable warnings causing compilation errors
- Fixed pytest `pytest_testnodeready` hook compatibility

---

## Quick Start

### Prerequisites
- **Python:** 3.10.1 to 3.11.9
- **Unity:** 6000.0 or later
- **OS:** Windows, macOS, or Linux

### Installation

```bash
# Clone this fork
git clone https://github.com/quanticsoul4772/ml-agents.git
cd ml-agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -e ./ml-agents-envs
pip install -e ./ml-agents

# Verify installation
python -c "import mlagents; print('ML-Agents version:', mlagents.trainers.__version__)"
```

### WSL Setup (Recommended for Windows)

```bash
# From Windows
wsl -d Ubuntu
cd /mnt/c/path/to/ml-agents
source wsl-setup.sh
```

### Training

```bash
# Train an agent
mlagents-learn config/ppo/3DBall.yaml --run-id=3DBall_01

# Monitor with TensorBoard
tensorboard --logdir=results
```

---

## Syncing with Upstream

This fork receives updates from Unity's upstream repository:

```bash
# Fetch upstream changes
git fetch upstream

# Merge into your branch
git checkout main
git merge upstream/develop

# Push to this fork only
git push origin main
```

---

## Project Structure

```
ml-agents/
├── ml-agents/              # Python training package
│   └── mlagents/trainers/  # Training algorithms (PPO, SAC, MA-POCA)
├── ml-agents-envs/         # Python environment interface
├── com.unity.ml-agents/    # Unity C# package (with performance optimizations)
├── Project/                # Unity example project (Input System migrated)
├── config/                 # Training configurations
└── AGENTS.md              # Development guide
```

---

## Documentation

- **[AGENTS.md](./AGENTS.md)** - Development guide with build, test, and training commands
- **[PROJECT-NOTES.md](./PROJECT-NOTES.md)** - Improvement notes and roadmap
- **[Unity Package Docs](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest)** - Official documentation

---

## Running Tests

```bash
# Python tests (requires Python 3.10-3.11)
pip install -r test_requirements.txt
pytest --cov=ml-agents --cov=ml-agents-envs -m "not slow"

# Unity C# tests (run from Unity Editor)
# Window → General → Test Runner → EditMode
```

### Quick Verification

To verify your installation works beyond basic imports:

```bash
# Run a quick training test (takes ~30 seconds)
mlagents-learn config/ppo/3DBall.yaml --run-id=test --max-steps=1000

# Or run the Python test suite
pytest ml-agents-envs/tests/ -v -x --tb=short
```

---

## Upstream Project

This fork is based on [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) - an open-source project that enables games and simulations to serve as environments for training intelligent agents.

**Upstream Features:**
- 17+ example Unity environments
- PPO, SAC, MA-POCA training algorithms
- Imitation learning (BC and GAIL)
- Curriculum learning
- Multi-agent training
- Gym and PettingZoo wrappers

For the full upstream documentation, see the [Unity ML-Agents Documentation](https://docs.unity3d.com/Packages/com.unity.ml-agents@latest).

---

## Community and Support

For help with ML-Agents (this fork or upstream):

- [Unity ML-Agents Discussions](https://discussions.unity.com/tag/ml-agents) - Q&A and community support
- [Discord](https://discord.com/channels/489222168727519232/1202574086115557446) - Real-time chat
- [GitHub Issues](https://github.com/Unity-Technologies/ml-agents/issues) - Bug reports (upstream)
- [ML-Agents Tutorials](https://www.youtube.com/playlist?list=PLzDRvYVwl53vehwiN_odYJkPBzcqFw110) - Video tutorials by CodeMonkeyUnity
- [Hugging Face Course](https://huggingface.co/learn/deep-rl-course/en/unit5/introduction) - Introduction to ML-Agents

---

## Contributing

Contributions to this fork are welcome! Please:
1. Fork this repository
2. Create a feature branch
3. Run `pre-commit run --all-files` before committing
4. Submit a pull request

---

## License

[Apache License 2.0](LICENSE.md)

---

## Citation

If you use this fork or Unity ML-Agents in research, please cite:

```bibtex
@article{juliani2020,
  title={Unity: A general platform for intelligent agents},
  author={Juliani, Arthur and Berges, Vincent-Pierre and Teng, Ervin and Cohen, Andrew and Harper, Jonathan and Elion, Chris and Goy, Chris and Gao, Yuan and Henry, Hunter and Mattar, Marwan and Lange, Danny},
  journal={arXiv preprint arXiv:1809.02627},
  year={2020}
}
```
