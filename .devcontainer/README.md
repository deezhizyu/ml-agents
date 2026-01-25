# ML-Agents DevContainer

This directory contains a complete development container configuration for ML-Agents, providing a consistent development environment across all platforms.

## What is a DevContainer?

A development container (DevContainer) is a Docker container configured specifically for development. It includes:
- All required tools and dependencies
- VS Code extensions
- Pre-configured settings
- Automated setup scripts

## Features

### Included Tools
- **Python 3.10** with pip, setuptools, wheel
- **Git** for version control
- **GitHub CLI** (gh) for repository operations
- **Docker-in-Docker** for building Unity environments

### VS Code Extensions
- Python language support (Pylance)
- Black formatter
- Flake8 linter
- Mypy type checker
- Ruff linter
- YAML/TOML support
- GitLens
- GitHub Copilot (if you have access)

### Automatic Setup
The `post-create.sh` script automatically:
1. Upgrades pip
2. Installs ml-agents and ml-agents-envs packages
3. Installs test dependencies
4. Sets up pre-commit hooks
5. Creates results/models directories

### Port Forwarding
- **Port 5005**: Unity ML-Agents communication
- **Port 6006**: TensorBoard visualization

## Quick Start

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [VS Code](https://code.visualstudio.com/)
- [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Usage

1. **Open in Container**:
   - Open this repository in VS Code
   - Press `F1` and select "Dev Containers: Reopen in Container"
   - Wait for container to build (first time takes 2-5 minutes)

2. **Verify Setup**:
   ```bash
   # Check Python
   python --version

   # Check packages
   pip list | grep mlagents

   # Check tools
   pre-commit --version
   pytest --version
   ```

3. **Start Developing**:
   ```bash
   # Run tests
   pytest --cov=ml-agents --cov=ml-agents-envs -m "not slow"

   # Train agent
   mlagents-learn config/ppo/3DBall.yaml --run-id=test

   # View TensorBoard
   tensorboard --logdir=results
   ```

## Configuration Files

### `devcontainer.json`
Main configuration file specifying:
- Base image (Python 3.10 on Debian Bullseye)
- VS Code extensions and settings
- Port forwarding
- Post-creation command
- Environment variables

### `post-create.sh`
Bash script that runs after container creation:
- Installs all dependencies
- Sets up development tools
- Creates necessary directories
- Verifies installation

## Customization

### Add Extensions
Edit `devcontainer.json`:
```json
"extensions": [
    "ms-python.python",
    "your-extension-id"
]
```

### Change Python Version
Edit `devcontainer.json`:
```json
"image": "mcr.microsoft.com/devcontainers/python:3.11-bullseye"
```

### Add System Packages
Edit `devcontainer.json`:
```json
"features": {
    "ghcr.io/devcontainers/features/your-feature:1": {}
}
```

### Modify Settings
Edit `settings` in `devcontainer.json` to customize VS Code behavior.

## Troubleshooting

### Container Build Fails
1. Check Docker is running: `docker ps`
2. Restart Docker Desktop
3. Clear Docker cache: `docker system prune -a`
4. Rebuild container: `Dev Containers: Rebuild Container`

### Packages Not Installed
1. Check post-create.sh ran successfully
2. Manually run: `bash .devcontainer/post-create.sh`
3. Check logs in VS Code: `Dev Containers: Show Container Log`

### Slow Performance
1. **Windows/Mac**: Allocate more resources to Docker Desktop
   - Settings → Resources → Increase CPU/Memory
2. **Use volumes**: Results and models directories use bind mounts for better performance

### Port Already in Use
1. Check for other containers: `docker ps`
2. Stop conflicting containers
3. Change ports in `devcontainer.json` if needed

### Extensions Not Loading
1. Reload window: `Developer: Reload Window`
2. Check extension compatibility
3. Manually install: `Extensions: Install Extensions`

## Performance Tips

### Build Caching
- First build takes longest (2-5 minutes)
- Subsequent builds use cache (30-60 seconds)
- Base image is cached and reused

### Volume Mounts
Results and models directories use bind mounts:
```json
"mounts": [
    "source=${localWorkspaceFolder}/results,target=/workspaces/ml-agents/results,type=bind,consistency=cached"
]
```

This ensures large training outputs don't slow down container.

### Resource Allocation
Recommended Docker resources:
- **CPU**: 4+ cores
- **Memory**: 8+ GB
- **Swap**: 2+ GB
- **Disk**: 20+ GB

## Comparison with Local Setup

| Aspect | Local Setup | DevContainer |
|--------|-------------|--------------|
| **Setup Time** | 10-15 min manual | 2-5 min automatic |
| **Consistency** | Varies by system | Identical everywhere |
| **Dependencies** | Manual management | Pre-installed |
| **Isolation** | System-wide | Containerized |
| **Cleanup** | Manual uninstall | Delete container |

## Integration with CI/CD

The devcontainer uses the same base image and setup process as CI:
- Same Python version (3.10.12)
- Same dependency versions
- Same test configuration

This ensures "works on my machine" issues are minimized.

## Advanced Usage

### Multiple Containers
Run multiple instances for different branches:
```bash
# Each workspace folder gets its own container
code /path/to/ml-agents
code /path/to/ml-agents-feature-branch
```

### Custom Dockerfile
Create `.devcontainer/Dockerfile` for more control:
```dockerfile
FROM mcr.microsoft.com/devcontainers/python:3.10-bullseye
RUN apt-get update && apt-get install -y custom-package
```

Reference in `devcontainer.json`:
```json
"dockerFile": "Dockerfile"
```

### Pre-build Images
Speed up container startup by pre-building:
```bash
docker build -t ml-agents-dev .devcontainer
```

## VS Code Features

### Integrated Terminal
- Terminal runs inside container
- Direct access to ml-agents commands
- Pre-configured with virtual environment

### IntelliSense
- Full Python type hints
- Import auto-completion
- Function signatures

### Debugging
- Set breakpoints in Python code
- Debug training runs
- Inspect variables

### Source Control
- Git integrated
- GitHub CLI available
- Commit/push from VS Code

## Security Considerations

### Sensitive Data
- Don't commit `.env` file
- Use `.env.example` template
- Mount secrets as volumes if needed

### Docker Socket
DevContainer has access to Docker socket for building Unity environments:
```json
"features": {
    "ghcr.io/devcontainers/features/docker-in-docker:2": {}
}
```

Be cautious when running untrusted code.

## Further Reading

- [VS Code DevContainers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [DevContainer Specification](https://containers.dev/)
- [Docker Documentation](https://docs.docker.com/)

## Support

If you encounter issues:
1. Check this README for troubleshooting
2. Review [GitHub Issues](https://github.com/Unity-Technologies/ml-agents/issues)
3. Ask in [Unity Discussions](https://discussions.unity.com/tag/ml-agents)
4. Join [Discord community](https://discord.com/channels/489222168727519232/1202574086115557446)
