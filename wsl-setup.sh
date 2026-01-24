#!/bin/bash
# ML-Agents WSL Environment Setup Script
# Run this script from within WSL to activate the ml-agents environment

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
if [ -f "$SCRIPT_DIR/.venv-wsl/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv-wsl/bin/activate"
    echo "✓ ML-Agents virtual environment activated"
    echo "  Python: $(python --version)"
    echo "  Location: $(which python)"
    echo ""
    echo "Available commands:"
    echo "  mlagents-learn config.yaml --run-id=MyRun  # Start training"
    echo "  python -m pytest ml-agents-envs/tests/     # Run envs tests"
    echo "  python -m pytest ml-agents/                # Run trainer tests"
else
    echo "✗ Virtual environment not found at $SCRIPT_DIR/.venv-wsl"
    echo "  Run: python3.10 -m venv .venv-wsl"
    echo "  Then: pip install -e ./ml-agents-envs -e ./ml-agents"
fi
