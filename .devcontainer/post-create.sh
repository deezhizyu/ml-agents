#!/bin/bash

# ML-Agents Devcontainer Post-Create Script
# This script runs after the devcontainer is created to set up the environment

set -e

echo "=========================================="
echo "ML-Agents Development Environment Setup"
echo "=========================================="

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Install ML-Agents packages in editable mode
echo "Installing ml-agents-envs..."
pip install -e ./ml-agents-envs

echo "Installing ml-agents..."
pip install -e ./ml-agents

# Install test dependencies
echo "Installing test dependencies..."
pip install -r test_requirements.txt

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pip install pre-commit
pre-commit install

# Create results directories
echo "Creating results directories..."
mkdir -p results models summaries

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="

echo "Python version:"
python --version

echo ""
echo "mlagents-learn available:"
which mlagents-learn

echo ""
echo "Pytest available:"
pytest --version

echo ""
echo "Pre-commit installed:"
pre-commit --version

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Quick Start:"
echo "  - Run tests: pytest --cov=ml-agents --cov=ml-agents-envs -m 'not slow'"
echo "  - Format code: pre-commit run --all-files"
echo "  - Train agent: mlagents-learn config/ppo/3DBall.yaml --run-id=test"
echo "  - View TensorBoard: tensorboard --logdir=results"
echo ""
