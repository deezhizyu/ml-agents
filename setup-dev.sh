#!/bin/bash

# ML-Agents Development Environment Setup Script
# Single command setup: ./setup-dev.sh

set -e

echo "=========================================="
echo "ML-Agents Development Environment Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "✓ Python $PYTHON_VERSION found"
else
    echo "✗ Python 3 not found. Please install Python 3.10.1 to 3.11.9"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo ""
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel -q
echo "✓ pip upgraded"

# Install packages
echo ""
echo "Installing ML-Agents packages..."
pip install -e ./ml-agents-envs -q
echo "✓ ml-agents-envs installed"

pip install -e ./ml-agents -q
echo "✓ ml-agents installed"

# Install test dependencies
echo ""
echo "Installing test dependencies..."
pip install -r test_requirements.txt -q
echo "✓ Test dependencies installed"

# Install pre-commit hooks
echo ""
echo "Setting up pre-commit hooks..."
pip install pre-commit -q
pre-commit install
echo "✓ Pre-commit hooks installed"

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
echo ""

echo "mlagents-learn available:"
which mlagents-learn

echo ""
echo "pytest available:"
pytest --version

echo ""
echo "pre-commit installed:"
pre-commit --version

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created (edit as needed)"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Quick Start Commands:"
echo "  • Run tests:      pytest --cov=ml-agents --cov=ml-agents-envs -m 'not slow'"
echo "  • Format code:    pre-commit run --all-files"
echo "  • Train agent:    mlagents-learn config/ppo/3DBall.yaml --run-id=test"
echo "  • View results:   tensorboard --logdir=results"
echo ""
echo "Virtual environment is active. To deactivate, run: deactivate"
echo ""
