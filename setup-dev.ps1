# ML-Agents Development Environment Setup Script (Windows PowerShell)
# Single command setup: .\setup-dev.ps1

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "ML-Agents Development Environment Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..."
try {
    $pythonVersion = python --version 2>&1 | Out-String
    Write-Host "✓ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.10.1 to 3.11.9" -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
Write-Host ""
if (-Not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..."
.\venv\Scripts\Activate.ps1
Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel --quiet
Write-Host "✓ pip upgraded" -ForegroundColor Green

# Install packages
Write-Host ""
Write-Host "Installing ML-Agents packages..."
pip install -e .\ml-agents-envs --quiet
Write-Host "✓ ml-agents-envs installed" -ForegroundColor Green

pip install -e .\ml-agents --quiet
Write-Host "✓ ml-agents installed" -ForegroundColor Green

# Install test dependencies
Write-Host ""
Write-Host "Installing test dependencies..."
pip install -r test_requirements.txt --quiet
Write-Host "✓ Test dependencies installed" -ForegroundColor Green

# Install pre-commit hooks
Write-Host ""
Write-Host "Setting up pre-commit hooks..."
pip install pre-commit --quiet
pre-commit install
Write-Host "✓ Pre-commit hooks installed" -ForegroundColor Green

# Verify installation
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Verifying Installation" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "mlagents-learn available:"
Get-Command mlagents-learn -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source

Write-Host ""
Write-Host "pytest available:"
pytest --version

Write-Host ""
Write-Host "pre-commit installed:"
pre-commit --version

# Create .env file if it doesn't exist
Write-Host ""
if (-Not (Test-Path ".env")) {
    Write-Host "Creating .env file from template..."
    Copy-Item .env.example .env
    Write-Host "✓ .env file created (edit as needed)" -ForegroundColor Green
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Quick Start Commands:"
Write-Host "  • Run tests:      pytest --cov=ml-agents --cov=ml-agents-envs -m 'not slow'"
Write-Host "  • Format code:    pre-commit run --all-files"
Write-Host "  • Train agent:    mlagents-learn config/ppo/3DBall.yaml --run-id=test"
Write-Host "  • View results:   tensorboard --logdir=results"
Write-Host ""
Write-Host "Virtual environment is active. To deactivate, run: deactivate"
Write-Host ""
