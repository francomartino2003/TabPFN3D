#!/bin/bash
# Setup script for MIT Engaging cluster
# Run this ONCE after cloning the repository

set -e  # Exit on error

echo "=========================================="
echo "Setting up TabPFN3D on MIT Engaging"
echo "=========================================="

# Load Python module (Sloan Python)
echo "Loading Python module..."
module load sloan/python/3.11.4

# Load newer GCC (needed for building some packages)
# numpy 2.x requires GCC >= 9.3, but we'll install numpy 1.x from wheels
echo "Loading GCC module (if needed for other packages)..."
module load gcc/9.3.0 2>/dev/null || echo "GCC module not available, using system GCC"

# Create virtual environment in $HOME
echo "Creating virtual environment..."
python -m venv $HOME/venv_tabpfn3d

# Activate venv
echo "Activating virtual environment..."
source $HOME/venv_tabpfn3d/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install numpy FIRST from wheel (precompiled) to avoid GCC version issues
# Pin to numpy < 2.0 for compatibility - must install before PyTorch
echo "Installing numpy < 2.0 (precompiled wheel)..."
pip install "numpy<2.0,>=1.24.0" --only-binary numpy

# Install PyTorch with CUDA support (for cluster GPUs)
# Using CUDA 11.8 for compatibility with cluster CUDA versions (11.2/11.3)
# Using --extra-index-url so other packages install from PyPI
# numpy is already installed, so PyTorch won't try to install numpy 2.x
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# Install pandas < 2.0 (compatible with numpy 1.x) before requirements.txt
# pandas 2.x requires numpy 2.x which requires GCC >= 9.3
echo "Installing pandas < 2.0 (compatible with numpy 1.x)..."
pip install "pandas>=1.5.0,<2.0"

# Install other dependencies (pandas already installed, so it won't reinstall)
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt || pip install -r requirements.txt --ignore-installed pandas

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment in future sessions:"
echo "  module load sloan/python/3.11.4"
echo "  source \$HOME/venv_tabpfn3d/bin/activate"
echo ""
