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

# Load newer GCC if available (needed for building some packages)
# Check if gcc module exists, if not continue with system gcc
if module avail gcc 2>&1 | grep -q "gcc"; then
    echo "Loading GCC module..."
    module load gcc/9.3.0 2>/dev/null || module load gcc/10.2.0 2>/dev/null || echo "Using system GCC"
else
    echo "Using system GCC"
fi

# Create virtual environment in $HOME
echo "Creating virtual environment..."
python -m venv $HOME/venv_tabpfn3d

# Activate venv
echo "Activating virtual environment..."
source $HOME/venv_tabpfn3d/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (for cluster GPUs)
# Using CUDA 11.8 for compatibility with cluster CUDA versions (11.2/11.3)
# Using --extra-index-url so other packages install from PyPI
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# Install numpy from wheel (precompiled) to avoid GCC version issues
# Pin to numpy < 2.0 for compatibility with older GCC
echo "Installing numpy (precompiled wheel)..."
pip install "numpy<2.0" --only-binary numpy

# Install other dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment in future sessions:"
echo "  module load sloan/python/3.11.4"
echo "  source \$HOME/venv_tabpfn3d/bin/activate"
echo ""
