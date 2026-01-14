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

# Install only essential packages for training (minimal set)
# Many packages in requirements.txt require numpy 2.x which needs GCC >= 9.3
echo "Installing essential packages for training..."
# scikit-learn: use wheel precompiled (compatible with numpy 1.x)
# scikit-learn 1.5.x is the last version compatible with numpy < 2.0
pip install "scikit-learn>=1.3.0,<1.6.0" --only-binary :all:
# pandas: install < 2.0 (compatible with numpy 1.x) before tabpfn
# pandas 1.5.x is the last version compatible with numpy < 2.0
pip install "pandas>=1.4.0,<2.0" --only-binary pandas
pip install "tabpfn>=6.0.0"
pip install "einops>=0.7.0"
# matplotlib: needed for training plots
pip install "matplotlib>=3.7.0"
# wandb is optional, uncomment if needed:
# pip install wandb>=0.15.0

# Pre-download TabPFN weights (optional but recommended)
echo "Pre-downloading TabPFN weights..."
python -c "from tabpfn import TabPFNClassifier; TabPFNClassifier()" 2>&1 | head -20 || echo "TabPFN weights download (will download automatically on first use if needed)"

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment in future sessions:"
echo "  module load sloan/python/3.11.4"
echo "  source \$HOME/venv_tabpfn3d/bin/activate"
echo ""
