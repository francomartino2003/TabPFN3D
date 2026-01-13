# Setup and Training on MIT Engaging Cluster

## 📋 Prerequisites

- MIT Kerberos credentials
- Access to `pi_dbertsim` partition
- SSH access to `eosloan.mit.edu`

## 🚀 Step-by-Step Setup

### 1. Connect to the cluster

```bash
ssh -K franco03@eosloan.mit.edu
```

The `-K` flag forwards your Kerberos credentials.

### 2. Clone repositories

```bash
# Clone main repository
git clone https://github.com/francomartino2003/TabPFN3D.git
cd TabPFN3D

# Clone TabPFN
git clone https://github.com/PriorLabs/tabpfn.git 00_TabPFN
```

### 3. Run setup script

```bash
# Make setup script executable
chmod +x setup_engaging.sh

# Run setup (this will create venv and install dependencies)
bash setup_engaging.sh
```

This will:
- Load Python 3.11.4 from Sloan module
- Create virtual environment in `$HOME/venv_tabpfn3d`
- Install PyTorch with CUDA support
- Install all dependencies from `requirements.txt`

### 4. Verify setup (optional)

```bash
module load sloan/python/3.11.4
source $HOME/venv_tabpfn3d/bin/activate
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 🏃 Running Training

### Debug/Smoke Test (recommended first)

Quick test to verify everything works (5-10 minutes):

```bash
sbatch train_temporal_encoder_debug.sbatch
```

Check status:
```bash
squeue -u franco03
```

View output:
```bash
tail -f logs/train_debug_*.out
```

### Full Training

Once debug test passes:

```bash
sbatch train_temporal_encoder.sbatch
```

Monitor:
```bash
squeue -u franco03
tail -f logs/train_*.out
```

## 📊 Job Management

```bash
# Check your jobs
squeue -u franco03

# Cancel a job
scancel <JOB_ID>

# Check job details
scontrol show job <JOB_ID>
```

## 📁 Important Directories

- Code: `$HOME/TabPFN3D/`
- Virtual environment: `$HOME/venv_tabpfn3d/`
- Training logs: `$HOME/TabPFN3D/logs/`
- Checkpoints: `$HOME/TabPFN3D/04_temporal_encoder/checkpoints/`

## ⚠️ Notes

- **Never run training directly on login nodes** - always use `sbatch`
- The scripts automatically:
  - Load the Python module
  - Activate the virtual environment
  - Set CUDA device visibility
  - Create necessary directories

- Training configuration can be modified in `04_temporal_encoder/training_config.py`
- Default training disables real dataset evaluation (`eval_real_datasets = False`)

## 🔧 Troubleshooting

If setup fails:
1. Check Python module: `module avail sloan/python`
2. Check disk quota: `quota -s`
3. Check GPU availability: `sinfo -p pi_dbertsim`

If training fails:
1. Check logs: `tail -f logs/train_*.err`
2. Verify CUDA: Check if PyTorch detects GPU in logs
3. Check memory: Increase `--mem` if OOM errors occur
