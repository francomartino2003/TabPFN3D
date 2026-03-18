# TabPFN3D

Research project extending TabPFN with 3D temporal encoders and custom synthetic data generators for time series classification.

---

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install PyTorch

Install PyTorch before the rest of the requirements, choosing the build that matches your hardware:

```bash
# GPU (CUDA 12.6)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CPU only
pip install torch torchvision
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Clone TabPFN into `00_TabPFN/`

`00_TabPFN/` is not committed to this repo. Clone the upstream repository and check out the exact commit used in this project:

```bash
git clone https://github.com/PriorLabs/tabpfn.git 00_TabPFN
cd 00_TabPFN
git checkout 0777167   # tabpfn 6.3.0
cd ..
pip install -e 00_TabPFN
```

### 5. Obtain the model weights

The TabPFN v2.5 weights are hosted on a gated HuggingFace repository and require accepting the terms of use:

1. Go to [huggingface.co/Prior-Labs/tabpfn_2_5](https://huggingface.co/Prior-Labs/tabpfn_2_5) and accept the license.
2. Authenticate locally:

```bash
pip install huggingface_hub
huggingface-cli login
```

Once authenticated, weights are downloaded automatically on the first run.

### 6. Download data and run benchmarks

From the project root, run the scripts in `01_real_data/` in this order:

| Step | Command | What it does |
|------|---------|--------------|
| 1 | `python 01_real_data/download.py` | Downloads 128 UCR + 30 UEA datasets via aeon; writes `datasets_summary.csv`. |
| 2 | `python 01_real_data/download_hc2_benchmarks.py` | Downloads HC2 SOTA accuracy tables (112 UCR, 26 UEA) into `01_real_data/benchmarks_hc2/`. |
| 3 | `sbatch benchmark_tabpfn.sbatch` | Pre-computes Models B and C (TabPFN standard and our ensemble pipeline, 30 runs each) on the cluster. Outputs CSVs in `01_real_data/benchmark_results/` that are loaded automatically by the final evaluation. |

Step 3 can also run locally with `python 01_real_data/benchmark_tabpfn.py --device auto`, but it is slow (several hours). On the cluster it takes ~4 hours with a single GPU.

For more info, see **[01_real_data/README.md](01_real_data/README.md)**.

### 7. Run final evaluation

After downloading a finetuned checkpoint (`best.pt`) from the cluster:

```bash
python 04_evaluation/final_evaluation.py 04_evaluation/checkpoints/best.pt
```

**Test 1** (ablation): loads pre-computed B and C from step 3, runs Model D live,
and produces Wilcoxon-test summaries and scatter plots.

**Test 2** (SOTA): compares D against the HC2 accuracy benchmarks from step 2.

Results are saved to `04_evaluation/results/<checkpoint_stem>/`.
See **[04_evaluation/README.md](04_evaluation/README.md)** for all options.

---

## Project Structure

| Folder | Description |
|---|---|
| `00_TabPFN/` | Upstream TabPFN clone (see step 4 above) |
| `01_real_data/` | UCR/UEA datasets and benchmarks; see `01_real_data/README.md` |
| `02_synthetic_generator_2D/` | 2D synthetic data generator |
| `03_synthetic_generator_3D/` | 3D synthetic data generator |
| `04_temporal_encoder/` | Temporal encoder architecture and training |
| `05_flattened_benchmark/` | Baseline benchmarks on flattened representations |
| `06_generator_experiments/` | Early generator experiments |
| `02_synthetic_data/` | Kernel + DAG generator, augmentation pipeline |
| `03_finetuning/` | TabPFN finetuning pipeline (see `03_finetuning/README.md`) |
| `04_evaluation/` | Final evaluation: ablation (B/C/D) and SOTA comparison (see `04_evaluation/README.md`) |
