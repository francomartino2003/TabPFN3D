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

### 6. Download real data and run TabPFN baselines

```bash
# Download UCR (128) + UEA (30) datasets; writes datasets_summary.csv
python 01_real_data/download.py

# Download HC2 SOTA accuracy tables (UCR + UEA standard subsets)
python 01_real_data/download_hc2_benchmarks.py
```

Then run the TabPFN baseline benchmark (three configurations, 30 seeds each,
on all PFN-eligible datasets):

| Configuration | Description |
|---|---|
| 1 ensemble | Standard TabPFN with 1 estimator (default sklearn preprocessing) |
| 8 ensembles | Standard TabPFN with 8 estimators (default sklearn preprocessing) |
| 8 ensembles, our inference | Pretrained TabPFN (no finetuning) run through our inference pipeline: per-channel normalisation, channel and class permutations per iteration, no SVD, no feature subsampling |

```bash
# Recommended — cluster (single GPU, UCR then UEA, ~36 h with 3 configs)
sbatch benchmark_tabpfn.sbatch

# Local alternative
python 01_real_data/benchmark_tabpfn.py --device auto
```

Results are saved to `01_real_data/benchmark_results/` and loaded automatically
by the final evaluation.  See **[01_real_data/README.md](01_real_data/README.md)**
for details on the PFN filter, subsampling, and dataset counts.

---

## Project Structure

| Folder | Description |
|---|---|
| `00_TabPFN/` | Upstream TabPFN clone (not committed; see step 4) |
| `01_real_data/` | UCR/UEA datasets, HC2 benchmarks, TabPFN baselines |
| `02_synthetic_data/` | Kernel-DAG generator and augmentation pipeline |
| `03_finetuning/` | Two-phase finetuning pipeline |
| `04_evaluation/` | Benchmark and evaluation scripts |

---

## 7  Synthetic data — `02_synthetic_data/`

The synthetic prior generates time series classification datasets from a
random kernel-DAG.  All hyperparameters live in `hyperparameters.py`.
See **[02_synthetic_data/README.md](02_synthetic_data/README.md)** for the
full methods description.

### Modules

| File | Purpose |
|---|---|
| `hyperparameters.py` | All sampling distributions (DAG topology, GP kernels, Conv1D, augmentation) |
| `dag_structure.py` | DAG builder: samples topology, assigns node types, wires edges, assigns roles |
| `generator.py` | `DatasetGenerator`: builds operations, propagates observations, extracts dataset |
| `augmentation.py` | `augment_dataset()`: feature permutation, class permutation, value transforms, temporal granularity, missing values |
| `visualize_transforms.py` | Utility plots for value and temporal transforms |

### Visualisations

Each script can be run directly to produce diagnostic plots under
`02_synthetic_data/visualizations/`:

```bash
# Sample and plot 5 datasets (series per class) + their DAG graphs
python 02_synthetic_data/generator.py --n 5 --seed 0

# Sample and plot 5 DAG topologies only
python 02_synthetic_data/dag_structure.py --n 5 --seed 0

# Plot value transforms (log, exp, squash, KDI, Kumaraswamy) and
# temporal granularity transforms (pooling, step-repeat) on real examples
python 02_synthetic_data/augmentation.py --n-series 6 --seed 42
```

---

## 8  Finetuning — `03_finetuning/`

Two-phase transfer learning on top of the pretrained TabPFN transformer.
See **[03_finetuning/README.md](03_finetuning/README.md)** for the full
architecture and training description.

### Modules

| File | Purpose |
|---|---|
| `model.py` | `build_overlap_model()`: patch encoder, global Conv1D encoder, overlap expansion, positional embeddings |
| `inference.py` | `evaluate_ensemble()`: multi-iteration ensemble inference (channel/class permutations, per-channel normalisation) |
| `worker_generator.py` | Continuous synthetic dataset producer; writes `.npz` batch files to a shared queue directory |
| `worker_trainer_v2.py` | Training loop: reads batches from the queue, computes loss, updates weights, saves checkpoints |
| `worker_evaluator_v2.py` | Periodic evaluation on held-out synthetic and real datasets; updates `best.pt` |
| `data_utils.py` | Batch serialisation helpers and real-data loaders used by generator and evaluator |

### Phase 1 — Encoder warm-up

Trains only the patch encoder and global Conv1D encoder; the pretrained
TabPFN transformer weights are frozen.

```bash
# Cluster (recommended)
sbatch step1_encoder.sbatch

# Local — start all three workers in separate terminals
python 03_finetuning/worker_generator.py  --out-dir /tmp/queue --max-datasets 50000
python 03_finetuning/worker_trainer_v2.py --queue-dir /tmp/queue --phase 1 \
    --checkpoint-dir 03_finetuning/checkpoints/phase1
python 03_finetuning/worker_evaluator_v2.py \
    --checkpoint-dir 03_finetuning/checkpoints/phase1
```

### Phase 2 — Full finetuning

All parameters (encoder + transformer) are trained jointly with a constant
learning rate.  Initialises from the Phase 1 `best.pt`.

```bash
# Cluster (recommended)
sbatch step2_finetune.sbatch

# Local
python 03_finetuning/worker_generator.py  --out-dir /tmp/queue --max-datasets 50000
python 03_finetuning/worker_trainer_v2.py --queue-dir /tmp/queue --phase 2 \
    --init-checkpoint 03_finetuning/checkpoints/phase1/best.pt \
    --checkpoint-dir  03_finetuning/checkpoints/phase2
python 03_finetuning/worker_evaluator_v2.py \
    --checkpoint-dir 03_finetuning/checkpoints/phase2
```

Checkpoints are written to `03_finetuning/checkpoints/phase{1,2}/`
(`best.pt` = best validation accuracy, `last.pt` = most recent step).

---

## 9  Evaluation — `04_evaluation/`

### `benchmark_ours.py` — benchmark our finetuned model

Evaluates the finetuned model in four configurations on all PFN-eligible
datasets (30 seeds each, accuracy and AUC averaged):

| Configuration | Description |
|---|---|
| Phase-1 best, 8 ensembles | Phase-1 checkpoint, 8 inference iterations |
| Phase-2 best, 1 ensemble | Phase-2 checkpoint, 1 inference iteration |
| Phase-2 best, 8 ensembles | Phase-2 checkpoint, 8 inference iterations |
| Phase-2 best, 8 ensembles + step-repeat | As above, with step-repeat preprocessing applied when the series is short |

```bash
# Cluster
sbatch benchmark_ours.sbatch

# Local
python 04_evaluation/benchmark_ours.py \
    --phase1-ckpt 03_finetuning/checkpoints/phase1/best.pt \
    --phase2-ckpt 03_finetuning/checkpoints/phase2/best.pt \
    --device auto
```

Results: `04_evaluation/results/benchmark_ours/{ucr,uea}_benchmark_ours.csv`

### `final_evaluation.py` — ablation and SOTA comparison

Compares TabPFN baselines (from `01_real_data/benchmark_results/`) against a
finetuned checkpoint on all PFN-eligible datasets, then compares against HC2
on the standard subset.

```bash
python 04_evaluation/final_evaluation.py \
    --phase2-ckpt 03_finetuning/checkpoints/phase2/best.pt \
    --device auto
```

### `preprocessing_eval.py` — step-repeat preprocessing ablation

Evaluates the effect of step-repeat preprocessing on a finetuned checkpoint.

```bash
python 04_evaluation/preprocessing_eval.py \
    --ckpt 03_finetuning/checkpoints/phase2/best.pt \
    --device auto
```
