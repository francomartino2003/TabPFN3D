# 04\_evaluation

Post-training evaluation of the **Overlap TabPFN** model (Model D) against
TabPFN baselines (Model B / C) and state-of-the-art time-series classifiers.

---

## Models compared

| ID | Name | Weights | Ensemble pipeline |
|----|------|---------|-------------------|
| **B** | TabPFN standard | Pretrained v2.5 | TabPFN default (SVD, fingerprint, feature-subsampling, 8 ensembles) |
| **C** | Pretrained + our pipeline | Pretrained v2.5 | Our channel-aware pipeline (no SVD, no fingerprint) |
| **D** | Overlap finetuned | Fine-tuned checkpoint | Our channel-aware pipeline, overlap tokenisation, temporal PE |

B and C are pre-computed by `01_real_data/benchmark_tabpfn.py` (30 seeds, averaged).
D is always run live from the provided checkpoint.

---

## Test 1 — Ablation study (B vs C vs D)

Evaluates every model on all PFN-eligible datasets
(`01_real_data/datasets_summary.csv`, `passes_pfn_filters=True`).

- ACC and AUC per dataset
- Wilcoxon signed-rank test for each pairwise comparison
  (C > B, D > C, D > B)
- Scatter plots

Ablation logic:
- **B vs C** isolates the contribution of the ensemble / preprocessing pipeline.
- **C vs D** isolates the contribution of the finetuning step.
- **B vs D** shows the combined improvement.

Output: `results/<run>/test1_ablation.csv`, `test1_ablation_scatter.png`

---

## Test 2 — SOTA comparison (D vs HC2)

Compares D against the HIVE-COTE 2.0 accuracy benchmarks on the standard
subset of UCR / UEA problems (those without variable-length series or missing
values, pre-computed over 30 resamples in the HC2 paper).

Benchmarks are loaded from `01_real_data/benchmarks_hc2/` (downloaded by
`python 01_real_data/download_hc2_benchmarks.py`).

Metrics: mean rank, rank-1 count (total and strict), mean accuracy.

Output: `results/<run>/test2_sota_rankings.csv`

---

## Usage

```bash
# From the project root:

# Standard run — 8 ensembles (matches benchmark_tabpfn.py, fair comparison)
python 04_evaluation/final_evaluation.py 04_evaluation/checkpoints/best.pt

# Custom number of ensemble iterations
python 04_evaluation/final_evaluation.py 04_evaluation/checkpoints/best.pt --n-ensemble 16

# Evaluate a specific device
python 04_evaluation/final_evaluation.py 04_evaluation/checkpoints/best.pt --device cuda

# UEA collection for Test 2
python 04_evaluation/final_evaluation.py 04_evaluation/checkpoints/best.pt --collection uea

# Skip Test 1 and reload from existing CSV
python 04_evaluation/final_evaluation.py 04_evaluation/checkpoints/best.pt --skip-test1

# Evaluate a custom subset of datasets
python 04_evaluation/final_evaluation.py 04_evaluation/checkpoints/best.pt --dataset-names "ECG200,FaceAll"

# Also plot training history (needs history.json next to checkpoint)
python 04_evaluation/final_evaluation.py 04_evaluation/checkpoints/best.pt --with-history

# Evaluate Model C live (if pre-computed C results are absent)
python 04_evaluation/final_evaluation.py 04_evaluation/checkpoints/best.pt --model-c
```

Results are saved to `04_evaluation/results/<checkpoint_stem>/`.

---

## Prerequisites

| What | Command |
|------|---------|
| UCR/UEA datasets | `python 01_real_data/download.py` |
| HC2 benchmarks | `python 01_real_data/download_hc2_benchmarks.py` |
| Pre-computed B/C | `sbatch benchmark_tabpfn.sbatch` (or `python 01_real_data/benchmark_tabpfn.py`) |
| Finetuned checkpoint | Produced by `sbatch train_overlap_v2.sbatch` |

---

## File structure

```
04_evaluation/
├── final_evaluation.py     # Main evaluation script (Test 1 + Test 2)
├── checkpoints/            # Trained model checkpoints (.pt files)
│   └── best.pt             # Best checkpoint from train_overlap_v2.sbatch
├── results/                # Output directory (created automatically)
│   └── <ckpt_stem>/
│       ├── test1_ablation.csv
│       ├── test1_ablation_scatter.png
│       └── test2_sota_rankings.csv
└── README.md
```

Note: `feat_raw_gpu_seed*.pt` is a feature-embedding lookup cache used internally
by `03_finetuning/model.py` and must remain in `03_finetuning/`.

---

## External paths used

| Path | Purpose |
|------|---------|
| `../00_TabPFN/src/` | TabPFN source |
| `../03_finetuning/` | `model.py`, `inference.py` |
| `../01_real_data/datasets_summary.csv` | Dataset metadata |
| `../01_real_data/data/{ucr,uea}/` | NPZ files |
| `../01_real_data/benchmark_results/` | Pre-computed B/C CSVs |
| `../01_real_data/benchmarks_hc2/` | HC2 SOTA accuracy tables |
