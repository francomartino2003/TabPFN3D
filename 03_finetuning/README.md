# 07\_finetuning

Fine-tuning pipeline for the **Overlap TabPFN** model: a TabPFN v2.5 backbone
adapted for time-series classification by replacing the input encoder with a
fresh MLP and using overlapping sliding-window tokenisation (window=16,
stride=8).

---

## Motivation

TabPFN v2.5 processes tabular data with a per-feature-group transformer (default
`features_per_group=3`). For time-series we instead slide a window of 16
timesteps across each channel with a stride of 8, producing overlapping tokens.
Each token receives a composite positional embedding:

```
token_emb(j, g) = feature_emb(j) + sinusoidal_PE(g)
```

where *j* is the original feature-channel index and *g* is the window position
within that channel. This lets the backbone attend across both channels and
time without modifying the pretrained attention weights.

The fresh encoder MLP (xavier-init) and the embedding projection are trained at
a higher learning rate (×10), while the pretrained backbone (24 layers + decoder)
is fine-tuned at a lower rate.

---

## Architecture (`model.py`)

| Component | Details |
|---|---|
| Backbone | TabPFN v2.5 (24 layers, emsize=192, nhead=3) |
| Encoder | MLP(32 → 64 → 192, GELU, no bias) — **fresh** |
| Tokenisation | Overlap window=16, stride=8 |
| Embedding | feature\_emb(j) + sinusoidal PE(g) — **fresh projection** |
| Missing values | Internal `NanHandlingEncoderStep` (indicator + mean imputation) |
| Legacy builder | `build_temporal_tabpfn_fpg8()` for loading old fpg=8 checkpoints |

---

## File structure

```
03_finetuning/
├── model.py               # Architecture, builders, overlap helpers, runtime state
├── data_utils.py          # FinetuneConfig, SyntheticDataGenerator, load_real_datasets
├── inference.py           # forward_single_dataset, deserialize_batch, evaluate_ensemble
├── worker_generator.py    # GPU 0 — generates & serialises synthetic batches
├── worker_trainer_v2.py   # GPU 1 — trains the model, saves last.pt
├── worker_evaluator_v2.py # GPU 2 — evaluates, saves best.pt, writes history.json
└── train_overlap_v2.sbatch  # Slurm job that orchestrates the three workers
```

Post-training evaluation lives in **[04_evaluation/](../04_evaluation/README.md)**.

### Module responsibilities

| Module | Exported symbols used by workers |
|---|---|
| `model.py` | `build_overlap_model`, `extract_patches`, `set_temporal_info`, `WINDOW`, `STRIDE`, `CHANNEL_GROUP`, `pad_to_group`, `build_temporal_tabpfn_fpg8` |
| `augmentation.py` | `augment_dataset` |
| `data_utils.py` | `FinetuneConfig`, `SyntheticDataGenerator`, `load_real_datasets` |
| `inference.py` | `forward_single_dataset`, `deserialize_batch`, `evaluate_ensemble` |

---

## Training pipeline (`train_overlap_v2.sbatch`)

The Slurm job allocates 3 GPUs on one node and runs three workers concurrently:

```
GPU 0  worker_generator.py  →  writes batch_XXXXXXXX.npz to $BATCH_DIR
GPU 1  worker_trainer_v2.py ←  reads .npz, trains, saves last.pt
GPU 2  worker_evaluator_v2.py ← polls last.pt, evaluates, saves best.pt
```

Key environment variables set in the sbatch script:

| Variable | Purpose |
|---|---|
| `BATCH_DIR` | Shared directory for .npz batch files |
| `CHECKPOINT_DIR` | Where `last.pt` and `best.pt` are saved |
| `LOG_DIR` | Where `history.json` is written by the evaluator |
| `N_STEPS` | Total training steps for the trainer |
| `BATCH_SIZE` | Datasets per batch (originals × 4 with augmentation) |

---

## External paths required

| Path | Used by |
|---|---|
| `../00_TabPFN/src/` | All workers (TabPFN source) |
| `../02_synthetic_data/` | DAG generator, hyperparameters, and augmentation pipeline |
| `../01_real_data/datasets_summary.csv` | `data_utils.load_real_datasets` |
| `../01_real_data/data/{ucr,uea}/` | NPZ files (one per dataset split) |

Run `python 01_real_data/download.py` from the project root before training
to create the NPZ files and `datasets_summary.csv`.

---

## Data augmentation (`augmentation.py`)

Each batch contains N/4 originals + 3N/4 augmented copies.
The augmentation pipeline (applied per dataset):

1. Random feature-channel permutation
2. Random class-label permutation
3. Per-feature value transform: none / log / exp / squash / KDI / Kumaraswamy
4. Dataset-level temporal granularity: identity (75%) / pooling (25%) / interpolation (10%)
5. Per-feature independent missing values (NaN, low probability)
6. Right zero-pad T to the next multiple of `group_size`

---

## Inference (`inference.py`)

### `forward_single_dataset`
Bypasses TabPFN's sklearn preprocessing pipeline entirely. Data is passed
directly to the model encoder — NaN values are handled internally by
TabPFN's `NanHandlingEncoderStep` (adds presence indicator + mean imputation
at projection time). Used during training and in the evaluator worker.

### `evaluate_ensemble`
Multi-iteration ensemble for final evaluation. Each iteration applies:
- Feature shuffle
- Class permutation
- Even iterations: robust (squashing) scaler
- Every 4th pair of iterations: global pooling 16/8 (mean + max + min → 3×m channels)
- Overlap expansion (or group-pad for fpg8 checkpoints)
Returns the averaged probability matrix.

---

## Final evaluation

Post-training evaluation is in `04_evaluation/final_evaluation.py`.

```bash
# From project root, after training:
python 04_evaluation/final_evaluation.py <path/to/best.pt> --device cuda
```

See **[04_evaluation/README.md](../04_evaluation/README.md)** for all options.
