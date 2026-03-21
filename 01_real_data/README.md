# 01 — Real-data evaluation

## Motivation

We evaluate our adapted PFN model on real-world time series classification (TSC)
benchmarks and compare it against TabPFN used as a flat-feature classifier on
flattened time series
([Hollmann et al., 2025](https://www.nature.com/articles/s41586-024-08328-6);
[PriorLabs/TabPFN](https://github.com/PriorLabs/tabpfn)).

As SOTA reference we use results from the HIVE-COTE 2.0 (HC2) paper
([Middlehurst et al., 2021](https://doi.org/10.1007/s10994-021-06057-9)),
downloaded from
[timeseriesclassification.com/HC2.php](https://www.timeseriesclassification.com/HC2.php).

---

## Data sources

| Archive | Datasets | Source | API |
|---------|----------|--------|-----|
| **UCR** (univariate TSC) | 128 total, 112 standard | [UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) | [`aeon`](https://github.com/aeon-toolkit/aeon) `load_classification` |
| **UEA** (multivariate TSC) | 30 total, 26 standard | [UEA Multivariate Archive](http://www.timeseriesclassification.com/) | [`aeon`](https://github.com/aeon-toolkit/aeon) `load_classification` |

The **standard** subsets (112 UCR, 26 UEA) are the equal-length, no-missing-value
problems used in the bake-off papers. The full archives add 16 UCR datasets with
variable length (11) or missing values (4), and 4 UEA variable-length datasets.

All datasets are downloaded with `download.py` via the `aeon` API.
Variable-length series are right-padded with NaN; missing values are kept as NaN
(TabPFN handles them internally through its `NanHandlingEncoderStep`).

---

## PFN filters and subsampling

TabPFN operates on 2D tabular data, so each time series `(n, m, t)` is flattened
to `(n, m×t)` features. We impose **PFN filter** limits to keep the
flattened feature count within the model's capacity:

| Limit | Value |
|-------|-------|
| `m × t` (total features) | ≤ 2 000 |
| Classes | ≤ 10 |

Datasets that do **not** pass these limits are excluded from TabPFN evaluation.
Full lists are in `datasets_summary.csv`.

### Train subsampling

Datasets with **≥ 1 000 training samples** (`subsample_train=True`) are
subsampled to exactly **1 000** training examples before any evaluation. The
subsample uses a **fixed seed (0)** so all benchmarks and the final evaluation
operate on the same subset, ensuring fair comparisons.

### Resulting counts

| Subset | UCR | UEA |
|--------|-----|-----|
| Total datasets | 128 | 30 |
| Pass PFN filter | **101** | **11** |
|  — of which subsampled to 1 000 train | 8 | 2 |
| Standard datasets | 112 | 26 |
| Standard ∩ PFN filter | **90** | **9** |

---

## Experiments

### Experiment 1 — TabPFN baseline (101 UCR + 11 UEA)

All 101 UCR and 11 UEA datasets that pass the PFN filter are benchmarked with
standard TabPFN v2.5 in two configurations:

| Label | Ensemble | Description |
|-------|----------|-------------|
| **e1** | 1 estimator | TabPFN default: SVD + squashing scaler on the first ensemble member |
| **e8** | 8 estimators | TabPFN default: distributes SVD and other preprocessors across 8 members |

Each dataset is run **30 times** with different random seeds and results are
averaged. Metrics: **accuracy** and **AUROC**.
Datasets with `subsample_train=True` are subsampled to 1 000 training examples
(fixed seed 0) for all runs.

Results are saved to `benchmark_results/` and automatically loaded by
`04_evaluation/final_evaluation.py` as baselines for the final comparison.

### Experiment 2 — Standard benchmark comparison with SOTA (90 UCR + 9 UEA)

The HC2 state-of-the-art results are only available for the **standard** subsets
(112 UCR, 26 UEA): problems with equal-length series and no missing values. Of
those, **90 UCR** and **9 UEA** also pass the PFN filter. We use this
intersection to compare our adapted PFN model against both TabPFN and HC2.

---

## Folder contents

### `download.py`

Downloads all **128 UCR** and **30 UEA** datasets via the `aeon` API. Saves
each dataset as `data/{ucr,uea}/<Name>_{train,test}.npz` (arrays `X`, `y`).
Generates `datasets_summary.csv` with one row per dataset:

| Column | Description |
|--------|-------------|
| `dataset` | Dataset name |
| `collection` | `UCR` or `UEA` |
| `train_shape`, `test_shape` | e.g. `100×1×500` |
| `n_classes` | Number of target classes |
| `is_variable_length` | Series were padded to equal length |
| `has_missings` | Contains original missing values |
| `is_standard` | In the 112 UCR or 26 UEA bake-off set |
| `passes_pfn_filters` | Passes PFN limits (`m×t ≤ 2000`, `labels ≤ 10`) |
| `subsample_train` | Train will be subsampled to 1 000 (n_train ≥ 1 000) |

```bash
python 01_real_data/download.py
```

### `download_hc2_benchmarks.py`

Downloads accuracy tables from the
[HC2 page](https://www.timeseriesclassification.com/HC2.php) for the 112 UCR
and 26 UEA standard problems, averaged over 30 resamples. Output goes to
`benchmarks_hc2/`.

```bash
python 01_real_data/download_hc2_benchmarks.py
```

### `benchmark_tabpfn.py`

Pre-computes the **e1** and **e8** TabPFN baselines on all PFN-eligible datasets.
Datasets with `subsample_train=True` are subsampled to 1 000 training examples
using a fixed seed (0). Each configuration × dataset is run 30 times and
averaged. Outputs:

- `benchmark_results/ucr_benchmark_tabpfn.csv`
- `benchmark_results/uea_benchmark_tabpfn.csv`

Columns: `dataset`, `collection`, `e1_acc_mean`, `e1_acc_std`, `e1_auc_mean`,
`e1_auc_std`, `e8_acc_mean`, `e8_acc_std`, `e8_auc_mean`, `e8_auc_std`, `n_runs`.

These CSVs are automatically loaded by `04_evaluation/final_evaluation.py` so
that only the finetuned model needs to run live during evaluation.

**Locally** (slower, uses `auto` device — MPS on Apple Silicon):
```bash
python 01_real_data/benchmark_tabpfn.py [--ucr-only] [--uea-only] [--n-runs 30] [--device auto]
```

**On the cluster** (recommended — single GPU, UCR and UEA as separate steps):
```bash
sbatch benchmark_tabpfn.sbatch
```

---

## References

- **UCR Archive:** Dau, H. A. et al. *The UCR time series archive.* IEEE/CAA J. Autom. Sinica 6(6), 2019.
- **UEA Archive:** Bagnall, A. et al. *The UEA multivariate time series classification archive, 2018.* arXiv:1811.00075.
- **HIVE-COTE 2.0:** Middlehurst, M., Large, J., Flynn, M. et al. *HIVE-COTE 2.0: a new meta ensemble for time series classification.* Mach Learn 110, 3211–3243 (2021). https://doi.org/10.1007/s10994-021-06057-9
- **TabPFN v2.5:** Hollmann, N. et al. *Accurate Predictions on Small Data with a Tabular Foundation Model.* arXiv:2501.02593 (2025). https://github.com/PriorLabs/tabpfn
- **aeon toolkit:** https://github.com/aeon-toolkit/aeon
