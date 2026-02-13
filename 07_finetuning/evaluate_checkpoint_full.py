#!/usr/bin/env python3
"""
Full evaluation of a finetuned TabPFN checkpoint vs baseline.

Produces:
  1. Per-dataset AUC and accuracy (baseline vs finetuned).
  2. Mean AUC / accuracy improvement.
  3. Wilcoxon signed-rank test (one-sided) for AUC and accuracy gains.
  4. Largest gain / largest loss.
  5. SOTA count: how many datasets is each model the best among all AEON benchmarks.

Usage:
    python evaluate_checkpoint_full.py --checkpoint path/to/checkpoint.pt [--device cpu]
"""

import sys
import pickle
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = Path(__file__).parent
REAL_DATA_PKL = SCRIPT_DIR.parent / "01_real_data" / "AEON" / "data" / "classification_datasets.pkl"
BENCHMARKS_DIR = SCRIPT_DIR.parent / "01_real_data" / "AEON" / "benchmarks"


# ============================================================================
# Data loading
# ============================================================================

def load_real_datasets(max_samples=1000, max_flat_features=500, max_classes=10):
    real_data_dir = SCRIPT_DIR.parent / '01_real_data'
    if str(real_data_dir) not in sys.path:
        sys.path.insert(0, str(real_data_dir))
    from src.data_loader import TimeSeriesDataset  # noqa

    with open(REAL_DATA_PKL, 'rb') as f:
        datasets_list = pickle.load(f)

    datasets = []
    for dataset in datasets_list:
        try:
            name = dataset.name
            X_train, y_train = dataset.X_train, dataset.y_train
            X_test, y_test = dataset.X_test, dataset.y_test
            if X_train is None or X_test is None:
                continue
            if X_train.ndim == 2:
                X_train = X_train[:, :, np.newaxis]
            if X_test.ndim == 2:
                X_test = X_test[:, :, np.newaxis]
            X_train = np.transpose(X_train, (0, 2, 1))
            X_test = np.transpose(X_test, (0, 2, 1))
            n_samples = X_train.shape[0] + X_test.shape[0]
            flat_features = X_train.shape[1] * X_train.shape[2]
            n_classes = len(np.unique(y_train))
            if n_samples > max_samples or flat_features > max_flat_features or n_classes > max_classes:
                continue
            X_tr = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
            X_te = X_test.reshape(X_test.shape[0], -1).astype(np.float32)
            le = LabelEncoder()
            le.fit(y_train)
            y_tr = le.transform(y_train).astype(np.int64)
            y_te = le.transform(y_test).astype(np.int64)
            if np.any(np.isnan(X_tr)):
                col_means = np.nan_to_num(np.nanmean(X_tr, axis=0), nan=0.0)
                for i in range(X_tr.shape[1]):
                    X_tr[:, i] = np.where(np.isnan(X_tr[:, i]), col_means[i], X_tr[:, i])
                    X_te[:, i] = np.where(np.isnan(X_te[:, i]), col_means[i], X_te[:, i])
            datasets.append({
                'name': name, 'X_train': X_tr, 'y_train': y_tr,
                'X_test': X_te, 'y_test': y_te, 'n_classes': n_classes,
                'n_features': flat_features,
            })
        except Exception:
            continue
    return datasets


def load_aeon_benchmarks():
    """Load AEON AUROC benchmark CSVs → {model_name: pd.Series(dataset→auc)}."""
    benchmarks = {}
    if not BENCHMARKS_DIR.exists():
        return benchmarks
    for fp in BENCHMARKS_DIR.glob("*_auroc.csv"):
        model = fp.stem.replace("_auroc", "")
        try:
            df = pd.read_csv(fp, index_col=0)
            benchmarks[model] = df.mean(axis=1)
        except Exception:
            continue
    return benchmarks


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model_state_dict, datasets, device, n_estimators=8, label="model"):
    from tabpfn import TabPFNClassifier
    results = []
    for i, data in enumerate(datasets):
        name = data['name']
        print(f"  [{i+1:2d}/{len(datasets)}] {name:35s}", end=' ', flush=True)
        try:
            clf = TabPFNClassifier(device=device, n_estimators=n_estimators,
                                   ignore_pretraining_limits=True)
            clf.fit(data['X_train'], data['y_train'])
            if model_state_dict is not None:
                clf.model_.load_state_dict(model_state_dict)
            clf.model_.eval()
            with torch.no_grad():
                proba = clf.predict_proba(data['X_test'])
                preds = proba.argmax(axis=1)
            acc = accuracy_score(data['y_test'], preds)
            try:
                auc = (roc_auc_score(data['y_test'], proba[:, 1])
                       if data['n_classes'] == 2
                       else roc_auc_score(data['y_test'], proba, multi_class='ovr'))
            except Exception:
                auc = None
            results.append({'name': name, 'accuracy': acc, 'auc': auc})
            auc_s = f"AUC={auc:.4f}" if auc is not None else ""
            print(f"Acc={acc:.4f} {auc_s}")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({'name': name, 'accuracy': None, 'auc': None})
    return results


# ============================================================================
# Analysis
# ============================================================================

def analyze(baseline_res, finetuned_res, aeon_benchmarks):
    b_df = pd.DataFrame(baseline_res).rename(columns={'accuracy': 'base_acc', 'auc': 'base_auc'})
    f_df = pd.DataFrame(finetuned_res).rename(columns={'accuracy': 'ft_acc', 'auc': 'ft_auc'})
    df = b_df.merge(f_df, on='name')

    # ── AUC analysis ──────────────────────────────────────────────────────
    v = df.dropna(subset=['base_auc', 'ft_auc']).copy()
    v['auc_diff'] = v['ft_auc'] - v['base_auc']
    v['acc_diff'] = v['ft_acc'] - v['base_acc']

    n = len(v)
    mean_base_auc = v['base_auc'].mean()
    mean_ft_auc = v['ft_auc'].mean()
    mean_base_acc = v['base_acc'].mean()
    mean_ft_acc = v['ft_acc'].mean()

    # Wilcoxon signed-rank (one-sided: alternative='greater')
    try:
        stat_auc, p_auc = wilcoxon(v['auc_diff'], alternative='greater')
    except Exception:
        stat_auc, p_auc = None, None
    try:
        stat_acc, p_acc = wilcoxon(v['acc_diff'], alternative='greater')
    except Exception:
        stat_acc, p_acc = None, None

    max_gain_auc = v.loc[v['auc_diff'].idxmax()]
    max_loss_auc = v.loc[v['auc_diff'].idxmin()]
    max_gain_acc = v.loc[v['acc_diff'].idxmax()]
    max_loss_acc = v.loc[v['acc_diff'].idxmin()]

    # ── SOTA analysis ─────────────────────────────────────────────────────
    sota_base, sota_ft, n_bench_datasets = _sota_analysis(v, aeon_benchmarks)

    # ── Print report ──────────────────────────────────────────────────────
    sep = "=" * 80
    print(f"\n{sep}")
    print("EVALUATION REPORT")
    print(f"{sep}\n")
    print(f"Datasets evaluated: {n}")

    print(f"\n--- AUC ---")
    print(f"  Mean AUC baseline:    {mean_base_auc:.4f}")
    print(f"  Mean AUC finetuned:   {mean_ft_auc:.4f}")
    print(f"  Mean AUC improvement: {mean_ft_auc - mean_base_auc:+.4f}")
    if p_auc is not None:
        print(f"  Wilcoxon signed-rank (one-sided, H1: ft > base): p = {p_auc:.6f}")
    print(f"  Largest gain:  {max_gain_auc['name']} ({max_gain_auc['auc_diff']:+.4f})")
    print(f"  Largest loss:  {max_loss_auc['name']} ({max_loss_auc['auc_diff']:+.4f})")

    print(f"\n--- Accuracy ---")
    print(f"  Mean Acc baseline:    {mean_base_acc:.4f}")
    print(f"  Mean Acc finetuned:   {mean_ft_acc:.4f}")
    print(f"  Mean Acc improvement: {mean_ft_acc - mean_base_acc:+.4f}")
    if p_acc is not None:
        print(f"  Wilcoxon signed-rank (one-sided, H1: ft > base): p = {p_acc:.6f}")
    print(f"  Largest gain:  {max_gain_acc['name']} ({max_gain_acc['acc_diff']*100:+.2f}%)")
    print(f"  Largest loss:  {max_loss_acc['name']} ({max_loss_acc['acc_diff']*100:+.2f}%)")

    if n_bench_datasets > 0:
        print(f"\n--- SOTA (best AUC among all AEON benchmarks) ---")
        print(f"  Benchmark datasets with AEON overlap: {n_bench_datasets}")
        print(f"  Baseline SOTA in:  {sota_base}/{n_bench_datasets} datasets")
        print(f"  Finetuned SOTA in: {sota_ft}/{n_bench_datasets} datasets")

    # ── Per-dataset table ─────────────────────────────────────────────────
    print(f"\n{'-'*85}")
    print(f"{'Dataset':<35} {'Base AUC':>9} {'FT AUC':>9} {'ΔAUC':>8} {'Base Acc':>9} {'FT Acc':>9} {'ΔAcc':>8}")
    print(f"{'-'*85}")
    for _, r in v.sort_values('auc_diff', ascending=False).iterrows():
        m = "↑" if r['auc_diff'] > 0.001 else ("↓" if r['auc_diff'] < -0.001 else "=")
        print(f"{m} {r['name']:<33} {r['base_auc']:>9.4f} {r['ft_auc']:>9.4f} {r['auc_diff']:>+8.4f} "
              f"{r['base_acc']:>9.4f} {r['ft_acc']:>9.4f} {r['acc_diff']:>+8.4f}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    out = SCRIPT_DIR / "eval_checkpoint_results.csv"
    v.to_csv(out, index=False)
    print(f"\nResults saved to {out}")

    return v


def _sota_analysis(v, aeon_benchmarks):
    """Count on how many datasets baseline/finetuned is SOTA among AEON models."""
    if not aeon_benchmarks:
        return 0, 0, 0

    # Build full results table
    all_results = {}
    for model, scores in aeon_benchmarks.items():
        all_results[model] = scores
    aeon_df = pd.DataFrame(all_results)

    # Add our two models
    tabpfn_idx = v.set_index('name')
    common = sorted(set(tabpfn_idx.index) & set(aeon_df.index))
    if not common:
        return 0, 0, 0

    aeon_sub = aeon_df.loc[common].copy()
    aeon_sub['TabPFN_baseline'] = tabpfn_idx.loc[common, 'base_auc']
    aeon_sub['TabPFN_finetuned'] = tabpfn_idx.loc[common, 'ft_auc']

    sota_base = 0
    sota_ft = 0
    for ds in common:
        row = aeon_sub.loc[ds].dropna()
        best = row.max()
        if tabpfn_idx.loc[ds, 'base_auc'] >= best:
            sota_base += 1
        if tabpfn_idx.loc[ds, 'ft_auc'] >= best:
            sota_ft += 1

    return sota_base, sota_ft, len(common)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Full evaluation of finetuned checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n-estimators', type=int, default=8)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    print(f"  Step: {ckpt.get('step', '?')}")

    datasets = load_real_datasets()
    print(f"  {len(datasets)} datasets loaded\n")

    # Load AEON benchmarks for SOTA comparison
    print("Loading AEON benchmarks...")
    aeon = load_aeon_benchmarks()
    print(f"  {len(aeon)} benchmark models\n")

    # Evaluate baseline
    print("=" * 60)
    print("BASELINE (pretrained TabPFN)")
    print("=" * 60)
    baseline = evaluate_model(None, datasets, args.device, args.n_estimators, "baseline")

    # Evaluate finetuned
    print("\n" + "=" * 60)
    print(f"FINETUNED (step {ckpt.get('step', '?')})")
    print("=" * 60)
    finetuned = evaluate_model(ckpt['model_state_dict'], datasets, args.device,
                               args.n_estimators, "finetuned")

    # Analysis
    analyze(baseline, finetuned, aeon)


if __name__ == "__main__":
    main()
