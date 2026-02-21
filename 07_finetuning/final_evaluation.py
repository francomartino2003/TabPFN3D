#!/usr/bin/env python3
"""
Final Evaluation Script — Three Tests

Test 1: TabPFN standard (B) vs V3 finetuned ensemble (D) on 55 datasets
        - Per-dataset ACC and AUC
        - Diff stats: mean, min, max
        - One-sided Wilcoxon Signed-Rank Test (D > B)
        - Scatter plots

Test 2: V3 finetuned ensemble (D) vs AEON benchmarks (ACC + AUC)
        - Rank 1 count, strict rank 1 count, mean position
        - Per-dataset breakdown

Test 3: Finetuning log analysis (first 50 steps of job 9272951)
        - Parse SynthEval Loss / AUC and real Acc / AUC from log
        - Spearman correlation between synth-eval and real-eval curves
        - Combined plot with dual axis

Usage:
    python final_evaluation.py checkpoints_v3/v3_400s/checkpoint_best_synth_loss.pt \
        --log checkpoints_v3/v3_400s/finetune_v3_9272951.out
"""

import sys
import argparse
import pickle
import time
import re
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as sp_stats
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ── Paths ──
sys.path.insert(0, str(Path(__file__).parent.parent / '11_final_generator'))
sys.path.insert(0, str(Path(__file__).parent.parent / '00_TabPFN' / 'src'))

from tabpfn import TabPFNClassifier
from tabpfn_temporal import build_temporal_tabpfn_fpg8, set_temporal_info, pad_to_group

# ═══════════════════════════════════════════════════════════════════════════════
# Constants (same as evaluate_v3_with_ensemble.py)
# ═══════════════════════════════════════════════════════════════════════════════

MAX_SAMPLES = 1000
MAX_T = 1024
MAX_M = 10
MAX_M_TIMES_T = 1200
MAX_CLASSES = 10
GROUP_SIZE = 8
N_ENSEMBLE_ITERS = 4
ENSEMBLE_SEED = 42
SOFTMAX_TEMPERATURE = 0.9

BENCHMARKS_DIR = Path(__file__).parent.parent / "01_real_data" / "AEON" / "benchmarks"
REAL_DATA_PKL = (Path(__file__).parent.parent / "01_real_data" / "AEON"
                 / "data" / "classification_datasets.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers (copied from evaluate_v3_with_ensemble.py to keep self-contained)
# ═══════════════════════════════════════════════════════════════════════════════

def _soft_clip(x, B=3.0):
    return x / np.sqrt(1.0 + (x / B) ** 2)


def temporal_squashing_scaler(X_train_flat, X_test_flat, m, T,
                              max_abs=3.0, q_low=25.0, q_high=75.0):
    X_tr = X_train_flat.copy()
    X_te = X_test_flat.copy()
    for j in range(m):
        c0, c1 = j * T, j * T + T
        vals = X_tr[:, c0:c1].ravel()
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            X_tr[:, c0:c1] = 0.0; X_te[:, c0:c1] = 0.0; continue
        median = np.median(finite)
        q_lo, q_hi = np.percentile(finite, q_low), np.percentile(finite, q_high)
        if q_hi != q_lo:
            scale = 1.0 / (q_hi - q_lo)
        else:
            vmin, vmax = np.min(finite), np.max(finite)
            if vmax != vmin:
                scale = 2.0 / (vmax - vmin)
            else:
                X_tr[:, c0:c1] = 0.0; X_te[:, c0:c1] = 0.0; continue
        X_tr[:, c0:c1] = _soft_clip((X_tr[:, c0:c1] - median) * scale, max_abs)
        X_te[:, c0:c1] = _soft_clip((X_te[:, c0:c1] - median) * scale, max_abs)
    return X_tr.astype(np.float32), X_te.astype(np.float32)


def shuffle_features(X_flat, m, T, perm):
    n = X_flat.shape[0]
    return X_flat.reshape(n, m, T)[:, perm, :].reshape(n, m * T)


def load_real_datasets():
    real_data_dir = Path(__file__).resolve().parent.parent / '01_real_data'
    if str(real_data_dir) not in sys.path:
        sys.path.insert(0, str(real_data_dir))
    from src.data_loader import TimeSeriesDataset  # noqa: F401

    if not REAL_DATA_PKL.exists():
        print(f"ERROR: Data not found at {REAL_DATA_PKL}"); return []
    with open(REAL_DATA_PKL, 'rb') as f:
        datasets_list = pickle.load(f)

    valid = []
    for ds in datasets_list:
        try:
            X_train, X_test = ds.X_train, ds.X_test
            if X_train is None or X_test is None: continue
            if X_train.ndim == 2: X_train = X_train[:, np.newaxis, :]
            if X_test.ndim == 2: X_test = X_test[:, np.newaxis, :]
            n = X_train.shape[0] + X_test.shape[0]
            m, T = X_train.shape[1], X_train.shape[2]
            if n > MAX_SAMPLES: continue
            if T > MAX_T or m > MAX_M or m * T > MAX_M_TIMES_T: continue
            n_classes = len(np.unique(ds.y_train))
            if n_classes > MAX_CLASSES or n_classes < 2: continue

            X_tr = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
            X_te = X_test.reshape(X_test.shape[0], -1).astype(np.float32)
            if np.any(np.isnan(X_tr)):
                col_means = np.nan_to_num(np.nanmean(X_tr, axis=0), nan=0.0)
                for i in range(X_tr.shape[1]):
                    X_tr[:, i] = np.where(np.isnan(X_tr[:, i]), col_means[i], X_tr[:, i])
                    X_te[:, i] = np.where(np.isnan(X_te[:, i]), col_means[i], X_te[:, i])

            le = LabelEncoder(); le.fit(ds.y_train)
            valid.append({
                'name': ds.name,
                'X_train': X_tr, 'X_test': X_te,
                'y_train': le.transform(ds.y_train).astype(np.int64),
                'y_test': le.transform(ds.y_test).astype(np.int64),
                'n_classes': n_classes, 'n_features': flat, 'm': m, 'T': T,
            })
        except Exception:
            continue
    return valid


def compute_metrics(y_true, proba, n_classes):
    preds = proba.argmax(axis=1)
    acc = accuracy_score(y_true, preds)
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, proba[:, 1])
        else:
            auc = roc_auc_score(y_true, proba, multi_class='ovr')
    except Exception:
        auc = None
    return acc, auc


# ═══════════════════════════════════════════════════════════════════════════════
# Model B: TabPFN standard (built-in preprocessing + ensemble)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_vanilla_standard(X_train, y_train, X_test, n_classes, device,
                              n_estimators=N_ENSEMBLE_ITERS):
    clf = TabPFNClassifier(
        device=device, n_estimators=n_estimators,
        ignore_pretraining_limits=True,
    )
    clf.fit(X_train, y_train)
    with torch.no_grad():
        proba = clf.predict_proba(X_test)
    del clf
    return proba


# ═══════════════════════════════════════════════════════════════════════════════
# Model D: V3 finetuned — manual ensemble
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_ensemble(model, X_train, y_train, X_test, n_classes, m, T, device,
                      n_iters=N_ENSEMBLE_ITERS, seed=ENSEMBLE_SEED):
    rng = np.random.RandomState(seed)
    proba_sum = np.zeros((X_test.shape[0], n_classes), dtype=np.float64)
    n_valid = 0

    for it in range(n_iters):
        try:
            feat_perm = rng.permutation(m)
            X_tr_perm = shuffle_features(X_train, m, T, feat_perm)
            X_te_perm = shuffle_features(X_test, m, T, feat_perm)

            class_perm = rng.permutation(n_classes)
            y_train_perm = class_perm[y_train]

            if it % 2 == 0:
                X_tr_proc, X_te_proc = temporal_squashing_scaler(
                    X_tr_perm, X_te_perm, m, T)
            else:
                X_tr_proc, X_te_proc = X_tr_perm.copy(), X_te_perm.copy()

            X_tr_padded, T_padded = pad_to_group(X_tr_proc, m, T, group_size=GROUP_SIZE)
            X_te_padded, _ = pad_to_group(X_te_proc, m, T, group_size=GROUP_SIZE)
            set_temporal_info(model, m, T_padded, group_size=GROUP_SIZE)

            X_tr_t = torch.as_tensor(X_tr_padded, dtype=torch.float32, device=device)
            y_tr_t = torch.as_tensor(y_train_perm, dtype=torch.float32, device=device)
            X_te_t = torch.as_tensor(X_te_padded, dtype=torch.float32, device=device)

            X_full = torch.cat([X_tr_t, X_te_t], dim=0).unsqueeze(1)
            y_in = y_tr_t.unsqueeze(1)

            output = model(X_full, y_in, only_return_standard_out=True,
                           categorical_inds=[[]])
            logits = output.squeeze(1) if output.ndim == 3 else output
            logits = logits[:, :n_classes]
            logits = logits[:, class_perm]
            proba = torch.softmax(logits / SOFTMAX_TEMPERATURE, dim=-1).cpu().numpy()
            proba_sum += proba
            n_valid += 1
            del X_tr_t, y_tr_t, X_te_t, X_full, y_in, output, logits, proba
        except Exception as e:
            print(f"    [Ensemble iter {it}] FAILED: {type(e).__name__}: {str(e)[:100]}")
            continue

    return proba_sum / n_valid if n_valid > 0 else None


# ═══════════════════════════════════════════════════════════════════════════════
# AEON benchmark loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_aeon_benchmarks_auroc():
    benchmarks = {}
    for fp in BENCHMARKS_DIR.glob("*_auroc.csv"):
        model_name = fp.stem.replace("_auroc", "")
        try:
            df = pd.read_csv(fp, index_col=0)
            benchmarks[model_name] = df.mean(axis=1)
        except Exception:
            continue
    return benchmarks


def load_aeon_benchmarks_acc():
    benchmarks = {}
    for fp in BENCHMARKS_DIR.glob("*_TESTFOLDS.csv"):
        model_name = fp.stem.replace("_TESTFOLDS", "")
        try:
            df = pd.read_csv(fp, index_col=0)
            benchmarks[model_name] = df.mean(axis=1)
        except Exception:
            continue
    return benchmarks


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: TabPFN standard (B) vs V3 finetuned ensemble (D)
# ═══════════════════════════════════════════════════════════════════════════════

def run_test1(datasets, ft_model, device, output_dir):
    print("\n" + "=" * 80)
    print("TEST 1: TabPFN standard (B) vs V3 finetuned ensemble (D)")
    print("=" * 80)

    results = []
    t0 = time.time()

    for i, ds in enumerate(datasets):
        name, m, T, nc = ds['name'], ds['m'], ds['T'], ds['n_classes']
        print(f"  [{i+1}/{len(datasets)}] {name} (m={m}, T={T}, cls={nc})", flush=True)

        res = {'name': name, 'm': m, 'T': T, 'n_classes': nc}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # B: TabPFN standard
        try:
            proba_b = evaluate_vanilla_standard(
                ds['X_train'], ds['y_train'], ds['X_test'], nc, device)
            acc_b, auc_b = compute_metrics(ds['y_test'], proba_b, nc)
            res['B_acc'], res['B_auc'] = acc_b, auc_b
            del proba_b
        except Exception as e:
            print(f"    [B] FAILED: {type(e).__name__}: {str(e)[:80]}")
            res['B_acc'], res['B_auc'] = None, None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # D: V3 finetuned ensemble
        try:
            with torch.no_grad():
                proba_d = evaluate_ensemble(
                    ft_model, ds['X_train'], ds['y_train'], ds['X_test'],
                    nc, m, T, device)
            if proba_d is not None:
                acc_d, auc_d = compute_metrics(ds['y_test'], proba_d, nc)
                res['D_acc'], res['D_auc'] = acc_d, auc_d
            else:
                res['D_acc'], res['D_auc'] = None, None
            del proba_d
        except Exception as e:
            print(f"    [D] FAILED: {type(e).__name__}: {str(e)[:80]}")
            res['D_acc'], res['D_auc'] = None, None

        def _f(v): return f"{v:.4f}" if v is not None else "FAIL"
        print(f"    B_auc={_f(res['B_auc'])} D_auc={_f(res['D_auc'])}  "
              f"B_acc={_f(res['B_acc'])} D_acc={_f(res['D_acc'])}")
        results.append(res)

    elapsed = time.time() - t0
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "test1_B_vs_D.csv", index=False)

    # ── Statistics ──
    valid = df.dropna(subset=['B_acc', 'D_acc', 'B_auc', 'D_auc'])
    n = len(valid)
    print(f"\n  Valid datasets: {n}/{len(datasets)}  ({elapsed:.0f}s)")

    for metric in ['acc', 'auc']:
        b_col, d_col = f'B_{metric}', f'D_{metric}'
        diffs = valid[d_col].values - valid[b_col].values
        mean_d, min_d, max_d = diffs.mean(), diffs.min(), diffs.max()
        median_d = np.median(diffs)

        # One-sided Wilcoxon: D > B  (alternative='greater')
        stat, pval = sp_stats.wilcoxon(valid[d_col].values, valid[b_col].values,
                                       alternative='greater')

        wins = (diffs > 0.001).sum()
        ties = ((diffs >= -0.001) & (diffs <= 0.001)).sum()
        losses = (diffs < -0.001).sum()

        label = metric.upper()
        print(f"\n  {label} differences (D - B):")
        print(f"    Mean: {mean_d:+.4f}   Median: {median_d:+.4f}")
        print(f"    Min:  {min_d:+.4f}   Max:    {max_d:+.4f}")
        print(f"    Win/Tie/Loss: {wins}/{ties}/{losses}")
        print(f"    Wilcoxon signed-rank (D > B): stat={stat:.1f}, p={pval:.6f}"
              f"  {'***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'}")

    # ── Plots ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, title in zip(axes, ['auc', 'acc'], ['AUC', 'Accuracy']):
        b_col, d_col = f'B_{metric}', f'D_{metric}'
        v = df.dropna(subset=[b_col, d_col])
        b_vals, d_vals = v[b_col].values, v[d_col].values
        diffs = d_vals - b_vals

        # Wilcoxon for this subset
        stat, pval = sp_stats.wilcoxon(d_vals, b_vals, alternative='greater')

        lo = min(b_vals.min(), d_vals.min()) - 0.02
        hi = max(b_vals.max(), d_vals.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, lw=1)
        colors = ['#2ecc71' if d > 0.001 else '#e74c3c' if d < -0.001 else '#95a5a6'
                  for d in diffs]
        ax.scatter(b_vals, d_vals, c=colors, s=40, alpha=0.7, edgecolors='white', linewidth=0.5)
        ax.set_xlabel(f'TabPFN standard (B) — {title}', fontsize=11)
        ax.set_ylabel(f'V3 finetuned ensemble (D) — {title}', fontsize=11)
        ax.set_title(f'{title}: D vs B  (mean Δ={diffs.mean():+.4f}, p={pval:.4f})',
                     fontsize=12, fontweight='bold')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        wins = (diffs > 0.001).sum()
        losses = (diffs < -0.001).sum()
        ax.text(0.05, 0.95, f'D wins: {wins}\nB wins: {losses}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_dir / "test1_B_vs_D_scatter.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: test1_B_vs_D.csv, test1_B_vs_D_scatter.png")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: V3 finetuned ensemble (D) vs AEON benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

def run_test2(test1_df, output_dir):
    print("\n\n" + "=" * 80)
    print("TEST 2: V3 finetuned ensemble (D) vs AEON benchmarks")
    print("=" * 80)

    our_name = "TabPFN_V3_ensemble"

    for metric_label, loader, our_col in [
        ("AUC (AUROC)", load_aeon_benchmarks_auroc, 'D_auc'),
        ("Accuracy", load_aeon_benchmarks_acc, 'D_acc'),
    ]:
        print(f"\n  ── {metric_label} ──")
        benchmarks = loader()
        if not benchmarks:
            print("    No benchmarks found, skipping."); continue

        aeon_df = pd.DataFrame(benchmarks)
        our_indexed = test1_df.set_index('name')

        common = sorted(set(aeon_df.index) & set(our_indexed.index))
        common = [d for d in common if pd.notna(our_indexed.loc[d, our_col])]
        if not common:
            print("    No common datasets, skipping."); continue

        aeon_common = aeon_df.loc[common].copy()
        aeon_common[our_name] = our_indexed.loc[common, our_col].values

        # Drop columns (models) that are all NaN for common datasets
        aeon_common = aeon_common.dropna(axis=1, how='all')

        n_models = len(aeon_common.columns)
        n_ds = len(common)

        rankings = aeon_common.rank(axis=1, ascending=False, method='min')

        all_stats = []
        for model_name in aeon_common.columns:
            model_ranks = rankings[model_name].dropna()
            model_vals = aeon_common[model_name].dropna()
            mean_rank = model_ranks.mean() if len(model_ranks) > 0 else float('inf')
            rank1 = int((model_ranks == 1).sum())
            rank1_strict = 0
            for dataset in model_ranks.index:
                if model_ranks[dataset] == 1 and (rankings.loc[dataset] == 1).sum() == 1:
                    rank1_strict += 1
            mean_val = model_vals.mean() if len(model_vals) > 0 else None
            all_stats.append({
                'model': model_name, 'mean_rank': mean_rank,
                'rank1': rank1, 'rank1_strict': rank1_strict,
                'mean_value': mean_val, 'n_datasets': int(model_vals.notna().sum()),
            })

        stats_df = pd.DataFrame(all_stats).sort_values('mean_rank')
        suffix = metric_label.split()[0].lower()
        stats_df.to_csv(output_dir / f"test2_aeon_rankings_{suffix}.csv", index=False)

        # Print ranking table
        print(f"\n  {'Model':<28} {'Mean Rank':>10} {'Rank 1':>7} {'Strict':>7} "
              f"{'Mean Val':>10} {'N':>4}")
        print(f"  {'-'*70}")

        for _, row in stats_df.iterrows():
            is_ours = row['model'] == our_name
            marker = ">>>" if is_ours else "   "
            v_str = f"{row['mean_value']:.4f}" if row['mean_value'] is not None else "N/A"
            print(f"  {marker} {row['model']:<25} {row['mean_rank']:>10.2f} "
                  f"{row['rank1']:>7} {row['rank1_strict']:>7} "
                  f"{v_str:>10} {row['n_datasets']:>4}")

        # Our summary
        our_row = stats_df[stats_df['model'] == our_name]
        if len(our_row) > 0:
            our_row = our_row.iloc[0]
            pos = int((stats_df['mean_rank'] < our_row['mean_rank']).sum()) + 1
            print(f"\n  >>> {our_name} — {metric_label}:")
            print(f"      Position: {pos}/{n_models}  (mean rank {our_row['mean_rank']:.2f})")
            print(f"      Rank 1: {our_row['rank1']}/{n_ds}  "
                  f"(strict: {our_row['rank1_strict']}/{n_ds})")
            v_str = f"{our_row['mean_value']:.4f}" if our_row['mean_value'] is not None else "N/A"
            print(f"      Mean {metric_label}: {v_str}")

        # Per-dataset breakdown
        our_ranks = rankings[our_name] if our_name in rankings.columns else None
        if our_ranks is not None:
            print(f"\n  Per-dataset breakdown ({metric_label}):")
            print(f"  {'Dataset':<30} {'Our Value':>10} {'Rank':>6} "
                  f"{'Best AEON':>10} {'Best Model':<20}")
            print(f"  {'-'*80}")
            for ds_name in common:
                our_val = aeon_common.loc[ds_name, our_name]
                our_rank = int(our_ranks[ds_name])
                aeon_only = aeon_common.loc[ds_name].drop(our_name)
                valid_aeon = aeon_only.dropna()
                if len(valid_aeon) > 0:
                    best_val = valid_aeon.max()
                    best_model = valid_aeon.idxmax()
                else:
                    best_val, best_model = None, "?"
                ov = f"{our_val:.4f}" if pd.notna(our_val) else "N/A"
                bv = f"{best_val:.4f}" if best_val is not None else "N/A"
                print(f"  {ds_name:<30} {ov:>10} {our_rank:>6}/{n_models}  "
                      f"{bv:>10} {best_model:<20}")

    print(f"\n  Saved: test2_aeon_rankings_auc.csv, test2_aeon_rankings_accuracy.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Finetuning log analysis — synth eval vs real eval correlation
# ═══════════════════════════════════════════════════════════════════════════════

def parse_log(log_path, max_steps=50):
    """
    Parse the finetuning log to extract per-step metrics.

    Each training step has a line like:
      Step     5 | Loss ... | SynthEval L=0.8230 A=0.7504 | LR ...
    followed by an eval line:
      >> EVAL step 6: Acc=0.8148  AUC=0.9231  (55/55 OK)

    Returns DataFrame with columns: step, synth_loss, synth_auc, real_acc, real_auc
    """
    with open(log_path, 'r') as f:
        lines = f.readlines()

    step_pattern = re.compile(
        r'Step\s+(\d+)\s+\|.*SynthEval L=([\d.]+)\s+A=([\d.]+)')
    eval_pattern = re.compile(
        r'>> EVAL step (\d+):\s+Acc=([\d.]+)\s+AUC=([\d.]+)')

    synth_data = {}
    real_data = {}

    for line in lines:
        m = step_pattern.search(line)
        if m:
            step = int(m.group(1))
            if step < max_steps:
                synth_data[step] = {
                    'synth_loss': float(m.group(2)),
                    'synth_auc': float(m.group(3)),
                }

        m = eval_pattern.search(line)
        if m:
            eval_step = int(m.group(1))
            train_step = eval_step - 1
            if train_step < max_steps:
                real_data[train_step] = {
                    'real_acc': float(m.group(2)),
                    'real_auc': float(m.group(3)),
                }

    rows = []
    for step in sorted(set(synth_data.keys()) & set(real_data.keys())):
        row = {'step': step}
        row.update(synth_data[step])
        row.update(real_data[step])
        rows.append(row)

    return pd.DataFrame(rows)


def run_test3(log_path, output_dir, max_steps=50):
    print("\n\n" + "=" * 80)
    print(f"TEST 3: Finetuning log analysis (first {max_steps} steps)")
    print("=" * 80)

    df = parse_log(log_path, max_steps=max_steps)
    if len(df) == 0:
        print("  ERROR: Could not parse any steps from log."); return
    print(f"  Parsed {len(df)} steps from log")
    df.to_csv(output_dir / "test3_log_parsed.csv", index=False)

    # ── Correlation analysis ──
    # We test: when synth eval improves (loss goes down, AUC goes up),
    # does real eval also improve (acc goes up, AUC goes up)?
    #
    # Use first-differences (Δ from step to step) to test co-movement.
    # Also test levels (Spearman on raw values).

    print(f"\n  ── Spearman rank correlations (levels, n={len(df)}) ──")

    pairs = [
        ('synth_loss', 'real_acc',  'SynthEval Loss vs Real Acc (expect negative)'),
        ('synth_loss', 'real_auc',  'SynthEval Loss vs Real AUC (expect negative)'),
        ('synth_auc',  'real_acc',  'SynthEval AUC  vs Real Acc (expect positive)'),
        ('synth_auc',  'real_auc',  'SynthEval AUC  vs Real AUC (expect positive)'),
    ]

    for col_x, col_y, label in pairs:
        rho, pval = sp_stats.spearmanr(df[col_x].values, df[col_y].values)
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
        print(f"    {label}")
        print(f"      ρ={rho:+.4f}  p={pval:.6f}  {sig}")

    # First-differences correlation
    d_synth_loss = df['synth_loss'].diff().dropna().values
    d_synth_auc  = df['synth_auc'].diff().dropna().values
    d_real_acc   = df['real_acc'].diff().dropna().values
    d_real_auc   = df['real_auc'].diff().dropna().values

    print(f"\n  ── Spearman rank correlations (first-differences Δ, n={len(d_synth_loss)}) ──")

    diff_pairs = [
        (d_synth_loss, d_real_acc,  'ΔSynthLoss vs ΔRealAcc  (expect negative)'),
        (d_synth_loss, d_real_auc,  'ΔSynthLoss vs ΔRealAUC  (expect negative)'),
        (d_synth_auc,  d_real_acc,  'ΔSynthAUC  vs ΔRealAcc  (expect positive)'),
        (d_synth_auc,  d_real_auc,  'ΔSynthAUC  vs ΔRealAUC  (expect positive)'),
    ]

    for x, y, label in diff_pairs:
        rho, pval = sp_stats.spearmanr(x, y)
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
        print(f"    {label}")
        print(f"      ρ={rho:+.4f}  p={pval:.6f}  {sig}")

    # ── Smoothed trend correlation (5-step rolling mean) ──
    window = 5
    if len(df) >= window + 2:
        sm_synth_loss = df['synth_loss'].rolling(window, center=True).mean().dropna()
        sm_synth_auc  = df['synth_auc'].rolling(window, center=True).mean().dropna()
        sm_real_acc   = df['real_acc'].rolling(window, center=True).mean().dropna()
        sm_real_auc   = df['real_auc'].rolling(window, center=True).mean().dropna()

        idx = sm_synth_loss.index.intersection(sm_real_acc.index)

        print(f"\n  ── Spearman correlations (5-step rolling mean, n={len(idx)}) ──")
        smooth_pairs = [
            (sm_synth_loss.loc[idx], sm_real_acc.loc[idx],
             'Smooth SynthLoss vs Smooth RealAcc'),
            (sm_synth_loss.loc[idx], sm_real_auc.loc[idx],
             'Smooth SynthLoss vs Smooth RealAUC'),
            (sm_synth_auc.loc[idx], sm_real_acc.loc[idx],
             'Smooth SynthAUC  vs Smooth RealAcc'),
            (sm_synth_auc.loc[idx], sm_real_auc.loc[idx],
             'Smooth SynthAUC  vs Smooth RealAUC'),
        ]
        for x, y, label in smooth_pairs:
            rho, pval = sp_stats.spearmanr(x.values, y.values)
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
            print(f"    {label}")
            print(f"      ρ={rho:+.4f}  p={pval:.6f}  {sig}")

    # ═══════════════════════════════════════════════════════════════════════
    # Plot: SynthEval (Loss, AUC) and Real (Acc, AUC) over steps
    # ═══════════════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    steps = df['step'].values

    # --- Top panel: SynthEval Loss (inverted for visual) + Real AUC ---
    ax1 = axes[0]
    color_sl = '#e74c3c'
    color_sa = '#3498db'
    color_ra = '#2ecc71'
    color_rau = '#9b59b6'

    ax1.plot(steps, df['synth_loss'], color=color_sl, alpha=0.3, linewidth=1)
    ax1.plot(steps, df['synth_loss'].rolling(5, center=True).mean(),
             color=color_sl, linewidth=2.5, label='SynthEval Loss (5-step avg)')
    ax1.set_ylabel('SynthEval Loss', color=color_sl, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_sl)
    ax1.invert_yaxis()

    ax1b = ax1.twinx()
    ax1b.plot(steps, df['real_auc'], color=color_rau, alpha=0.3, linewidth=1)
    ax1b.plot(steps, df['real_auc'].rolling(5, center=True).mean(),
              color=color_rau, linewidth=2.5, label='Real AUC (5-step avg)')
    ax1b.set_ylabel('Real AUC (55 datasets)', color=color_rau, fontsize=12)
    ax1b.tick_params(axis='y', labelcolor=color_rau)

    lines1a = ax1.get_legend_handles_labels()
    lines1b = ax1b.get_legend_handles_labels()
    ax1.legend(lines1a[0] + lines1b[0], lines1a[1] + lines1b[1],
               loc='lower left', fontsize=10)
    ax1.set_title('SynthEval Loss ↓  ↔  Real AUC ↑  (co-movement)',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.2)

    # --- Bottom panel: SynthEval AUC + Real Acc + Real AUC ---
    ax2 = axes[1]
    ax2.plot(steps, df['synth_auc'], color=color_sa, alpha=0.3, linewidth=1)
    ax2.plot(steps, df['synth_auc'].rolling(5, center=True).mean(),
             color=color_sa, linewidth=2.5, label='SynthEval AUC (5-step avg)')
    ax2.set_ylabel('SynthEval AUC', color=color_sa, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_sa)

    ax2b = ax2.twinx()
    ax2b.plot(steps, df['real_acc'], color=color_ra, alpha=0.3, linewidth=1)
    ax2b.plot(steps, df['real_acc'].rolling(5, center=True).mean(),
              color=color_ra, linewidth=2.5, label='Real Acc (5-step avg)')
    ax2b.plot(steps, df['real_auc'], color=color_rau, alpha=0.3, linewidth=1)
    ax2b.plot(steps, df['real_auc'].rolling(5, center=True).mean(),
              color=color_rau, linewidth=2.5, linestyle='--',
              label='Real AUC (5-step avg)')
    ax2b.set_ylabel('Real Eval (55 datasets)', fontsize=12)

    lines2a = ax2.get_legend_handles_labels()
    lines2b = ax2b.get_legend_handles_labels()
    ax2.legend(lines2a[0] + lines2b[0], lines2a[1] + lines2b[1],
               loc='lower left', fontsize=10)
    ax2.set_title('SynthEval AUC  ↔  Real Acc & AUC  (co-movement)',
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_dir / "test3_synth_vs_real.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ── Scatter: smoothed synth loss vs smoothed real AUC ──
    if len(df) >= window + 2:
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

        sm_sl = df['synth_loss'].rolling(window, center=True).mean()
        sm_sa = df['synth_auc'].rolling(window, center=True).mean()
        sm_ra = df['real_acc'].rolling(window, center=True).mean()
        sm_rau = df['real_auc'].rolling(window, center=True).mean()
        valid_idx = sm_sl.dropna().index

        scatter_configs = [
            (sm_sl.loc[valid_idx], sm_rau.loc[valid_idx],
             'SynthEval Loss (smoothed)', 'Real AUC (smoothed)',
             'SynthLoss vs Real AUC'),
            (sm_sa.loc[valid_idx], sm_rau.loc[valid_idx],
             'SynthEval AUC (smoothed)', 'Real AUC (smoothed)',
             'SynthAUC vs Real AUC'),
            (sm_sa.loc[valid_idx], sm_ra.loc[valid_idx],
             'SynthEval AUC (smoothed)', 'Real Acc (smoothed)',
             'SynthAUC vs Real Acc'),
        ]

        for ax, (x, y, xlabel, ylabel, title) in zip(axes2, scatter_configs):
            rho, pval = sp_stats.spearmanr(x.values, y.values)
            c = df['step'].loc[valid_idx].values
            sc = ax.scatter(x, y, c=c, cmap='viridis', s=30, alpha=0.8,
                            edgecolors='white', linewidth=0.3)
            plt.colorbar(sc, ax=ax, label='Step')

            # Linear fit for visual
            z = np.polyfit(x.values, y.values, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.5, linewidth=1.5)

            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
            ax.set_title(f'{title}\nρ={rho:+.3f}, p={pval:.4f} {sig}',
                         fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        fig2.savefig(output_dir / "test3_scatter_correlations.png", dpi=200,
                     bbox_inches='tight')
        plt.close(fig2)

    print(f"\n  Saved: test3_log_parsed.csv, test3_synth_vs_real.png, "
          f"test3_scatter_correlations.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Final 3-test evaluation')
    parser.add_argument('checkpoint', type=str,
                        help='Path to checkpoint .pt file')
    parser.add_argument('--log', type=str, default=None,
                        help='Path to SLURM log (.out) for Test 3')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--n-ensemble', type=int, default=N_ENSEMBLE_ITERS)
    parser.add_argument('--max-log-steps', type=int, default=50,
                        help='Max steps to parse from log for Test 3')
    parser.add_argument('--skip-test1', action='store_true',
                        help='Skip Test 1 (load from CSV if exists)')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        checkpoint_path = Path(__file__).parent / args.checkpoint
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}"); return

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    output_dir = checkpoint_path.parent / "final_eval_results"
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("FINAL EVALUATION — 3 TESTS")
    print("=" * 80)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device:     {device}")
    print(f"  Output:     {output_dir}")

    # ── Load data ──
    print("\nLoading real datasets...")
    datasets = load_real_datasets()
    print(f"  {len(datasets)} datasets meet constraints")

    # ── Build model ──
    print("\nBuilding temporal TabPFN (fpg=8) + loading checkpoint...")
    ft_model, ft_clf, _ = build_temporal_tabpfn_fpg8(device=device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ft_model.load_state_dict(ckpt['model_state_dict'])
    ft_model.eval()
    print(f"  Step: {ckpt.get('step', '?')}, version: {ckpt.get('version', '?')}")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 1
    # ══════════════════════════════════════════════════════════════════════
    test1_csv = output_dir / "test1_B_vs_D.csv"
    if args.skip_test1 and test1_csv.exists():
        print(f"\n  Skipping Test 1 — loading from {test1_csv}")
        test1_df = pd.read_csv(test1_csv)
    else:
        test1_df = run_test1(datasets, ft_model, device, output_dir)

    # ══════════════════════════════════════════════════════════════════════
    # TEST 2
    # ══════════════════════════════════════════════════════════════════════
    run_test2(test1_df, output_dir)

    # ══════════════════════════════════════════════════════════════════════
    # TEST 3
    # ══════════════════════════════════════════════════════════════════════
    log_path = args.log
    if log_path is None:
        candidate = checkpoint_path.parent / "finetune_v3_9272951.out"
        if candidate.exists():
            log_path = str(candidate)
    if log_path and Path(log_path).exists():
        run_test3(log_path, output_dir, max_steps=args.max_log_steps)
    else:
        print("\n  Skipping Test 3 — no log file found. Use --log <path>")

    print("\n\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
