#!/usr/bin/env python3
"""
Evaluate finetuned TabPFN checkpoint and compare with baseline.
Uses the SAME datasets and constraints as training (55 datasets).

Usage:
    python evaluate_finetuned_comparison.py                                    # uses default checkpoint
    python evaluate_finetuned_comparison.py checkpoints/checkpoint_step700.pt  # specific checkpoint
"""

import sys
import argparse
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier

# Default paths
DEFAULT_CHECKPOINT = Path(__file__).parent / "checkpoints" / "checkpoint_step700.pt"
OUTPUT_DIR = Path(__file__).parent / "comparison_results"
REAL_DATA_PKL = Path(__file__).parent.parent / "01_real_data" / "AEON" / "data" / "classification_datasets.pkl"

# Same constraints as training (finetune_tabpfn.py FinetuneConfig)
MAX_SAMPLES = 1000
MAX_FLAT_FEATURES = 500
MAX_CLASSES = 10

# Same n_estimators as training evaluation
N_ESTIMATORS = 4


def load_real_datasets():
    """Load datasets using SAME constraints as training."""
    # Add path for pickle loading
    real_data_dir = Path(__file__).resolve().parent.parent / '01_real_data'
    if str(real_data_dir) not in sys.path:
        sys.path.insert(0, str(real_data_dir))
    
    from src.data_loader import TimeSeriesDataset  # noqa: F401 - needed for pickle
    
    # Load pickle
    if not REAL_DATA_PKL.exists():
        print(f"ERROR: Data not found at {REAL_DATA_PKL}")
        return []
    
    with open(REAL_DATA_PKL, 'rb') as f:
        datasets_list = pickle.load(f)
    
    datasets = []
    skipped_reasons = []
    
    for dataset in datasets_list:
        try:
            name = dataset.name
            X_train = dataset.X_train
            y_train = dataset.y_train
            X_test = dataset.X_test
            y_test = dataset.y_test
            
            if X_train is None or X_test is None:
                skipped_reasons.append((name, "None data"))
                continue
            
            # Ensure 3D
            if X_train.ndim == 2:
                X_train = X_train[:, :, np.newaxis]
            if X_test.ndim == 2:
                X_test = X_test[:, :, np.newaxis]
            
            # Shape: (n_samples, length, n_channels) -> (n_samples, n_channels, length)
            X_train = np.transpose(X_train, (0, 2, 1))
            X_test = np.transpose(X_test, (0, 2, 1))
            
            n_samples = X_train.shape[0] + X_test.shape[0]
            n_channels = X_train.shape[1]
            length = X_train.shape[2]
            flat_features = n_channels * length
            n_classes = len(np.unique(y_train))
            
            # Apply SAME constraints as training
            if n_samples > MAX_SAMPLES:
                skipped_reasons.append((name, f"samples={n_samples} > {MAX_SAMPLES}"))
                continue
            if flat_features > MAX_FLAT_FEATURES:
                skipped_reasons.append((name, f"features={flat_features} > {MAX_FLAT_FEATURES}"))
                continue
            if n_classes > MAX_CLASSES:
                skipped_reasons.append((name, f"classes={n_classes} > {MAX_CLASSES}"))
                continue
            
            # Flatten
            X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
            X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float32)
            
            # Encode labels
            le = LabelEncoder()
            le.fit(y_train)
            y_train_enc = le.transform(y_train).astype(np.int64)
            y_test_enc = le.transform(y_test).astype(np.int64)
            
            # Handle missing values
            if np.any(np.isnan(X_train_flat)):
                col_means = np.nanmean(X_train_flat, axis=0)
                col_means = np.nan_to_num(col_means, nan=0.0)
                for i in range(X_train_flat.shape[1]):
                    X_train_flat[:, i] = np.where(np.isnan(X_train_flat[:, i]), col_means[i], X_train_flat[:, i])
                    X_test_flat[:, i] = np.where(np.isnan(X_test_flat[:, i]), col_means[i], X_test_flat[:, i])
            
            datasets.append({
                'name': name,
                'X_train': X_train_flat,
                'y_train': y_train_enc,
                'X_test': X_test_flat,
                'y_test': y_test_enc,
                'n_classes': n_classes,
                'n_samples': n_samples,
                'n_features': flat_features,
            })
            
        except Exception as e:
            skipped_reasons.append((dataset.name if hasattr(dataset, 'name') else 'unknown', str(e)))
            continue
    
    print(f"\nSkipped {len(skipped_reasons)} datasets due to constraints:")
    for name, reason in skipped_reasons[:10]:
        print(f"  - {name}: {reason}")
    if len(skipped_reasons) > 10:
        print(f"  ... and {len(skipped_reasons) - 10} more")
    
    return datasets


def evaluate_model(clf, data, device):
    """Evaluate a model on a single dataset."""
    try:
        # Fit
        clf.fit(data['X_train'], data['y_train'])
        
        # Predict
        with torch.no_grad():
            proba = clf.predict_proba(data['X_test'])
            preds = proba.argmax(axis=1)
        
        # Metrics
        acc = accuracy_score(data['y_test'], preds)
        
        try:
            if data['n_classes'] == 2:
                auc = roc_auc_score(data['y_test'], proba[:, 1])
            else:
                auc = roc_auc_score(data['y_test'], proba, multi_class='ovr')
        except Exception:
            auc = None
        
        return {'accuracy': acc, 'auc': auc}
        
    except Exception as e:
        return {'accuracy': None, 'auc': None, 'error': str(e)}


def run_comparison(checkpoint_path, datasets, device='mps'):
    """Run both baseline and finetuned evaluation on all datasets."""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"  Step: {checkpoint.get('step', 'unknown')}")
    
    results = []
    
    for i, data in enumerate(datasets):
        name = data['name']
        print(f"  [{i+1}/{len(datasets)}] {name}...", end=' ', flush=True)
        
        try:
            # ===== BASELINE (fresh TabPFN, no finetuning) =====
            clf_baseline = TabPFNClassifier(
                device=device,
                n_estimators=N_ESTIMATORS,
                ignore_pretraining_limits=True,
            )
            baseline_result = evaluate_model(clf_baseline, data, device)
            del clf_baseline
            
            # ===== FINETUNED =====
            clf_finetuned = TabPFNClassifier(
                device=device,
                n_estimators=N_ESTIMATORS,
                ignore_pretraining_limits=True,
            )
            # Fit first (initializes internal model)
            clf_finetuned.fit(data['X_train'], data['y_train'])
            # Load finetuned weights
            clf_finetuned.model_.load_state_dict(checkpoint['model_state_dict'])
            clf_finetuned.model_.eval()
            
            # Predict with finetuned
            with torch.no_grad():
                proba = clf_finetuned.predict_proba(data['X_test'])
                preds = proba.argmax(axis=1)
            
            finetuned_acc = accuracy_score(data['y_test'], preds)
            try:
                if data['n_classes'] == 2:
                    finetuned_auc = roc_auc_score(data['y_test'], proba[:, 1])
                else:
                    finetuned_auc = roc_auc_score(data['y_test'], proba, multi_class='ovr')
            except:
                finetuned_auc = None
            
            del clf_finetuned
            
            # Record
            results.append({
                'name': name,
                'n_samples': data['n_samples'],
                'n_features': data['n_features'],
                'n_classes': data['n_classes'],
                'baseline_acc': baseline_result['accuracy'],
                'baseline_auc': baseline_result['auc'],
                'finetuned_acc': finetuned_acc,
                'finetuned_auc': finetuned_auc,
            })
            
            if baseline_result['auc'] and finetuned_auc:
                diff = finetuned_auc - baseline_result['auc']
                marker = "↑" if diff > 0.001 else ("↓" if diff < -0.001 else "=")
                print(f"Base={baseline_result['auc']:.4f} Fine={finetuned_auc:.4f} ({marker}{diff:+.4f})")
            else:
                print(f"Acc: {finetuned_acc:.4f}")
                
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                'name': name,
                'n_samples': data['n_samples'],
                'n_features': data['n_features'],
                'n_classes': data['n_classes'],
                'baseline_acc': None,
                'baseline_auc': None,
                'finetuned_acc': None,
                'finetuned_auc': None,
            })
    
    return results


def analyze_results(results):
    """Analyze and display comparison results."""
    df = pd.DataFrame(results)
    
    # Filter to successful evaluations
    valid = df[df['finetuned_auc'].notna() & df['baseline_auc'].notna()].copy()
    
    # Calculate differences
    valid['auc_diff'] = valid['finetuned_auc'] - valid['baseline_auc']
    valid['acc_diff'] = valid['finetuned_acc'] - valid['baseline_acc']
    
    print("\n" + "=" * 80)
    print("COMPARISON: FINETUNED vs BASELINE (same 55 datasets as training)")
    print("=" * 80)
    
    # Summary stats
    mean_baseline_auc = valid['baseline_auc'].mean()
    mean_finetuned_auc = valid['finetuned_auc'].mean()
    mean_diff = valid['auc_diff'].mean()
    
    mean_baseline_acc = valid['baseline_acc'].mean()
    mean_finetuned_acc = valid['finetuned_acc'].mean()
    
    print(f"\nDatasets evaluated: {len(valid)}/{len(df)}")
    print(f"n_estimators: {N_ESTIMATORS}")
    print(f"\nMean AUC - Baseline:  {mean_baseline_auc:.4f}")
    print(f"Mean AUC - Finetuned: {mean_finetuned_auc:.4f}")
    print(f"Mean Improvement:     {mean_diff:+.4f} ({mean_diff*100:+.2f}pp)")
    
    print(f"\nMean Acc - Baseline:  {mean_baseline_acc:.4f}")
    print(f"Mean Acc - Finetuned: {mean_finetuned_acc:.4f}")
    
    # Win/Tie/Loss
    wins = (valid['auc_diff'] > 0.001).sum()
    ties = ((valid['auc_diff'] >= -0.001) & (valid['auc_diff'] <= 0.001)).sum()
    losses = (valid['auc_diff'] < -0.001).sum()
    
    print(f"\nWin/Tie/Loss: {wins}/{ties}/{losses}")
    
    # Biggest improvements
    print("\n" + "-" * 60)
    print("TOP 10 IMPROVEMENTS:")
    print("-" * 60)
    top_improved = valid.nlargest(10, 'auc_diff')
    for _, row in top_improved.iterrows():
        print(f"  {row['name']:35s}: {row['baseline_auc']:.4f} → {row['finetuned_auc']:.4f} ({row['auc_diff']:+.4f})")
    
    # Biggest degradations
    print("\n" + "-" * 60)
    print("TOP 10 DEGRADATIONS:")
    print("-" * 60)
    top_degraded = valid.nsmallest(10, 'auc_diff')
    for _, row in top_degraded.iterrows():
        print(f"  {row['name']:35s}: {row['baseline_auc']:.4f} → {row['finetuned_auc']:.4f} ({row['auc_diff']:+.4f})")
    
    # Full table sorted by diff
    print("\n" + "-" * 60)
    print("ALL DATASETS (sorted by improvement):")
    print("-" * 60)
    sorted_df = valid.sort_values('auc_diff', ascending=False)
    for _, row in sorted_df.iterrows():
        marker = "↑" if row['auc_diff'] > 0.001 else ("↓" if row['auc_diff'] < -0.001 else "=")
        print(f"  {marker} {row['name']:35s}: {row['baseline_auc']:.4f} → {row['finetuned_auc']:.4f} ({row['auc_diff']:+.4f})")
    
    return valid


def main():
    parser = argparse.ArgumentParser(description="Evaluate finetuned TabPFN checkpoint vs baseline")
    parser.add_argument('checkpoint', nargs='?', default=None,
                        help=f'Path to checkpoint (default: {DEFAULT_CHECKPOINT})')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, mps, cuda, cpu')
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT
    
    print("=" * 60)
    print("FINETUNED TABPFN EVALUATION (apples-to-apples)")
    print("=" * 60)
    print(f"Constraints: max_samples={MAX_SAMPLES}, max_features={MAX_FLAT_FEATURES}, max_classes={MAX_CLASSES}")
    print(f"n_estimators: {N_ESTIMATORS} (same as training eval)")
    
    # Check checkpoint exists
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    # Create output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load datasets with same constraints as training
    print("\nLoading datasets with training constraints...")
    datasets = load_real_datasets()
    print(f"\nLoaded {len(datasets)} datasets (should be 55)")
    
    # Determine device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Run comparison (baseline computed fresh, not from old CSV)
    results = run_comparison(checkpoint_path, datasets, device=device)
    
    # Analyze
    comparison_df = analyze_results(results)
    
    # Save results
    step_str = checkpoint_path.stem.replace('checkpoint_', '')
    output_path = OUTPUT_DIR / f"finetuned_comparison_{step_str}.csv"
    comparison_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
