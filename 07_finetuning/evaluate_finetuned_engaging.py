#!/usr/bin/env python3
"""
Evaluate finetuned TabPFN checkpoint and compare with baseline.
Run this on Engaging where the data is already loaded.

Usage:
    python evaluate_finetuned_engaging.py --checkpoint checkpoints/checkpoint_step650.pt
"""

import sys
import pickle
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier

# Paths (adjust for Engaging)
SCRIPT_DIR = Path(__file__).parent
REAL_DATA_PKL = SCRIPT_DIR.parent / "01_real_data" / "AEON" / "data" / "classification_datasets.pkl"


def load_real_datasets(max_samples=1000, max_flat_features=500, max_classes=10):
    """Load datasets from pickle file."""
    real_data_dir = SCRIPT_DIR.parent / '01_real_data'
    if str(real_data_dir) not in sys.path:
        sys.path.insert(0, str(real_data_dir))
    
    from src.data_loader import TimeSeriesDataset  # noqa
    
    print(f"Loading datasets from {REAL_DATA_PKL}...")
    with open(REAL_DATA_PKL, 'rb') as f:
        datasets_list = pickle.load(f)
    
    print(f"Loaded {len(datasets_list)} total datasets")
    
    datasets = []
    for dataset in datasets_list:
        try:
            name = dataset.name
            X_train = dataset.X_train
            y_train = dataset.y_train
            X_test = dataset.X_test
            y_test = dataset.y_test
            
            if X_train is None or X_test is None:
                continue
            
            # Ensure 3D
            if X_train.ndim == 2:
                X_train = X_train[:, :, np.newaxis]
            if X_test.ndim == 2:
                X_test = X_test[:, :, np.newaxis]
            
            # Transpose
            X_train = np.transpose(X_train, (0, 2, 1))
            X_test = np.transpose(X_test, (0, 2, 1))
            
            n_samples = X_train.shape[0] + X_test.shape[0]
            n_channels = X_train.shape[1]
            length = X_train.shape[2]
            flat_features = n_channels * length
            n_classes = len(np.unique(y_train))
            
            # Check constraints
            if n_samples > max_samples:
                continue
            if flat_features > max_flat_features:
                continue
            if n_classes > max_classes:
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
                'n_features': flat_features,
            })
            
        except Exception as e:
            continue
    
    print(f"Filtered to {len(datasets)} valid datasets")
    return datasets


def evaluate_model(model_state_dict, datasets, device='cuda', n_estimators=8, model_name="model"):
    """Evaluate a model on all datasets."""
    results = []
    
    for i, data in enumerate(datasets):
        name = data['name']
        print(f"  [{i+1}/{len(datasets)}] {name}...", end=' ', flush=True)
        
        try:
            clf = TabPFNClassifier(
                device=device,
                n_estimators=n_estimators,
                ignore_pretraining_limits=True,
            )
            
            clf.fit(data['X_train'], data['y_train'])
            
            if model_state_dict is not None:
                clf.model_.load_state_dict(model_state_dict)
            clf.model_.eval()
            
            with torch.no_grad():
                proba = clf.predict_proba(data['X_test'])
                preds = proba.argmax(axis=1)
            
            acc = accuracy_score(data['y_test'], preds)
            
            try:
                if data['n_classes'] == 2:
                    auc = roc_auc_score(data['y_test'], proba[:, 1])
                else:
                    auc = roc_auc_score(data['y_test'], proba, multi_class='ovr')
            except Exception:
                auc = None
            
            results.append({
                'name': name,
                'accuracy': acc,
                'auc': auc,
            })
            
            if auc:
                print(f"AUC={auc:.4f}")
            else:
                print(f"Acc={acc:.4f}")
                
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                'name': name,
                'accuracy': None,
                'auc': None,
            })
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to finetuned checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--n-estimators', type=int, default=8, help='Number of estimators')
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    print(f"  Step: {checkpoint.get('step', 'unknown')}")
    
    # Load datasets
    datasets = load_real_datasets()
    
    # Evaluate baseline (no finetuning)
    print("\n" + "=" * 60)
    print("EVALUATING BASELINE (pretrained TabPFN)")
    print("=" * 60)
    baseline_results = evaluate_model(None, datasets, device=args.device, n_estimators=args.n_estimators, model_name="baseline")
    
    # Evaluate finetuned
    print("\n" + "=" * 60)
    print("EVALUATING FINETUNED MODEL")
    print("=" * 60)
    finetuned_results = evaluate_model(checkpoint['model_state_dict'], datasets, device=args.device, n_estimators=args.n_estimators, model_name="finetuned")
    
    # Create comparison DataFrame
    baseline_df = pd.DataFrame(baseline_results).rename(columns={'accuracy': 'baseline_acc', 'auc': 'baseline_auc'})
    finetuned_df = pd.DataFrame(finetuned_results).rename(columns={'accuracy': 'finetuned_acc', 'auc': 'finetuned_auc'})
    
    comparison = baseline_df.merge(finetuned_df, on='name')
    comparison['auc_diff'] = comparison['finetuned_auc'] - comparison['baseline_auc']
    comparison['acc_diff'] = comparison['finetuned_acc'] - comparison['baseline_acc']
    
    # Print results
    print("\n" + "=" * 80)
    print("COMPARISON: FINETUNED vs BASELINE")
    print("=" * 80)
    
    valid = comparison[comparison['finetuned_auc'].notna() & comparison['baseline_auc'].notna()]
    
    mean_baseline = valid['baseline_auc'].mean()
    mean_finetuned = valid['finetuned_auc'].mean()
    mean_diff = valid['auc_diff'].mean()
    
    print(f"\nMean AUC - Baseline:  {mean_baseline:.4f}")
    print(f"Mean AUC - Finetuned: {mean_finetuned:.4f}")
    print(f"Mean Improvement:     {mean_diff:+.4f} ({mean_diff*100:+.2f}pp)")
    
    wins = (valid['auc_diff'] > 0.001).sum()
    ties = ((valid['auc_diff'] >= -0.001) & (valid['auc_diff'] <= 0.001)).sum()
    losses = (valid['auc_diff'] < -0.001).sum()
    
    print(f"\nWin/Tie/Loss: {wins}/{ties}/{losses}")
    
    # Top improvements
    print("\n" + "-" * 60)
    print("TOP 10 IMPROVEMENTS:")
    print("-" * 60)
    top_improved = valid.nlargest(10, 'auc_diff')
    for _, row in top_improved.iterrows():
        print(f"  {row['name']:35s}: {row['baseline_auc']:.4f} → {row['finetuned_auc']:.4f} ({row['auc_diff']:+.4f})")
    
    # Top degradations
    print("\n" + "-" * 60)
    print("TOP 10 DEGRADATIONS:")
    print("-" * 60)
    top_degraded = valid.nsmallest(10, 'auc_diff')
    for _, row in top_degraded.iterrows():
        print(f"  {row['name']:35s}: {row['baseline_auc']:.4f} → {row['finetuned_auc']:.4f} ({row['auc_diff']:+.4f})")
    
    # All datasets
    print("\n" + "-" * 60)
    print("ALL DATASETS (sorted by improvement):")
    print("-" * 60)
    sorted_df = valid.sort_values('auc_diff', ascending=False)
    for _, row in sorted_df.iterrows():
        marker = "↑" if row['auc_diff'] > 0.001 else ("↓" if row['auc_diff'] < -0.001 else "=")
        print(f"  {marker} {row['name']:35s}: {row['baseline_auc']:.4f} → {row['finetuned_auc']:.4f} ({row['auc_diff']:+.4f})")
    
    # Save results
    output_path = SCRIPT_DIR / "finetuned_comparison_results.csv"
    comparison.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
