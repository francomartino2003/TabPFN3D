#!/usr/bin/env python3
"""
Evaluate ALL checkpoints on ALL real datasets.

For each checkpoint, evaluates on all real datasets and saves:
- Per-dataset accuracy and AUC
- Mean accuracy and AUC
- Comparison to baseline (pre-trained model)

Usage:
    python evaluate_checkpoints.py --checkpoint-dir checkpoints --device cuda
"""

import sys
import os
import pickle
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "00_TabPFN" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "01_real_data"))

from tabpfn import TabPFNClassifier


def load_real_datasets(max_samples=1000, max_flat_features=500, max_classes=10):
    """Load and filter ALL real datasets."""
    datasets_path = Path(__file__).parent.parent / "01_real_data" / "AEON" / "data" / "classification_datasets.pkl"
    
    # Import the module so pickle can find the class
    from src.data_loader import TimeSeriesDataset  # noqa: F401
    
    with open(datasets_path, 'rb') as f:
        all_datasets = pickle.load(f)
    
    valid_datasets = []
    
    for ds in all_datasets:
        try:
            X_train = ds.X_train
            X_test = ds.X_test
            y_train = ds.y_train
            y_test = ds.y_test
            
            if X_train is None or X_test is None:
                continue
            
            # Ensure 3D
            if X_train.ndim == 2:
                X_train = X_train[:, :, np.newaxis]
            if X_test.ndim == 2:
                X_test = X_test[:, :, np.newaxis]
            
            # Transpose to (n_samples, n_channels, length)
            X_train = np.transpose(X_train, (0, 2, 1))
            X_test = np.transpose(X_test, (0, 2, 1))
            
            n_samples = X_train.shape[0] + X_test.shape[0]
            n_channels = X_train.shape[1]
            length = X_train.shape[2]
            flat_features = n_channels * length
            n_classes = len(np.unique(y_train))
            
            if n_samples > max_samples:
                continue
            if flat_features > max_flat_features:
                continue
            if n_classes > max_classes:
                continue
            
            # Flatten
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            # Handle NaN
            train_mean = np.nanmean(X_train_flat, axis=0)
            train_mean = np.where(np.isnan(train_mean), 0, train_mean)
            
            X_train_flat = np.where(np.isnan(X_train_flat), train_mean, X_train_flat)
            X_test_flat = np.where(np.isnan(X_test_flat), train_mean, X_test_flat)
            X_train_flat = np.nan_to_num(X_train_flat, nan=0.0, posinf=1e10, neginf=-1e10)
            X_test_flat = np.nan_to_num(X_test_flat, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Encode labels
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_test_enc = le.transform(y_test)
            
            valid_datasets.append({
                'name': ds.name,
                'X_train': X_train_flat.astype(np.float32),
                'X_test': X_test_flat.astype(np.float32),
                'y_train': y_train_enc,
                'y_test': y_test_enc,
                'n_classes': n_classes,
                'n_features': flat_features,
                'n_samples': n_samples,
            })
            
        except Exception as e:
            continue
    
    return valid_datasets


def evaluate_checkpoint(checkpoint_path, datasets, device='cuda', n_estimators=8):
    """Evaluate a single checkpoint on ALL datasets."""
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint['model_state_dict']
    step = checkpoint.get('step', 0)
    
    results = []
    
    for data in datasets:
        result = {'name': data['name'], 'n_classes': data['n_classes']}
        try:
            eval_clf = TabPFNClassifier(
                device=device,
                n_estimators=n_estimators,
                ignore_pretraining_limits=True,
            )
            
            # Fit to get preprocessing and then load finetuned weights
            eval_clf.fit(data['X_train'], data['y_train'])
            eval_clf.model_.load_state_dict(model_state)
            eval_clf.model_.eval()
            
            proba = eval_clf.predict_proba(data['X_test'])
            preds = proba.argmax(axis=1)
            
            acc = accuracy_score(data['y_test'], preds)
            result['accuracy'] = float(acc)
            
            try:
                if data['n_classes'] == 2:
                    auc = roc_auc_score(data['y_test'], proba[:, 1])
                else:
                    auc = roc_auc_score(data['y_test'], proba, multi_class='ovr')
                result['auc'] = float(auc)
            except Exception:
                result['auc'] = None
            
            result['status'] = 'success'
                
        except Exception as e:
            result['accuracy'] = None
            result['auc'] = None
            result['status'] = 'failed'
            result['error'] = str(e)[:100]
        
        results.append(result)
    
    return {
        'step': step,
        'checkpoint': str(checkpoint_path.name),
        'results': results,
    }


def evaluate_baseline(datasets, device='cuda', n_estimators=8):
    """Evaluate the baseline (pre-trained) model on ALL datasets."""
    
    results = []
    
    for data in datasets:
        result = {'name': data['name'], 'n_classes': data['n_classes']}
        try:
            clf = TabPFNClassifier(
                device=device,
                n_estimators=n_estimators,
                ignore_pretraining_limits=True,
            )
            
            clf.fit(data['X_train'], data['y_train'])
            proba = clf.predict_proba(data['X_test'])
            preds = proba.argmax(axis=1)
            
            acc = accuracy_score(data['y_test'], preds)
            result['accuracy'] = float(acc)
            
            try:
                if data['n_classes'] == 2:
                    auc = roc_auc_score(data['y_test'], proba[:, 1])
                else:
                    auc = roc_auc_score(data['y_test'], proba, multi_class='ovr')
                result['auc'] = float(auc)
            except Exception:
                result['auc'] = None
            
            result['status'] = 'success'
                
        except Exception as e:
            result['accuracy'] = None
            result['auc'] = None
            result['status'] = 'failed'
            result['error'] = str(e)[:100]
        
        results.append(result)
    
    return {
        'step': 0,
        'checkpoint': 'baseline',
        'results': results,
    }


def summarize_results(eval_result):
    """Compute summary statistics for an evaluation result."""
    results = eval_result['results']
    
    aucs = [r['auc'] for r in results if r.get('auc') is not None]
    accs = [r['accuracy'] for r in results if r.get('accuracy') is not None]
    
    return {
        'step': eval_result['step'],
        'checkpoint': eval_result['checkpoint'],
        'n_datasets': len(results),
        'n_successful': len(aucs),
        'mean_auc': float(np.mean(aucs)) if aucs else None,
        'std_auc': float(np.std(aucs)) if aucs else None,
        'mean_acc': float(np.mean(accs)) if accs else None,
        'std_acc': float(np.std(accs)) if accs else None,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate checkpoints on all real datasets')
    parser.add_argument('--checkpoint-dir', type=str, 
                        default='checkpoints',
                        help='Directory containing checkpoints')
    parser.add_argument('--output-dir', type=str,
                        default='eval_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-estimators', type=int, default=8)
    args = parser.parse_args()
    
    print("=" * 60)
    print("CHECKPOINT EVALUATION ON ALL REAL DATASETS")
    print("=" * 60)
    
    # Load datasets
    print("\nLoading ALL real datasets...")
    datasets = load_real_datasets()
    print(f"Loaded {len(datasets)} datasets")
    
    for ds in datasets:
        print(f"  - {ds['name']}: {ds['n_samples']} samples, {ds['n_features']} features, {ds['n_classes']} classes")
    
    # Find checkpoints
    checkpoint_dir = Path(__file__).parent / args.checkpoint_dir
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_step*.pt"))
    print(f"\nFound {len(checkpoints)} checkpoints")
    
    # Create output directory
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate baseline
    print("\n" + "-" * 40)
    print("Evaluating BASELINE (pre-trained)...")
    baseline = evaluate_baseline(datasets, args.device, args.n_estimators)
    baseline_summary = summarize_results(baseline)
    print(f"  Mean AUC: {baseline_summary['mean_auc']:.4f} ± {baseline_summary['std_auc']:.4f}")
    print(f"  Mean Acc: {baseline_summary['mean_acc']:.4f} ± {baseline_summary['std_acc']:.4f}")
    print(f"  Successful: {baseline_summary['n_successful']}/{baseline_summary['n_datasets']}")
    
    all_results = [baseline]
    all_summaries = [baseline_summary]
    
    # Evaluate each checkpoint
    for ckpt_path in checkpoints:
        print(f"\nEvaluating {ckpt_path.name}...")
        result = evaluate_checkpoint(ckpt_path, datasets, args.device, args.n_estimators)
        summary = summarize_results(result)
        
        delta_auc = (summary['mean_auc'] or 0) - (baseline_summary['mean_auc'] or 0)
        delta_acc = (summary['mean_acc'] or 0) - (baseline_summary['mean_acc'] or 0)
        
        print(f"  Step {result['step']:4d}: AUC {summary['mean_auc']:.4f} ({delta_auc:+.4f}), "
              f"Acc {summary['mean_acc']:.4f} ({delta_acc:+.4f})")
        
        all_results.append(result)
        all_summaries.append(summary)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    detailed_results = {
        'timestamp': timestamp,
        'n_datasets': len(datasets),
        'dataset_names': [ds['name'] for ds in datasets],
        'evaluations': all_results,
    }
    
    with open(output_dir / f'detailed_results_{timestamp}.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Save summaries
    with open(output_dir / f'summaries_{timestamp}.json', 'w') as f:
        json.dump(all_summaries, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Step':>8} | {'AUC':>8} | {'Δ AUC':>8} | {'Acc':>8} | {'Δ Acc':>8} | {'N':>4}")
    print("-" * 60)
    
    for s in all_summaries:
        delta_auc = (s['mean_auc'] or 0) - (baseline_summary['mean_auc'] or 0)
        delta_acc = (s['mean_acc'] or 0) - (baseline_summary['mean_acc'] or 0)
        step_label = "Base" if s['step'] == 0 else str(s['step'])
        
        auc_str = f"{s['mean_auc']:.4f}" if s['mean_auc'] else "N/A"
        acc_str = f"{s['mean_acc']:.4f}" if s['mean_acc'] else "N/A"
        
        print(f"{step_label:>8} | {auc_str:>8} | {delta_auc:+.4f} | {acc_str:>8} | {delta_acc:+.4f} | {s['n_successful']:>4}")
    
    # Find best checkpoint
    if len(all_summaries) > 1:
        valid_summaries = [s for s in all_summaries[1:] if s['mean_auc'] is not None]
        if valid_summaries:
            best = max(valid_summaries, key=lambda x: x['mean_auc'])
            print(f"\nBest checkpoint: Step {best['step']} with AUC {best['mean_auc']:.4f}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
