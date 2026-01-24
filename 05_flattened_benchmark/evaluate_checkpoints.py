#!/usr/bin/env python3
"""
Evaluate existing checkpoints on real datasets.
This script loads saved checkpoints and evaluates them with the corrected evaluation code.
"""

import sys
import os
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "00_TabPFN" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "01_real_data" / "src"))

from tabpfn import TabPFNClassifier


def load_real_datasets(max_samples=1000, max_flat_features=500, max_classes=10):
    """Load and filter real datasets."""
    datasets_path = Path(__file__).parent.parent / "01_real_data" / "data" / "classification_datasets.pkl"
    
    # Temporarily add src to path for unpickling
    src_path = str(Path(__file__).parent.parent / "01_real_data" / "src")
    sys.path.insert(0, src_path)
    
    with open(datasets_path, 'rb') as f:
        all_datasets = pickle.load(f)
    
    sys.path.remove(src_path)
    
    valid_datasets = []
    
    for ds in all_datasets:
        try:
            X_train = ds.X_train  # (n_samples, n_channels, length)
            X_test = ds.X_test
            y_train = ds.y_train
            y_test = ds.y_test
            
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
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            # Handle missing values
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
            })
            
        except Exception as e:
            continue
    
    return valid_datasets


def evaluate_checkpoint(checkpoint_path, datasets, device='cuda'):
    """Evaluate a single checkpoint on all datasets."""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint['model_state_dict']
    step = checkpoint.get('step', 0)
    
    all_aucs = []
    all_accs = []
    
    for data in datasets:
        try:
            # Create fresh classifier
            eval_clf = TabPFNClassifier(
                device=device,
                n_estimators=8,
                ignore_pretraining_limits=True,
            )
            
            # Call fit to set up preprocessing and inference engine
            eval_clf.fit(data['X_train'], data['y_train'])
            
            # NOW copy the finetuned weights (after fit initialized everything)
            eval_clf.model_.load_state_dict(model_state)
            eval_clf.model_.eval()
            
            # Predict
            proba = eval_clf.predict_proba(data['X_test'])
            preds = proba.argmax(axis=1)
            
            # Accuracy
            acc = accuracy_score(data['y_test'], preds)
            all_accs.append(acc)
            
            # AUC
            try:
                if data['n_classes'] == 2:
                    auc = roc_auc_score(data['y_test'], proba[:, 1])
                else:
                    auc = roc_auc_score(data['y_test'], proba, multi_class='ovr')
                all_aucs.append(auc)
            except Exception:
                pass
                
        except Exception as e:
            print(f"  Error on {data['name']}: {type(e).__name__}")
            continue
    
    return {
        'step': step,
        'mean_auc': np.mean(all_aucs) if all_aucs else 0.0,
        'mean_acc': np.mean(all_accs) if all_accs else 0.0,
        'n_evaluated': len(all_aucs),
    }


def evaluate_baseline(datasets, device='cuda'):
    """Evaluate the baseline (pre-trained) model."""
    
    all_aucs = []
    all_accs = []
    
    for data in datasets:
        try:
            clf = TabPFNClassifier(
                device=device,
                n_estimators=8,
                ignore_pretraining_limits=True,
            )
            
            clf.fit(data['X_train'], data['y_train'])
            proba = clf.predict_proba(data['X_test'])
            preds = proba.argmax(axis=1)
            
            acc = accuracy_score(data['y_test'], preds)
            all_accs.append(acc)
            
            try:
                if data['n_classes'] == 2:
                    auc = roc_auc_score(data['y_test'], proba[:, 1])
                else:
                    auc = roc_auc_score(data['y_test'], proba, multi_class='ovr')
                all_aucs.append(auc)
            except Exception:
                pass
                
        except Exception as e:
            continue
    
    return {
        'step': 0,
        'mean_auc': np.mean(all_aucs) if all_aucs else 0.0,
        'mean_acc': np.mean(all_accs) if all_accs else 0.0,
        'n_evaluated': len(all_aucs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str, 
                        default='results/finetune_checkpoints',
                        help='Directory containing checkpoints')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    print("=" * 60)
    print("CHECKPOINT EVALUATION")
    print("=" * 60)
    
    # Load datasets
    print("\nLoading real datasets...")
    datasets = load_real_datasets()
    print(f"Loaded {len(datasets)} datasets")
    
    # Find checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_step*.pt"))
    print(f"Found {len(checkpoints)} checkpoints")
    
    # Evaluate baseline first
    print("\n" + "-" * 40)
    print("Evaluating BASELINE (pre-trained)...")
    baseline = evaluate_baseline(datasets, args.device)
    print(f"  Baseline AUC: {baseline['mean_auc']:.4f}, Acc: {baseline['mean_acc']:.4f}")
    
    # Evaluate each checkpoint
    results = [baseline]
    
    for ckpt_path in checkpoints:
        print(f"\nEvaluating {ckpt_path.name}...")
        result = evaluate_checkpoint(ckpt_path, datasets, args.device)
        results.append(result)
        
        delta_auc = result['mean_auc'] - baseline['mean_auc']
        delta_acc = result['mean_acc'] - baseline['mean_acc']
        
        print(f"  Step {result['step']:4d}: AUC {result['mean_auc']:.4f} ({delta_auc:+.4f}), "
              f"Acc {result['mean_acc']:.4f} ({delta_acc:+.4f})")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Step':>6} | {'AUC':>8} | {'Δ AUC':>8} | {'Acc':>8} | {'Δ Acc':>8}")
    print("-" * 50)
    
    for r in results:
        delta_auc = r['mean_auc'] - baseline['mean_auc']
        delta_acc = r['mean_acc'] - baseline['mean_acc']
        step_label = "Base" if r['step'] == 0 else f"{r['step']}"
        print(f"{step_label:>6} | {r['mean_auc']:.4f}   | {delta_auc:+.4f}   | {r['mean_acc']:.4f}   | {delta_acc:+.4f}")
    
    # Best checkpoint
    best = max(results[1:], key=lambda x: x['mean_auc']) if len(results) > 1 else results[0]
    print(f"\nBest checkpoint: Step {best['step']} with AUC {best['mean_auc']:.4f}")


if __name__ == "__main__":
    main()
