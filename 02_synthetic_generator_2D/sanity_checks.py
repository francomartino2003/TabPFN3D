"""
Sanity Checks for the Synthetic Dataset Generator.

This script evaluates the quality and diversity of generated datasets
through multiple statistical and ML-based checks.

Run with:
    python sanity_checks.py

Or import specific checks:
    from sanity_checks import run_all_checks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import warnings
import time
import json
import os

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

# Import generator
try:
    from generator import SyntheticDatasetGenerator, SyntheticDataset
    from config import PriorConfig
except ImportError:
    from .generator import SyntheticDatasetGenerator, SyntheticDataset
    from .config import PriorConfig

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Using RandomForest as replacement.")


@dataclass
class DatasetStats:
    """Statistics for a single generated dataset."""
    dataset_id: int
    n_samples: int
    n_features: int
    n_classes: int
    train_ratio: float
    
    # Feature statistics
    n_categorical: int
    n_continuous: int
    n_relevant: int
    n_irrelevant: int
    missing_pct: float
    
    # Model performances (accuracy)
    baseline_acc: float
    logistic_acc: float
    rf_acc: float
    xgb_acc: float
    
    # AUC scores (if binary/multiclass)
    baseline_auc: Optional[float] = None
    logistic_auc: Optional[float] = None
    rf_auc: Optional[float] = None
    xgb_auc: Optional[float] = None
    
    # Feature importance stats
    n_zero_importance: int = 0
    top_feature_importance: float = 0.0
    importance_entropy: float = 0.0
    
    # Model rankings (1 = best)
    rank_logistic: int = 0
    rank_rf: int = 0
    rank_xgb: int = 0


def prepare_data(dataset: SyntheticDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare dataset for ML: handle NaN, encode labels, split train/test.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = dataset.X.copy()
    y = dataset.y.copy()
    
    # Handle missing values
    if np.any(np.isnan(X)):
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
    
    # Encode labels for classification
    if dataset.is_classification:
        le = LabelEncoder()
        y = le.fit_transform(y.astype(int))
    
    # Split using dataset's train_ratio
    train_ratio = dataset.config.train_ratio
    test_size = 1.0 - train_ratio
    
    # Ensure we have at least some samples in each split
    n_samples = len(y)
    min_test = max(2, int(n_samples * 0.1))
    min_train = max(2, int(n_samples * 0.1))
    
    if n_samples < min_train + min_test:
        test_size = 0.3
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if dataset.is_classification else None
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def train_models(X_train: np.ndarray, X_test: np.ndarray, 
                 y_train: np.ndarray, y_test: np.ndarray,
                 n_classes: int) -> Dict[str, Dict[str, float]]:
    """
    Train baseline and ML models, return performance metrics.
    """
    results = {}
    
    # Baseline: Majority class
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    results['baseline'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': None
    }
    
    # Logistic Regression
    try:
        lr = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        y_proba = lr.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        if n_classes == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            try:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            except:
                auc = None
        results['logistic'] = {'accuracy': acc, 'auc': auc}
    except Exception as e:
        results['logistic'] = {'accuracy': results['baseline']['accuracy'], 'auc': None}
    
    # Random Forest
    try:
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        if n_classes == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            try:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            except:
                auc = None
        results['rf'] = {'accuracy': acc, 'auc': auc, 'importances': rf.feature_importances_}
    except Exception as e:
        results['rf'] = {'accuracy': results['baseline']['accuracy'], 'auc': None, 'importances': None}
    
    # XGBoost (or fallback to RF)
    if HAS_XGBOOST:
        try:
            xgb = XGBClassifier(
                n_estimators=50, max_depth=5, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbosity=0,
                use_label_encoder=False, eval_metric='logloss'
            )
            xgb.fit(X_train, y_train)
            y_pred = xgb.predict(X_test)
            y_proba = xgb.predict_proba(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            if n_classes == 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                try:
                    auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                except:
                    auc = None
            results['xgb'] = {'accuracy': acc, 'auc': auc, 'importances': xgb.feature_importances_}
        except Exception as e:
            results['xgb'] = results['rf'].copy()
    else:
        results['xgb'] = results['rf'].copy()
    
    return results


def compute_feature_importance_stats(importances: np.ndarray) -> Dict[str, float]:
    """Compute statistics about feature importances."""
    if importances is None:
        return {'n_zero': 0, 'top_importance': 0, 'entropy': 0}
    
    # Normalize
    imp = importances / (importances.sum() + 1e-10)
    
    # Number of zero-importance features
    n_zero = np.sum(imp < 0.01)
    
    # Top feature importance
    top_imp = np.max(imp)
    
    # Entropy of importance distribution
    imp_nonzero = imp[imp > 1e-10]
    if len(imp_nonzero) > 0:
        entropy = -np.sum(imp_nonzero * np.log(imp_nonzero + 1e-10))
    else:
        entropy = 0
    
    return {'n_zero': n_zero, 'top_importance': top_imp, 'entropy': entropy}


def sanity_check_1_baseline_vs_models(stats_list: List[DatasetStats]) -> Dict[str, Any]:
    """
    SANITY CHECK 1: Compare baseline vs ML models.
    
    Expected: Wide distribution of performances, no single dominant model.
    """
    print("\n" + "="*70)
    print("SANITY CHECK 1: Baseline vs Models")
    print("="*70)
    
    # Categorize datasets by difficulty
    categories = {
        'very_easy': [],    # All models >> baseline
        'easy': [],         # XGB/RF >> Logistic >> baseline
        'medium': [],       # Spread between models
        'hard': [],         # All models ≈ baseline
    }
    
    for s in stats_list:
        baseline = s.baseline_acc
        best_model = max(s.logistic_acc, s.rf_acc, s.xgb_acc)
        spread = best_model - baseline
        
        if spread > 0.3 and s.logistic_acc > baseline + 0.2:
            categories['very_easy'].append(s)
        elif spread > 0.15:
            categories['easy'].append(s)
        elif spread > 0.05:
            categories['medium'].append(s)
        else:
            categories['hard'].append(s)
    
    total = len(stats_list)
    print(f"\nDataset difficulty distribution (n={total}):")
    for cat, items in categories.items():
        pct = 100 * len(items) / total
        bar = "█" * int(pct / 2)
        print(f"  {cat:12s}: {len(items):3d} ({pct:5.1f}%) {bar}")
    
    # Red flags
    red_flags = []
    
    # Check if Logistic always wins (everything is linear)
    logistic_wins = sum(1 for s in stats_list if s.logistic_acc >= s.xgb_acc and s.logistic_acc >= s.rf_acc)
    if logistic_wins / total > 0.7:
        red_flags.append(f"⚠️ Logistic wins {100*logistic_wins/total:.0f}% of time → too linear!")
    
    # Check if XGB always perfect
    xgb_perfect = sum(1 for s in stats_list if s.xgb_acc > 0.95)
    if xgb_perfect / total > 0.5:
        red_flags.append(f"⚠️ XGB perfect {100*xgb_perfect/total:.0f}% of time → target too direct!")
    
    # Check if all models fail
    all_fail = sum(1 for s in stats_list if max(s.logistic_acc, s.rf_acc, s.xgb_acc) - s.baseline_acc < 0.02)
    if all_fail / total > 0.5:
        red_flags.append(f"⚠️ All models fail {100*all_fail/total:.0f}% of time → too much noise!")
    
    if red_flags:
        print("\n🚨 RED FLAGS:")
        for flag in red_flags:
            print(f"  {flag}")
    else:
        print("\n✅ No red flags detected!")
    
    return {
        'categories': {k: len(v) for k, v in categories.items()},
        'red_flags': red_flags,
        'logistic_win_rate': logistic_wins / total,
        'xgb_perfect_rate': xgb_perfect / total,
        'all_fail_rate': all_fail / total
    }


def sanity_check_2_model_rankings(stats_list: List[DatasetStats]) -> Dict[str, Any]:
    """
    SANITY CHECK 2: Check variability in model rankings.
    
    Expected: No single model dominates, varied rankings across datasets.
    """
    print("\n" + "="*70)
    print("SANITY CHECK 2: Model Ranking Variability")
    print("="*70)
    
    ranking_counts = {
        'logistic_best': 0,
        'rf_best': 0,
        'xgb_best': 0,
        'all_similar': 0,  # Spread < 0.05
    }
    
    for s in stats_list:
        accs = {'logistic': s.logistic_acc, 'rf': s.rf_acc, 'xgb': s.xgb_acc}
        sorted_models = sorted(accs.items(), key=lambda x: -x[1])
        
        spread = sorted_models[0][1] - sorted_models[-1][1]
        
        if spread < 0.05:
            ranking_counts['all_similar'] += 1
        else:
            best = sorted_models[0][0]
            ranking_counts[f'{best}_best'] += 1
    
    total = len(stats_list)
    print(f"\nModel ranking distribution (n={total}):")
    for name, count in ranking_counts.items():
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"  {name:15s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    # Check for dominance
    max_dominance = max(ranking_counts['logistic_best'], ranking_counts['rf_best'], ranking_counts['xgb_best'])
    dominance_rate = max_dominance / total
    
    if dominance_rate > 0.6:
        print(f"\n⚠️ Warning: One model dominates {100*dominance_rate:.0f}% of datasets")
    else:
        print(f"\n✅ Good variability: No model dominates > 60%")
    
    return {
        'rankings': ranking_counts,
        'max_dominance_rate': dominance_rate
    }


def sanity_check_3_difficulty_spectrum(stats_list: List[DatasetStats]) -> Dict[str, Any]:
    """
    SANITY CHECK 3: Check the spectrum of dataset difficulty.
    
    Expected: Wide distribution from easy to hard datasets.
    """
    print("\n" + "="*70)
    print("SANITY CHECK 3: Difficulty Spectrum")
    print("="*70)
    
    # Use XGB accuracy as difficulty measure
    xgb_accs = [s.xgb_acc for s in stats_list]
    baseline_accs = [s.baseline_acc for s in stats_list]
    
    # Compute lift over baseline
    lifts = [x - b for x, b in zip(xgb_accs, baseline_accs)]
    
    percentiles = [10, 25, 50, 75, 90]
    xgb_pcts = np.percentile(xgb_accs, percentiles)
    lift_pcts = np.percentile(lifts, percentiles)
    
    print(f"\nXGBoost Accuracy Percentiles:")
    for p, v in zip(percentiles, xgb_pcts):
        print(f"  P{p:02d}: {v:.3f}")
    
    print(f"\nLift over Baseline Percentiles:")
    for p, v in zip(percentiles, lift_pcts):
        print(f"  P{p:02d}: {v:+.3f}")
    
    # Check distribution shape
    print(f"\nDistribution stats:")
    print(f"  XGB Accuracy: mean={np.mean(xgb_accs):.3f}, std={np.std(xgb_accs):.3f}")
    print(f"  Lift: mean={np.mean(lifts):.3f}, std={np.std(lifts):.3f}")
    
    # Create histogram bins
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(xgb_accs, bins=bins)
    
    print(f"\nXGB Accuracy Histogram:")
    for i in range(len(bins)-1):
        pct = 100 * hist[i] / len(xgb_accs)
        bar = "█" * int(pct / 2)
        print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]:3d} ({pct:5.1f}%) {bar}")
    
    return {
        'xgb_percentiles': dict(zip(percentiles, xgb_pcts.tolist())),
        'lift_percentiles': dict(zip(percentiles, lift_pcts.tolist())),
        'xgb_mean': np.mean(xgb_accs),
        'xgb_std': np.std(xgb_accs)
    }


def sanity_check_4_feature_relevance(stats_list: List[DatasetStats]) -> Dict[str, Any]:
    """
    SANITY CHECK 4: Check feature importance distribution.
    
    Expected: Many irrelevant features, few dominant ones, varied across datasets.
    """
    print("\n" + "="*70)
    print("SANITY CHECK 4: Feature Relevance")
    print("="*70)
    
    zero_importance_pcts = [s.n_zero_importance / s.n_features * 100 for s in stats_list if s.n_features > 0]
    top_importances = [s.top_feature_importance for s in stats_list]
    entropies = [s.importance_entropy for s in stats_list]
    
    print(f"\n% Features with ~0 importance:")
    print(f"  Mean: {np.mean(zero_importance_pcts):.1f}%")
    print(f"  Median: {np.median(zero_importance_pcts):.1f}%")
    print(f"  Range: [{np.min(zero_importance_pcts):.1f}%, {np.max(zero_importance_pcts):.1f}%]")
    
    print(f"\nTop feature importance (normalized):")
    print(f"  Mean: {np.mean(top_importances):.3f}")
    print(f"  Median: {np.median(top_importances):.3f}")
    
    print(f"\nImportance entropy (higher = more spread):")
    print(f"  Mean: {np.mean(entropies):.3f}")
    print(f"  Median: {np.median(entropies):.3f}")
    
    # Red flags
    if np.mean(zero_importance_pcts) < 10:
        print("\n⚠️ Warning: Almost all features are relevant → graph too dense?")
    elif np.mean(top_importances) > 0.7:
        print("\n⚠️ Warning: Single feature dominates → possible structural bug")
    else:
        print("\n✅ Feature relevance distribution looks reasonable")
    
    return {
        'zero_importance_pct_mean': np.mean(zero_importance_pcts),
        'top_importance_mean': np.mean(top_importances),
        'entropy_mean': np.mean(entropies)
    }


def sanity_check_5_label_permutation(datasets: List[SyntheticDataset], n_test: int = 10) -> Dict[str, Any]:
    """
    SANITY CHECK 5: Label permutation test.
    
    If permuting labels doesn't hurt performance → there's leakage!
    """
    print("\n" + "="*70)
    print("SANITY CHECK 5: Label Permutation Test")
    print("="*70)
    
    # Test on subset
    test_datasets = datasets[:n_test]
    
    results = []
    for i, ds in enumerate(test_datasets):
        try:
            X_train, X_test, y_train, y_test = prepare_data(ds)
            
            # Normal training
            rf = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            normal_acc = accuracy_score(y_test, rf.predict(X_test))
            
            # Permuted labels
            y_train_perm = np.random.permutation(y_train)
            rf_perm = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
            rf_perm.fit(X_train, y_train_perm)
            perm_acc = accuracy_score(y_test, rf_perm.predict(X_test))
            
            # Baseline
            baseline_acc = 1.0 / len(np.unique(y_test))  # Random guess
            
            drop = normal_acc - perm_acc
            results.append({
                'dataset_id': i,
                'normal_acc': normal_acc,
                'permuted_acc': perm_acc,
                'drop': drop,
                'baseline': baseline_acc
            })
            
            print(f"  Dataset {i}: Normal={normal_acc:.3f}, Permuted={perm_acc:.3f}, Drop={drop:+.3f}")
            
        except Exception as e:
            print(f"  Dataset {i}: Error - {e}")
    
    if results:
        avg_drop = np.mean([r['drop'] for r in results])
        print(f"\nAverage drop after permutation: {avg_drop:+.3f}")
        
        if avg_drop < 0.05:
            print("⚠️ WARNING: Labels don't matter much → possible leakage!")
        else:
            print("✅ Labels matter significantly → no obvious leakage")
    
    return {'results': results, 'avg_drop': np.mean([r['drop'] for r in results]) if results else 0}


def sanity_check_6_data_size_robustness(datasets: List[SyntheticDataset], n_test: int = 10) -> Dict[str, Any]:
    """
    SANITY CHECK 6: Check if models improve with more data.
    
    Expected: Performance increases with data size, with diminishing returns.
    """
    print("\n" + "="*70)
    print("SANITY CHECK 6: Data Size Robustness")
    print("="*70)
    
    fractions = [0.2, 0.5, 1.0]
    test_datasets = [d for d in datasets[:20] if d.X.shape[0] >= 100][:n_test]
    
    all_curves = []
    
    for i, ds in enumerate(test_datasets):
        try:
            X_train, X_test, y_train, y_test = prepare_data(ds)
            
            curve = []
            for frac in fractions:
                n_use = max(10, int(len(y_train) * frac))
                X_sub, y_sub = X_train[:n_use], y_train[:n_use]
                
                rf = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
                rf.fit(X_sub, y_sub)
                acc = accuracy_score(y_test, rf.predict(X_test))
                curve.append(acc)
            
            all_curves.append(curve)
            improvement = curve[-1] - curve[0]
            print(f"  Dataset {i}: 20%={curve[0]:.3f}, 50%={curve[1]:.3f}, 100%={curve[2]:.3f}, Δ={improvement:+.3f}")
            
        except Exception as e:
            print(f"  Dataset {i}: Error - {e}")
    
    if all_curves:
        avg_curve = np.mean(all_curves, axis=0)
        print(f"\nAverage learning curve:")
        for frac, acc in zip(fractions, avg_curve):
            print(f"  {int(frac*100):3d}% data: {acc:.3f}")
        
        avg_improvement = avg_curve[-1] - avg_curve[0]
        if avg_improvement < 0.02:
            print("\n⚠️ Warning: No improvement with more data → weak signal?")
        elif avg_improvement > 0.3:
            print("\n⚠️ Warning: Too much improvement → target too direct?")
        else:
            print("\n✅ Reasonable learning curve")
    
    return {'curves': all_curves, 'fractions': fractions}


def sanity_check_7_mutual_information(datasets: List[SyntheticDataset], n_test: int = 10) -> Dict[str, Any]:
    """
    SANITY CHECK 7: Check mutual information between features and target.
    
    Expected: Few features with high MI, many with ~0 MI.
    """
    print("\n" + "="*70)
    print("SANITY CHECK 7: Mutual Information Analysis")
    print("="*70)
    
    test_datasets = datasets[:n_test]
    
    all_mi_stats = []
    
    for i, ds in enumerate(test_datasets):
        try:
            X = ds.X.copy()
            y = ds.y.copy()
            
            # Handle missing values
            if np.any(np.isnan(X)):
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
            
            # Encode labels
            if ds.is_classification:
                le = LabelEncoder()
                y = le.fit_transform(y.astype(int))
            
            # Compute MI
            mi = mutual_info_classif(X, y, random_state=42)
            mi_normalized = mi / (mi.max() + 1e-10)
            
            # Stats
            high_mi = np.sum(mi_normalized > 0.5)
            low_mi = np.sum(mi_normalized < 0.1)
            
            all_mi_stats.append({
                'high_mi': high_mi,
                'low_mi': low_mi,
                'max_mi': mi_normalized.max(),
                'mean_mi': mi_normalized.mean()
            })
            
            print(f"  Dataset {i}: High MI={high_mi}, Low MI={low_mi}, Max={mi_normalized.max():.3f}")
            
        except Exception as e:
            print(f"  Dataset {i}: Error - {e}")
    
    if all_mi_stats:
        avg_high = np.mean([s['high_mi'] for s in all_mi_stats])
        avg_low = np.mean([s['low_mi'] for s in all_mi_stats])
        print(f"\nAverage: High MI features={avg_high:.1f}, Low MI features={avg_low:.1f}")
        
        if avg_low < 2:
            print("⚠️ Warning: Almost no irrelevant features")
        else:
            print("✅ Good mix of relevant and irrelevant features")
    
    return {'stats': all_mi_stats}


def sanity_check_8_invariances(datasets: List[SyntheticDataset], n_test: int = 5) -> Dict[str, Any]:
    """
    SANITY CHECK 8: Check invariance to permutations and rescaling.
    
    Expected: Performance invariant to column/row order, rescaling.
    """
    print("\n" + "="*70)
    print("SANITY CHECK 8: Invariance Checks")
    print("="*70)
    
    test_datasets = datasets[:n_test]
    results = []
    
    for i, ds in enumerate(test_datasets):
        try:
            X_train, X_test, y_train, y_test = prepare_data(ds)
            
            # Original
            rf = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            original_acc = accuracy_score(y_test, rf.predict(X_test))
            
            # Column permutation
            perm_cols = np.random.permutation(X_train.shape[1])
            X_train_perm = X_train[:, perm_cols]
            X_test_perm = X_test[:, perm_cols]
            rf_perm = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
            rf_perm.fit(X_train_perm, y_train)
            col_perm_acc = accuracy_score(y_test, rf_perm.predict(X_test_perm))
            
            # Row shuffle (same split, different order)
            shuffle_idx = np.random.permutation(len(y_train))
            X_train_shuf = X_train[shuffle_idx]
            y_train_shuf = y_train[shuffle_idx]
            rf_shuf = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
            rf_shuf.fit(X_train_shuf, y_train_shuf)
            row_shuf_acc = accuracy_score(y_test, rf_shuf.predict(X_test))
            
            # Rescaling (different scale per feature)
            scales = np.random.uniform(0.5, 2.0, X_train.shape[1])
            X_train_scale = X_train * scales
            X_test_scale = X_test * scales
            rf_scale = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
            rf_scale.fit(X_train_scale, y_train)
            scale_acc = accuracy_score(y_test, rf_scale.predict(X_test_scale))
            
            results.append({
                'original': original_acc,
                'col_perm': col_perm_acc,
                'row_shuf': row_shuf_acc,
                'rescale': scale_acc
            })
            
            print(f"  Dataset {i}: Orig={original_acc:.3f}, ColPerm={col_perm_acc:.3f}, "
                  f"RowShuf={row_shuf_acc:.3f}, Rescale={scale_acc:.3f}")
            
        except Exception as e:
            print(f"  Dataset {i}: Error - {e}")
    
    if results:
        # Check invariances
        col_diffs = [abs(r['original'] - r['col_perm']) for r in results]
        row_diffs = [abs(r['original'] - r['row_shuf']) for r in results]
        scale_diffs = [abs(r['original'] - r['rescale']) for r in results]
        
        print(f"\nAverage differences from original:")
        print(f"  Column permutation: {np.mean(col_diffs):.4f}")
        print(f"  Row shuffle: {np.mean(row_diffs):.4f}")
        print(f"  Rescaling: {np.mean(scale_diffs):.4f}")
        
        if max(np.mean(col_diffs), np.mean(row_diffs)) > 0.05:
            print("\n⚠️ Warning: Results depend on ordering → possible artifacts")
        else:
            print("\n✅ Results are invariant to permutations")
    
    return {'results': results}


def generate_dataset_stats(dataset: SyntheticDataset, dataset_id: int) -> DatasetStats:
    """Generate comprehensive statistics for a single dataset."""
    
    # Basic stats
    n_samples = dataset.X.shape[0]
    n_features = dataset.X.shape[1]
    n_classes = dataset.n_classes if dataset.is_classification else 0
    train_ratio = dataset.config.train_ratio
    
    # Feature types
    feature_types = dataset.metadata.get('feature_types', {})
    n_categorical = sum(1 for t in feature_types.values() if t == 'categorical')
    n_continuous = n_features - n_categorical
    
    n_relevant = dataset.metadata.get('n_relevant_features', 0)
    n_irrelevant = dataset.metadata.get('n_irrelevant_features', 0)
    
    missing_pct = 100 * np.isnan(dataset.X).mean() if np.any(np.isnan(dataset.X)) else 0
    
    # Train models
    try:
        X_train, X_test, y_train, y_test = prepare_data(dataset)
        model_results = train_models(X_train, X_test, y_train, y_test, n_classes)
        
        baseline_acc = model_results['baseline']['accuracy']
        logistic_acc = model_results['logistic']['accuracy']
        rf_acc = model_results['rf']['accuracy']
        xgb_acc = model_results['xgb']['accuracy']
        
        # AUC
        baseline_auc = model_results['baseline']['auc']
        logistic_auc = model_results['logistic']['auc']
        rf_auc = model_results['rf']['auc']
        xgb_auc = model_results['xgb']['auc']
        
        # Feature importance
        importances = model_results.get('xgb', {}).get('importances')
        if importances is None:
            importances = model_results.get('rf', {}).get('importances')
        
        imp_stats = compute_feature_importance_stats(importances)
        
        # Rankings
        accs = [('logistic', logistic_acc), ('rf', rf_acc), ('xgb', xgb_acc)]
        accs_sorted = sorted(accs, key=lambda x: -x[1])
        ranks = {name: i+1 for i, (name, _) in enumerate(accs_sorted)}
        
    except Exception as e:
        print(f"  Warning: Error processing dataset {dataset_id}: {e}")
        baseline_acc = logistic_acc = rf_acc = xgb_acc = 0.0
        baseline_auc = logistic_auc = rf_auc = xgb_auc = None
        imp_stats = {'n_zero': 0, 'top_importance': 0, 'entropy': 0}
        ranks = {'logistic': 0, 'rf': 0, 'xgb': 0}
    
    return DatasetStats(
        dataset_id=dataset_id,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        train_ratio=train_ratio,
        n_categorical=n_categorical,
        n_continuous=n_continuous,
        n_relevant=n_relevant,
        n_irrelevant=n_irrelevant,
        missing_pct=missing_pct,
        baseline_acc=baseline_acc,
        logistic_acc=logistic_acc,
        rf_acc=rf_acc,
        xgb_acc=xgb_acc,
        baseline_auc=baseline_auc,
        logistic_auc=logistic_auc,
        rf_auc=rf_auc,
        xgb_auc=xgb_auc,
        n_zero_importance=imp_stats['n_zero'],
        top_feature_importance=imp_stats['top_importance'],
        importance_entropy=imp_stats['entropy'],
        rank_logistic=ranks['logistic'],
        rank_rf=ranks['rf'],
        rank_xgb=ranks['xgb']
    )


def print_dataset_overview(stats_list: List[DatasetStats]):
    """Print overview of generated datasets."""
    print("\n" + "="*70)
    print("DATASET OVERVIEW")
    print("="*70)
    
    n = len(stats_list)
    
    # Size distributions
    n_samples = [s.n_samples for s in stats_list]
    n_features = [s.n_features for s in stats_list]
    n_classes = [s.n_classes for s in stats_list]
    train_ratios = [s.train_ratio for s in stats_list]
    
    print(f"\nGenerated {n} classification datasets:")
    print(f"\n  Samples:   min={min(n_samples):5d}, max={max(n_samples):5d}, median={np.median(n_samples):7.0f}")
    print(f"  Features:  min={min(n_features):5d}, max={max(n_features):5d}, median={np.median(n_features):7.0f}")
    print(f"  Classes:   min={min(n_classes):5d}, max={max(n_classes):5d}, median={np.median(n_classes):7.0f}")
    print(f"  Train %:   min={min(train_ratios)*100:5.1f}%, max={max(train_ratios)*100:5.1f}%, median={np.median(train_ratios)*100:5.1f}%")
    
    # Feature types
    n_categorical = [s.n_categorical for s in stats_list]
    n_continuous = [s.n_continuous for s in stats_list]
    n_relevant = [s.n_relevant for s in stats_list]
    n_irrelevant = [s.n_irrelevant for s in stats_list]
    missing_pcts = [s.missing_pct for s in stats_list]
    
    print(f"\nFeature statistics:")
    print(f"  Categorical:  mean={np.mean(n_categorical):5.1f}, median={np.median(n_categorical):5.1f}")
    print(f"  Continuous:   mean={np.mean(n_continuous):5.1f}, median={np.median(n_continuous):5.1f}")
    print(f"  Relevant:     mean={np.mean(n_relevant):5.1f}, median={np.median(n_relevant):5.1f}")
    print(f"  Irrelevant:   mean={np.mean(n_irrelevant):5.1f}, median={np.median(n_irrelevant):5.1f}")
    print(f"  Missing %:    mean={np.mean(missing_pcts):5.1f}%, max={max(missing_pcts):5.1f}%")


def run_all_checks(n_datasets: int = 100, seed: int = 42, save_results: bool = True) -> Dict[str, Any]:
    """
    Run all sanity checks on generated datasets.
    
    Args:
        n_datasets: Number of datasets to generate
        seed: Random seed
        save_results: Whether to save results to JSON
        
    Returns:
        Dictionary with all check results
    """
    print("="*70)
    print("SYNTHETIC DATASET GENERATOR - SANITY CHECKS")
    print("="*70)
    
    # Configure prior for classification only
    prior = PriorConfig(
        prob_classification=1.0,  # Only classification
        n_rows_range=(100, 5000),  # Reasonable sizes for testing
        n_features_range=(5, 50),
    )
    
    # Generate datasets
    print(f"\nGenerating {n_datasets} classification datasets...")
    generator = SyntheticDatasetGenerator(prior=prior, seed=seed)
    
    datasets = []
    stats_list = []
    
    start_time = time.time()
    for i in range(n_datasets):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (n_datasets - i - 1)
            print(f"  Generated {i+1}/{n_datasets} datasets... (ETA: {eta:.0f}s)")
        
        dataset = generator.generate()
        datasets.append(dataset)
        
        stats = generate_dataset_stats(dataset, i)
        stats_list.append(stats)
    
    elapsed = time.time() - start_time
    print(f"\nGeneration complete in {elapsed:.1f}s ({elapsed/n_datasets:.2f}s per dataset)")
    
    # Print overview
    print_dataset_overview(stats_list)
    
    # Run all checks
    results = {}
    
    results['check_1_baseline'] = sanity_check_1_baseline_vs_models(stats_list)
    results['check_2_rankings'] = sanity_check_2_model_rankings(stats_list)
    results['check_3_difficulty'] = sanity_check_3_difficulty_spectrum(stats_list)
    results['check_4_features'] = sanity_check_4_feature_relevance(stats_list)
    results['check_5_permutation'] = sanity_check_5_label_permutation(datasets, n_test=10)
    results['check_6_data_size'] = sanity_check_6_data_size_robustness(datasets, n_test=10)
    results['check_7_mutual_info'] = sanity_check_7_mutual_information(datasets, n_test=10)
    results['check_8_invariances'] = sanity_check_8_invariances(datasets, n_test=5)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_red_flags = results['check_1_baseline'].get('red_flags', [])
    
    if all_red_flags:
        print("\n🚨 RED FLAGS FOUND:")
        for flag in all_red_flags:
            print(f"  {flag}")
    else:
        print("\n✅ ALL SANITY CHECKS PASSED!")
    
    print("\nKey metrics:")
    print(f"  - XGB accuracy range: [{results['check_3_difficulty']['xgb_percentiles'][10]:.2f}, "
          f"{results['check_3_difficulty']['xgb_percentiles'][90]:.2f}]")
    print(f"  - Model ranking variability: {results['check_2_rankings']['max_dominance_rate']:.1%} max dominance")
    print(f"  - Feature relevance: {results['check_4_features']['zero_importance_pct_mean']:.1f}% zero importance")
    print(f"  - Label permutation drop: {results['check_5_permutation']['avg_drop']:+.3f}")
    
    # Save results
    if save_results:
        output_file = 'sanity_check_results.json'
        
        # Convert numpy to python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        results_json = convert(results)
        
        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run sanity checks on synthetic dataset generator")
    parser.add_argument('--n', type=int, default=100, help="Number of datasets to generate")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--no-save', action='store_true', help="Don't save results to JSON")
    
    args = parser.parse_args()
    
    run_all_checks(n_datasets=args.n, seed=args.seed, save_results=not args.no_save)


