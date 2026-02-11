#!/usr/bin/env python3
"""
Compare TabPFN (baseline and finetuned) against AEON benchmark models.
Calculates mean rank, #rank1, #rank1_strict (state of the art).

Uses AUC ROC metric from AEON bakeoff benchmarks.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Paths
BENCHMARKS_DIR = Path(__file__).parent.parent / "01_real_data" / "AEON" / "benchmarks"
TABPFN_RESULTS = Path(__file__).parent / "comparison_results" / "finetuned_comparison_step700.csv"


def load_aeon_benchmarks() -> Dict[str, pd.DataFrame]:
    """Load all AEON benchmark AUROC files."""
    benchmarks = {}
    
    for filepath in BENCHMARKS_DIR.glob("*_auroc.csv"):
        model_name = filepath.stem.replace("_auroc", "")
        
        try:
            df = pd.read_csv(filepath, index_col=0)
            # Average across resamples (columns are resample indices)
            df_mean = df.mean(axis=1)
            benchmarks[model_name] = df_mean
        except Exception as e:
            print(f"  Warning: Could not load {filepath.name}: {e}")
    
    return benchmarks


def load_tabpfn_results() -> pd.DataFrame:
    """Load TabPFN results."""
    if not TABPFN_RESULTS.exists():
        raise FileNotFoundError(f"TabPFN results not found at {TABPFN_RESULTS}")
    
    return pd.read_csv(TABPFN_RESULTS)


def create_comparison_table(benchmarks: Dict[str, pd.Series], tabpfn_df: pd.DataFrame) -> pd.DataFrame:
    """Create a unified table with all models and TabPFN results."""
    
    # Get list of datasets that TabPFN was evaluated on
    tabpfn_datasets = set(tabpfn_df['name'].values)
    
    # Create a DataFrame with all models
    all_results = {}
    
    # Add AEON benchmark models
    for model_name, scores in benchmarks.items():
        all_results[model_name] = scores
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Find common datasets between AEON and TabPFN
    aeon_datasets = set(df.index)
    common_datasets = tabpfn_datasets & aeon_datasets
    
    print(f"\nDataset overlap:")
    print(f"  TabPFN evaluated on: {len(tabpfn_datasets)} datasets")
    print(f"  AEON benchmarks have: {len(aeon_datasets)} datasets")
    print(f"  Common datasets: {len(common_datasets)}")
    
    # Show which TabPFN datasets are missing from AEON
    missing_from_aeon = tabpfn_datasets - aeon_datasets
    if missing_from_aeon:
        print(f"\n  TabPFN datasets NOT in AEON benchmarks ({len(missing_from_aeon)}):")
        for name in sorted(missing_from_aeon):
            print(f"    - {name}")
    
    # Filter to common datasets
    common_datasets_list = sorted(list(common_datasets))
    df = df.loc[df.index.isin(common_datasets)]
    
    # Add TabPFN baseline and finetuned
    tabpfn_indexed = tabpfn_df.set_index('name')
    
    df['TabPFN_baseline'] = tabpfn_indexed.loc[common_datasets_list, 'baseline_auc']
    df['TabPFN_finetuned'] = tabpfn_indexed.loc[common_datasets_list, 'finetuned_auc']
    
    return df


def compute_rankings(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rankings for each dataset (higher AUC = rank 1)."""
    # Rank in descending order (higher AUC = better = rank 1)
    rankings = results_df.rank(axis=1, ascending=False, method='min')
    return rankings


def analyze_model(rankings: pd.DataFrame, model_name: str, results_df: pd.DataFrame) -> Dict:
    """Analyze ranking statistics for a single model."""
    if model_name not in rankings.columns:
        return None
    
    model_ranks = rankings[model_name]
    model_aucs = results_df[model_name]
    
    # Mean rank
    mean_rank = model_ranks.mean()
    
    # Rank 1 (including ties)
    rank1_count = (model_ranks == 1).sum()
    
    # Rank 1 strict (sole winner, no ties)
    rank1_strict_count = 0
    for dataset in rankings.index:
        # Check if this model is sole rank 1
        if model_ranks[dataset] == 1:
            # Count how many models have rank 1 for this dataset
            n_rank1 = (rankings.loc[dataset] == 1).sum()
            if n_rank1 == 1:
                rank1_strict_count += 1
    
    # Mean AUC
    mean_auc = model_aucs.mean()
    
    return {
        'model': model_name,
        'mean_rank': mean_rank,
        'rank1_count': rank1_count,
        'rank1_strict_count': rank1_strict_count,
        'mean_auc': mean_auc,
        'n_datasets': len(model_ranks),
    }


def main():
    print("=" * 80)
    print("COMPARE TabPFN vs AEON BENCHMARKS")
    print("=" * 80)
    
    # Load AEON benchmarks
    print("\nLoading AEON benchmarks...")
    benchmarks = load_aeon_benchmarks()
    print(f"  Loaded {len(benchmarks)} models")
    
    # Load TabPFN results
    print("\nLoading TabPFN results...")
    tabpfn_df = load_tabpfn_results()
    print(f"  Loaded {len(tabpfn_df)} datasets")
    
    # Create unified table
    print("\nCreating comparison table...")
    results_df = create_comparison_table(benchmarks, tabpfn_df)
    
    # Compute rankings
    print("\nComputing rankings...")
    rankings = compute_rankings(results_df)
    
    n_models = len(results_df.columns)
    n_datasets = len(results_df)
    
    print(f"\nTotal: {n_models} models, {n_datasets} datasets")
    
    # Analyze all models
    print("\n" + "=" * 80)
    print("RANKING ANALYSIS (on common datasets)")
    print("=" * 80)
    
    all_stats = []
    for model_name in results_df.columns:
        stats = analyze_model(rankings, model_name, results_df)
        if stats:
            all_stats.append(stats)
    
    # Sort by mean rank
    stats_df = pd.DataFrame(all_stats).sort_values('mean_rank')
    
    # Print table
    print(f"\n{'Model':<25} {'Mean Rank':>10} {'Rank 1':>8} {'Rank 1 Strict':>14} {'Mean AUC':>10}")
    print("-" * 70)
    
    for _, row in stats_df.iterrows():
        is_tabpfn = 'TabPFN' in row['model']
        marker = ">>>" if is_tabpfn else "   "
        print(f"{marker} {row['model']:<22} {row['mean_rank']:>10.2f} {int(row['rank1_count']):>8} {int(row['rank1_strict_count']):>14} {row['mean_auc']:>10.4f}")
    
    # Summary for TabPFN
    print("\n" + "=" * 80)
    print("TABPFN SUMMARY")
    print("=" * 80)
    
    tabpfn_baseline = stats_df[stats_df['model'] == 'TabPFN_baseline'].iloc[0]
    tabpfn_finetuned = stats_df[stats_df['model'] == 'TabPFN_finetuned'].iloc[0]
    
    baseline_rank_pos = (stats_df['mean_rank'] < tabpfn_baseline['mean_rank']).sum() + 1
    finetuned_rank_pos = (stats_df['mean_rank'] < tabpfn_finetuned['mean_rank']).sum() + 1
    
    print(f"\nTabPFN Baseline:")
    print(f"  Mean Rank: {tabpfn_baseline['mean_rank']:.2f} (position {baseline_rank_pos}/{n_models} in model ranking)")
    print(f"  Rank 1 (tied): {int(tabpfn_baseline['rank1_count'])}/{n_datasets} datasets")
    print(f"  Rank 1 (strict/SOTA): {int(tabpfn_baseline['rank1_strict_count'])}/{n_datasets} datasets")
    print(f"  Mean AUC: {tabpfn_baseline['mean_auc']:.4f}")
    
    print(f"\nTabPFN Finetuned (step 700):")
    print(f"  Mean Rank: {tabpfn_finetuned['mean_rank']:.2f} (position {finetuned_rank_pos}/{n_models} in model ranking)")
    print(f"  Rank 1 (tied): {int(tabpfn_finetuned['rank1_count'])}/{n_datasets} datasets")
    print(f"  Rank 1 (strict/SOTA): {int(tabpfn_finetuned['rank1_strict_count'])}/{n_datasets} datasets")
    print(f"  Mean AUC: {tabpfn_finetuned['mean_auc']:.4f}")
    
    print(f"\nImprovement from finetuning:")
    print(f"  Mean Rank: {tabpfn_baseline['mean_rank']:.2f} -> {tabpfn_finetuned['mean_rank']:.2f} ({tabpfn_finetuned['mean_rank'] - tabpfn_baseline['mean_rank']:+.2f})")
    print(f"  Rank 1 (tied): {int(tabpfn_baseline['rank1_count'])} -> {int(tabpfn_finetuned['rank1_count'])} ({int(tabpfn_finetuned['rank1_count'] - tabpfn_baseline['rank1_count']):+d})")
    print(f"  Rank 1 (strict): {int(tabpfn_baseline['rank1_strict_count'])} -> {int(tabpfn_finetuned['rank1_strict_count'])} ({int(tabpfn_finetuned['rank1_strict_count'] - tabpfn_baseline['rank1_strict_count']):+d})")
    print(f"  Mean AUC: {tabpfn_baseline['mean_auc']:.4f} -> {tabpfn_finetuned['mean_auc']:.4f} ({tabpfn_finetuned['mean_auc'] - tabpfn_baseline['mean_auc']:+.4f})")
    
    # Top 10 models by mean rank
    print("\n" + "=" * 80)
    print("TOP 10 MODELS BY MEAN RANK")
    print("=" * 80)
    top10 = stats_df.head(10)
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        is_tabpfn = 'TabPFN' in row['model']
        marker = "***" if is_tabpfn else "   "
        print(f"{i:2d}. {marker} {row['model']:<22} Mean Rank: {row['mean_rank']:.2f}, AUC: {row['mean_auc']:.4f}")
    
    # Ranking by mean AUC
    print("\n" + "=" * 80)
    print("ALL MODELS BY MEAN AUC (descending)")
    print("=" * 80)
    stats_by_auc = stats_df.sort_values('mean_auc', ascending=False).reset_index(drop=True)
    
    print(f"\n{'Pos':<4} {'Model':<25} {'Mean AUC':>10} {'Mean Rank':>10} {'Rank1':>6} {'SOTA':>6}")
    print("-" * 65)
    
    for i, row in stats_by_auc.iterrows():
        pos = i + 1
        is_tabpfn = 'TabPFN' in row['model']
        marker = ">>>" if is_tabpfn else "   "
        print(f"{marker} {pos:<3} {row['model']:<25} {row['mean_auc']:>10.4f} {row['mean_rank']:>10.2f} {int(row['rank1_count']):>6} {int(row['rank1_strict_count']):>6}")
    
    # Find TabPFN positions in AUC ranking
    baseline_auc_pos = stats_by_auc[stats_by_auc['model'] == 'TabPFN_baseline'].index[0] + 1
    finetuned_auc_pos = stats_by_auc[stats_by_auc['model'] == 'TabPFN_finetuned'].index[0] + 1
    
    print(f"\n--- TabPFN positions by Mean AUC ---")
    print(f"TabPFN Baseline:  {baseline_auc_pos}/{n_models}")
    print(f"TabPFN Finetuned: {finetuned_auc_pos}/{n_models}")
    
    # Per-dataset ranking improvement analysis
    print("\n" + "=" * 80)
    print("PER-DATASET RANKING: TabPFN Baseline vs Finetuned")
    print("=" * 80)
    
    # Get rankings for both TabPFN versions
    baseline_ranks = rankings['TabPFN_baseline']
    finetuned_ranks = rankings['TabPFN_finetuned']
    baseline_aucs = results_df['TabPFN_baseline']
    finetuned_aucs = results_df['TabPFN_finetuned']
    
    # Create per-dataset comparison
    per_dataset = pd.DataFrame({
        'dataset': rankings.index,
        'baseline_rank': baseline_ranks.values,
        'finetuned_rank': finetuned_ranks.values,
        'rank_improvement': baseline_ranks.values - finetuned_ranks.values,  # positive = better
        'baseline_auc': baseline_aucs.values,
        'finetuned_auc': finetuned_aucs.values,
        'auc_improvement': finetuned_aucs.values - baseline_aucs.values,
    })
    
    # Sort by rank improvement
    per_dataset_sorted = per_dataset.sort_values('rank_improvement', ascending=False)
    
    print(f"\n{'Dataset':<35} {'Base Rank':>10} {'Fine Rank':>10} {'Rank Δ':>8} {'AUC Δ':>8}")
    print("-" * 75)
    
    for _, row in per_dataset_sorted.iterrows():
        rank_delta = row['rank_improvement']
        auc_delta = row['auc_improvement']
        marker = "↑↑" if rank_delta >= 5 else ("↑" if rank_delta > 0 else ("↓" if rank_delta < 0 else "="))
        print(f"{marker} {row['dataset']:<33} {int(row['baseline_rank']):>10} {int(row['finetuned_rank']):>10} {rank_delta:>+8.0f} {auc_delta:>+8.4f}")
    
    # Summary
    improved = (per_dataset['rank_improvement'] > 0).sum()
    same = (per_dataset['rank_improvement'] == 0).sum()
    worse = (per_dataset['rank_improvement'] < 0).sum()
    
    print(f"\nSummary: Improved rank in {improved} datasets, same in {same}, worse in {worse}")
    print(f"Mean rank improvement: {per_dataset['rank_improvement'].mean():+.2f}")
    
    # Top improvements
    print("\n--- TOP 10 RANK IMPROVEMENTS ---")
    top_improved = per_dataset_sorted.head(10)
    for _, row in top_improved.iterrows():
        print(f"  {row['dataset']:<30}: rank {int(row['baseline_rank'])} -> {int(row['finetuned_rank'])} ({row['rank_improvement']:+.0f}), AUC {row['baseline_auc']:.4f} -> {row['finetuned_auc']:.4f}")
    
    # Worst degradations
    print("\n--- TOP 10 RANK DEGRADATIONS ---")
    worst = per_dataset_sorted.tail(10).iloc[::-1]
    for _, row in worst.iterrows():
        print(f"  {row['dataset']:<30}: rank {int(row['baseline_rank'])} -> {int(row['finetuned_rank'])} ({row['rank_improvement']:+.0f}), AUC {row['baseline_auc']:.4f} -> {row['finetuned_auc']:.4f}")
    
    # Save detailed results
    output_dir = Path(__file__).parent / "comparison_results"
    output_dir.mkdir(exist_ok=True)
    
    # Save full ranking table
    rankings.to_csv(output_dir / "aeon_rankings.csv")
    stats_df.to_csv(output_dir / "aeon_model_stats.csv", index=False)
    per_dataset_sorted.to_csv(output_dir / "per_dataset_ranking_comparison.csv", index=False)
    
    print(f"\nResults saved to:")
    print(f"  {output_dir / 'aeon_rankings.csv'}")
    print(f"  {output_dir / 'aeon_model_stats.csv'}")
    print(f"  {output_dir / 'per_dataset_ranking_comparison.csv'}")


if __name__ == "__main__":
    main()
