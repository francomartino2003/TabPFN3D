#!/usr/bin/env python3
"""
Compare TabPFN (baseline and finetuned) against AEON benchmark models using ACCURACY.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Paths
BENCHMARKS_DIR = Path(__file__).parent.parent / "01_real_data" / "AEON" / "benchmarks"
TABPFN_RESULTS = Path(__file__).parent / "comparison_results" / "finetuned_comparison_step700.csv"


def load_aeon_benchmarks_accuracy():
    """Load all AEON benchmark accuracy files (TESTFOLDS)."""
    benchmarks = {}
    
    for filepath in BENCHMARKS_DIR.glob("*_TESTFOLDS.csv"):
        model_name = filepath.stem.replace("_TESTFOLDS", "")
        
        try:
            df = pd.read_csv(filepath, index_col=0)
            # Average across folds (columns are fold indices)
            df_mean = df.mean(axis=1)
            benchmarks[model_name] = df_mean
        except Exception as e:
            print(f"  Warning: Could not load {filepath.name}: {e}")
    
    return benchmarks


def load_tabpfn_results():
    """Load TabPFN results."""
    return pd.read_csv(TABPFN_RESULTS)


def create_comparison_table(benchmarks, tabpfn_df):
    """Create a unified table with all models and TabPFN results."""
    
    tabpfn_datasets = set(tabpfn_df['name'].values)
    
    # Create DataFrame with all models
    all_results = {}
    for model_name, scores in benchmarks.items():
        all_results[model_name] = scores
    
    df = pd.DataFrame(all_results)
    aeon_datasets = set(df.index)
    common_datasets = tabpfn_datasets & aeon_datasets
    
    print(f"\nDataset overlap:")
    print(f"  TabPFN evaluated on: {len(tabpfn_datasets)} datasets")
    print(f"  AEON benchmarks have: {len(aeon_datasets)} datasets")
    print(f"  Common datasets: {len(common_datasets)}")
    
    # Filter to common datasets
    common_datasets_list = sorted(list(common_datasets))
    df = df.loc[df.index.isin(common_datasets)]
    
    # Add TabPFN
    tabpfn_indexed = tabpfn_df.set_index('name')
    df['TabPFN_baseline'] = tabpfn_indexed.loc[common_datasets_list, 'baseline_acc']
    df['TabPFN_finetuned'] = tabpfn_indexed.loc[common_datasets_list, 'finetuned_acc']
    
    return df


def compute_rankings(results_df):
    """Compute rankings for each dataset (higher accuracy = rank 1)."""
    rankings = results_df.rank(axis=1, ascending=False, method='min')
    return rankings


def analyze_model(rankings, model_name, results_df):
    """Analyze ranking statistics for a single model."""
    if model_name not in rankings.columns:
        return None
    
    model_ranks = rankings[model_name]
    model_accs = results_df[model_name]
    
    mean_rank = model_ranks.mean()
    rank1_count = (model_ranks == 1).sum()
    
    rank1_strict_count = 0
    for dataset in rankings.index:
        if model_ranks[dataset] == 1:
            n_rank1 = (rankings.loc[dataset] == 1).sum()
            if n_rank1 == 1:
                rank1_strict_count += 1
    
    mean_acc = model_accs.mean()
    
    return {
        'model': model_name,
        'mean_rank': mean_rank,
        'rank1_count': rank1_count,
        'rank1_strict_count': rank1_strict_count,
        'mean_acc': mean_acc,
        'n_datasets': len(model_ranks),
    }


def main():
    print("=" * 80)
    print("COMPARE TabPFN vs AEON BENCHMARKS (ACCURACY)")
    print("=" * 80)
    
    print("\nLoading AEON benchmarks (accuracy)...")
    benchmarks = load_aeon_benchmarks_accuracy()
    print(f"  Loaded {len(benchmarks)} models")
    
    print("\nLoading TabPFN results...")
    tabpfn_df = load_tabpfn_results()
    print(f"  Loaded {len(tabpfn_df)} datasets")
    
    print("\nCreating comparison table...")
    results_df = create_comparison_table(benchmarks, tabpfn_df)
    
    print("\nComputing rankings...")
    rankings = compute_rankings(results_df)
    
    n_models = len(results_df.columns)
    n_datasets = len(results_df)
    
    print(f"\nTotal: {n_models} models, {n_datasets} datasets")
    
    # Analyze all models
    all_stats = []
    for model_name in results_df.columns:
        stats = analyze_model(rankings, model_name, results_df)
        if stats:
            all_stats.append(stats)
    
    stats_df = pd.DataFrame(all_stats).sort_values('mean_rank')
    
    # Print ranking by mean rank
    print("\n" + "=" * 80)
    print("ALL MODELS BY MEAN RANK (ACCURACY)")
    print("=" * 80)
    
    print(f"\n{'Model':<25} {'Mean Rank':>10} {'Rank 1':>8} {'SOTA':>6} {'Mean Acc':>10}")
    print("-" * 65)
    
    for _, row in stats_df.iterrows():
        is_tabpfn = 'TabPFN' in row['model']
        marker = ">>>" if is_tabpfn else "   "
        print(f"{marker} {row['model']:<22} {row['mean_rank']:>10.2f} {int(row['rank1_count']):>8} {int(row['rank1_strict_count']):>6} {row['mean_acc']:>10.4f}")
    
    # Ranking by mean accuracy
    print("\n" + "=" * 80)
    print("ALL MODELS BY MEAN ACCURACY (descending)")
    print("=" * 80)
    
    stats_by_acc = stats_df.sort_values('mean_acc', ascending=False).reset_index(drop=True)
    
    print(f"\n{'Pos':<4} {'Model':<25} {'Mean Acc':>10} {'Mean Rank':>10} {'Rank1':>6} {'SOTA':>6}")
    print("-" * 65)
    
    for i, row in stats_by_acc.iterrows():
        pos = i + 1
        is_tabpfn = 'TabPFN' in row['model']
        marker = ">>>" if is_tabpfn else "   "
        print(f"{marker} {pos:<3} {row['model']:<25} {row['mean_acc']:>10.4f} {row['mean_rank']:>10.2f} {int(row['rank1_count']):>6} {int(row['rank1_strict_count']):>6}")
    
    # TabPFN summary
    print("\n" + "=" * 80)
    print("TABPFN SUMMARY (ACCURACY)")
    print("=" * 80)
    
    tabpfn_baseline = stats_df[stats_df['model'] == 'TabPFN_baseline'].iloc[0]
    tabpfn_finetuned = stats_df[stats_df['model'] == 'TabPFN_finetuned'].iloc[0]
    
    baseline_rank_pos = (stats_df['mean_rank'] < tabpfn_baseline['mean_rank']).sum() + 1
    finetuned_rank_pos = (stats_df['mean_rank'] < tabpfn_finetuned['mean_rank']).sum() + 1
    
    baseline_acc_pos = (stats_by_acc['mean_acc'] > tabpfn_baseline['mean_acc']).sum() + 1
    finetuned_acc_pos = (stats_by_acc['mean_acc'] > tabpfn_finetuned['mean_acc']).sum() + 1
    
    print(f"\nTabPFN Baseline:")
    print(f"  Mean Rank: {tabpfn_baseline['mean_rank']:.2f} (position {baseline_rank_pos}/{n_models})")
    print(f"  Mean Accuracy: {tabpfn_baseline['mean_acc']:.4f} (position {baseline_acc_pos}/{n_models})")
    print(f"  Rank 1 (tied): {int(tabpfn_baseline['rank1_count'])}/{n_datasets}")
    print(f"  Rank 1 (SOTA): {int(tabpfn_baseline['rank1_strict_count'])}/{n_datasets}")
    
    print(f"\nTabPFN Finetuned:")
    print(f"  Mean Rank: {tabpfn_finetuned['mean_rank']:.2f} (position {finetuned_rank_pos}/{n_models})")
    print(f"  Mean Accuracy: {tabpfn_finetuned['mean_acc']:.4f} (position {finetuned_acc_pos}/{n_models})")
    print(f"  Rank 1 (tied): {int(tabpfn_finetuned['rank1_count'])}/{n_datasets}")
    print(f"  Rank 1 (SOTA): {int(tabpfn_finetuned['rank1_strict_count'])}/{n_datasets}")
    
    print(f"\nImprovement from finetuning:")
    print(f"  Mean Rank: {tabpfn_baseline['mean_rank']:.2f} -> {tabpfn_finetuned['mean_rank']:.2f} ({tabpfn_finetuned['mean_rank'] - tabpfn_baseline['mean_rank']:+.2f})")
    print(f"  Mean Accuracy: {tabpfn_baseline['mean_acc']:.4f} -> {tabpfn_finetuned['mean_acc']:.4f} ({tabpfn_finetuned['mean_acc'] - tabpfn_baseline['mean_acc']:+.4f})")
    
    # Per-dataset analysis
    print("\n" + "=" * 80)
    print("PER-DATASET RANKING: TabPFN Baseline vs Finetuned (ACCURACY)")
    print("=" * 80)
    
    baseline_ranks = rankings['TabPFN_baseline']
    finetuned_ranks = rankings['TabPFN_finetuned']
    baseline_accs = results_df['TabPFN_baseline']
    finetuned_accs = results_df['TabPFN_finetuned']
    
    per_dataset = pd.DataFrame({
        'dataset': rankings.index,
        'baseline_rank': baseline_ranks.values,
        'finetuned_rank': finetuned_ranks.values,
        'rank_improvement': baseline_ranks.values - finetuned_ranks.values,
        'baseline_acc': baseline_accs.values,
        'finetuned_acc': finetuned_accs.values,
        'acc_improvement': finetuned_accs.values - baseline_accs.values,
    })
    
    per_dataset_sorted = per_dataset.sort_values('rank_improvement', ascending=False)
    
    print(f"\n{'Dataset':<35} {'Base Rank':>10} {'Fine Rank':>10} {'Rank Δ':>8} {'Acc Δ':>8}")
    print("-" * 75)
    
    for _, row in per_dataset_sorted.iterrows():
        rank_delta = row['rank_improvement']
        acc_delta = row['acc_improvement']
        marker = "↑↑" if rank_delta >= 5 else ("↑" if rank_delta > 0 else ("↓" if rank_delta < 0 else "="))
        print(f"{marker} {row['dataset']:<33} {int(row['baseline_rank']):>10} {int(row['finetuned_rank']):>10} {rank_delta:>+8.0f} {acc_delta:>+8.4f}")
    
    improved = (per_dataset['rank_improvement'] > 0).sum()
    same = (per_dataset['rank_improvement'] == 0).sum()
    worse = (per_dataset['rank_improvement'] < 0).sum()
    
    print(f"\nSummary: Improved rank in {improved} datasets, same in {same}, worse in {worse}")
    print(f"Mean rank improvement: {per_dataset['rank_improvement'].mean():+.2f}")
    
    print("\n--- TOP 10 RANK IMPROVEMENTS (ACCURACY) ---")
    top_improved = per_dataset_sorted.head(10)
    for _, row in top_improved.iterrows():
        print(f"  {row['dataset']:<30}: rank {int(row['baseline_rank'])} -> {int(row['finetuned_rank'])} ({row['rank_improvement']:+.0f}), Acc {row['baseline_acc']:.4f} -> {row['finetuned_acc']:.4f}")
    
    print("\n--- TOP 10 RANK DEGRADATIONS (ACCURACY) ---")
    worst = per_dataset_sorted.tail(10).iloc[::-1]
    for _, row in worst.iterrows():
        print(f"  {row['dataset']:<30}: rank {int(row['baseline_rank'])} -> {int(row['finetuned_rank'])} ({row['rank_improvement']:+.0f}), Acc {row['baseline_acc']:.4f} -> {row['finetuned_acc']:.4f}")
    
    # Save
    output_dir = Path(__file__).parent / "comparison_results"
    output_dir.mkdir(exist_ok=True)
    stats_by_acc.to_csv(output_dir / "aeon_model_stats_accuracy.csv", index=False)
    per_dataset_sorted.to_csv(output_dir / "per_dataset_ranking_comparison_accuracy.csv", index=False)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
