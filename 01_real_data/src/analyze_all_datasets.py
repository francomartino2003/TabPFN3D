"""
Main script to analyze all datasets
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
import pickle
from typing import List, Dict
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

from src.data_loader import TimeSeriesDataset
from src.time_series_statistics import TimeSeriesStatistics
from src.load_classification_datasets import load_all_classification_datasets



def analyze_classification_datasets(output_dir: Path, 
                                   max_datasets: int = None,
                                   save_datasets: bool = True) -> pd.DataFrame:
    """
    Analyze all classification datasets
    
    Args:
        output_dir: Directory to save results
        max_datasets: Maximum number of datasets to analyze
        save_datasets: If True, saves loaded datasets
        
    Returns:
        DataFrame with aggregated statistics
    """
    print("=" * 80)
    print("CLASSIFICATION DATASETS ANALYSIS")
    print("=" * 80)
    
    # Ensure output_dir is relative to project root directory
    if not output_dir.is_absolute():
        # If we're in src/, go up one level
        base_dir = Path(__file__).parent.parent
        output_dir = base_dir / output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    datasets_path = output_dir / "classification_datasets.pkl"
    if datasets_path.exists() and save_datasets:
        print(f"\nLoading datasets from {datasets_path}...")
        with open(datasets_path, 'rb') as f:
            datasets = pickle.load(f)
        print(f"Loaded {len(datasets)} datasets")
    else:
        print("\nDownloading classification datasets...")
        datasets = load_all_classification_datasets(
            max_datasets=max_datasets,
            save_path=datasets_path if save_datasets else None,
            verbose=True
        )
    
    if not datasets:
        print("Could not load classification datasets")
        return pd.DataFrame()
    
    # Calculate basic (lightweight) statistics
    print("\nCalculating basic statistics...")
    all_stats = []
    simple_stats = []
    
    for dataset in tqdm(datasets, desc="Computing stats"):
        stats = TimeSeriesStatistics.compute_dataset_stats_simple(dataset)
        simple_stats.append(stats)
        all_stats.append(dataset.get_info())
    
    # Check if they have benchmarks (using downloaded benchmarks)
    print("\nChecking benchmarks...")
    base_dir = Path(__file__).parent.parent
    benchmarks_dir = base_dir / "AEON" / "benchmarks"
    
    # Get list of datasets with benchmarks from CSV files
    # Structure: each CSV has first column with dataset name (no header)
    # Other columns are different model runs
    datasets_with_benchmarks = set()
    if benchmarks_dir.exists():
        # Search in any metrics folder
        for metric_dir in benchmarks_dir.iterdir():
            if metric_dir.is_dir():
                for csv_file in metric_dir.glob("*.csv"):
                    try:
                        # Read CSV: first column is "Resamples," then dataset names are in the first column of each row
                        # Structure: Resamples,0,1,2,... (header)
                        #           DatasetName,value1,value2,... (data)
                        df_bench = pd.read_csv(csv_file)
                        
                        if len(df_bench) > 0:
                            # First column contains dataset names
                            first_col_name = df_bench.columns[0]
                            # Get dataset names from first column
                            dataset_names = df_bench[first_col_name].astype(str).tolist()
                            
                            # Normalize names (lowercase, no spaces, no underscores, no hyphens)
                            normalized_names = [str(name).lower().replace('_', '').replace(' ', '').replace('-', '') 
                                              for name in dataset_names 
                                              if pd.notna(name) and str(name).strip() and str(name).lower() != 'resamples']
                            datasets_with_benchmarks.update(normalized_names)
                    except Exception as e:
                        continue
    
    # Add benchmark information to statistics
    for stats in simple_stats:
        dataset_name_normalized = stats['name'].lower().replace('_', '').replace(' ', '')
        stats['has_benchmark'] = dataset_name_normalized in datasets_with_benchmarks
    
    # Create DataFrame for summary (only for printing)
    df_stats = pd.DataFrame(simple_stats)
    
    # Save statistics to JSON
    stats_path = output_dir / "classification_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(simple_stats, f, indent=2, default=str)
    print(f"Statistics saved to {stats_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("CLASSIFICATION DATASETS SUMMARY")
    print("=" * 80)
    print(f"\nTotal datasets: {len(datasets)}")
    print(f"\nShape distribution:")
    if 'length' in df_stats.columns and 'n_dimensions' in df_stats.columns:
        print(df_stats[['n_samples', 'length', 'n_dimensions']].describe())
    else:
        print(df_stats[['n_samples', 'n_timesteps', 'n_channels']].describe())
    print(f"\nMissing values:")
    print(f"  Average: {df_stats['missing_pct'].mean():.2f}%")
    print(f"  Maximum: {df_stats['missing_pct'].max():.2f}%")
    print(f"  Datasets with missing values: {(df_stats['missing_pct'] > 0).sum()}")
    
    if 'n_classes' in df_stats.columns:
        valid_classes = df_stats['n_classes'].dropna()
        if len(valid_classes) > 0:
            print(f"\nClasses:")
            print(f"  Average: {valid_classes.mean():.1f}")
            print(f"  Range: {valid_classes.min():.0f} - {valid_classes.max():.0f}")
    
    if 'has_benchmark' in df_stats.columns:
        print(f"\nBenchmarks:")
        print(f"  Datasets with benchmarks: {df_stats['has_benchmark'].sum()}")
    
    return df_stats


def main():
    """Main function"""
    # Directories - ensure they are relative to project root directory
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "AEON" / "data"
    
    # Analyze classification datasets
    classification_df = analyze_classification_datasets(
        data_dir,
        max_datasets=None,  # None = all
        save_datasets=True
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED")
    print("=" * 80)
    print(f"\nResults saved to: {data_dir}")
    print(f"Total datasets analyzed: {len(classification_df)}")
    print(f"JSON file: {data_dir / 'classification_stats.json'}")


if __name__ == "__main__":
    main()

