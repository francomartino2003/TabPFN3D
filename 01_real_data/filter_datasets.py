"""
Filter datasets according to specific criteria
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict

from src.data_loader import TimeSeriesDataset

def filter_datasets(
    stats_json_path: Path,
    datasets_pkl_path: Path,
    output_json_path: Path,
    output_pkl_path: Path
):
    """
    Filter datasets according to criteria:
    - n_samples (train + test) <= 10000
    - n_dimensions (features) <= 15
    - length (steps) <= 1000
    - n_classes <= 10
    """
    # Load statistics
    print("Loading statistics...")
    with open(stats_json_path, 'r') as f:
        all_stats = json.load(f)
    
    # Load datasets
    print("Loading datasets...")
    with open(datasets_pkl_path, 'rb') as f:
        all_datasets = pickle.load(f)
    
    # Crear diccionario de datasets por nombre
    datasets_dict = {ds.name: ds for ds in all_datasets}
    
    # Filter datasets
    print("\nFiltering datasets...")
    filtered_stats = []
    filtered_names = []
    
    for stats in all_stats:
        name = stats['name']
        
        # Get values
        n_samples = stats.get('n_samples', 0)
        n_dimensions = stats.get('n_dimensions', 0)
        length = stats.get('length', 0)
        n_classes = stats.get('n_classes', None)
        
        # Apply filters
        if (n_samples <= 10000 and 
            n_dimensions <= 15 and 
            length <= 1000 and 
            (n_classes is None or n_classes <= 10)):
            filtered_stats.append(stats)
            filtered_names.append(name)
    
    print(f"\nTotal datasets: {len(all_stats)}")
    print(f"Datasets that meet criteria: {len(filtered_stats)}")
    
    # Save names in JSON
    print(f"\nSaving names to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(filtered_names, f, indent=2)
    
    # Calculate statistics of filtered datasets
    if filtered_stats:
        df = pd.DataFrame(filtered_stats)
        
        print("\n" + "=" * 80)
        print("FILTERED DATASETS STATISTICS")
        print("=" * 80)
        
        print(f"\nTotal filtered datasets: {len(filtered_stats)}")
        
        print(f"\nNumber of samples (train + test):")
        print(f"  Min: {df['n_samples'].min()}")
        print(f"  Max: {df['n_samples'].max()}")
        print(f"  Mean: {df['n_samples'].mean():.1f}")
        print(f"  Median: {df['n_samples'].median():.1f}")
        
        print(f"\nNumber of dimensions (features):")
        print(f"  Min: {df['n_dimensions'].min()}")
        print(f"  Max: {df['n_dimensions'].max()}")
        print(f"  Mean: {df['n_dimensions'].mean():.1f}")
        print(f"  Median: {df['n_dimensions'].median():.1f}")
        
        print(f"\nTemporal length (steps):")
        print(f"  Min: {df['length'].min()}")
        print(f"  Max: {df['length'].max()}")
        print(f"  Mean: {df['length'].mean():.1f}")
        print(f"  Median: {df['length'].median():.1f}")
        
        valid_classes = df['n_classes'].dropna()
        if len(valid_classes) > 0:
            print(f"\nNumber of classes:")
            print(f"  Min: {valid_classes.min()}")
            print(f"  Max: {valid_classes.max()}")
            print(f"  Mean: {valid_classes.mean():.1f}")
            print(f"  Median: {valid_classes.median():.1f}")
        
        print(f"\nComplete distribution:")
        print(df[['n_samples', 'n_dimensions', 'length', 'n_classes']].describe())
    
    # Filter datasets from pkl
    print(f"\nSaving filtered datasets to {output_pkl_path}...")
    filtered_datasets = [datasets_dict[name] for name in filtered_names if name in datasets_dict]
    
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(filtered_datasets, f)
    
    print(f"Saved {len(filtered_datasets)} filtered datasets")
    
    return filtered_names, filtered_stats


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    data_dir = base_dir / "AEON" / "data"
    
    stats_json_path = data_dir / "classification_stats.json"
    datasets_pkl_path = data_dir / "classification_datasets.pkl"
    output_json_path = data_dir / "filtered_dataset_names.json"
    output_pkl_path = data_dir / "classification_datasets.pkl"  # Overwrite original
    
    filter_datasets(
        stats_json_path,
        datasets_pkl_path,
        output_json_path,
        output_pkl_path
    )





