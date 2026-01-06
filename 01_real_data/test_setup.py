"""
Test script to verify that configuration works correctly
"""
import sys
from pathlib import Path

print("=" * 80)
print("CONFIGURATION TEST - TabPFN 3D")
print("=" * 80)

# Check imports
print("\n1. Checking imports...")
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from statsmodels.tsa.stattools import adfuller
    print("   ✓ Basic imports OK")
except ImportError as e:
    print(f"   ✗ Error in basic imports: {e}")
    sys.exit(1)

try:
    from aeon.datasets import load_classification, get_dataset_names
    print("   ✓ aeon installed correctly")
except ImportError as e:
    print(f"   ✗ aeon not installed: {e}")
    print("   Install with: pip install aeon")
    sys.exit(1)

# Check project modules
print("\n2. Checking project modules...")
try:
    from src.data_loader import TimeSeriesDataset
    from src.time_series_statistics import TimeSeriesStatistics
    from src.load_classification_datasets import (
        get_all_classification_datasets,
        load_single_classification_dataset
    )
    print("   ✓ Project modules OK")
except ImportError as e:
    print(f"   ✗ Error in project modules: {e}")
    sys.exit(1)

# Check folder structure
print("\n3. Checking folder structure...")
folders = [
    "data/real/classification",
    "data/real/forecasting",
    "data/real/metadata",
    "src",
    "notebooks"
]

all_ok = True
for folder in folders:
    path = Path(folder)
    if path.exists():
        print(f"   ✓ {folder}")
    else:
        print(f"   ✗ {folder} - DOES NOT EXIST")
        all_ok = False

if not all_ok:
    print("\n   Creating missing folders...")
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
    print("   ✓ Folders created")

# Test loading a dataset
print("\n4. Testing example dataset loading...")
try:
    dataset = load_single_classification_dataset("GunPoint", verbose=True)
    if dataset:
        print(f"   ✓ Dataset loaded: {dataset}")
        print(f"   ✓ Shape: {dataset.X.shape}")
        print(f"   ✓ Info: {dataset.get_info()}")
        
        # Test statistics calculation
        print("\n5. Testing statistics calculation...")
        stats_dict = TimeSeriesStatistics.compute_dataset_stats(dataset)
        print(f"   ✓ Statistics calculated")
        print(f"   ✓ Number of samples: {stats_dict['n_samples']}")
        print(f"   ✓ Temporal length: {stats_dict['n_timesteps']}")
        print(f"   ✓ Number of channels: {stats_dict['n_channels']}")
    else:
        print("   ✗ Could not load dataset")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Check list of available datasets
print("\n6. Checking list of available datasets...")
try:
    dataset_names = get_all_classification_datasets()
    print(f"   ✓ {len(dataset_names)} datasets available")
    print(f"   ✓ First 5: {dataset_names[:5]}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 80)
print("CONFIGURATION COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("1. Run: python src/analyze_all_datasets.py")
print("2. Or open: notebooks/01_statistical_analysis.ipynb")
print("\n")

