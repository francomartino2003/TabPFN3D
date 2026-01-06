# TabPFN 3D - Real Data Analysis (AEON)

This directory contains code and data for analyzing real (non-synthetic) time series that will be used to validate the final model. The data comes from the AEON archive (UCR/UEA Time Series Classification Archive).

## Structure

```
01_real_data/
├── src/                          # Source code
│   ├── data_loader.py            # TimeSeriesDataset class to encapsulate data
│   ├── load_classification_datasets.py  # Load UCR/UEA classification datasets
│   ├── load_forecasting_datasets.py    # Load Monash forecasting datasets
│   ├── time_series_statistics.py        # Statistics calculation
│   └── analyze_all_datasets.py         # Main analysis script
├── AEON/                          # AEON data (UCR/UEA Archive)
│   ├── benchmarks/               # Bakeoff 2017 benchmarks
│   │   ├── accuracy/
│   │   ├── auroc/
│   │   ├── balacc/
│   │   ├── f1/
│   │   ├── fittime/
│   │   ├── logloss/
│   │   ├── memoryusage/
│   │   └── predicttime/
│   └── data/                      # Datasets and statistics
│       ├── classification_datasets.pkl
│       ├── classification_stats.csv
│       ├── classification_stats.json
│       ├── classification_distributions.png
│       └── classification_relationships.png
└── notebooks/
    └── 01_statistical_analysis.ipynb  # Interactive analysis
```

## Usage

### 1. Download and analyze classification datasets

```bash
python src/analyze_all_datasets.py
```

This:
- Downloads all UCR/UEA classification datasets using `aeon`
- Saves train and test separately in `AEON/data/classification_datasets.pkl`
- Calculates basic statistics (shape, length, dimensions, classes, etc.)
- Saves results in CSV and JSON
- Generates visualizations

### 2. Interactive analysis

Open `notebooks/01_statistical_analysis.ipynb` for interactive analysis.

## Datasets

### Classification (UCR/UEA Archive)
- Source: https://www.timeseriesclassification.com/dataset.php
- Total: ~181 datasets
- Loaded using `aeon.datasets.load_classification`
- Train and test are saved separately

### Forecasting (Monash Archive)
- Source: https://forecastingdata.org/
- Pending implementation

## Calculated Statistics

For each dataset, the following are calculated:
- `n_samples`: Total number of samples (train + test)
- `length`: Temporal length (timesteps)
- `n_dimensions`: Number of dimensions/variables
- `n_classes`: Number of classes (for classification)
- `train_size`: Training set size
- `test_size`: Test set size
- `missing_pct`: Percentage of missing values
- `has_variable_length`: Whether it has variable length series (padded)

## Benchmarks

Bakeoff 2017 benchmarks are saved in `AEON/benchmarks/` with metrics:
- accuracy
- auroc
- balacc
- f1
- logloss
- fittime
- predicttime
- memoryusage
