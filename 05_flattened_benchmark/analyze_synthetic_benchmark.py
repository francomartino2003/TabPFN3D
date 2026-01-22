"""
Detailed analysis of synthetic dataset benchmark results.

This script:
1. Regenerates synthetic datasets with FULL metadata
2. Runs TabPFN benchmark on each
3. Analyzes correlations between generation hyperparameters and AUC ROC
4. Prints comprehensive tables showing relationships

Key questions:
- Why do some datasets have AUC <= 0.5?
- What parameters correlate with good/bad performance?
"""

import json
import time
import warnings
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
GENERATOR_3D_DIR = PROJECT_ROOT / "03_synthetic_generator_3D"
RESULTS_DIR = Path(__file__).parent / "results"
USABLE_DATASETS_FILE = RESULTS_DIR / "usable_datasets.json"

sys.path.insert(0, str(GENERATOR_3D_DIR))


@dataclass
class DatasetAnalysis:
    """Full analysis of a synthetic dataset."""
    # Basic info
    id: int
    name: str
    
    # Shape
    n_samples: int
    n_features: int
    length: int
    n_flat_features: int
    
    # Classification
    n_classes: int
    class_balance: float  # Min class proportion / max class proportion
    
    # Generation config
    n_nodes: int
    density: float
    n_time_inputs: int
    n_state_inputs: int
    n_roots: int
    
    # State inputs
    state_lags: List[int]  # Lags of each state input
    mean_state_lag: float
    max_state_lag: int
    
    # Time activations
    time_activations: List[str]
    has_periodic: bool  # sin, cos
    has_nonlinear: bool  # tanh, exp, log, etc.
    
    # Transformations
    transform_probs: Dict[str, float]
    dominant_transform: str
    prob_nn: float
    prob_tree: float
    prob_disc: float
    
    # Target config
    target_offset_type: str
    target_offset: int
    is_forecasting: bool  # target_offset != 0
    
    # Sample mode
    sample_mode: str
    window_stride: int
    
    # Post-processing
    apply_warping: bool
    warping_intensity: float
    apply_quantization: bool
    apply_missing: bool
    
    # Feature-target relationship
    feature_nodes: List[int]
    target_node: int
    target_in_features: bool  # For forecasting scenarios
    
    # Noise parameters
    edge_noise_prob: float
    noise_scale: float
    state_alpha: float
    
    # Distance preference
    spatial_distance_alpha: float
    
    # Tree params (if applicable)
    tree_depth: int
    
    # Benchmark results
    accuracy: Optional[float] = None
    roc_auc: Optional[float] = None
    train_time: Optional[float] = None
    benchmark_status: str = "pending"


def load_real_stats() -> Dict:
    """Load real dataset statistics for prior configuration."""
    if not USABLE_DATASETS_FILE.exists():
        raise FileNotFoundError(f"Usable datasets file not found: {USABLE_DATASETS_FILE}")
    
    with open(USABLE_DATASETS_FILE, 'r') as f:
        data = json.load(f)
    
    datasets = data['datasets']
    
    return {
        'n_samples': {
            'p25': int(np.percentile([d['n_samples'] for d in datasets], 25)),
            'p75': int(np.percentile([d['n_samples'] for d in datasets], 75)),
        },
        'n_features': {
            'max': max(d['n_dimensions'] for d in datasets),
            'prob_univariate': sum(1 for d in datasets if d['n_dimensions'] == 1) / len(datasets),
        },
        'length': {
            'p25': max(10, int(np.percentile([d['length'] for d in datasets], 25))),
            'p75': int(np.percentile([d['length'] for d in datasets], 75)),
        },
    }


def create_prior(real_stats: Dict):
    """Create PriorConfig3D matching real data distributions."""
    from config import PriorConfig3D
    
    return PriorConfig3D(
        # Size constraints
        max_samples=10000,
        max_features=12,
        max_t_subseq=500,
        max_T_total=600,
        max_classes=6,
        max_complexity=50_000_000,
        
        # Sample size
        n_samples_range=(real_stats['n_samples']['p25'], real_stats['n_samples']['p75']),
        
        # Features
        prob_univariate=real_stats['n_features']['prob_univariate'],
        n_features_beta_a=1.0,
        n_features_beta_b=3.0,
        n_features_range=(2, real_stats['n_features']['max']),
        
        # Temporal
        T_total_range=(real_stats['length']['p25'], real_stats['length']['p75'] + 50),
        t_subseq_range=(real_stats['length']['p25'], real_stats['length']['p75']),
        
        # Graph (v3: distance preference controls complexity)
        n_nodes_range=(8, 30),
        density_range=(0.1, 0.6),
        n_roots_range=(4, 18),
        min_roots_fraction=0.25,
        max_roots_fraction=0.60,
        
        # Roots (v3: favor state inputs)
        min_time_inputs=2,
        min_state_inputs=2,
        time_fraction_range=(0.20, 0.40),
        
        # State inputs
        state_lag_range=(1, 8),
        state_lag_distribution='geometric',
        state_lag_geometric_p=0.4,
        state_alpha_range=(0.3, 0.8),
        
        # Transformations
        prob_nn_transform=0.50,
        prob_tree_transform=0.30,
        prob_discretization=0.20,
        prob_identity_activation=0.5,
        
        # Noise (v3: variable per dataset)
        prob_edge_noise=0.05,
        noise_scale_range=(0.001, 0.08),
        prob_low_noise_dataset=0.3,
        low_noise_scale_max=0.005,
        
        # Distance preference (v3)
        distance_alpha=1.5,
        
        # Sample mode
        prob_iid_mode=0.5,
        prob_sliding_window_mode=0.3,
        prob_mixed_mode=0.2,
        
        # Classification only
        prob_classification=1.0,
        force_classification=True,
        min_samples_per_class=25,
        max_target_offset=20,
        prob_future=0.75,
        
        # Post-processing
        prob_warping=0.15,
        warping_intensity_range=(0.1, 0.25),
        prob_quantization=0.1,
        prob_missing_values=0.0,
        
        train_ratio_range=(0.4, 0.7),
    )


def generate_datasets_with_full_metadata(n_datasets: int, seed: int = 42) -> List[DatasetAnalysis]:
    """Generate datasets and extract ALL configuration metadata."""
    from generator import SyntheticDatasetGenerator3D
    from config import DatasetConfig3D
    
    real_stats = load_real_stats()
    prior = create_prior(real_stats)
    generator = SyntheticDatasetGenerator3D(prior=prior, seed=seed)
    
    analyses = []
    
    for i, dataset in enumerate(tqdm(generator.generate_many(n_datasets), 
                                      total=n_datasets, desc="Generating")):
        config = dataset.config
        metadata = dataset.metadata
        
        # Calculate class balance
        unique, counts = np.unique(dataset.y, return_counts=True)
        class_balance = counts.min() / counts.max() if len(counts) > 1 else 1.0
        
        # Extract state lags from state_configs
        state_lags = [lag for (_, lag) in config.state_configs]
        
        # Check time activations
        has_periodic = any(a in ['sin_1', 'sin_2', 'sin_3', 'sin_5', 'cos_1', 'cos_2', 'cos_3', 'cos_5'] 
                         for a in config.time_activations)
        has_nonlinear = any(a in ['tanh', 'exp_decay', 'log', 'quadratic', 'cubic'] 
                          for a in config.time_activations)
        
        # Dominant transform
        dominant_transform = max(config.transform_probs.keys(), 
                                key=lambda k: config.transform_probs[k])
        
        # Target in features?
        feature_nodes = metadata.get('feature_nodes', [])
        target_node = metadata.get('target_node', -1)
        target_in_features = target_node in feature_nodes
        
        analysis = DatasetAnalysis(
            id=i,
            name=f"synthetic_{i:04d}",
            
            # Shape
            n_samples=dataset.shape[0],
            n_features=dataset.shape[1],
            length=dataset.shape[2],
            n_flat_features=dataset.shape[1] * dataset.shape[2],
            
            # Classification
            n_classes=dataset.n_classes,
            class_balance=class_balance,
            
            # Generation config
            n_nodes=config.n_nodes,
            density=config.density,
            n_time_inputs=config.n_time_inputs,
            n_state_inputs=config.n_state_inputs,
            n_roots=config.n_time_inputs + config.n_state_inputs,
            
            # State inputs
            state_lags=state_lags,
            mean_state_lag=np.mean(state_lags) if state_lags else 0,
            max_state_lag=max(state_lags) if state_lags else 0,
            
            # Time activations
            time_activations=config.time_activations,
            has_periodic=has_periodic,
            has_nonlinear=has_nonlinear,
            
            # Transformations
            transform_probs=config.transform_probs,
            dominant_transform=dominant_transform,
            prob_nn=config.transform_probs.get('nn', 0),
            prob_tree=config.transform_probs.get('tree', 0),
            prob_disc=config.transform_probs.get('discretization', 0),
            
            # Target config (derive type from offset value)
            target_offset_type='within' if config.target_offset == 0 else ('future' if config.target_offset > 0 else 'past'),
            target_offset=config.target_offset,
            is_forecasting=config.target_offset != 0,
            
            # Sample mode
            sample_mode=config.sample_mode,
            window_stride=config.window_stride,
            
            # Post-processing
            apply_warping=config.apply_warping,
            warping_intensity=config.warping_intensity,
            apply_quantization=config.apply_quantization,
            apply_missing=config.apply_missing,
            
            # Feature-target relationship
            feature_nodes=feature_nodes,
            target_node=target_node,
            target_in_features=target_in_features,
            
            # Noise
            edge_noise_prob=config.edge_noise_prob,
            noise_scale=config.noise_scale,
            state_alpha=config.state_alpha,
            
            # Distance preference
            spatial_distance_alpha=config.spatial_distance_alpha,
            
            # Tree
            tree_depth=config.tree_depth,
        )
        
        # Store X and y for benchmarking
        analysis._X = dataset.X
        analysis._y = dataset.y
        
        analyses.append(analysis)
    
    return analyses


def flatten_3d_to_2d(X: np.ndarray) -> np.ndarray:
    """Flatten 3D array to 2D."""
    n_samples, n_features, n_timesteps = X.shape
    return X.reshape(n_samples, n_features * n_timesteps)


def benchmark_dataset(analysis: DatasetAnalysis, test_ratio: float = 0.3) -> DatasetAnalysis:
    """Run TabPFN on a single dataset."""
    from tabpfn import TabPFNClassifier
    
    X = analysis._X
    y = analysis._y
    
    # Stratified split
    unique_classes = np.unique(y)
    train_indices = []
    test_indices = []
    
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        n_cls = len(cls_indices)
        n_cls_test = max(1, int(n_cls * test_ratio))
        
        np.random.shuffle(cls_indices)
        train_indices.extend(cls_indices[:-n_cls_test])
        test_indices.extend(cls_indices[-n_cls_test:])
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # Flatten
    X_train_flat = flatten_3d_to_2d(X_train)
    X_test_flat = flatten_3d_to_2d(X_test)
    
    # Handle NaN
    X_train_flat = np.nan_to_num(X_train_flat, nan=0.0)
    X_test_flat = np.nan_to_num(X_test_flat, nan=0.0)
    
    # Encode labels
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Subsample if needed
    if len(y_train_enc) > 10000:
        idx = np.random.choice(len(y_train_enc), size=10000, replace=False)
        X_train_flat = X_train_flat[idx]
        y_train_enc = y_train_enc[idx]
    
    try:
        start_time = time.time()
        clf = TabPFNClassifier()
        clf.fit(X_train_flat, y_train_enc)
        
        y_pred = clf.predict(X_test_flat)
        y_proba = clf.predict_proba(X_test_flat)
        
        train_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test_enc, y_pred)
        
        if analysis.n_classes == 2:
            roc_auc = roc_auc_score(y_test_enc, y_proba[:, 1])
        else:
            try:
                roc_auc = roc_auc_score(y_test_enc, y_proba, multi_class='ovr', average='weighted')
            except:
                roc_auc = None
        
        analysis.accuracy = accuracy
        analysis.roc_auc = roc_auc
        analysis.train_time = train_time
        analysis.benchmark_status = "success"
        
    except Exception as e:
        analysis.benchmark_status = f"error: {str(e)}"
    
    return analysis


def analyze_correlations(analyses: List[DatasetAnalysis]) -> pd.DataFrame:
    """Convert to DataFrame for correlation analysis."""
    data = []
    
    for a in analyses:
        if a.benchmark_status != "success":
            continue
            
        row = {
            'id': a.id,
            'roc_auc': a.roc_auc,
            'accuracy': a.accuracy,
            'n_samples': a.n_samples,
            'n_features': a.n_features,
            'length': a.length,
            'n_flat_features': a.n_flat_features,
            'n_classes': a.n_classes,
            'class_balance': a.class_balance,
            'n_nodes': a.n_nodes,
            'density': a.density,
            'n_time_inputs': a.n_time_inputs,
            'n_state_inputs': a.n_state_inputs,
            'n_roots': a.n_roots,
            'mean_state_lag': a.mean_state_lag,
            'max_state_lag': a.max_state_lag,
            'has_periodic': int(a.has_periodic),
            'has_nonlinear': int(a.has_nonlinear),
            'prob_nn': a.prob_nn,
            'prob_tree': a.prob_tree,
            'prob_disc': a.prob_disc,
            'target_offset': a.target_offset,
            'is_forecasting': int(a.is_forecasting),
            'target_offset_type': a.target_offset_type,
            'sample_mode': a.sample_mode,
            'window_stride': a.window_stride,
            'apply_warping': int(a.apply_warping),
            'warping_intensity': a.warping_intensity,
            'apply_quantization': int(a.apply_quantization),
            'target_in_features': int(a.target_in_features),
            'edge_noise_prob': a.edge_noise_prob,
            'noise_scale': a.noise_scale,
            'state_alpha': a.state_alpha,
            'spatial_distance_alpha': a.spatial_distance_alpha,
            'tree_depth': a.tree_depth,
        }
        data.append(row)
    
    return pd.DataFrame(data)


def print_detailed_results(analyses: List[DatasetAnalysis]):
    """Print detailed per-dataset results."""
    print("\n" + "=" * 120)
    print("DETAILED PER-DATASET RESULTS")
    print("=" * 120)
    
    # Sort by AUC ROC
    sorted_analyses = sorted(
        [a for a in analyses if a.roc_auc is not None],
        key=lambda x: x.roc_auc
    )
    
    print(f"\n{'ID':>4} | {'AUC':>6} | {'ACC':>6} | {'Shape':<20} | {'Cls':>3} | {'Bal':>5} | "
          f"{'Nodes':>5} | {'Mode':<12} | {'TargOff':>8} | {'Forecast':>8} | {'NN':>4} | {'Tree':>4}")
    print("-" * 120)
    
    for a in sorted_analyses:
        shape_str = f"({a.n_samples},{a.n_features},{a.length})"
        target_off = f"{a.target_offset_type}({a.target_offset})"
        
        # Color indicator for AUC
        if a.roc_auc <= 0.5:
            status = "❌"
        elif a.roc_auc < 0.6:
            status = "⚠️"
        else:
            status = "✅"
        
        print(f"{a.id:>4} | {a.roc_auc:.4f} | {a.accuracy:.4f} | {shape_str:<20} | {a.n_classes:>3} | "
              f"{a.class_balance:.3f} | {a.n_nodes:>5} | {a.sample_mode:<12} | {target_off:>8} | "
              f"{'Yes' if a.is_forecasting else 'No':>8} | {a.prob_nn:.2f} | {a.prob_tree:.2f} {status}")


def print_correlation_analysis(df: pd.DataFrame):
    """Print correlation analysis between parameters and performance."""
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS: Parameters vs ROC AUC")
    print("=" * 80)
    
    # Numeric correlations
    numeric_cols = [
        'n_samples', 'n_features', 'length', 'n_flat_features', 'n_classes',
        'class_balance', 'n_nodes', 'density', 'n_time_inputs', 'n_state_inputs',
        'mean_state_lag', 'max_state_lag', 'prob_nn', 'prob_tree', 'prob_disc',
        'target_offset', 'is_forecasting', 'has_periodic', 'has_nonlinear',
        'apply_warping', 'warping_intensity', 'apply_quantization',
        'edge_noise_prob', 'noise_scale', 'state_alpha', 'tree_depth'
    ]
    
    correlations = []
    for col in numeric_cols:
        if col in df.columns and df[col].std() > 0:
            corr = df['roc_auc'].corr(df[col])
            correlations.append((col, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\nCorrelation with ROC AUC (sorted by absolute value):")
    print("-" * 50)
    for col, corr in correlations:
        indicator = "+" if corr > 0 else "-"
        bar_len = int(abs(corr) * 30)
        bar = "█" * bar_len
        print(f"{col:<25} {corr:>7.4f} {indicator} {bar}")


def print_group_analysis(df: pd.DataFrame):
    """Analyze performance by categorical groups."""
    print("\n" + "=" * 80)
    print("GROUP ANALYSIS: Average AUC by Category")
    print("=" * 80)
    
    # By sample mode
    print("\n📊 By Sample Mode:")
    print(df.groupby('sample_mode')['roc_auc'].agg(['mean', 'std', 'count']).round(4))
    
    # By target offset type
    print("\n📊 By Target Offset Type:")
    print(df.groupby('target_offset_type')['roc_auc'].agg(['mean', 'std', 'count']).round(4))
    
    # By forecasting
    print("\n📊 By Forecasting (target_offset != 0):")
    print(df.groupby('is_forecasting')['roc_auc'].agg(['mean', 'std', 'count']).round(4))
    
    # By n_classes
    print("\n📊 By Number of Classes:")
    print(df.groupby('n_classes')['roc_auc'].agg(['mean', 'std', 'count']).round(4))
    
    # By n_features (univariate vs multivariate)
    df['is_univariate'] = df['n_features'] == 1
    print("\n📊 By Univariate vs Multivariate:")
    print(df.groupby('is_univariate')['roc_auc'].agg(['mean', 'std', 'count']).round(4))
    
    # High AUC vs Low AUC comparison
    print("\n" + "=" * 80)
    print("HIGH vs LOW AUC COMPARISON")
    print("=" * 80)
    
    df['auc_category'] = pd.cut(df['roc_auc'], bins=[0, 0.5, 0.65, 0.8, 1.0], 
                                labels=['Poor (≤0.5)', 'Fair (0.5-0.65)', 'Good (0.65-0.8)', 'Excellent (>0.8)'])
    
    comparison_cols = ['n_samples', 'n_features', 'length', 'n_classes', 'class_balance', 
                       'n_nodes', 'density', 'mean_state_lag', 'is_forecasting']
    
    print("\nMean values by AUC category:")
    print(df.groupby('auc_category')[comparison_cols].mean().round(3))


def print_poor_auc_analysis(analyses: List[DatasetAnalysis]):
    """Deep dive into datasets with poor AUC."""
    print("\n" + "=" * 80)
    print("🔍 DEEP DIVE: Datasets with AUC ≤ 0.5 (Worse than random)")
    print("=" * 80)
    
    poor_datasets = [a for a in analyses if a.roc_auc is not None and a.roc_auc <= 0.5]
    
    if not poor_datasets:
        print("\n✅ No datasets with AUC ≤ 0.5!")
        return
    
    print(f"\nFound {len(poor_datasets)} datasets with poor AUC:\n")
    
    for a in poor_datasets:
        print(f"\n--- Dataset {a.id} (AUC={a.roc_auc:.4f}, ACC={a.accuracy:.4f}) ---")
        print(f"  Shape: ({a.n_samples}, {a.n_features}, {a.length}) → {a.n_flat_features} flat features")
        print(f"  Classes: {a.n_classes}, Balance: {a.class_balance:.3f}")
        print(f"  DAG: {a.n_nodes} nodes, density={a.density:.2f}")
        print(f"  Roots: {a.n_time_inputs} time + {a.n_state_inputs} state")
        print(f"  State lags: {a.state_lags} (mean={a.mean_state_lag:.1f})")
        print(f"  Time activations: {a.time_activations}")
        print(f"  Transforms: NN={a.prob_nn:.2f}, Tree={a.prob_tree:.2f}, Disc={a.prob_disc:.2f}")
        print(f"  Target: node={a.target_node}, offset={a.target_offset_type}({a.target_offset})")
        print(f"  Features: nodes={a.feature_nodes}, target_in_features={a.target_in_features}")
        print(f"  Sample mode: {a.sample_mode}, stride={a.window_stride}")
        print(f"  Noise: edge_prob={a.edge_noise_prob:.3f}, scale={a.noise_scale:.4f}")
        print(f"  Post-proc: warping={a.apply_warping} ({a.warping_intensity:.2f}), quant={a.apply_quantization}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze synthetic dataset benchmark")
    parser.add_argument('--n-datasets', type=int, default=20, 
                        help='Number of datasets to generate and analyze')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 80)
    print("SYNTHETIC DATASET BENCHMARK ANALYSIS")
    print("=" * 80)
    
    np.random.seed(args.seed)
    
    # Generate datasets
    print(f"\n[1/4] Generating {args.n_datasets} synthetic datasets with full metadata...")
    analyses = generate_datasets_with_full_metadata(args.n_datasets, args.seed)
    
    # Benchmark each
    print(f"\n[2/4] Running TabPFN benchmark on each dataset...")
    for analysis in tqdm(analyses, desc="Benchmarking"):
        benchmark_dataset(analysis)
    
    # Convert to DataFrame
    print(f"\n[3/4] Analyzing results...")
    df = analyze_correlations(analyses)
    
    # Print all analyses
    print_detailed_results(analyses)
    print_correlation_analysis(df)
    print_group_analysis(df)
    print_poor_auc_analysis(analyses)
    
    # Save results
    print(f"\n[4/4] Saving detailed results...")
    output_path = RESULTS_DIR / "synthetic_analysis_detailed.csv"
    df.to_csv(output_path, index=False)
    print(f"    Saved to: {output_path}")
    
    # Summary stats
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    successful = df[df['roc_auc'].notna()]
    print(f"Total datasets: {len(analyses)}")
    print(f"Successful: {len(successful)}")
    print(f"Mean AUC: {successful['roc_auc'].mean():.4f} ± {successful['roc_auc'].std():.4f}")
    print(f"Mean Accuracy: {successful['accuracy'].mean():.4f} ± {successful['accuracy'].std():.4f}")
    print(f"AUC ≤ 0.5: {(successful['roc_auc'] <= 0.5).sum()} datasets")
    print(f"AUC ≤ 0.6: {(successful['roc_auc'] <= 0.6).sum()} datasets")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
