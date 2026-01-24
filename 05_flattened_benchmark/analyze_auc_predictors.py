"""
Generate synthetic datasets with FULL metadata, benchmark them,
and analyze what predicts AUC using interpretable models.
"""

import sys
import json
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / '03_synthetic_generator_3D'))
sys.path.insert(0, str(Path(__file__).parent.parent / '00_TabPFN' / 'src'))

from config import PriorConfig3D, DatasetConfig3D
from generator import SyntheticDatasetGenerator3D
from tabpfn import TabPFNClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def create_prior() -> PriorConfig3D:
    """Create prior with current config (v5: optimized t_ratio, less noise)."""
    return PriorConfig3D(
        # Size constraints
        max_samples=1000,
        max_features=12,
        max_t_subseq=500,
        max_T_total=600,
        max_classes=6,
        
        # Samples
        n_samples_range=(200, 3000),
        
        # Features
        prob_univariate=0.65,
        n_features_beta_a=1.5,
        n_features_beta_b=4.0,
        n_features_range=(2, 8),
        
        # Temporal - T_total will be optimized in config.py
        T_total_range=(100, 400),
        t_subseq_range=(50, 300),
        
        # Graph - FEW inputs, random mix
        n_nodes_range=(5, 15),
        density_range=(0.1, 0.6),
        n_roots_range=(2, 6),
        min_roots_fraction=0.25,
        max_roots_fraction=0.60,
        
        # Root input distribution - random (IID forces >=1 state in config.py)
        min_time_inputs=0,
        min_state_inputs=0,  # IID mode will force >=1 state input
        time_fraction_range=(0.0, 1.0),
        
        # State params - shorter lags
        state_lag_range=(1, 5),
        state_lag_geometric_p=0.6,  # Higher p = shorter lags
        state_alpha_range=(0.3, 0.8),  # Legacy (not used)
        
        # Transformations
        prob_nn_transform=0.40,
        prob_tree_transform=0.45,
        prob_discretization=0.15,
        
        # Noise - REDUCED
        prob_edge_noise=0.03,
        noise_scale_range=(0.001, 0.02),
        prob_low_noise_dataset=0.4,
        low_noise_scale_max=0.003,
        
        # Distance preference
        distance_alpha=1.5,
        
        # Sampling mode - MORE IID
        prob_iid_mode=0.60,
        prob_sliding_window_mode=0.25,
        prob_mixed_mode=0.15,
        
        # Target - smaller offset for better t_ratio
        prob_classification=1.0,
        force_classification=True,
        min_samples_per_class=25,
        max_target_offset=15,
        prob_future=0.75,
        
        # Train/test
        train_ratio_range=(0.4, 0.7),
    )


@dataclass
class DatasetFeatures:
    """All features extracted from a generated dataset for analysis."""
    id: int
    
    # Basic shape
    n_samples: int
    n_features: int
    t_subseq: int
    n_classes: int
    flat_features: int
    
    # Graph structure
    n_nodes: int
    n_roots: int
    density: float
    
    # Input types
    n_time_inputs: int
    n_state_inputs: int
    time_input_ratio: float  # n_time / n_roots
    
    # Target info
    target_node: int
    target_offset: int
    target_offset_abs: int
    is_future: bool
    is_within: bool
    
    # DAG distances
    target_depth: int  # distance from inputs to target
    min_feature_to_target_dist: int
    max_feature_to_target_dist: int
    mean_feature_to_target_dist: float
    
    # State input details
    mean_state_lag: float
    max_state_lag: int
    state_alpha: float
    
    # Noise
    noise_scale: float
    is_low_noise: bool
    
    # Temporal
    T_total: int
    t_ratio: float  # t_subseq / T_total
    
    # Sampling mode
    sample_mode: str
    is_iid: bool
    is_sliding: bool
    
    # Complexity proxies
    n_internal_nodes: int  # n_nodes - n_roots
    edges_per_node: float
    
    # Time activations
    n_unique_time_activations: int
    has_linear_time: bool
    has_sin_time: bool
    has_exp_time: bool
    
    # Results (filled after benchmark)
    auc: Optional[float] = None
    accuracy: Optional[float] = None
    error: Optional[str] = None


def extract_features(dataset, config: DatasetConfig3D, dataset_id: int) -> DatasetFeatures:
    """Extract all analyzable features from a generated dataset."""
    meta = dataset.metadata
    
    # Basic
    n_samples = dataset.X.shape[0]
    n_features = dataset.X.shape[1]
    t_subseq = dataset.X.shape[2]
    n_classes = dataset.n_classes
    
    # Graph
    n_nodes = meta.get('n_nodes', config.n_nodes)
    n_time = config.n_time_inputs
    n_state = config.n_state_inputs
    n_roots = n_time + n_state  # Computed from inputs
    density = config.density
    
    # Inputs - already have n_time and n_state
    time_ratio = n_time / max(n_roots, 1)
    
    # Target
    target_node = meta.get('target_node', -1)
    target_offset = config.target_offset
    
    # Feature nodes
    feature_nodes = meta.get('feature_nodes', [])
    
    # DAG distances - compute from adjacency if available
    dag = meta.get('dag_adjacency', None)
    if dag is not None and target_node >= 0:
        target_depth = _compute_depth(dag, target_node, n_roots)
        distances = [_compute_distance(dag, f, target_node) for f in feature_nodes]
        min_dist = min(distances) if distances else 0
        max_dist = max(distances) if distances else 0
        mean_dist = np.mean(distances) if distances else 0
    else:
        target_depth = n_nodes // 2
        min_dist = max_dist = 1
        mean_dist = 1.0
    
    # State lags from config.state_configs (list of tuples: (source_node, lag))
    state_configs = config.state_configs if hasattr(config, 'state_configs') else []
    if state_configs:
        lags = [sc[1] if isinstance(sc, tuple) else 1 for sc in state_configs]
        mean_lag = np.mean(lags) if lags else 0
        max_lag = max(lags) if lags else 0
    else:
        mean_lag = 0
        max_lag = 0
    
    state_alpha = config.state_alpha
    
    # Noise
    noise_scale = config.noise_scale
    is_low_noise = noise_scale < 0.01
    
    # Temporal
    T_total = config.T_total
    t_ratio = t_subseq / max(T_total, 1)
    
    # Mode
    sample_mode = config.sample_mode
    
    # Time activations
    time_acts = config.time_activations if hasattr(config, 'time_activations') else []
    n_unique_acts = len(set(time_acts))
    has_linear = any('linear' in str(a) for a in time_acts)
    has_sin = any('sin' in str(a) for a in time_acts)
    has_exp = any('exp' in str(a) for a in time_acts)
    
    return DatasetFeatures(
        id=dataset_id,
        n_samples=n_samples,
        n_features=n_features,
        t_subseq=t_subseq,
        n_classes=n_classes,
        flat_features=n_features * t_subseq,
        n_nodes=n_nodes,
        n_roots=n_roots,
        density=density,
        n_time_inputs=n_time,
        n_state_inputs=n_state,
        time_input_ratio=time_ratio,
        target_node=target_node,
        target_offset=target_offset,
        target_offset_abs=abs(target_offset),
        is_future=target_offset > 0,
        is_within=target_offset == 0,
        target_depth=target_depth,
        min_feature_to_target_dist=min_dist,
        max_feature_to_target_dist=max_dist,
        mean_feature_to_target_dist=mean_dist,
        mean_state_lag=mean_lag,
        max_state_lag=max_lag,
        state_alpha=state_alpha,
        noise_scale=noise_scale,
        is_low_noise=is_low_noise,
        T_total=T_total,
        t_ratio=t_ratio,
        sample_mode=sample_mode,
        is_iid=(sample_mode == 'iid'),
        is_sliding=(sample_mode == 'sliding_window'),
        n_internal_nodes=n_nodes - n_roots,
        edges_per_node=density * (n_nodes - 1),
        n_unique_time_activations=n_unique_acts,
        has_linear_time=has_linear,
        has_sin_time=has_sin,
        has_exp_time=has_exp,
    )


def _compute_depth(dag, node, n_roots):
    """Compute depth of node from root inputs."""
    # Simple BFS from roots
    if dag is None:
        return 0
    try:
        # dag is adjacency matrix or dict
        visited = set()
        queue = [(i, 0) for i in range(n_roots)]
        while queue:
            n, d = queue.pop(0)
            if n == node:
                return d
            if n in visited:
                continue
            visited.add(n)
            # Get children
            if hasattr(dag, 'shape'):
                children = np.where(dag[n] > 0)[0]
            else:
                children = dag.get(n, [])
            for c in children:
                queue.append((c, d + 1))
        return len(visited)
    except:
        return 0


def _compute_distance(dag, node1, node2):
    """Compute shortest path distance between two nodes."""
    if dag is None or node1 == node2:
        return 0
    try:
        # BFS
        visited = {node1: 0}
        queue = [node1]
        while queue:
            n = queue.pop(0)
            if n == node2:
                return visited[n]
            # Get neighbors (both directions)
            if hasattr(dag, 'shape'):
                neighbors = list(np.where(dag[n] > 0)[0]) + list(np.where(dag[:, n] > 0)[0])
            else:
                neighbors = dag.get(n, [])
            for nb in neighbors:
                if nb not in visited:
                    visited[nb] = visited[n] + 1
                    queue.append(nb)
        return 10  # Not connected
    except:
        return 1


def benchmark_dataset(X: np.ndarray, y: np.ndarray, sample_mode: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Run TabPFN on a single dataset and return AUC, accuracy."""
    try:
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))
        
        # Train/test split based on mode
        if sample_mode in ['sliding_window', 'mixed']:
            # Temporal split
            n_train = int(0.7 * n_samples)
            train_idx = np.arange(n_train)
            test_idx = np.arange(n_train, n_samples)
        else:
            # Stratified split
            train_idx, test_idx = train_test_split(
                np.arange(n_samples), test_size=0.3, stratify=y, random_state=42
            )
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Flatten
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Handle missing values
        train_means = np.nanmean(X_train_flat, axis=0)
        train_means = np.nan_to_num(train_means, nan=0.0)
        
        X_train_flat = np.where(np.isnan(X_train_flat), train_means, X_train_flat)
        X_test_flat = np.where(np.isnan(X_test_flat), train_means, X_test_flat)
        X_train_flat = np.nan_to_num(X_train_flat, nan=0.0)
        X_test_flat = np.nan_to_num(X_test_flat, nan=0.0)
        
        # Check for constant features
        stds = np.std(X_train_flat, axis=0)
        if np.all(stds < 1e-8):
            return None, None, "All features constant"
        
        # Encode labels
        le = LabelEncoder()
        le.fit(y_train)
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)
        
        # TabPFN
        clf = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)
        clf.fit(X_train_flat, y_train_enc)
        
        y_pred = clf.predict(X_test_flat)
        acc = accuracy_score(y_test_enc, y_pred)
        
        # AUC
        try:
            if n_classes == 2:
                y_proba = clf.predict_proba(X_test_flat)[:, 1]
                auc = roc_auc_score(y_test_enc, y_proba)
            else:
                y_proba = clf.predict_proba(X_test_flat)
                auc = roc_auc_score(y_test_enc, y_proba, multi_class='ovr', average='weighted')
        except:
            auc = None
        
        return auc, acc, None
        
    except Exception as e:
        return None, None, str(e)[:50]


def generate_and_benchmark(n_datasets: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate datasets, extract features, run benchmark."""
    print(f"\n{'='*70}")
    print(f"GENERATING {n_datasets} DATASETS WITH FULL METADATA")
    print(f"{'='*70}\n")
    
    prior = create_prior()
    generator = SyntheticDatasetGenerator3D(prior, seed=seed)
    
    all_features = []
    
    for i in tqdm(range(n_datasets), desc="Generating & Benchmarking"):
        try:
            # Generate
            config = DatasetConfig3D.sample_from_prior(prior, np.random.default_rng(seed + i))
            dataset = generator.generate(config)
            
            # Extract features
            features = extract_features(dataset, config, i)
            
            # Benchmark
            auc, acc, error = benchmark_dataset(dataset.X, dataset.y, features.sample_mode)
            features.auc = auc
            features.accuracy = acc
            features.error = error
            
            all_features.append(asdict(features))
            
            # Progress print every 10
            if (i + 1) % 10 == 0:
                valid = [f for f in all_features if f['auc'] is not None]
                if valid:
                    mean_auc = np.mean([f['auc'] for f in valid])
                    print(f"  [{i+1}/{n_datasets}] Mean AUC so far: {mean_auc:.3f}")
                    
        except Exception as e:
            print(f"  Error on dataset {i}: {e}")
            continue
    
    return pd.DataFrame(all_features)


def analyze_predictors(df: pd.DataFrame):
    """Analyze what predicts AUC using interpretable models."""
    
    print(f"\n{'='*70}")
    print("ANALYSIS: WHAT PREDICTS AUC?")
    print(f"{'='*70}\n")
    
    # Filter valid
    df_valid = df[df['auc'].notna()].copy()
    print(f"Valid datasets: {len(df_valid)}/{len(df)}")
    
    # Feature columns (numeric only)
    feature_cols = [
        'n_samples', 'n_features', 't_subseq', 'n_classes', 'flat_features',
        'n_nodes', 'n_roots', 'density',
        'n_time_inputs', 'n_state_inputs', 'time_input_ratio',
        'target_offset', 'target_offset_abs', 'is_future', 'is_within',
        'target_depth', 'min_feature_to_target_dist', 'max_feature_to_target_dist',
        'mean_feature_to_target_dist',
        'mean_state_lag', 'max_state_lag', 'state_alpha',
        'noise_scale', 'is_low_noise',
        'T_total', 't_ratio',
        'is_iid', 'is_sliding',
        'n_internal_nodes', 'edges_per_node',
        'n_unique_time_activations', 'has_linear_time', 'has_sin_time', 'has_exp_time'
    ]
    
    # Prepare X, y
    X = df_valid[feature_cols].values.astype(float)
    y = df_valid['auc'].values
    
    # Handle NaN in features
    X = np.nan_to_num(X, nan=0.0)
    
    # 1. Correlations
    print("\n" + "="*50)
    print("1. CORRELATIONS WITH AUC")
    print("="*50)
    correlations = []
    for i, col in enumerate(feature_cols):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        if not np.isnan(corr):
            correlations.append((col, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print("\nTop correlations (sorted by |r|):")
    for col, corr in correlations[:15]:
        bar = "+" * int(abs(corr) * 20) if corr > 0 else "-" * int(abs(corr) * 20)
        print(f"  {col:35s}: r = {corr:+.3f}  {bar}")
    
    # 2. Linear Regression
    print("\n" + "="*50)
    print("2. LINEAR REGRESSION COEFFICIENTS")
    print("="*50)
    
    # Standardize
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    lr = Ridge(alpha=1.0)
    lr.fit(X_std, y)
    print(f"\nR² score: {lr.score(X_std, y):.3f}")
    
    coef_importance = list(zip(feature_cols, lr.coef_))
    coef_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    print("\nTop coefficients (standardized):")
    for col, coef in coef_importance[:15]:
        bar = "+" * int(abs(coef) * 50) if coef > 0 else "-" * int(abs(coef) * 50)
        print(f"  {col:35s}: β = {coef:+.4f}  {bar}")
    
    # 3. Decision Tree
    print("\n" + "="*50)
    print("3. DECISION TREE (depth=4)")
    print("="*50)
    
    dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42)
    dt.fit(X, y)
    print(f"\nR² score: {dt.score(X, y):.3f}")
    print("\nTree structure:")
    tree_text = export_text(dt, feature_names=feature_cols, max_depth=4)
    print(tree_text)
    
    # 4. Random Forest Feature Importance
    print("\n" + "="*50)
    print("4. RANDOM FOREST FEATURE IMPORTANCE")
    print("="*50)
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    print(f"\nR² score: {rf.score(X, y):.3f}")
    
    importance = list(zip(feature_cols, rf.feature_importances_))
    importance.sort(key=lambda x: x[1], reverse=True)
    print("\nTop features by importance:")
    for col, imp in importance[:15]:
        bar = "█" * int(imp * 100)
        print(f"  {col:35s}: {imp:.4f}  {bar}")
    
    # 5. Group analysis
    print("\n" + "="*50)
    print("5. GROUP ANALYSIS")
    print("="*50)
    
    # By sample mode
    print("\nBy sample_mode:")
    for mode in df_valid['sample_mode'].unique():
        subset = df_valid[df_valid['sample_mode'] == mode]
        print(f"  {mode:20s}: AUC = {subset['auc'].mean():.3f} ± {subset['auc'].std():.3f} (n={len(subset)})")
    
    # By input type
    print("\nBy input type:")
    df_valid['input_type'] = df_valid.apply(
        lambda r: 'time_only' if r['n_state_inputs']==0 else ('state_only' if r['n_time_inputs']==0 else 'mixed'),
        axis=1
    )
    for itype in ['time_only', 'state_only', 'mixed']:
        subset = df_valid[df_valid['input_type'] == itype]
        if len(subset) > 0:
            print(f"  {itype:20s}: AUC = {subset['auc'].mean():.3f} ± {subset['auc'].std():.3f} (n={len(subset)})")
    
    # By offset type
    print("\nBy target offset:")
    for otype in ['within (0)', 'future (>0)', 'past (<0)']:
        if 'within' in otype:
            subset = df_valid[df_valid['target_offset'] == 0]
        elif 'future' in otype:
            subset = df_valid[df_valid['target_offset'] > 0]
        else:
            subset = df_valid[df_valid['target_offset'] < 0]
        if len(subset) > 0:
            print(f"  {otype:20s}: AUC = {subset['auc'].mean():.3f} ± {subset['auc'].std():.3f} (n={len(subset)})")
    
    # By n_classes
    print("\nBy n_classes:")
    for nc in sorted(df_valid['n_classes'].unique()):
        subset = df_valid[df_valid['n_classes'] == nc]
        print(f"  {nc} classes: AUC = {subset['auc'].mean():.3f} ± {subset['auc'].std():.3f} (n={len(subset)})")
    
    # 6. High vs Low AUC comparison
    print("\n" + "="*50)
    print("6. HIGH AUC vs LOW AUC COMPARISON")
    print("="*50)
    
    high_auc = df_valid[df_valid['auc'] >= 0.8]
    low_auc = df_valid[df_valid['auc'] <= 0.5]
    
    print(f"\nHigh AUC (≥0.8): n={len(high_auc)}, mean={high_auc['auc'].mean():.3f}")
    print(f"Low AUC (≤0.5): n={len(low_auc)}, mean={low_auc['auc'].mean():.3f}")
    
    if len(high_auc) > 0 and len(low_auc) > 0:
        print("\nMean values comparison:")
        for col in feature_cols[:20]:
            high_mean = high_auc[col].mean()
            low_mean = low_auc[col].mean()
            diff = high_mean - low_mean
            if abs(diff) > 0.1 * (abs(high_mean) + abs(low_mean) + 0.01):
                direction = "↑" if diff > 0 else "↓"
                print(f"  {col:30s}: HIGH={high_mean:8.2f}  LOW={low_mean:8.2f}  {direction}")
    
    return df_valid


def visualize_key_relationships(df: pd.DataFrame, output_dir: Path):
    """Create visualizations of key relationships."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_valid = df[df['auc'].notna()].copy()
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Key variables to plot
    vars_to_plot = [
        ('n_state_inputs', 'State Inputs'),
        ('n_time_inputs', 'Time Inputs'),
        ('noise_scale', 'Noise Scale'),
        ('n_classes', 'N Classes'),
        ('target_offset_abs', '|Target Offset|'),
        ('n_nodes', 'N Nodes'),
        ('mean_feature_to_target_dist', 'Feature-Target Dist'),
        ('state_alpha', 'State Alpha'),
        ('t_subseq', 'Sequence Length'),
        ('n_samples', 'N Samples'),
        ('density', 'DAG Density'),
        ('time_input_ratio', 'Time/Root Ratio'),
    ]
    
    for ax, (var, label) in zip(axes.flat, vars_to_plot):
        if var in df_valid.columns:
            ax.scatter(df_valid[var], df_valid['auc'], alpha=0.5, s=30)
            
            # Add trend line
            x = df_valid[var].values
            y = df_valid['auc'].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 5:
                z = np.polyfit(x[mask], y[mask], 1)
                p = np.poly1d(z)
                x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                ax.plot(x_line, p(x_line), 'r-', alpha=0.7, linewidth=2)
                
                corr = np.corrcoef(x[mask], y[mask])[0, 1]
                ax.set_title(f'{label}\nr={corr:.2f}', fontsize=11)
            else:
                ax.set_title(label, fontsize=11)
            
            ax.set_xlabel(var, fontsize=9)
            ax.set_ylabel('AUC', fontsize=9)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('AUC vs Generator Parameters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'auc_relationships.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Box plots by category
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # By sample mode
    modes = df_valid['sample_mode'].unique()
    data = [df_valid[df_valid['sample_mode'] == m]['auc'].dropna() for m in modes]
    axes[0].boxplot(data, labels=modes)
    axes[0].set_ylabel('AUC')
    axes[0].set_title('AUC by Sample Mode')
    axes[0].grid(True, alpha=0.3)
    
    # By input type
    df_valid['input_type'] = df_valid.apply(
        lambda r: 'time_only' if r['n_state_inputs']==0 else ('state_only' if r['n_time_inputs']==0 else 'mixed'),
        axis=1
    )
    itypes = ['time_only', 'state_only', 'mixed']
    data = [df_valid[df_valid['input_type'] == t]['auc'].dropna() for t in itypes if len(df_valid[df_valid['input_type'] == t]) > 0]
    labels = [t for t in itypes if len(df_valid[df_valid['input_type'] == t]) > 0]
    axes[1].boxplot(data, labels=labels)
    axes[1].set_ylabel('AUC')
    axes[1].set_title('AUC by Input Type')
    axes[1].grid(True, alpha=0.3)
    
    # By n_classes
    classes = sorted(df_valid['n_classes'].unique())
    data = [df_valid[df_valid['n_classes'] == c]['auc'].dropna() for c in classes]
    axes[2].boxplot(data, labels=classes)
    axes[2].set_ylabel('AUC')
    axes[2].set_title('AUC by N Classes')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'auc_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-datasets', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='results/auc_analysis')
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate and benchmark
    df = generate_and_benchmark(args.n_datasets, args.seed)
    
    # Save raw data
    df.to_csv(output_dir / 'dataset_features.csv', index=False)
    print(f"\nSaved features to {output_dir / 'dataset_features.csv'}")
    
    # Analyze
    df_analyzed = analyze_predictors(df)
    
    # Visualize
    visualize_key_relationships(df, output_dir)
    
    # Summary
    valid = df[df['auc'].notna()]
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total datasets: {len(df)}")
    print(f"Valid (with AUC): {len(valid)}")
    print(f"Mean AUC: {valid['auc'].mean():.3f} ± {valid['auc'].std():.3f}")
    print(f"Mean Accuracy: {valid['accuracy'].mean():.3f} ± {valid['accuracy'].std():.3f}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
