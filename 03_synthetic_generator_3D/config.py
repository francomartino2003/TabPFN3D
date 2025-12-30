"""
Configuration for 3D Synthetic Dataset Generator (Time Series Classification).

This module defines hyperparameters for generating synthetic time series 
classification datasets. The key difference from 2D is:
- Input: (n_samples, n_features, n_timesteps) 
- Output: (n_samples,) classification labels

The generation process:
1. Create a base DAG (like 2D)
2. Unroll it T times to create temporal structure
3. Define temporal connections between time steps
4. Select features and target across time
5. Generate observations by propagating noise
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


def _safe_randint(rng: np.random.Generator, low: int, high: int) -> int:
    """Sample integer from range, handling low >= high case."""
    if low >= high:
        return low
    return rng.integers(low, high + 1)


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    """Sample from log-uniform distribution."""
    if low == high:
        return low
    return np.exp(rng.uniform(np.log(low), np.log(high)))


@dataclass
class PriorConfig3D:
    """
    Prior configuration for 3D time series dataset generation.
    
    These hyperparameters control the distribution from which datasets are sampled.
    Designed to cover realistic time series classification scenarios.
    """
    
    # === Size parameters ===
    # Number of observations (samples)
    n_samples_range: Tuple[int, int] = (50, 10000)
    n_samples_log_uniform: bool = True
    
    # Number of features (channels/variables)
    # In time series, often just 1 feature (univariate), sometimes up to 15
    n_features_range: Tuple[int, int] = (1, 15)
    # Probability of univariate (1 feature) - very common in time series
    prob_univariate: float = 0.4
    # If not univariate, use this distribution
    n_features_log_uniform: bool = False  # Linear is fine for small range
    
    # Number of timesteps (increased min for more realistic series)
    n_timesteps_range: Tuple[int, int] = (100, 1000)
    n_timesteps_log_uniform: bool = True
    
    # === Base graph parameters (inherited from 2D concept) ===
    # Number of nodes in the base DAG
    # AUMENTADO: grafos más grandes = menos correlación espacial entre features
    n_nodes_range: Tuple[int, int] = (20, 100)
    n_nodes_log_uniform: bool = True
    
    # Graph density (Gamma distribution)
    # Con más nodos, podemos mantener density similar
    density_gamma_shape: float = 2.0
    density_gamma_scale: float = 1.0
    
    # Probability of disconnected subgraphs (for irrelevant features)
    prob_disconnected_subgraph: float = 0.2
    n_disconnected_subgraphs_range: Tuple[int, int] = (1, 3)
    
    # === Temporal connection parameters ===
    # NUEVO: Primero se define un "tipo de proceso temporal" dominante
    # Esto prioriza procesos simples y realistas (alta autocorrelación)
    
    # Probabilidades para cada TIPO DE PROCESO (suma ~1.0)
    # 1. pure_ar: Solo self-connections → genera series suaves tipo AR
    prob_process_pure_ar: float = 0.35
    
    # 2. simple_lag: Self-connections con múltiples lags (AR de orden alto)
    prob_process_simple_lag: float = 0.25
    
    # 3. cross_channel: Principalmente cross-connections entre canales
    prob_process_cross_channel: float = 0.15
    
    # 4. mixed_simple: Mezcla de self y cross, pero simple
    prob_process_mixed_simple: float = 0.15
    
    # 5. complex: Todos los tipos de conexiones (casos raros/complejos)
    prob_process_complex: float = 0.10
    
    # === Parámetros dentro de cada tipo de proceso ===
    
    # Probabilidades de conexiones (usadas en procesos mixed/complex)
    prob_temporal_self: float = 0.40       # Aumentado: self-connections dominantes
    prob_temporal_cross: float = 0.25      # Cross entre nodos
    prob_temporal_many_to_one: float = 0.10
    prob_temporal_one_to_many: float = 0.10
    prob_temporal_broadcast_multiskip: float = 0.10  # AR multi-lag
    prob_temporal_conditional_lag: float = 0.03      # Reducido: muy raro
    prob_temporal_conditional_dest: float = 0.02     # Reducido: muy raro
    
    # Temporal skip parameters - prioriza lags cortos
    temporal_skip_range: Tuple[int, int] = (1, 15)
    temporal_skip_geometric_p: float = 0.5  # Aumentado: lag=1 muy probable (~50%)
    
    # Multi-skip decay factor for broadcast patterns (AR decay)
    multiskip_decay_range: Tuple[float, float] = (0.7, 0.98)  # Más alto = más memoria
    
    # Number of temporal connection patterns per dataset
    n_temporal_patterns_range: Tuple[int, int] = (2, 10)  # Reducido: menos complejidad
    
    # Probability of partial time range (connection only active for subsequence)
    prob_partial_time_range: float = 0.1  # Reducido: la mayoría son totales
    
    # For conditional connections
    n_conditions_range: Tuple[int, int] = (2, 3)  # Simplificado
    
    # === Transformation parameters (same as 2D) ===
    prob_nn_transform: float = 0.5
    prob_tree_transform: float = 0.2
    prob_discretization: float = 0.2
    prob_identity: float = 0.1
    
    nn_hidden_range: Tuple[int, int] = (1, 4)
    nn_width_range: Tuple[int, int] = (1, 8)
    
    activations: List[str] = field(default_factory=lambda: [
        'identity', 'log', 'sigmoid', 'tanh', 'sin', 'cos',
        'abs', 'square', 'sqrt', 'relu', 'softplus',
        'step', 'exp_neg', 'gaussian'
    ])
    
    n_categories_range: Tuple[int, int] = (2, 10)
    tree_depth_range: Tuple[int, int] = (1, 4)
    tree_n_splits_range: Tuple[int, int] = (1, 8)
    
    # === Noise parameters (IGUAL QUE 2D) ===
    # La autocorrelación viene de las CONEXIONES TEMPORALES, no del ruido
    noise_types: List[str] = field(default_factory=lambda: [
        'normal', 'uniform', 'laplace', 'mixture'
    ])
    noise_scale_range: Tuple[float, float] = (0.01, 1.0)  # Igual que 2D
    noise_scale_log_uniform: bool = True
    prob_edge_noise: float = 0.8  # Igual que 2D
    
    # === Ruido temporal (DESACTIVADO - correlación viene de conexiones) ===
    # Mantenemos los parámetros por compatibilidad pero no se usan
    prob_correlated_noise: float = 0.0  # SIN ruido AR - igual que 2D
    noise_ar_coef_range: Tuple[float, float] = (0.0, 0.0)  # No se usa
    
    # === Feature/Target selection parameters ===
    # Where in the time series to extract feature window
    # 0.0 = start, 1.0 = end
    feature_window_start_range: Tuple[float, float] = (0.0, 0.7)
    feature_window_length_range: Tuple[float, float] = (0.2, 1.0)  # Fraction of total T
    
    # Target position relative to feature window
    # Options: 'before', 'within', 'after'
    # Probabilities for each
    prob_target_before: float = 0.1   # Rare - target before features
    prob_target_within: float = 0.4   # Common - target during features
    prob_target_after: float = 0.5    # Most common - prediction task
    
    # How far after feature window to place target (if 'after')
    target_offset_range: Tuple[int, int] = (1, 50)
    
    # === Post-processing ===
    prob_warping: float = 0.3
    warping_intensity_range: Tuple[float, float] = (0.1, 1.0)
    
    prob_quantization: float = 0.2
    n_quantization_bins_range: Tuple[int, int] = (2, 30)
    
    prob_missing_values: float = 0.2
    missing_rate_range: Tuple[float, float] = (0.01, 0.15)
    
    # === Target parameters ===
    max_classes: int = 10
    prob_classification: float = 1.0  # Always classification for this generator
    
    # === Train/Test split ===
    train_ratio_range: Tuple[float, float] = (0.5, 0.8)
    train_ratio_beta_a: float = 2.0
    train_ratio_beta_b: float = 2.0
    
    def sample_hyperparams(self, rng: Optional[np.random.Generator] = None) -> 'DatasetConfig3D':
        """Sample a specific dataset configuration from this prior."""
        if rng is None:
            rng = np.random.default_rng()
        return DatasetConfig3D.sample_from_prior(self, rng)


@dataclass
class TemporalConnectionConfig:
    """
    Configuration for temporal connections between time steps.
    
    Supports rich connection patterns:
    - 'self': node -> same node at different time
    - 'cross': node -> different node at different time
    - 'many_to_one': multiple nodes -> one hub
    - 'one_to_many': one node -> multiple targets
    - 'broadcast_multiskip': one node -> itself at multiple skips with decay
    - 'conditional_lag': skip depends on value (lag switching)
    - 'conditional_dest': target depends on value (destination switching)
    """
    
    # Type of connection pattern
    connection_type: str
    
    # Which nodes are connected (indices in base graph)
    source_nodes: List[int]
    target_nodes: List[int]
    
    # Temporal skip (how many timesteps forward) or list for multi-skip
    skip: int
    
    # For multi-skip patterns: list of all skips
    skip_values: Optional[List[int]] = None
    
    # For multi-skip: weights for each skip (decay effect)
    skip_weights: Optional[List[float]] = None
    
    # For multi-skip: decay factor
    decay_factor: float = 0.8
    
    # Transformation to apply
    transform_type: str = 'nn'
    
    # Weight/strength of connection
    weight: float = 1.0
    
    # Time range where this connection is active (fraction of T)
    # (0.0, 1.0) = active for all time
    time_range: Tuple[float, float] = (0.0, 1.0)
    
    # For conditional patterns: thresholds for switching
    condition_thresholds: Optional[List[float]] = None
    
    # For conditional_lag: skip options for each condition
    conditional_skips: Optional[List[int]] = None
    
    # For conditional_dest: target options for each condition
    conditional_targets: Optional[List[List[int]]] = None
    
    # Unique pattern ID
    pattern_id: str = ""
    
    def get_active_timesteps(self, n_timesteps: int) -> List[int]:
        """Get timesteps where this connection is active as source."""
        t_start = int(self.time_range[0] * n_timesteps)
        t_end = int(self.time_range[1] * n_timesteps)
        return list(range(t_start, t_end))
    
    def get_all_skips(self) -> List[int]:
        """Get all skip values for this connection."""
        if self.skip_values:
            return self.skip_values
        return [self.skip]
    
    def get_skip_weight(self, skip: int) -> float:
        """Get weight for a specific skip value."""
        if self.skip_weights and self.skip_values:
            try:
                idx = self.skip_values.index(skip)
                return self.skip_weights[idx]
            except ValueError:
                pass
        return self.weight


@dataclass
class DatasetConfig3D:
    """
    Configuration for a specific 3D dataset instance.
    """
    
    # Size
    n_samples: int
    n_features: int
    n_timesteps: int
    
    # Base graph structure
    n_nodes: int
    density: float
    n_disconnected_subgraphs: int
    
    # Temporal structure
    temporal_connections: List[TemporalConnectionConfig]
    n_temporal_connections: int
    
    # Transformation settings
    edge_transform_probs: Dict[str, float]
    nn_hidden: int
    nn_width: int
    allowed_activations: List[str]
    n_categories: int
    tree_depth: int
    tree_n_splits: int
    
    # Noise settings
    noise_type: str
    noise_scale: float
    edge_noise_prob: float
    has_correlated_noise: bool
    noise_ar_coef: float
    
    # Feature/Target selection
    feature_window_start: int  # Timestep where feature window starts
    feature_window_end: int    # Timestep where feature window ends
    target_timestep: int       # Timestep for target
    target_position: str       # 'before', 'within', 'after'
    
    # Post-processing
    apply_warping: bool
    warping_intensity: float
    apply_quantization: bool
    n_quantization_bins: int
    apply_missing: bool
    missing_rate: float
    
    # Target
    is_classification: bool
    n_classes: int
    
    # Train/Test split
    train_ratio: float
    
    # Random seed
    seed: Optional[int] = None
    
    @classmethod
    def sample_from_prior(cls, prior: PriorConfig3D, rng: np.random.Generator) -> 'DatasetConfig3D':
        """Sample a dataset configuration from a prior."""
        
        # Use module-level helper functions
        log_uniform = lambda low, high: _log_uniform(rng, low, high)
        safe_randint = lambda low, high: _safe_randint(rng, low, high)
        
        # Features - sample first to determine minimum samples needed
        if rng.random() < prior.prob_univariate:
            n_features = 1
        elif prior.n_features_log_uniform:
            n_features = int(log_uniform(*prior.n_features_range))
        else:
            n_features = safe_randint(*prior.n_features_range)
        
        # Classes - sample early to determine minimum samples
        n_classes = rng.integers(2, prior.max_classes + 1)
        
        # Sample sizes - ADJUST based on complexity (features * classes)
        # Minimum samples = max(base_min, samples_per_class * n_classes, samples_per_dim * n_features)
        # This ensures enough data for learning
        samples_per_class = 20  # At least 20 samples per class
        samples_per_dim = 10    # At least 10 samples per feature dimension
        
        min_samples_for_complexity = max(
            prior.n_samples_range[0],
            samples_per_class * n_classes,
            samples_per_dim * n_features * n_classes
        )
        min_samples_for_complexity = min(min_samples_for_complexity, prior.n_samples_range[1])
        
        if prior.n_samples_log_uniform:
            n_samples = int(log_uniform(min_samples_for_complexity, prior.n_samples_range[1]))
        else:
            n_samples = safe_randint(min_samples_for_complexity, prior.n_samples_range[1])
        
        if prior.n_timesteps_log_uniform:
            n_timesteps = int(log_uniform(*prior.n_timesteps_range))
        else:
            n_timesteps = safe_randint(*prior.n_timesteps_range)
        
        # Base graph
        if prior.n_nodes_log_uniform:
            n_nodes = int(log_uniform(*prior.n_nodes_range))
        else:
            n_nodes = safe_randint(*prior.n_nodes_range)
        n_nodes = max(n_nodes, n_features + 2)
        
        density = rng.gamma(prior.density_gamma_shape, prior.density_gamma_scale)
        
        has_disconnected = rng.random() < prior.prob_disconnected_subgraph
        n_disconnected = safe_randint(*prior.n_disconnected_subgraphs_range) if has_disconnected else 0
        
        # Sample temporal connections
        temporal_connections = cls._sample_temporal_connections(
            n_nodes, n_timesteps, prior, rng
        )
        
        # Transformation probabilities
        edge_transform_probs = {
            'nn': prior.prob_nn_transform,
            'tree': prior.prob_tree_transform,
            'discretization': prior.prob_discretization,
            'identity': prior.prob_identity
        }
        
        n_activations = rng.integers(3, len(prior.activations) + 1)
        allowed_activations = list(rng.choice(prior.activations, size=n_activations, replace=False))
        
        nn_hidden = safe_randint(*prior.nn_hidden_range)
        nn_width = safe_randint(*prior.nn_width_range)
        n_categories = safe_randint(*prior.n_categories_range)
        tree_depth = safe_randint(*prior.tree_depth_range)
        tree_n_splits = safe_randint(*prior.tree_n_splits_range)
        
        # Noise
        noise_type = rng.choice(prior.noise_types)
        if prior.noise_scale_log_uniform:
            noise_scale = log_uniform(*prior.noise_scale_range)
        else:
            noise_scale = rng.uniform(*prior.noise_scale_range)
        
        has_correlated_noise = rng.random() < prior.prob_correlated_noise
        noise_ar_coef = rng.uniform(*prior.noise_ar_coef_range) if has_correlated_noise else 0.0
        
        # Feature window and target position
        feature_start_frac = rng.uniform(*prior.feature_window_start_range)
        feature_length_frac = rng.uniform(*prior.feature_window_length_range)
        
        # Ensure window fits
        max_length = 1.0 - feature_start_frac
        feature_length_frac = min(feature_length_frac, max_length)
        
        feature_window_start = int(feature_start_frac * n_timesteps)
        feature_window_end = int((feature_start_frac + feature_length_frac) * n_timesteps)
        feature_window_end = max(feature_window_end, feature_window_start + 5)  # At least 5 timesteps
        feature_window_end = min(feature_window_end, n_timesteps)
        
        # Target position
        target_probs = [prior.prob_target_before, prior.prob_target_within, prior.prob_target_after]
        target_probs = np.array(target_probs) / sum(target_probs)
        target_position = rng.choice(['before', 'within', 'after'], p=target_probs)
        
        if target_position == 'before':
            target_timestep = rng.integers(0, max(1, feature_window_start))
        elif target_position == 'within':
            if feature_window_start < feature_window_end:
                target_timestep = rng.integers(feature_window_start, feature_window_end)
            else:
                target_timestep = feature_window_start
        else:  # after
            offset = safe_randint(*prior.target_offset_range)
            offset = max(0, offset)  # Ensure non-negative
            target_timestep = min(feature_window_end + offset, n_timesteps - 1)
            # Ensure target is actually after feature window (but still valid)
            target_timestep = max(target_timestep, min(feature_window_end, n_timesteps - 1))
        
        # Final safety check: ensure target_timestep is valid
        target_timestep = min(target_timestep, n_timesteps - 1)
        target_timestep = max(0, target_timestep)
        
        # Post-processing
        apply_warping = rng.random() < prior.prob_warping
        warping_intensity = rng.uniform(*prior.warping_intensity_range)
        
        apply_quantization = rng.random() < prior.prob_quantization
        n_quantization_bins = safe_randint(*prior.n_quantization_bins_range)
        
        apply_missing = rng.random() < prior.prob_missing_values
        missing_rate = rng.uniform(*prior.missing_rate_range)
        
        # Target: n_classes already sampled at the beginning
        
        # Train/test split
        beta_sample = rng.beta(prior.train_ratio_beta_a, prior.train_ratio_beta_b)
        train_ratio = prior.train_ratio_range[0] + beta_sample * (
            prior.train_ratio_range[1] - prior.train_ratio_range[0]
        )
        
        return cls(
            n_samples=n_samples,
            n_features=n_features,
            n_timesteps=n_timesteps,
            n_nodes=n_nodes,
            density=density,
            n_disconnected_subgraphs=n_disconnected,
            temporal_connections=temporal_connections,
            n_temporal_connections=len(temporal_connections),
            edge_transform_probs=edge_transform_probs,
            nn_hidden=nn_hidden,
            nn_width=nn_width,
            allowed_activations=allowed_activations,
            n_categories=n_categories,
            tree_depth=tree_depth,
            tree_n_splits=tree_n_splits,
            noise_type=noise_type,
            noise_scale=noise_scale,
            edge_noise_prob=prior.prob_edge_noise,
            has_correlated_noise=has_correlated_noise,
            noise_ar_coef=noise_ar_coef,
            feature_window_start=feature_window_start,
            feature_window_end=feature_window_end,
            target_timestep=target_timestep,
            target_position=target_position,
            apply_warping=apply_warping,
            warping_intensity=warping_intensity,
            apply_quantization=apply_quantization,
            n_quantization_bins=n_quantization_bins,
            apply_missing=apply_missing,
            missing_rate=missing_rate,
            is_classification=True,
            n_classes=n_classes,
            train_ratio=train_ratio,
            seed=int(rng.integers(0, 2**31))
        )
    
    @staticmethod
    def _sample_temporal_connections(
        n_nodes: int, 
        n_timesteps: int,
        prior: PriorConfig3D,
        rng: np.random.Generator
    ) -> List[TemporalConnectionConfig]:
        """
        Sample temporal connections between nodes across time.
        
        NUEVO ENFOQUE: Primero se elige un "tipo de proceso" dominante,
        luego se generan conexiones apropiadas para ese tipo.
        Esto prioriza procesos simples y realistas.
        
        Tipos de proceso:
        - pure_ar: Solo self-connections (genera alta ACF)
        - simple_lag: Self con múltiples lags
        - cross_channel: Cross-connections entre canales
        - mixed_simple: Mezcla simple de self y cross
        - complex: Todos los tipos (raro)
        """
        connections = []
        
        # Use module-level helper
        safe_randint = lambda low, high: _safe_randint(rng, low, high)
        
        # PASO 1: Elegir tipo de proceso
        process_types = ['pure_ar', 'simple_lag', 'cross_channel', 'mixed_simple', 'complex']
        process_probs = [
            prior.prob_process_pure_ar,
            prior.prob_process_simple_lag,
            prior.prob_process_cross_channel,
            prior.prob_process_mixed_simple,
            prior.prob_process_complex
        ]
        process_probs = np.array(process_probs) / sum(process_probs)
        process_type = rng.choice(process_types, p=process_probs)
        
        # PASO 2: Generar conexiones según el tipo de proceso
        n_patterns = safe_randint(*prior.n_temporal_patterns_range)
        n_patterns = min(n_patterns, n_nodes * 2)
        
        # Transformaciones simples para preservar señal (más identity)
        simple_transform_types = ['identity', 'nn']
        simple_transform_probs = [0.6, 0.4]  # Mayoría identity
        
        complex_transform_types = ['nn', 'tree', 'identity']
        complex_transform_probs = [0.4, 0.3, 0.3]
        
        def sample_skip() -> int:
            """Sample lag priorizando valores pequeños."""
            skip = rng.geometric(prior.temporal_skip_geometric_p)
            return max(1, min(skip, min(prior.temporal_skip_range[1], n_timesteps // 3)))
        
        def sample_time_range() -> Tuple[float, float]:
            """Sample time range (mayoría completa)."""
            if rng.random() < prior.prob_partial_time_range:
                start = rng.uniform(0, 0.5)
                length = rng.uniform(0.3, 1.0 - start)
                return (start, start + length)
            return (0.0, 1.0)
        
        if process_type == 'pure_ar':
            # === PURE AR: Solo self-connections, genera series muy suaves ===
            # Cada nodo se conecta consigo mismo con lag=1 (como AR(1))
            for node in range(min(n_nodes, 10)):  # Conectar los primeros nodos
                connections.append(TemporalConnectionConfig(
                    connection_type='self',
                    source_nodes=[node],
                    target_nodes=[node],
                    skip=1,  # Siempre lag=1 para AR(1)
                    transform_type='identity',  # Preserva señal
                    weight=rng.uniform(0.8, 1.0),  # Peso alto
                    time_range=(0.0, 1.0),
                    pattern_id=f"ar_{node}"
                ))
            # Opcionalmente añadir algunos lags extras
            for i in range(min(3, n_patterns - n_nodes)):
                node = rng.integers(0, n_nodes)
                skip = rng.integers(2, 4)  # Lags 2-3
                connections.append(TemporalConnectionConfig(
                    connection_type='self',
                    source_nodes=[node],
                    target_nodes=[node],
                    skip=skip,
                    transform_type='identity',
                    weight=rng.uniform(0.3, 0.6),  # Peso menor para lags mayores
                    time_range=(0.0, 1.0),
                    pattern_id=f"ar_lag{skip}_{i}"
                ))
        
        elif process_type == 'simple_lag':
            # === SIMPLE LAG: Self-connections con múltiples lags (AR de orden alto) ===
            for i in range(n_patterns):
                node = rng.integers(0, n_nodes)
                skip = sample_skip()
                # Usar broadcast multiskip para algunos
                if i < n_patterns // 2:
                    # AR multi-lag con decay
                    max_skip = min(6, n_timesteps // 5)
                    n_skips = rng.integers(2, max_skip + 1)
                    skip_values = list(range(1, n_skips + 1))
                    decay = rng.uniform(*prior.multiskip_decay_range)
                    skip_weights = [decay ** (s - 1) for s in skip_values]
                    total_w = sum(skip_weights)
                    skip_weights = [w / total_w for w in skip_weights]
                    
                    connections.append(TemporalConnectionConfig(
                        connection_type='broadcast_multiskip',
                        source_nodes=[node],
                        target_nodes=[node],
                        skip=1,
                        skip_values=skip_values,
                        skip_weights=skip_weights,
                        decay_factor=decay,
                        transform_type=rng.choice(simple_transform_types, p=simple_transform_probs),
                        weight=rng.uniform(0.6, 1.0),
                        time_range=(0.0, 1.0),
                        pattern_id=f"multilag_{i}"
                    ))
                else:
                    # Self simple
                    connections.append(TemporalConnectionConfig(
                        connection_type='self',
                        source_nodes=[node],
                        target_nodes=[node],
                        skip=skip,
                        transform_type=rng.choice(simple_transform_types, p=simple_transform_probs),
                        weight=rng.uniform(0.5, 1.0),
                        time_range=sample_time_range(),
                        pattern_id=f"self_{i}"
                    ))
        
        elif process_type == 'cross_channel':
            # === CROSS CHANNEL: Principalmente cross-connections ===
            # Primero algunas self para mantener continuidad
            for node in range(min(3, n_nodes)):
                connections.append(TemporalConnectionConfig(
                    connection_type='self',
                    source_nodes=[node],
                    target_nodes=[node],
                    skip=1,
                    transform_type='identity',
                    weight=0.9,
                    time_range=(0.0, 1.0),
                    pattern_id=f"self_base_{node}"
                ))
            
            # Luego cross-connections
            for i in range(n_patterns):
                if n_nodes >= 2:
                    source = rng.integers(0, n_nodes)
                    target = rng.integers(0, n_nodes)
                    while target == source:
                        target = rng.integers(0, n_nodes)
                    connections.append(TemporalConnectionConfig(
                        connection_type='cross',
                        source_nodes=[source],
                        target_nodes=[target],
                        skip=sample_skip(),
                        transform_type=rng.choice(simple_transform_types, p=simple_transform_probs),
                        weight=rng.uniform(0.4, 0.8),
                        time_range=sample_time_range(),
                        pattern_id=f"cross_{i}"
                    ))
        
        elif process_type == 'mixed_simple':
            # === MIXED SIMPLE: Mezcla de self y cross, pero simple ===
            for i in range(n_patterns):
                if rng.random() < 0.6:  # 60% self
                    node = rng.integers(0, n_nodes)
                    connections.append(TemporalConnectionConfig(
                        connection_type='self',
                        source_nodes=[node],
                        target_nodes=[node],
                        skip=sample_skip(),
                        transform_type=rng.choice(simple_transform_types, p=simple_transform_probs),
                        weight=rng.uniform(0.5, 1.0),
                        time_range=sample_time_range(),
                        pattern_id=f"self_{i}"
                    ))
                elif n_nodes >= 2:  # 40% cross
                    source = rng.integers(0, n_nodes)
                    target = rng.integers(0, n_nodes)
                    while target == source:
                        target = rng.integers(0, n_nodes)
                    connections.append(TemporalConnectionConfig(
                        connection_type='cross',
                        source_nodes=[source],
                        target_nodes=[target],
                        skip=sample_skip(),
                        transform_type=rng.choice(simple_transform_types, p=simple_transform_probs),
                        weight=rng.uniform(0.4, 0.8),
                        time_range=sample_time_range(),
                        pattern_id=f"cross_{i}"
                    ))
        
        else:  # complex
            # === COMPLEX: Todos los tipos de conexiones ===
            conn_types = [
                'self', 'cross', 'many_to_one', 'one_to_many',
                'broadcast_multiskip', 'conditional_lag', 'conditional_dest'
            ]
            conn_probs = [
                prior.prob_temporal_self,
                prior.prob_temporal_cross,
                prior.prob_temporal_many_to_one,
                prior.prob_temporal_one_to_many,
                prior.prob_temporal_broadcast_multiskip,
                prior.prob_temporal_conditional_lag,
                prior.prob_temporal_conditional_dest
            ]
            conn_probs = np.array(conn_probs) / sum(conn_probs)
            
            for i in range(n_patterns):
                conn_type = rng.choice(conn_types, p=conn_probs)
                time_range = sample_time_range()
                base_skip = sample_skip()
                transform_type = rng.choice(complex_transform_types, p=complex_transform_probs)
                weight = rng.uniform(0.5, 1.5)
                
                if conn_type == 'self':
                    node = rng.integers(0, n_nodes)
                    connections.append(TemporalConnectionConfig(
                        connection_type=conn_type,
                        source_nodes=[node],
                        target_nodes=[node],
                        skip=base_skip,
                        transform_type=transform_type,
                        weight=weight,
                        time_range=time_range,
                        pattern_id=f"self_{i}"
                    ))
                
                elif conn_type == 'cross' and n_nodes >= 2:
                    source = rng.integers(0, n_nodes)
                    target = rng.integers(0, n_nodes)
                    while target == source:
                        target = rng.integers(0, n_nodes)
                    connections.append(TemporalConnectionConfig(
                        connection_type=conn_type,
                        source_nodes=[source],
                        target_nodes=[target],
                        skip=base_skip,
                        transform_type=transform_type,
                        weight=weight,
                        time_range=time_range,
                        pattern_id=f"cross_{i}"
                    ))
                
                elif conn_type == 'many_to_one':
                    n_sources = min(rng.integers(2, 5), n_nodes)
                    sources = list(rng.choice(n_nodes, size=n_sources, replace=False))
                    target = rng.integers(0, n_nodes)
                    connections.append(TemporalConnectionConfig(
                        connection_type=conn_type,
                        source_nodes=sources,
                        target_nodes=[target],
                        skip=base_skip,
                        transform_type=transform_type,
                        weight=weight,
                        time_range=time_range,
                        pattern_id=f"many2one_{i}"
                    ))
                
                elif conn_type == 'one_to_many':
                    source = rng.integers(0, n_nodes)
                    n_targets = min(rng.integers(2, 5), n_nodes)
                    targets = list(rng.choice(n_nodes, size=n_targets, replace=False))
                    connections.append(TemporalConnectionConfig(
                        connection_type=conn_type,
                        source_nodes=[source],
                        target_nodes=targets,
                        skip=base_skip,
                        transform_type=transform_type,
                        weight=weight,
                        time_range=time_range,
                        pattern_id=f"one2many_{i}"
                    ))
                
                elif conn_type == 'broadcast_multiskip':
                    node = rng.integers(0, n_nodes)
                    max_skip = min(8, n_timesteps // 4)
                    n_skips = rng.integers(2, min(5, max_skip + 1))
                    skip_values = list(range(1, n_skips + 1))
                    decay = rng.uniform(*prior.multiskip_decay_range)
                    skip_weights = [decay ** (s - 1) for s in skip_values]
                    total_w = sum(skip_weights)
                    skip_weights = [w / total_w for w in skip_weights]
                    
                    connections.append(TemporalConnectionConfig(
                        connection_type=conn_type,
                        source_nodes=[node],
                        target_nodes=[node],
                        skip=1,
                        skip_values=skip_values,
                        skip_weights=skip_weights,
                        decay_factor=decay,
                        transform_type=transform_type,
                        weight=weight,
                        time_range=time_range,
                        pattern_id=f"broadcast_{i}"
                    ))
                
                elif conn_type == 'conditional_lag':
                    source = rng.integers(0, n_nodes)
                    target = rng.integers(0, n_nodes)
                    n_conditions = safe_randint(*prior.n_conditions_range)
                    thresholds = sorted(rng.uniform(-1, 1, size=n_conditions - 1).tolist())
                    max_skip = min(10, n_timesteps // 4)
                    conditional_skips = sorted(rng.integers(1, max_skip + 1, size=n_conditions).tolist())
                    
                    connections.append(TemporalConnectionConfig(
                        connection_type=conn_type,
                        source_nodes=[source],
                        target_nodes=[target],
                        skip=conditional_skips[0],
                        condition_thresholds=thresholds,
                        conditional_skips=conditional_skips,
                        transform_type='tree',
                        weight=weight,
                        time_range=time_range,
                        pattern_id=f"condlag_{i}"
                    ))
                
                elif conn_type == 'conditional_dest':
                    source = rng.integers(0, n_nodes)
                    n_conditions = safe_randint(*prior.n_conditions_range)
                    thresholds = sorted(rng.uniform(-1, 1, size=n_conditions - 1).tolist())
                    conditional_targets = []
                    for _ in range(n_conditions):
                        n_t = rng.integers(1, min(3, n_nodes + 1))
                        targets = list(rng.choice(n_nodes, size=n_t, replace=False))
                        conditional_targets.append(targets)
                    
                    connections.append(TemporalConnectionConfig(
                        connection_type=conn_type,
                        source_nodes=[source],
                        target_nodes=list(range(n_nodes)),
                        skip=base_skip,
                        condition_thresholds=thresholds,
                        conditional_targets=conditional_targets,
                        transform_type='tree',
                        weight=weight,
                        time_range=time_range,
                        pattern_id=f"conddest_{i}"
                    ))
        
        # Garantizar al menos algunas self-connections para continuidad temporal
        if not any(c.connection_type == 'self' for c in connections):
            for node in range(min(3, n_nodes)):
                connections.append(TemporalConnectionConfig(
                    connection_type='self',
                    source_nodes=[node],
                    target_nodes=[node],
                    skip=1,
                    transform_type='identity',
                    weight=0.9,
                    time_range=(0.0, 1.0),
                    pattern_id=f"self_fallback_{node}"
                ))
        
        return connections
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_timesteps': self.n_timesteps,
            'n_nodes': self.n_nodes,
            'density': self.density,
            'n_disconnected_subgraphs': self.n_disconnected_subgraphs,
            'n_temporal_connections': self.n_temporal_connections,
            'edge_transform_probs': self.edge_transform_probs,
            'nn_hidden': self.nn_hidden,
            'nn_width': self.nn_width,
            'allowed_activations': self.allowed_activations,
            'n_categories': self.n_categories,
            'tree_depth': self.tree_depth,
            'tree_n_splits': self.tree_n_splits,
            'noise_type': self.noise_type,
            'noise_scale': self.noise_scale,
            'edge_noise_prob': self.edge_noise_prob,
            'has_correlated_noise': self.has_correlated_noise,
            'noise_ar_coef': self.noise_ar_coef,
            'feature_window_start': self.feature_window_start,
            'feature_window_end': self.feature_window_end,
            'target_timestep': self.target_timestep,
            'target_position': self.target_position,
            'apply_warping': self.apply_warping,
            'warping_intensity': self.warping_intensity,
            'apply_quantization': self.apply_quantization,
            'n_quantization_bins': self.n_quantization_bins,
            'apply_missing': self.apply_missing,
            'missing_rate': self.missing_rate,
            'is_classification': self.is_classification,
            'n_classes': self.n_classes,
            'train_ratio': self.train_ratio,
            'seed': self.seed
        }

