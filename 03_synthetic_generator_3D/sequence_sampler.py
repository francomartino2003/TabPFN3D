"""
Sequence Sampler for 3D Synthetic Data.

Handles extraction of feature subsequences and targets from propagated values.

Modes:
- IID: Each sample is an independent sequence
- Sliding Window: Multiple overlapping windows from one long sequence
- Mixed: Windows from multiple long sequences
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

from config import DatasetConfig3D
from temporal_propagator import TemporalPropagatedValues


@dataclass
class Sample3D:
    """
    A single 3D sample with features and target.
    
    Attributes:
        X: Feature tensor of shape (n_features, t_subseq)
        y: Target value (scalar)
        feature_nodes: List of node IDs used as features
        target_node: Node ID of target
        t_start: Start timestep of feature window
        t_end: End timestep of feature window (exclusive)
        t_target: Timestep of target
        sequence_id: ID of source sequence (for tracking)
    """
    X: np.ndarray
    y: float
    feature_nodes: List[int]
    target_node: int
    t_start: int
    t_end: int
    t_target: int
    sequence_id: int = 0


@dataclass
class FeatureTargetSelection:
    """
    Selection of which nodes are features and which is target.
    
    Attributes:
        feature_nodes: List of node IDs to use as features
        target_node: Node ID for the target
        target_offset: Time offset from end of feature window
        is_classification: Whether target should be discretized
        n_classes: Number of classes (for classification)
    """
    feature_nodes: List[int]
    target_node: int
    target_offset: int
    is_classification: bool
    n_classes: int


class SequenceSampler:
    """
    Extracts samples from propagated temporal values.
    
    Handles all three sampling modes: IID, sliding window, mixed.
    """
    
    def __init__(
        self,
        config: DatasetConfig3D,
        selection: FeatureTargetSelection,
        rng: np.random.Generator
    ):
        """
        Initialize the sampler.
        
        Args:
            config: Dataset configuration
            selection: Feature/target node selection
            rng: Random number generator
        """
        self.config = config
        self.selection = selection
        self.rng = rng
        
        self.t_subseq = config.t_subseq
        self.target_offset = selection.target_offset
    
    def sample_iid(
        self,
        propagated_list: List[TemporalPropagatedValues],
        n_samples: int
    ) -> List[Sample3D]:
        """
        Sample from independent sequences (IID mode).
        
        Each sample comes from a different propagated sequence.
        
        Args:
            propagated_list: List of propagated value objects
            n_samples: Number of samples to generate
            
        Returns:
            List of Sample3D objects
        """
        samples = []
        
        for i in range(min(n_samples, len(propagated_list))):
            propagated = propagated_list[i]
            
            # Random window start
            max_start = propagated.T - self.t_subseq - abs(self.target_offset) - 1
            if max_start < 0:
                max_start = 0
            
            t_start = self.rng.integers(0, max(1, max_start + 1))
            t_end = t_start + self.t_subseq
            
            # Target timestep
            t_target = self._compute_target_timestep(t_start, t_end, propagated.T)
            
            # Extract features and target
            sample = self._extract_sample(
                propagated, t_start, t_end, t_target, 
                sample_idx=0, sequence_id=i
            )
            samples.append(sample)
        
        return samples
    
    def sample_iid_batch(
        self,
        propagated: 'TemporalPropagatedValues',
        n_samples: int
    ) -> List[Sample3D]:
        """
        Sample from a batch of independent sequences (efficient IID mode).
        
        Each row in the propagated batch is an independent sequence.
        We extract one window per row.
        
        Args:
            propagated: Single propagated values object with n_samples rows
            n_samples: Number of samples to generate
            
        Returns:
            List of Sample3D objects
        """
        samples = []
        actual_samples = min(n_samples, propagated.n_samples)
        
        for sample_idx in range(actual_samples):
            # Random window start for this sample
            max_start = propagated.T - self.t_subseq - abs(self.target_offset) - 1
            if max_start < 0:
                max_start = 0
            
            t_start = self.rng.integers(0, max(1, max_start + 1))
            t_end = t_start + self.t_subseq
            
            # Target timestep
            t_target = self._compute_target_timestep(t_start, t_end, propagated.T)
            
            # Extract features and target for this sample
            sample = self._extract_sample(
                propagated, t_start, t_end, t_target, 
                sample_idx=sample_idx, sequence_id=sample_idx
            )
            samples.append(sample)
        
        return samples
    
    def sample_sliding_window(
        self,
        propagated: TemporalPropagatedValues,
        n_samples: int,
        stride: int = 1
    ) -> List[Sample3D]:
        """
        Sample using sliding window over a long sequence.
        
        Windows can overlap based on stride.
        
        Args:
            propagated: Single long propagated sequence
            n_samples: Maximum number of samples
            stride: Step between consecutive windows
            
        Returns:
            List of Sample3D objects
        """
        samples = []
        
        # Calculate valid window positions
        max_start = propagated.T - self.t_subseq - abs(self.target_offset) - 1
        if max_start < 0:
            return samples
        
        # Generate window positions
        positions = list(range(0, max_start + 1, stride))
        
        # Limit to n_samples
        if len(positions) > n_samples:
            # Randomly select n_samples positions
            positions = self.rng.choice(positions, size=n_samples, replace=False)
            positions = sorted(positions)
        
        # Extract samples for each position
        for i, t_start in enumerate(positions):
            t_end = t_start + self.t_subseq
            t_target = self._compute_target_timestep(t_start, t_end, propagated.T)
            
            # For sliding window, each window produces one sample per batch element
            for sample_idx in range(propagated.n_samples):
                sample = self._extract_sample(
                    propagated, t_start, t_end, t_target,
                    sample_idx=sample_idx, sequence_id=0
                )
                samples.append(sample)
                
                if len(samples) >= n_samples:
                    return samples
        
        return samples
    
    def sample_mixed(
        self,
        propagated_list: List[TemporalPropagatedValues],
        n_samples: int,
        stride: int = 1
    ) -> List[Sample3D]:
        """
        Sample from multiple long sequences (mixed mode).
        
        Combines sliding windows from multiple sequences.
        
        Args:
            propagated_list: List of long sequences
            n_samples: Total number of samples
            stride: Window stride
            
        Returns:
            List of Sample3D objects
        """
        samples = []
        samples_per_sequence = n_samples // len(propagated_list) + 1
        
        for seq_id, propagated in enumerate(propagated_list):
            seq_samples = self.sample_sliding_window(
                propagated, samples_per_sequence, stride
            )
            
            # Update sequence IDs
            for s in seq_samples:
                s.sequence_id = seq_id
            
            samples.extend(seq_samples)
            
            if len(samples) >= n_samples:
                break
        
        # Trim to exact count
        return samples[:n_samples]
    
    def _compute_target_timestep(
        self, 
        t_start: int, 
        t_end: int, 
        T_total: int
    ) -> int:
        """
        Compute target timestep based on offset type.
        
        Args:
            t_start: Start of feature window
            t_end: End of feature window
            T_total: Total sequence length
            
        Returns:
            Target timestep
        """
        offset_type = self.config.target_offset_type
        offset = self.target_offset
        
        if offset_type == 'within':
            # Target within the window
            t_target = t_start + self.t_subseq // 2
        elif offset_type in ['future_near', 'future_far']:
            # Target after the window
            t_target = t_end + offset
        else:  # past
            # Target before the window
            t_target = t_start + offset  # offset is negative
        
        # Clamp to valid range
        t_target = max(0, min(t_target, T_total - 1))
        
        return t_target
    
    def _extract_sample(
        self,
        propagated: TemporalPropagatedValues,
        t_start: int,
        t_end: int,
        t_target: int,
        sample_idx: int,
        sequence_id: int
    ) -> Sample3D:
        """
        Extract a single sample from propagated values.
        
        Args:
            propagated: Propagated values
            t_start, t_end: Feature window
            t_target: Target timestep
            sample_idx: Index within batch
            sequence_id: ID of source sequence
            
        Returns:
            Sample3D object
        """
        n_features = len(self.selection.feature_nodes)
        t_len = t_end - t_start
        
        # Extract features: (n_features, t_len)
        X = np.zeros((n_features, t_len))
        for i, node_id in enumerate(self.selection.feature_nodes):
            for t_idx, t in enumerate(range(t_start, t_end)):
                value = propagated.get_value(t, node_id)
                if len(value) > sample_idx:
                    X[i, t_idx] = value[sample_idx]
        
        # Extract target
        target_value = propagated.get_value(t_target, self.selection.target_node)
        if len(target_value) > sample_idx:
            y = target_value[sample_idx]
        else:
            y = 0.0
        
        return Sample3D(
            X=X,
            y=y,
            feature_nodes=self.selection.feature_nodes,
            target_node=self.selection.target_node,
            t_start=t_start,
            t_end=t_end,
            t_target=t_target,
            sequence_id=sequence_id
        )


def discretize_targets(
    samples: List[Sample3D], 
    n_classes: int,
    rng: np.random.Generator
) -> List[Sample3D]:
    """
    Discretize continuous targets into classes.
    
    Uses quantile-based discretization for balanced classes.
    
    Args:
        samples: List of samples with continuous targets
        n_classes: Number of classes
        rng: Random number generator
        
    Returns:
        Samples with discretized targets
    """
    if not samples:
        return samples
    
    # Collect all target values
    y_values = np.array([s.y for s in samples])
    
    # Quantile-based discretization
    quantiles = np.linspace(0, 100, n_classes + 1)
    thresholds = np.percentile(y_values, quantiles[1:-1])
    thresholds = np.unique(thresholds)
    
    # Discretize
    y_discrete = np.digitize(y_values, thresholds)
    
    # Handle case where we get fewer classes than desired
    if len(np.unique(y_discrete)) < 2:
        # Fallback: median split
        median = np.median(y_values)
        y_discrete = (y_values > median).astype(int)
    
    # Update samples
    for sample, y_new in zip(samples, y_discrete):
        sample.y = float(y_new)
    
    return samples


def samples_to_arrays(
    samples: List[Sample3D]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert list of samples to numpy arrays.
    
    Args:
        samples: List of Sample3D objects
        
    Returns:
        Tuple of (X, y) where:
        - X has shape (n_samples, n_features, t_subseq)
        - y has shape (n_samples,)
    """
    if not samples:
        return np.array([]), np.array([])
    
    n_samples = len(samples)
    n_features = samples[0].X.shape[0]
    t_subseq = samples[0].X.shape[1]
    
    X = np.zeros((n_samples, n_features, t_subseq))
    y = np.zeros(n_samples)
    
    for i, sample in enumerate(samples):
        X[i] = sample.X
        y[i] = sample.y
    
    return X, y

