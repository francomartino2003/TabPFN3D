"""
Rich Temporal Connections for 3D Synthetic Dataset Generation.

This module defines sophisticated temporal connection patterns that can
capture realistic time series dynamics:

1. **Self-connections**: node_i(t) -> node_i(t+k)
2. **Cross-connections**: node_i(t) -> node_j(t+k) where i != j
3. **Many-to-one (Hub)**: multiple nodes -> one target across time
4. **One-to-many (Broadcast)**: one node -> multiple targets
   - Same node at different skips (with decay)
   - Cross connections at same skip
   - Mixtures
5. **Conditional connections (Tree-like)**:
   - Lag switching: skip depends on value
   - Destination switching: target depends on value
6. **Partial connections**: Only active for subsequence of T

Each connection pattern defines:
- Which nodes are involved
- Skip pattern (fixed, multi-skip, or conditional)
- Time range (all T or subsequence)
- Transformation to apply
- Optional decay for distant connections
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
import numpy as np


class ConnectionType(Enum):
    """Types of temporal connections."""
    SELF = "self"                    # node_i -> node_i at different time
    CROSS = "cross"                  # node_i -> node_j at different time
    MANY_TO_ONE = "many_to_one"      # multiple nodes -> one node
    ONE_TO_MANY = "one_to_many"      # one node -> multiple nodes
    BROADCAST_MULTISKIP = "broadcast_multiskip"  # one node -> same node at multiple skips
    CONDITIONAL_LAG = "conditional_lag"          # skip depends on value
    CONDITIONAL_DEST = "conditional_dest"        # destination depends on value


class SkipPattern(Enum):
    """Patterns for temporal skip behavior."""
    FIXED = "fixed"           # Single fixed skip value
    MULTI = "multi"           # Multiple skip values (with weights/decay)
    GEOMETRIC_DECAY = "geometric_decay"  # Multiple skips with geometric decay
    CONDITIONAL = "conditional"  # Skip depends on node value


@dataclass
class TemporalConnectionPattern:
    """
    Defines a pattern for temporal connections.
    
    This is a "rule" that gets applied across time to create actual edges.
    """
    # Connection type
    connection_type: ConnectionType
    
    # Source nodes (in base graph)
    source_nodes: List[int]
    
    # Target nodes (in base graph)
    target_nodes: List[int]
    
    # Skip pattern
    skip_pattern: SkipPattern = SkipPattern.FIXED
    
    # For FIXED: single skip value
    # For MULTI/GEOMETRIC_DECAY: list of skip values
    # For CONDITIONAL: min and max skip
    skip_values: Union[int, List[int], Tuple[int, int]] = 1
    
    # Weights for multi-skip (optional, defaults to equal or decay)
    skip_weights: Optional[List[float]] = None
    
    # Decay factor for GEOMETRIC_DECAY (weight = decay^skip)
    decay_factor: float = 0.8
    
    # Time range: (start_fraction, end_fraction) of total T
    # (0.0, 1.0) means active for all time
    time_range: Tuple[float, float] = (0.0, 1.0)
    
    # Transform type to apply
    transform_type: str = "nn"
    
    # For conditional connections: thresholds for switching
    # If value < thresholds[0]: use option 0
    # If thresholds[0] <= value < thresholds[1]: use option 1, etc.
    condition_thresholds: Optional[List[float]] = None
    
    # For CONDITIONAL_LAG: skip options for each condition
    conditional_skips: Optional[List[int]] = None
    
    # For CONDITIONAL_DEST: target options for each condition
    conditional_targets: Optional[List[List[int]]] = None
    
    # Noise scale multiplier for this connection
    noise_scale: float = 1.0
    
    # Unique ID for this pattern
    pattern_id: str = ""
    
    def get_active_timesteps(self, n_timesteps: int) -> List[int]:
        """Get the timesteps where this connection is active (as source)."""
        t_start = int(self.time_range[0] * n_timesteps)
        t_end = int(self.time_range[1] * n_timesteps)
        return list(range(t_start, t_end))
    
    def get_skip_values_and_weights(self) -> Tuple[List[int], List[float]]:
        """Get all skip values and their weights."""
        if self.skip_pattern == SkipPattern.FIXED:
            if isinstance(self.skip_values, int):
                return [self.skip_values], [1.0]
            else:
                return [self.skip_values[0]], [1.0]
        
        elif self.skip_pattern == SkipPattern.MULTI:
            skips = self.skip_values if isinstance(self.skip_values, list) else [self.skip_values]
            if self.skip_weights:
                weights = self.skip_weights
            else:
                weights = [1.0 / len(skips)] * len(skips)
            return skips, weights
        
        elif self.skip_pattern == SkipPattern.GEOMETRIC_DECAY:
            skips = self.skip_values if isinstance(self.skip_values, list) else list(range(1, self.skip_values + 1))
            weights = [self.decay_factor ** (s - 1) for s in skips]
            # Normalize
            total = sum(weights)
            weights = [w / total for w in weights]
            return skips, weights
        
        elif self.skip_pattern == SkipPattern.CONDITIONAL:
            # Return range of possible skips
            if isinstance(self.skip_values, tuple):
                min_skip, max_skip = self.skip_values
                skips = list(range(min_skip, max_skip + 1))
            else:
                skips = self.conditional_skips if self.conditional_skips else [1]
            return skips, [1.0 / len(skips)] * len(skips)
        
        return [1], [1.0]


class TemporalConnectionSampler:
    """
    Samples rich temporal connection patterns for a dataset.
    """
    
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
    
    def sample_patterns(
        self,
        n_base_nodes: int,
        n_timesteps: int,
        n_patterns: int,
        config_probs: Dict[str, float]
    ) -> List[TemporalConnectionPattern]:
        """
        Sample a set of temporal connection patterns.
        
        Args:
            n_base_nodes: Number of nodes in base graph
            n_timesteps: Total number of timesteps
            n_patterns: Number of patterns to sample
            config_probs: Probabilities for each connection type
            
        Returns:
            List of TemporalConnectionPattern
        """
        patterns = []
        
        # Normalize probabilities
        conn_types = [
            ConnectionType.SELF,
            ConnectionType.CROSS,
            ConnectionType.MANY_TO_ONE,
            ConnectionType.ONE_TO_MANY,
            ConnectionType.BROADCAST_MULTISKIP,
            ConnectionType.CONDITIONAL_LAG,
            ConnectionType.CONDITIONAL_DEST,
        ]
        
        probs = [
            config_probs.get('self', 0.25),
            config_probs.get('cross', 0.20),
            config_probs.get('many_to_one', 0.15),
            config_probs.get('one_to_many', 0.15),
            config_probs.get('broadcast_multiskip', 0.10),
            config_probs.get('conditional_lag', 0.08),
            config_probs.get('conditional_dest', 0.07),
        ]
        probs = np.array(probs) / sum(probs)
        
        for i in range(n_patterns):
            conn_type = self.rng.choice(conn_types, p=probs)
            pattern = self._sample_pattern(conn_type, n_base_nodes, n_timesteps, i)
            patterns.append(pattern)
        
        return patterns
    
    def _sample_pattern(
        self,
        conn_type: ConnectionType,
        n_base_nodes: int,
        n_timesteps: int,
        pattern_idx: int
    ) -> TemporalConnectionPattern:
        """Sample a single pattern of the given type."""
        
        # Sample time range (most are full, some are partial)
        if self.rng.random() < 0.2:  # 20% chance of partial time range
            start = self.rng.uniform(0, 0.5)
            length = self.rng.uniform(0.3, 1.0 - start)
            time_range = (start, start + length)
        else:
            time_range = (0.0, 1.0)
        
        # Sample transform type
        transform_types = ['nn', 'tree', 'identity']
        transform_probs = [0.5, 0.3, 0.2]
        transform_type = self.rng.choice(transform_types, p=transform_probs)
        
        # Sample base skip using geometric (smaller more likely)
        base_skip = self.rng.geometric(0.3)  # E[skip] ≈ 3
        base_skip = max(1, min(base_skip, min(20, n_timesteps // 2)))
        
        if conn_type == ConnectionType.SELF:
            return self._sample_self_connection(n_base_nodes, base_skip, time_range, transform_type, pattern_idx)
        
        elif conn_type == ConnectionType.CROSS:
            return self._sample_cross_connection(n_base_nodes, base_skip, time_range, transform_type, pattern_idx)
        
        elif conn_type == ConnectionType.MANY_TO_ONE:
            return self._sample_many_to_one(n_base_nodes, base_skip, time_range, transform_type, pattern_idx)
        
        elif conn_type == ConnectionType.ONE_TO_MANY:
            return self._sample_one_to_many(n_base_nodes, base_skip, time_range, transform_type, pattern_idx)
        
        elif conn_type == ConnectionType.BROADCAST_MULTISKIP:
            return self._sample_broadcast_multiskip(n_base_nodes, n_timesteps, time_range, transform_type, pattern_idx)
        
        elif conn_type == ConnectionType.CONDITIONAL_LAG:
            return self._sample_conditional_lag(n_base_nodes, n_timesteps, time_range, transform_type, pattern_idx)
        
        elif conn_type == ConnectionType.CONDITIONAL_DEST:
            return self._sample_conditional_dest(n_base_nodes, base_skip, time_range, transform_type, pattern_idx)
        
        # Default fallback
        return self._sample_self_connection(n_base_nodes, base_skip, time_range, transform_type, pattern_idx)
    
    def _sample_self_connection(self, n_nodes, skip, time_range, transform, idx) -> TemporalConnectionPattern:
        """Self connection: node connects to itself at future time."""
        node = self.rng.integers(0, n_nodes)
        return TemporalConnectionPattern(
            connection_type=ConnectionType.SELF,
            source_nodes=[node],
            target_nodes=[node],
            skip_pattern=SkipPattern.FIXED,
            skip_values=skip,
            time_range=time_range,
            transform_type=transform,
            pattern_id=f"self_{idx}"
        )
    
    def _sample_cross_connection(self, n_nodes, skip, time_range, transform, idx) -> TemporalConnectionPattern:
        """Cross connection: node connects to different node."""
        if n_nodes < 2:
            return self._sample_self_connection(n_nodes, skip, time_range, transform, idx)
        
        source = self.rng.integers(0, n_nodes)
        target = self.rng.integers(0, n_nodes)
        while target == source:
            target = self.rng.integers(0, n_nodes)
        
        return TemporalConnectionPattern(
            connection_type=ConnectionType.CROSS,
            source_nodes=[source],
            target_nodes=[target],
            skip_pattern=SkipPattern.FIXED,
            skip_values=skip,
            time_range=time_range,
            transform_type=transform,
            pattern_id=f"cross_{idx}"
        )
    
    def _sample_many_to_one(self, n_nodes, skip, time_range, transform, idx) -> TemporalConnectionPattern:
        """Multiple source nodes connect to one target."""
        n_sources = min(self.rng.integers(2, 5), n_nodes)
        sources = list(self.rng.choice(n_nodes, size=n_sources, replace=False))
        target = self.rng.integers(0, n_nodes)
        
        return TemporalConnectionPattern(
            connection_type=ConnectionType.MANY_TO_ONE,
            source_nodes=sources,
            target_nodes=[target],
            skip_pattern=SkipPattern.FIXED,
            skip_values=skip,
            time_range=time_range,
            transform_type=transform,
            pattern_id=f"many2one_{idx}"
        )
    
    def _sample_one_to_many(self, n_nodes, skip, time_range, transform, idx) -> TemporalConnectionPattern:
        """One source connects to multiple targets at same skip."""
        source = self.rng.integers(0, n_nodes)
        n_targets = min(self.rng.integers(2, 5), n_nodes)
        targets = list(self.rng.choice(n_nodes, size=n_targets, replace=False))
        
        return TemporalConnectionPattern(
            connection_type=ConnectionType.ONE_TO_MANY,
            source_nodes=[source],
            target_nodes=targets,
            skip_pattern=SkipPattern.FIXED,
            skip_values=skip,
            time_range=time_range,
            transform_type=transform,
            pattern_id=f"one2many_{idx}"
        )
    
    def _sample_broadcast_multiskip(self, n_nodes, n_timesteps, time_range, transform, idx) -> TemporalConnectionPattern:
        """One node connects to itself at multiple skips with decay."""
        source = self.rng.integers(0, n_nodes)
        
        # Sample multiple skips (1, 2, 3, ...) with geometric decay
        max_skip = min(10, n_timesteps // 3)
        n_skips = self.rng.integers(2, min(6, max_skip + 1))
        skips = list(range(1, n_skips + 1))
        
        # Decay factor
        decay = self.rng.uniform(0.5, 0.95)
        
        return TemporalConnectionPattern(
            connection_type=ConnectionType.BROADCAST_MULTISKIP,
            source_nodes=[source],
            target_nodes=[source],  # Self at multiple skips
            skip_pattern=SkipPattern.GEOMETRIC_DECAY,
            skip_values=skips,
            decay_factor=decay,
            time_range=time_range,
            transform_type=transform,
            pattern_id=f"broadcast_{idx}"
        )
    
    def _sample_conditional_lag(self, n_nodes, n_timesteps, time_range, transform, idx) -> TemporalConnectionPattern:
        """Skip depends on node value (lag switching)."""
        source = self.rng.integers(0, n_nodes)
        target = self.rng.integers(0, n_nodes)
        
        # Sample 2-4 conditions with different skips
        n_conditions = self.rng.integers(2, 5)
        
        # Thresholds for switching (percentiles)
        thresholds = sorted(self.rng.uniform(-1, 1, size=n_conditions - 1).tolist())
        
        # Different skips for each condition
        max_skip = min(15, n_timesteps // 3)
        conditional_skips = sorted(self.rng.integers(1, max_skip + 1, size=n_conditions).tolist())
        
        return TemporalConnectionPattern(
            connection_type=ConnectionType.CONDITIONAL_LAG,
            source_nodes=[source],
            target_nodes=[target],
            skip_pattern=SkipPattern.CONDITIONAL,
            skip_values=(1, max_skip),
            condition_thresholds=thresholds,
            conditional_skips=conditional_skips,
            time_range=time_range,
            transform_type='tree',  # Tree-like decisions
            pattern_id=f"condlag_{idx}"
        )
    
    def _sample_conditional_dest(self, n_nodes, skip, time_range, transform, idx) -> TemporalConnectionPattern:
        """Destination depends on node value (destination switching)."""
        source = self.rng.integers(0, n_nodes)
        
        # Sample 2-4 conditions with different targets
        n_conditions = self.rng.integers(2, min(5, n_nodes + 1))
        
        # Thresholds
        thresholds = sorted(self.rng.uniform(-1, 1, size=n_conditions - 1).tolist())
        
        # Different targets for each condition
        all_targets = list(range(n_nodes))
        conditional_targets = []
        for _ in range(n_conditions):
            n_targets = self.rng.integers(1, min(3, n_nodes + 1))
            targets = list(self.rng.choice(all_targets, size=n_targets, replace=False))
            conditional_targets.append(targets)
        
        return TemporalConnectionPattern(
            connection_type=ConnectionType.CONDITIONAL_DEST,
            source_nodes=[source],
            target_nodes=all_targets,  # All possible targets
            skip_pattern=SkipPattern.FIXED,
            skip_values=skip,
            condition_thresholds=thresholds,
            conditional_targets=conditional_targets,
            time_range=time_range,
            transform_type='tree',
            pattern_id=f"conddest_{idx}"
        )


def instantiate_temporal_edges(
    patterns: List[TemporalConnectionPattern],
    n_timesteps: int,
    n_base_nodes: int,
    node_values: Optional[Dict[int, np.ndarray]] = None
) -> List[Tuple[int, int, int, int, TemporalConnectionPattern, float]]:
    """
    Instantiate actual edges from patterns.
    
    For non-conditional patterns, creates edges for all valid timesteps.
    For conditional patterns, requires node_values to determine skip/target.
    
    Args:
        patterns: List of connection patterns
        n_timesteps: Total number of timesteps
        n_base_nodes: Number of nodes in base graph
        node_values: Optional dict of node values (for conditional patterns)
        
    Returns:
        List of (from_t, to_t, from_base, to_base, pattern, weight) tuples
    """
    edges = []
    
    for pattern in patterns:
        active_timesteps = pattern.get_active_timesteps(n_timesteps)
        
        if pattern.connection_type == ConnectionType.CONDITIONAL_LAG:
            # For conditional lag, we need node values
            # If not provided, use deterministic behavior
            if node_values is None:
                # Use middle skip
                skips = pattern.conditional_skips or [1]
                skip = skips[len(skips) // 2]
                for t in active_timesteps:
                    if t + skip < n_timesteps:
                        for src in pattern.source_nodes:
                            for tgt in pattern.target_nodes:
                                edges.append((t, t + skip, src, tgt, pattern, 1.0))
            else:
                # Skip determined at runtime by row_generator
                skips, weights = pattern.get_skip_values_and_weights()
                for skip, weight in zip(skips, weights):
                    for t in active_timesteps:
                        if t + skip < n_timesteps:
                            for src in pattern.source_nodes:
                                for tgt in pattern.target_nodes:
                                    edges.append((t, t + skip, src, tgt, pattern, weight))
        
        elif pattern.connection_type == ConnectionType.CONDITIONAL_DEST:
            # Similar - destinations determined at runtime
            skips, weights = pattern.get_skip_values_and_weights()
            for skip, weight in zip(skips, weights):
                for t in active_timesteps:
                    if t + skip < n_timesteps:
                        for src in pattern.source_nodes:
                            # All potential targets
                            for targets in (pattern.conditional_targets or [pattern.target_nodes]):
                                for tgt in targets:
                                    if tgt < n_base_nodes:
                                        edges.append((t, t + skip, src, tgt, pattern, weight))
        
        else:
            # Standard patterns - enumerate all edges
            skips, weights = pattern.get_skip_values_and_weights()
            
            for skip, weight in zip(skips, weights):
                for t in active_timesteps:
                    if t + skip < n_timesteps:
                        for src in pattern.source_nodes:
                            if src < n_base_nodes:
                                for tgt in pattern.target_nodes:
                                    if tgt < n_base_nodes:
                                        edges.append((t, t + skip, src, tgt, pattern, weight))
    
    return edges

