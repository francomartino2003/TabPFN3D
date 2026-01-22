"""
Temporal Input Generators for 3D Synthetic Data.

NEW DESIGN (v2):
- Two types of root inputs: TIME and STATE
- No direct noise inputs as roots
- State inputs: value of node X at t-k
- When t-k < 0, use noise initialization (provides sample variability)
- Each state input has a specific source node and lag

Input types:
1. Time inputs: Deterministic functions of normalized time u = t/T
2. State inputs: Value of specific node at t-k (with noise init if t-k < 0)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from config import DatasetConfig3D


class NoiseGenerator:
    """
    Generates noise for state initialization (when t-k < 0).
    
    Not used as direct root input anymore, only for initialization.
    """
    
    def __init__(self, config: DatasetConfig3D, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.noise_type = config.noise_type
        self.sigma = config.init_sigma
        self.a = config.init_a
    
    def generate(self, n_samples: int, n_values: int = 1) -> np.ndarray:
        """
        Generate noise values for initialization.
        
        Args:
            n_samples: Number of samples
            n_values: Number of noise values to generate per sample
            
        Returns:
            Array of shape (n_samples, n_values) or (n_samples,) if n_values=1
        """
        shape = (n_samples, n_values) if n_values > 1 else (n_samples,)
        
        if self.noise_type == 'normal':
            return self.rng.normal(0, self.sigma, size=shape)
        elif self.noise_type == 'uniform':
            return self.rng.uniform(-self.a, self.a, size=shape)
        elif self.noise_type == 'mixed':
            if n_values > 1:
                output = np.zeros(shape)
                for i in range(n_values):
                    if self.rng.random() < 0.5:
                        output[:, i] = self.rng.normal(0, self.sigma, size=n_samples)
                    else:
                        output[:, i] = self.rng.uniform(-self.a, self.a, size=n_samples)
                return output
            else:
                if self.rng.random() < 0.5:
                    return self.rng.normal(0, self.sigma, size=shape)
                else:
                    return self.rng.uniform(-self.a, self.a, size=shape)
        else:
            return self.rng.normal(0, self.sigma, size=shape)


class TimeInputGenerator:
    """
    Generates time-dependent inputs.
    
    Input is u = t/T (normalized time in [0, 1]).
    Various activation functions transform u into input values.
    """
    
    def __init__(self, config: DatasetConfig3D, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.activations = config.time_activations
        self.params = config.time_activation_params
    
    def generate(self, t: int, T: int) -> np.ndarray:
        """
        Generate time input values for a specific timestep.
        
        Args:
            t: Current timestep (0-indexed)
            T: Total number of timesteps
            
        Returns:
            Array of shape (n_time_inputs,) with values for each time input
        """
        u = t / max(T - 1, 1)  # Normalized time in [0, 1]
        
        values = []
        for i, activation in enumerate(self.activations):
            val = self._apply_activation(u, activation, i)
            values.append(val)
        
        return np.array(values)
    
    def generate_all_timesteps(self, T: int) -> np.ndarray:
        """
        Generate time inputs for all timesteps.
        
        Args:
            T: Total number of timesteps
            
        Returns:
            Array of shape (T, n_time_inputs)
        """
        return np.array([self.generate(t, T) for t in range(T)])
    
    def _apply_activation(self, u: float, activation: str, idx: int) -> float:
        """Apply a time activation function."""
        
        if activation == 'constant':
            return 1.0
        elif activation == 'linear':
            return u
        elif activation == 'quadratic':
            return u ** 2
        elif activation == 'cubic':
            return u ** 3
        elif activation == 'tanh':
            beta = self.params.get(f'tanh_beta_{idx}', 1.0)
            return np.tanh(beta * (2 * u - 1))
        elif activation.startswith('sin_'):
            k = int(activation.split('_')[1])
            return np.sin(2 * np.pi * k * u)
        elif activation.startswith('cos_'):
            k = int(activation.split('_')[1])
            return np.cos(2 * np.pi * k * u)
        elif activation == 'exp_decay':
            gamma = self.params.get(f'exp_gamma_{idx}', 1.0)
            return np.exp(-gamma * u)
        elif activation == 'log':
            return np.log(u + 0.1)
        else:
            return u


@dataclass
class StateInputConfig:
    """Configuration for a single state input."""
    source_node_id: int  # ID of the node whose past value to use
    lag: int             # How many timesteps back (k in t-k)
    root_node_id: int    # ID of the root node that receives this state


class StateInputManager:
    """
    Manages state inputs (node X at t-k).
    
    NEW DESIGN:
    - Each state input references a specific non-root node
    - When t-k < 0, use noise (provides initial variability)
    - When t-k >= 0, use the actual value of the source node at that timestep
    """
    
    def __init__(self, config: DatasetConfig3D, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.alpha = config.state_alpha
        self.n_state_inputs = config.n_state_inputs
        
        # Noise generator for initialization
        self.noise_gen = NoiseGenerator(config, rng)
        
        # State configurations: will be set after DAG is built
        # List of StateInputConfig
        self.state_configs: List[StateInputConfig] = []
        
        # Cache for noise initialization (per sample)
        self._init_noise_cache: Dict[int, np.ndarray] = {}
    
    def configure_states(
        self, 
        state_root_ids: List[int], 
        non_root_node_ids: List[int],
        lags: List[int],
        node_distances: Optional[Dict[int, int]] = None
    ):
        """
        Configure state inputs after DAG is built.
        
        Args:
            state_root_ids: IDs of root nodes that are state inputs
            non_root_node_ids: IDs of non-root nodes that can be state sources
            lags: List of lags (one per state input)
            node_distances: Optional dict mapping node_id -> distance to target
                           If provided, prefer nodes CLOSER to target (smaller distance)
        """
        self.state_configs = []
        
        for i, (root_id, lag) in enumerate(zip(state_root_ids, lags)):
            if not non_root_node_ids:
                # Fallback: use the first root as source (shouldn't happen normally)
                source_id = state_root_ids[0] if state_root_ids else 0
            elif node_distances is None:
                # No distance info: uniform random selection
                source_id = self.rng.choice(non_root_node_ids)
            else:
                # Distance-weighted selection: prefer CLOSER nodes to target
                # prob ∝ 1 / (1 + distance^alpha) where alpha from config
                alpha = getattr(self.config, 'state_source_distance_alpha', 2.0)
                distances = np.array([node_distances.get(nid, 100) for nid in non_root_node_ids])
                # Closer = higher probability
                weights = 1.0 / (1.0 + np.power(distances, alpha))
                probs = weights / weights.sum()
                source_id = self.rng.choice(non_root_node_ids, p=probs)
            
            self.state_configs.append(StateInputConfig(
                source_node_id=source_id,
                lag=lag,
                root_node_id=root_id
            ))
    
    def get_state_value(
        self, 
        state_config: StateInputConfig,
        t: int,
        n_samples: int,
        history: Dict[Tuple[int, int], np.ndarray]  # (t, node_id) -> values
    ) -> np.ndarray:
        """
        Get the value for a state input at timestep t.
        
        Args:
            state_config: Configuration for this state input
            t: Current timestep
            n_samples: Number of samples
            history: Dict of (timestep, node_id) -> values array
            
        Returns:
            Array of shape (n_samples,)
        """
        t_source = t - state_config.lag
        
        if t_source < 0:
            # Before sequence start: use noise initialization
            cache_key = (state_config.root_node_id, state_config.source_node_id)
            if cache_key not in self._init_noise_cache:
                self._init_noise_cache[cache_key] = self.noise_gen.generate(n_samples)
            return self._init_noise_cache[cache_key]
        else:
            # Get value from history
            key = (t_source, state_config.source_node_id)
            if key in history:
                raw_value = history[key]
                # Normalize with tanh
                return np.tanh(self.alpha * raw_value)
            else:
                # Fallback: noise (shouldn't happen if history is built correctly)
                return self.noise_gen.generate(n_samples)
    
    def reset_cache(self):
        """Reset the initialization noise cache (call between IID samples)."""
        self._init_noise_cache = {}


@dataclass
class TemporalInputManager:
    """
    Manages all temporal inputs (time and state).
    
    NEW DESIGN: Only time and state inputs, no noise roots.
    """
    
    time_generator: TimeInputGenerator
    state_manager: StateInputManager
    noise_gen: NoiseGenerator  # For any remaining noise needs
    
    # Node ID assignments (set when DAG is built)
    time_node_ids: List[int] = None
    state_node_ids: List[int] = None
    
    @classmethod
    def from_config(cls, config: DatasetConfig3D, rng: np.random.Generator) -> 'TemporalInputManager':
        """Create from dataset config."""
        return cls(
            time_generator=TimeInputGenerator(config, rng),
            state_manager=StateInputManager(config, rng),
            noise_gen=NoiseGenerator(config, rng),
            time_node_ids=[],
            state_node_ids=[],
        )
    
    def assign_root_nodes(
        self, 
        root_node_ids: List[int],
        non_root_node_ids: List[int],
        dag: Optional[Any] = None,
        target_node: Optional[int] = None
    ):
        """
        Assign root nodes to input types.
        
        Args:
            root_node_ids: List of all root node IDs in the DAG
            non_root_node_ids: List of non-root node IDs (for state sources)
            dag: Optional DAG object for computing distances
            target_node: Optional target node for distance-based source selection
        """
        n_time = len(self.time_generator.activations)
        n_state = self.state_manager.n_state_inputs
        
        total_needed = n_time + n_state
        
        # Adjust if not enough roots
        if len(root_node_ids) < total_needed:
            scale = len(root_node_ids) / total_needed
            n_time = max(1, int(n_time * scale))
            n_state = max(1, len(root_node_ids) - n_time)
        
        # Shuffle and assign
        shuffled = list(root_node_ids)
        self.state_manager.rng.shuffle(shuffled)
        
        self.time_node_ids = shuffled[:n_time]
        self.state_node_ids = shuffled[n_time:n_time + n_state]
        
        # Get lags from config
        lags = [cfg[1] for cfg in self.state_manager.config.state_configs]
        # Extend if needed
        while len(lags) < len(self.state_node_ids):
            lags.append(self.state_manager.rng.integers(1, 5))
        lags = lags[:len(self.state_node_ids)]
        
        # Compute distances to target if DAG and target are provided
        node_distances = None
        if dag is not None and target_node is not None:
            node_distances = self._compute_distances_to_target(dag, target_node)
        
        # Configure state manager with distance info (closer to target = preferred)
        self.state_manager.configure_states(
            self.state_node_ids,
            non_root_node_ids,
            lags,
            node_distances
        )
    
    def _compute_distances_to_target(self, dag: Any, target_node: int) -> Dict[int, int]:
        """
        Compute shortest path distance from each node to the target.
        
        Uses BFS on undirected graph (edges treated as bidirectional).
        Nodes closer to target will be preferred as state sources.
        """
        # Build undirected adjacency
        adjacency = {nid: set() for nid in dag.nodes}
        for nid, node in dag.nodes.items():
            for parent_id in node.parents:
                adjacency[nid].add(parent_id)
                adjacency[parent_id].add(nid)
        
        # BFS from target
        distances = {target_node: 0}
        queue = [target_node]
        
        while queue:
            current = queue.pop(0)
            current_dist = distances[current]
            
            for neighbor in adjacency.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        
        # Nodes not reachable get max distance
        max_dist = len(dag.nodes)
        for nid in dag.nodes:
            if nid not in distances:
                distances[nid] = max_dist
        
        return distances
    
    def generate_inputs_for_timestep(
        self, 
        t: int, 
        T: int, 
        n_samples: int,
        history: Dict[Tuple[int, int], np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """
        Generate all root node inputs for a specific timestep.
        
        Args:
            t: Current timestep
            T: Total timesteps
            n_samples: Number of samples
            history: Dict of (timestep, node_id) -> values for state lookups
            
        Returns:
            Dict mapping node_id -> input values array of shape (n_samples,)
        """
        inputs = {}
        
        # Time inputs (same value for all samples at this timestep)
        if self.time_node_ids:
            time_values = self.time_generator.generate(t, T)
            for i, node_id in enumerate(self.time_node_ids):
                if i < len(time_values):
                    inputs[node_id] = np.full(n_samples, time_values[i])
        
        # State inputs (from history or noise init)
        for state_config in self.state_manager.state_configs:
            value = self.state_manager.get_state_value(
                state_config, t, n_samples, history
            )
            inputs[state_config.root_node_id] = value
        
        return inputs
    
    def get_all_root_node_ids(self) -> List[int]:
        """Get all root node IDs (time + state)."""
        return (self.time_node_ids or []) + (self.state_node_ids or [])
    
    def reset_for_new_sequence(self):
        """Reset caches for generating a new independent sequence."""
        self.state_manager.reset_cache()
