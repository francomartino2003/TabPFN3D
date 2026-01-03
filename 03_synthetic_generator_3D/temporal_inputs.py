"""
Temporal Input Generators for 3D Synthetic Data.

Three types of inputs to the DAG root nodes:
1. Noise inputs: Random values (like 2D generator)
2. Time inputs: Functions of normalized time u = t/T
3. State inputs: Memory from previous timestep

Each input type has its own generator class.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from config import DatasetConfig3D


class NoiseInputGenerator:
    """
    Generates noise inputs for root nodes.
    
    Same as 2D generator: Normal, Uniform, or Mixed distributions.
    """
    
    def __init__(self, config: DatasetConfig3D, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.noise_type = config.noise_type
        self.sigma = config.init_sigma
        self.a = config.init_a
    
    def generate(self, n_samples: int, n_inputs: int) -> np.ndarray:
        """
        Generate noise values for all noise inputs.
        
        Args:
            n_samples: Number of samples
            n_inputs: Number of noise input nodes
            
        Returns:
            Array of shape (n_samples, n_inputs)
        """
        if self.noise_type == 'normal':
            return self.rng.normal(0, self.sigma, size=(n_samples, n_inputs))
        
        elif self.noise_type == 'uniform':
            return self.rng.uniform(-self.a, self.a, size=(n_samples, n_inputs))
        
        elif self.noise_type == 'mixed':
            # For each input, randomly choose normal or uniform
            output = np.zeros((n_samples, n_inputs))
            for i in range(n_inputs):
                if self.rng.random() < 0.5:
                    output[:, i] = self.rng.normal(0, self.sigma, size=n_samples)
                else:
                    output[:, i] = self.rng.uniform(-self.a, self.a, size=n_samples)
            return output
        
        else:
            # Default to normal
            return self.rng.normal(0, self.sigma, size=(n_samples, n_inputs))


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
        u = t / T  # Normalized time in [0, 1]
        
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
            # tanh(β(2u-1)) with β from params
            beta = self.params.get(f'tanh_beta_{idx}', 1.0)
            return np.tanh(beta * (2 * u - 1))
        
        elif activation.startswith('sin_'):
            # sin(2πku) where k is extracted from activation name
            k = int(activation.split('_')[1])
            return np.sin(2 * np.pi * k * u)
        
        elif activation.startswith('cos_'):
            # cos(2πku)
            k = int(activation.split('_')[1])
            return np.cos(2 * np.pi * k * u)
        
        elif activation == 'exp_decay':
            # exp(-γu) with γ from params
            gamma = self.params.get(f'exp_gamma_{idx}', 1.0)
            return np.exp(-gamma * u)
        
        else:
            # Default to linear
            return u


class StateInputGenerator:
    """
    Generates state (memory) inputs from previous timestep.
    
    State inputs are the values of designated "state nodes" from t-1,
    normalized using tanh(α·s_{t-1}) to keep values in [-1, 1].
    """
    
    def __init__(self, config: DatasetConfig3D, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.alpha = config.state_alpha
        self.n_state_inputs = config.n_state_inputs
        
        # For storing state node IDs (set later when DAG is built)
        self.state_node_ids: List[int] = []
    
    def set_state_nodes(self, node_ids: List[int]):
        """Set which nodes are state nodes."""
        self.state_node_ids = node_ids
    
    def generate_initial(self, n_samples: int) -> np.ndarray:
        """
        Generate initial state values for t=0.
        
        At t=0 there's no previous state, so we use noise.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Array of shape (n_samples, n_state_inputs)
        """
        # Use noise generator for initial state
        noise_gen = NoiseInputGenerator(self.config, self.rng)
        return noise_gen.generate(n_samples, self.n_state_inputs)
    
    def normalize_state(self, state_values: np.ndarray) -> np.ndarray:
        """
        Normalize state values using tanh(α·s).
        
        Args:
            state_values: Raw state values from previous timestep
            
        Returns:
            Normalized values in [-1, 1]
        """
        return np.tanh(self.alpha * state_values)
    
    def extract_state_values(self, propagated_values: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Extract state values from propagated node values.
        
        Args:
            propagated_values: Dict mapping node_id -> values array
            
        Returns:
            Array of shape (n_samples, n_state_inputs)
        """
        if not self.state_node_ids:
            raise ValueError("State node IDs not set. Call set_state_nodes first.")
        
        n_samples = len(list(propagated_values.values())[0])
        state_values = np.zeros((n_samples, len(self.state_node_ids)))
        
        for i, node_id in enumerate(self.state_node_ids):
            if node_id in propagated_values:
                state_values[:, i] = propagated_values[node_id]
            else:
                # Fallback: use zeros (shouldn't happen if DAG is built correctly)
                state_values[:, i] = 0.0
        
        return state_values


@dataclass
class TemporalInputManager:
    """
    Manages all three types of temporal inputs.
    
    Coordinates noise, time, and state inputs for each timestep.
    
    Supports noise_only_at_t0 mode where noise is generated once at t=0
    and then held constant (or propagated as state) for all timesteps.
    This increases autocorrelation in generated series.
    """
    
    noise_generator: NoiseInputGenerator
    time_generator: TimeInputGenerator
    state_generator: StateInputGenerator
    
    # Node ID assignments (set when DAG is built)
    noise_node_ids: List[int] = None
    time_node_ids: List[int] = None
    state_node_ids: List[int] = None
    
    # noise_only_at_t0 mode
    noise_only_at_t0: bool = False
    _cached_noise: Optional[np.ndarray] = None  # Cached noise from t=0
    
    @classmethod
    def from_config(cls, config: DatasetConfig3D, rng: np.random.Generator) -> 'TemporalInputManager':
        """Create from dataset config."""
        return cls(
            noise_generator=NoiseInputGenerator(config, rng),
            time_generator=TimeInputGenerator(config, rng),
            state_generator=StateInputGenerator(config, rng),
            noise_node_ids=[],
            time_node_ids=[],
            state_node_ids=[],
            noise_only_at_t0=getattr(config, 'noise_only_at_t0', False),
            _cached_noise=None
        )
    
    def assign_root_nodes(self, root_node_ids: List[int]):
        """
        Assign root nodes to input types.
        
        Args:
            root_node_ids: List of all root node IDs in the DAG
        """
        n_noise = self.noise_generator.config.n_noise_inputs
        n_time = len(self.time_generator.activations)
        n_state = self.state_generator.n_state_inputs
        
        total_needed = n_noise + n_time + n_state
        
        if len(root_node_ids) < total_needed:
            # Not enough root nodes - adjust proportionally
            scale = len(root_node_ids) / total_needed
            n_noise = max(1, int(n_noise * scale))
            n_time = max(1, int(n_time * scale))
            n_state = max(1, int(n_state * scale))
            
            # Adjust to exactly match
            while n_noise + n_time + n_state > len(root_node_ids):
                if n_state > 1:
                    n_state -= 1
                elif n_time > 1:
                    n_time -= 1
                else:
                    n_noise -= 1
        
        # INTERLEAVE assignment (not sequential blocks)
        # This ensures noise/time/state inputs are mixed in the DAG topology
        # instead of being grouped together
        
        shuffled = list(root_node_ids)
        np.random.shuffle(shuffled)
        
        # Create assignment order: interleave types
        # e.g., [noise, time, state, noise, time, state, ...]
        assignments = []
        for i in range(max(n_noise, n_time, n_state)):
            if i < n_noise:
                assignments.append('noise')
            if i < n_time:
                assignments.append('time')
            if i < n_state:
                assignments.append('state')
        
        # Shuffle the assignment order for more randomness
        np.random.shuffle(assignments)
        
        # Assign nodes according to interleaved order
        self.noise_node_ids = []
        self.time_node_ids = []
        self.state_node_ids = []
        
        for i, assignment in enumerate(assignments):
            if i >= len(shuffled):
                break
            node_id = shuffled[i]
            if assignment == 'noise':
                self.noise_node_ids.append(node_id)
            elif assignment == 'time':
                self.time_node_ids.append(node_id)
            else:  # state
                self.state_node_ids.append(node_id)
        
        # Set state nodes in state generator
        self.state_generator.set_state_nodes(self.state_node_ids)
    
    def generate_inputs_for_timestep(
        self, 
        t: int, 
        T: int, 
        n_samples: int,
        previous_state: Optional[np.ndarray] = None
    ) -> Dict[int, np.ndarray]:
        """
        Generate all root node inputs for a specific timestep.
        
        Args:
            t: Current timestep
            T: Total timesteps
            n_samples: Number of samples
            previous_state: State values from t-1 (None for t=0)
            
        Returns:
            Dict mapping node_id -> input values array of shape (n_samples,)
        """
        inputs = {}
        
        # Noise inputs
        if self.noise_node_ids:
            if self.noise_only_at_t0:
                # In noise_only_at_t0 mode: generate once at t=0, reuse thereafter
                if t == 0 or self._cached_noise is None:
                    noise_values = self.noise_generator.generate(n_samples, len(self.noise_node_ids))
                    self._cached_noise = noise_values
                else:
                    # Reuse cached noise (same noise at each timestep = more autocorrelation)
                    noise_values = self._cached_noise
                    # Handle n_samples mismatch
                    if noise_values.shape[0] != n_samples:
                        noise_values = self.noise_generator.generate(n_samples, len(self.noise_node_ids))
            else:
                # Standard mode: fresh noise each timestep
                noise_values = self.noise_generator.generate(n_samples, len(self.noise_node_ids))
            
            for i, node_id in enumerate(self.noise_node_ids):
                inputs[node_id] = noise_values[:, i]
        
        # Time inputs (same value for all samples at this timestep)
        if self.time_node_ids:
            time_values = self.time_generator.generate(t, T)
            for i, node_id in enumerate(self.time_node_ids):
                if i < len(time_values):
                    # Broadcast to all samples
                    inputs[node_id] = np.full(n_samples, time_values[i])
        
        # State inputs
        if self.state_node_ids:
            if t == 0 or previous_state is None:
                # Initial state: use noise
                state_values = self.state_generator.generate_initial(n_samples)
            else:
                # Normalize previous state
                state_values = self.state_generator.normalize_state(previous_state)
            
            for i, node_id in enumerate(self.state_node_ids):
                if i < state_values.shape[1]:
                    inputs[node_id] = state_values[:, i]
        
        return inputs
    
    def get_all_root_node_ids(self) -> List[int]:
        """Get all root node IDs (noise + time + state)."""
        return (self.noise_node_ids or []) + (self.time_node_ids or []) + (self.state_node_ids or [])

