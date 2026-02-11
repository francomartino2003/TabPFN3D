"""
Temporal Input Generators for 3D Synthetic Data.

v4 Design:
- 1 base TIME input: u = t/T (normalized, no activation)
- 0-7 extra TIME inputs with activations (sin, cos, tanh, etc.)
- MEMORY vector: sampled ONCE per sequence (1-8 dimensions)
  - For IID: each sample gets its own MEMORY (gives variability)
  - For sliding/mixed: one MEMORY per long sequence

Input types:
1. TIME base: u = t/T (provides linear temporal dependency)
2. TIME extra: f(u) with various activations (periodic, S-curves, etc.)
3. MEMORY: fixed vector per sequence (provides sample variability)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

from config import DatasetConfig3D


# Time activation functions
TIME_ACTIVATIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    'sin': lambda u: np.sin(2 * np.pi * u),           # [-1, 1] periodic
    'cos': lambda u: np.cos(2 * np.pi * u),           # [-1, 1] periodic
    'sin2': lambda u: np.sin(4 * np.pi * u),          # [-1, 1] higher freq
    'cos2': lambda u: np.cos(4 * np.pi * u),          # [-1, 1] higher freq
    'tanh': lambda u: np.tanh(4 * (u - 0.5)),         # [-1, 1] S-curve
    'sigmoid': lambda u: 1 / (1 + np.exp(-10 * (u - 0.5))),  # [0, 1] S-curve
    'sqrt': lambda u: np.sqrt(np.clip(u, 0, 1)),      # [0, 1] concave
    'square': lambda u: u ** 2,                        # [0, 1] convex
    'exp': lambda u: np.exp(u) - 1,                   # [0, e-1] exponential
    'log': lambda u: np.log1p(u),                     # [0, log(2)] logarithmic
}


class MemoryGenerator:
    """
    Generates MEMORY vectors for sequences.
    
    v4 Design:
    - MEMORY is sampled ONCE per sequence
    - Provides variability between samples (especially in IID mode)
    - Does NOT change within a sequence
    """
    
    def __init__(self, config: DatasetConfig3D, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.memory_dim = config.memory_dim
        self.noise_type = config.memory_noise_type
        self.sigma = config.memory_sigma
        self.a = config.memory_a
    
        # Current memory vectors (shape: n_samples, memory_dim)
        self._memory: Optional[np.ndarray] = None
    
    def sample_memory(self, n_samples: int) -> np.ndarray:
        """
        Sample new MEMORY vectors for n_samples.
        
        Args:
            n_samples: Number of samples (sequences)
            
        Returns:
            Array of shape (n_samples, memory_dim)
        """
        if self.noise_type == 'normal':
            self._memory = self.rng.normal(0, self.sigma, size=(n_samples, self.memory_dim))
        elif self.noise_type == 'uniform':
            self._memory = self.rng.uniform(-self.a, self.a, size=(n_samples, self.memory_dim))
        else:
            # Default to normal
            self._memory = self.rng.normal(0, self.sigma, size=(n_samples, self.memory_dim))
        
        return self._memory
    
    def get_memory(self) -> np.ndarray:
        """Get current MEMORY vectors."""
        if self._memory is None:
            raise ValueError("Memory not yet sampled. Call sample_memory first.")
        return self._memory
    
    def get_memory_component(self, idx: int) -> np.ndarray:
        """
        Get a specific component of the MEMORY vector for all samples.
        
        Args:
            idx: Index of the memory component (0 to memory_dim-1)
            
        Returns:
            Array of shape (n_samples,)
        """
        if self._memory is None:
            raise ValueError("Memory not yet sampled. Call sample_memory first.")
        return self._memory[:, idx]
    
    def reset(self):
        """Reset memory (call before new sequence in IID mode)."""
        self._memory = None


class TimeInputGenerator:
    """
    Generates TIME inputs.
    
    v4 Design:
    - 1 base input: u = t/T (normalized time, no activation)
    - 0-7 extra inputs: f(u) with various activations
    - Same values for all samples at timestep t
    """
    
    def __init__(self, config: DatasetConfig3D, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.n_extra = config.n_extra_time_inputs
        self.activations = config.time_input_activations
    
    def generate_base(self, t: int, T: int, n_samples: int) -> np.ndarray:
        """
        Generate base TIME input (u = t/T) for timestep t.
        
        Args:
            t: Current timestep (0-indexed)
            T: Total number of timesteps
            n_samples: Number of samples
            
        Returns:
            Array of shape (n_samples,) with value u = t/T
        """
        u = t / max(T - 1, 1)  # Normalized time in [0, 1]
        return np.full(n_samples, u, dtype=np.float32)
    
    def generate_extra(self, t: int, T: int, n_samples: int, activation_idx: int) -> np.ndarray:
        """
        Generate extra TIME input with activation for timestep t.
        
        Args:
            t: Current timestep (0-indexed)
            T: Total number of timesteps
            n_samples: Number of samples
            activation_idx: Index into self.activations
            
        Returns:
            Array of shape (n_samples,) with activated time value
        """
        u = t / max(T - 1, 1)  # Normalized time in [0, 1]
        
        if activation_idx >= len(self.activations):
            return np.full(n_samples, u, dtype=np.float32)
        
        activation_name = self.activations[activation_idx]
        if activation_name in TIME_ACTIVATIONS:
            value = TIME_ACTIVATIONS[activation_name](u)
        else:
            value = u  # Fallback to linear
        
        return np.full(n_samples, value, dtype=np.float32)
    
    def get_total_time_inputs(self) -> int:
        """Get total number of TIME inputs (1 base + n_extra)."""
        return 1 + self.n_extra


@dataclass
class TemporalInputManager:
    """
    Manages all temporal inputs (TIME and MEMORY).
    
    v4 Design:
    - 1 base TIME input (u = t/T)
    - 0-7 extra TIME inputs with activations
    - memory_dim MEMORY inputs (fixed per sequence)
    - Total roots = 1 + n_extra_time_inputs + memory_dim
    """
    
    time_generator: TimeInputGenerator
    memory_generator: MemoryGenerator
    config: DatasetConfig3D
    
    # Node ID assignments (set when DAG is built)
    time_base_node_id: Optional[int] = None      # Base TIME (u = t/T)
    time_extra_node_ids: List[int] = None        # Extra TIME with activations
    memory_node_ids: List[int] = None            # MEMORY inputs
    
    @classmethod
    def from_config(cls, config: DatasetConfig3D, rng: np.random.Generator) -> 'TemporalInputManager':
        """Create from dataset config."""
        return cls(
            time_generator=TimeInputGenerator(config, rng),
            memory_generator=MemoryGenerator(config, rng),
            config=config,
            time_base_node_id=None,
            time_extra_node_ids=[],
            memory_node_ids=[],
        )
    
    def assign_root_nodes(self, root_node_ids: List[int]):
        """
        Assign root nodes to input types.
        
        Order: [base TIME, extra TIME..., MEMORY...]
        
        Args:
            root_node_ids: List of all root node IDs in the DAG
        """
        n_extra_time = self.config.n_extra_time_inputs
        memory_dim = self.config.memory_dim
        total_needed = 1 + n_extra_time + memory_dim
        
        if len(root_node_ids) < total_needed:
            raise ValueError(f"Need {total_needed} roots but only got {len(root_node_ids)}")
        
        idx = 0
        
        # First root is base TIME (u = t/T)
        self.time_base_node_id = root_node_ids[idx]
        idx += 1
        
        # Next are extra TIME inputs with activations
        self.time_extra_node_ids = root_node_ids[idx:idx + n_extra_time]
        idx += n_extra_time
        
        # Rest are MEMORY
        self.memory_node_ids = root_node_ids[idx:idx + memory_dim]
    
    def sample_memory_for_sequences(self, n_samples: int):
        """
        Sample MEMORY vectors for n_samples sequences.
        Call this BEFORE generating timesteps.
        
        For IID mode: n_samples = number of independent sequences
        For sliding/mixed: n_samples = batch size for the long sequence
        """
        self.memory_generator.sample_memory(n_samples)
    
    def generate_inputs_for_timestep(
        self, 
        t: int, 
        T: int, 
        n_samples: int
    ) -> Dict[int, np.ndarray]:
        """
        Generate all root node inputs for a specific timestep.
        
        Args:
            t: Current timestep
            T: Total timesteps
            n_samples: Number of samples
            
        Returns:
            Dict mapping node_id -> input values array of shape (n_samples,)
        """
        inputs = {}
        
        # Base TIME input (u = t/T)
        if self.time_base_node_id is not None:
            inputs[self.time_base_node_id] = self.time_generator.generate_base(t, T, n_samples)
        
        # Extra TIME inputs with activations
        for i, node_id in enumerate(self.time_extra_node_ids or []):
            inputs[node_id] = self.time_generator.generate_extra(t, T, n_samples, i)
        
        # MEMORY inputs (fixed per sequence)
        for i, node_id in enumerate(self.memory_node_ids or []):
            inputs[node_id] = self.memory_generator.get_memory_component(i)
        
        return inputs
    
    def get_all_root_node_ids(self) -> List[int]:
        """Get all root node IDs (TIME base + TIME extra + MEMORY)."""
        result = []
        if self.time_base_node_id is not None:
            result.append(self.time_base_node_id)
        result.extend(self.time_extra_node_ids or [])
        result.extend(self.memory_node_ids or [])
        return result
    
    def get_time_node_ids(self) -> List[int]:
        """Get all TIME node IDs (base + extra)."""
        result = []
        if self.time_base_node_id is not None:
            result.append(self.time_base_node_id)
        result.extend(self.time_extra_node_ids or [])
        return result
    
    def reset_for_new_sequence(self):
        """Reset state for generating a new independent sequence."""
        self.memory_generator.reset()
