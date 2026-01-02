"""
Wrapper for importing 2D generator components.

This module handles the import of DAG and transformation components from
02_synthetic_generator_2D without conflicting with the 3D config.
"""

import sys
import os

# Path to 2D generator
_2d_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        '..', '02_synthetic_generator_2D')
_2d_path = os.path.abspath(_2d_path)

# Save original path and modules
_original_path = sys.path.copy()
_original_modules = {k: v for k, v in sys.modules.items() if 'config' in k.lower()}

# Remove any conflicting config modules
for key in list(sys.modules.keys()):
    if key == 'config' or key.startswith('config.'):
        del sys.modules[key]

# Set path to only include 2D directory for imports
sys.path = [_2d_path] + [p for p in sys.path if '03_synthetic_generator_3D' not in p]

try:
    # Import 2D modules
    from dag_builder import DAG, Node, DAGBuilder
    from transformations import EdgeTransformation, TransformationFactory, DiscretizationTransformation
    from post_processing import Warper, MissingValueInjector
    
    # Alias for compatibility
    DAGNode = Node
finally:
    # Restore original path
    sys.path = _original_path
    
    # Remove the 2D config from modules to avoid conflicts
    if 'config' in sys.modules:
        del sys.modules['config']

# Export DAGNode alias
DAGNode = Node

