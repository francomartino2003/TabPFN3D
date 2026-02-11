#!/usr/bin/env python3
"""
Analyze NN activations: which are monotonic, and min/max ranges.

For sin/cos, we need to know the theoretical min/max given input range.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

# All NN activations
ACTIVATIONS = {
    'identity': lambda x: x,
    'relu': lambda x: np.maximum(0, x),
    'tanh': lambda x: np.tanh(x),
    'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
    'softplus': lambda x: np.log1p(np.exp(np.clip(x, -500, 500))),
    'sin': lambda x: np.sin(x),
    'cos': lambda x: np.cos(x),
    'abs': lambda x: np.abs(x),
    'square': lambda x: x ** 2,
    'log': lambda x: np.log(np.abs(x) + 1e-6),
    'step': lambda x: (x > 0).astype(float),
    'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x),
    'elu': lambda x: np.where(x > 0, x, np.exp(x) - 1),
    'rank': lambda x: np.array([0.5] * len(x)) if len(x) <= 1 else np.arange(len(x)) / (len(x) - 1),
    'power': lambda x: np.sign(x) * np.power(np.abs(x) + 1e-10, 2.0),  # p=2 for analysis
    'mod': lambda x: np.mod(x, 2.0),
}

def analyze_activation(name: str, fn, input_min: float, input_max: float) -> Dict:
    """Analyze an activation function."""
    # Test on range
    x_test = np.linspace(input_min, input_max, 1000)
    y_test = fn(x_test)
    
    # Check monotonicity
    # For periodic functions (sin, cos), check if they're truly monotonic
    # by looking at derivative sign changes
    is_monotonic = True
    is_increasing = True
    is_decreasing = True
    
    # Special case: sin/cos are periodic, not monotonic
    if name in ['sin', 'cos']:
        is_monotonic = False
        is_increasing = False
        is_decreasing = False
    else:
        for i in range(1, len(y_test)):
            if y_test[i] > y_test[i-1]:
                is_decreasing = False
            elif y_test[i] < y_test[i-1]:
                is_increasing = False
            else:
                pass  # Equal, still monotonic
        
        if not is_increasing and not is_decreasing:
            is_monotonic = False
    
    # Get min/max
    y_min = np.min(y_test)
    y_max = np.max(y_test)
    
    # For sin/cos, get theoretical bounds
    theoretical_min = None
    theoretical_max = None
    
    if name == 'sin':
        theoretical_min = -1.0
        theoretical_max = 1.0
    elif name == 'cos':
        theoretical_min = -1.0
        theoretical_max = 1.0
    elif name == 'tanh':
        theoretical_min = -1.0
        theoretical_max = 1.0
    elif name == 'sigmoid':
        theoretical_min = 0.0
        theoretical_max = 1.0
    elif name == 'abs':
        theoretical_min = 0.0
        theoretical_max = max(abs(input_min), abs(input_max))
    elif name == 'square':
        theoretical_min = 0.0
        theoretical_max = max(input_min**2, input_max**2)
    elif name == 'relu':
        theoretical_min = 0.0
        theoretical_max = max(0, input_max)
    elif name == 'leaky_relu':
        theoretical_min = min(0, 0.01 * input_min)
        theoretical_max = max(0, input_max)
    elif name == 'elu':
        theoretical_min = -1.0  # exp(x) - 1 when x << 0
        theoretical_max = max(0, input_max)
    elif name == 'softplus':
        theoretical_min = 0.0
        theoretical_max = np.log1p(np.exp(input_max))
    elif name == 'log':
        theoretical_min = np.log(abs(input_min) + 1e-6)
        theoretical_max = np.log(abs(input_max) + 1e-6)
    elif name == 'step':
        theoretical_min = 0.0
        theoretical_max = 1.0
    elif name == 'mod':
        theoretical_min = 0.0
        theoretical_max = 2.0
    elif name == 'identity':
        theoretical_min = input_min
        theoretical_max = input_max
    elif name == 'power':
        # For p=2, sign(x) * |x|^2
        theoretical_min = min(0, -input_max**2) if input_min < 0 else 0
        theoretical_max = max(input_min**2, input_max**2)
    elif name == 'rank':
        theoretical_min = 0.0
        theoretical_max = 1.0
    
    return {
        'name': name,
        'is_monotonic': is_monotonic,
        'is_increasing': is_increasing,
        'is_decreasing': is_decreasing,
        'empirical_min': y_min,
        'empirical_max': y_max,
        'theoretical_min': theoretical_min,
        'theoretical_max': theoretical_max,
        'input_range': (input_min, input_max),
    }

def main():
    print("="*80)
    print("NN ACTIVATION ANALYSIS")
    print("="*80)
    print("\nAssuming input range: [0, 1] (after normalization)\n")
    
    input_min = 0.0
    input_max = 1.0
    
    results = []
    
    for name, fn in ACTIVATIONS.items():
        result = analyze_activation(name, fn, input_min, input_max)
        results.append(result)
    
    # Sort by monotonicity
    monotonic = [r for r in results if r['is_monotonic']]
    non_monotonic = [r for r in results if not r['is_monotonic']]
    
    print("MONOTONIC ACTIVATIONS:")
    print("-" * 80)
    for r in sorted(monotonic, key=lambda x: x['name']):
        direction = "increasing" if r['is_increasing'] else "decreasing"
        theo_range = f"[{r['theoretical_min']:.3f}, {r['theoretical_max']:.3f}]"
        print(f"  {r['name']:15s} | {direction:10s} | Range: {theo_range}")
    
    print("\nNON-MONOTONIC ACTIVATIONS:")
    print("-" * 80)
    if non_monotonic:
        for r in sorted(non_monotonic, key=lambda x: x['name']):
            theo_range = f"[{r['theoretical_min']:.3f}, {r['theoretical_max']:.3f}]"
            emp_range = f"[{r['empirical_min']:.3f}, {r['empirical_max']:.3f}]"
            print(f"  {r['name']:15s} | Range: {theo_range} (empirical: {emp_range})")
    else:
        print("  (none)")
    
    print("\n" + "="*80)
    print("SUMMARY FOR NORMALIZATION:")
    print("="*80)
    print("\nFor min-max normalization to [0, 1]:")
    print("\n1. MONOTONIC (can use theoretical min/max):")
    for r in sorted(monotonic, key=lambda x: x['name']):
        print(f"   {r['name']:15s}: min={r['theoretical_min']:.3f}, max={r['theoretical_max']:.3f}")
    
    print("\n2. NON-MONOTONIC (periodic or need special handling):")
    if non_monotonic:
        for r in sorted(non_monotonic, key=lambda x: x['name']):
            if r['theoretical_min'] is not None:
                print(f"   {r['name']:15s}: min={r['theoretical_min']:.3f}, max={r['theoretical_max']:.3f} (FIXED RANGE)")
            else:
                print(f"   {r['name']:15s}: evaluate at input_min and input_max")
    else:
        print("   (none)")
    
    print("\n3. SPECIAL CASES:")
    print("   sin/cos: Always [-1, 1] regardless of input")
    print("   rank: Always [0, 1] (normalizes across samples)")
    print("   step: Always {0, 1}")

if __name__ == '__main__':
    main()
