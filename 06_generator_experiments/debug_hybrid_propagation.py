"""
Debug script to understand why t_features and z_features are boring.
Trace propagation step by step through each layer.
"""

import numpy as np
import matplotlib.pyplot as plt
from dag_hybrid_generator import HybridDAGGenerator, HybridDAGConfig


def debug_propagation(seed=42):
    """Debug propagation through the network."""
    
    config = HybridDAGConfig(
        z_dim_range=(4, 4),  # Fixed for debugging
        z_std_range=(0.5, 0.5),
        z_nn_layers_range=(3, 3),
        z_nn_nodes_range=(6, 6),
        t_nn_layers_range=(3, 3),
        t_nn_nodes_range=(6, 6),
        noise_std_range=(0.01, 0.01),
        n_roots_range=(4, 4),
        n_dag_nodes_range=(10, 10),
        seq_length=200,
    )
    
    gen = HybridDAGGenerator(config, seed=seed)
    cfg = gen.sampled_config
    
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"z_dim: {cfg.z_dim}, z_std: {cfg.z_std}")
    print(f"z_nn_layers: {cfg.z_nn_layers}, nodes: {cfg.z_nn_nodes_per_layer}")
    print(f"z_nn_activations: {cfg.z_nn_activations}")
    print()
    print(f"t_nn_layers: {cfg.t_nn_layers}, nodes: {cfg.t_nn_nodes_per_layer}")
    print(f"t_nn_activations: {cfg.t_nn_activations}")
    print()
    
    # =========================================================================
    # DEBUG t_nn propagation
    # =========================================================================
    print("=" * 70)
    print("t_nn PROPAGATION (step by step)")
    print("=" * 70)
    
    T = cfg.seq_length
    t = np.linspace(0, 1, T)  # Input: normalized time [0, 1]
    t_input = t.reshape(-1, 1)  # (T, 1)
    
    print(f"\nInput t: shape={t_input.shape}, range=[{t_input.min():.3f}, {t_input.max():.3f}]")
    
    x = t_input
    t_intermediates = [x.copy()]
    
    for layer_idx in range(cfg.t_nn_layers):
        W = gen.t_nn_weights[layer_idx]
        b = gen.t_nn_biases[layer_idx]
        act = cfg.t_nn_activations[layer_idx]
        
        print(f"\n--- Layer {layer_idx} ---")
        print(f"  W shape: {W.shape}, W range: [{W.min():.3f}, {W.max():.3f}]")
        print(f"  b shape: {b.shape}, b range: [{b.min():.3f}, {b.max():.3f}]")
        print(f"  activation: {act}")
        
        # Linear transform
        pre_act = x @ W.T + b
        print(f"  pre-activation: shape={pre_act.shape}, range=[{pre_act.min():.3f}, {pre_act.max():.3f}]")
        print(f"  pre-activation std per neuron: {np.std(pre_act, axis=0)}")
        
        # Activation
        x = gen._activation(pre_act, act)
        print(f"  post-activation: shape={x.shape}, range=[{x.min():.3f}, {x.max():.3f}]")
        print(f"  post-activation std per neuron: {np.std(x, axis=0)}")
        
        # Check for saturation
        if act in ['sigmoid', 'tanh']:
            saturated_high = (np.abs(x) > 0.99).mean() * 100
            print(f"  SATURATION WARNING: {saturated_high:.1f}% of values near ±1")
        
        t_intermediates.append(x.copy())
    
    # Final t_features (with raw t concatenated)
    t_features = np.concatenate([t_input, x], axis=1)
    print(f"\nFinal t_features: shape={t_features.shape}")
    
    # =========================================================================
    # DEBUG z_nn propagation
    # =========================================================================
    print("\n" + "=" * 70)
    print("z_nn PROPAGATION (step by step)")
    print("=" * 70)
    
    n_samples = 5
    z = gen.rng.normal(0, cfg.z_std, (n_samples, cfg.z_dim))
    
    print(f"\nInput z: shape={z.shape}, range=[{z.min():.3f}, {z.max():.3f}]")
    print(f"  z per sample: {z}")
    
    x = z
    z_intermediates = [x.copy()]
    
    for layer_idx in range(cfg.z_nn_layers):
        W = gen.z_nn_weights[layer_idx]
        b = gen.z_nn_biases[layer_idx]
        act = cfg.z_nn_activations[layer_idx]
        
        print(f"\n--- Layer {layer_idx} ---")
        print(f"  W shape: {W.shape}, W range: [{W.min():.3f}, {W.max():.3f}]")
        print(f"  b shape: {b.shape}, b range: [{b.min():.3f}, {b.max():.3f}]")
        print(f"  activation: {act}")
        
        # Linear transform
        pre_act = x @ W.T + b
        print(f"  pre-activation: shape={pre_act.shape}")
        print(f"  pre-activation values:\n{pre_act}")
        
        # Activation
        x = gen._activation(pre_act, act)
        print(f"  post-activation values:\n{x}")
        
        z_intermediates.append(x.copy())
    
    z_features = x
    print(f"\nFinal z_features: shape={z_features.shape}")
    print(f"z_features:\n{z_features}")
    
    # =========================================================================
    # VISUALIZE
    # =========================================================================
    fig, axes = plt.subplots(2, len(t_intermediates), figsize=(4 * len(t_intermediates), 8))
    
    # t_nn intermediates
    for i, inter in enumerate(t_intermediates):
        ax = axes[0, i]
        for j in range(min(6, inter.shape[1])):
            ax.plot(inter[:, j], label=f'n{j}')
        ax.set_title(f't_nn layer {i-1 if i > 0 else "input"}')
        ax.legend(fontsize=6)
    
    # z_nn intermediates (bar plot per sample)
    for i, inter in enumerate(z_intermediates):
        ax = axes[1, i]
        x_pos = np.arange(inter.shape[1])
        width = 0.15
        for s in range(n_samples):
            ax.bar(x_pos + s * width, inter[s], width, label=f's{s}')
        ax.set_title(f'z_nn layer {i-1 if i > 0 else "input"}')
        ax.legend(fontsize=6)
    
    plt.tight_layout()
    plt.savefig('/Users/franco/Documents/TabPFN3D/06_generator_experiments/debug_propagation.png', dpi=150)
    plt.close()
    
    print("\n" + "=" * 70)
    print("Saved visualization to debug_propagation.png")
    print("=" * 70)
    
    return gen, t_intermediates, z_intermediates


if __name__ == "__main__":
    debug_propagation(seed=42)
