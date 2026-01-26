"""
Benchmark TabPFN on generated synthetic datasets.
"""

import numpy as np
import os
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier
import matplotlib.pyplot as plt

from dataset_generator import (
    generate_dataset, GeneratedDataset, 
    RandomNNConfig, DatasetConfig
)


def visualize_dataset(dataset: GeneratedDataset, filepath: str, title: str, result: dict = None):
    """Visualize a generated dataset with selected features and target."""
    X = np.concatenate([dataset.X_train, dataset.X_test], axis=0)
    y = np.concatenate([dataset.y_train, dataset.y_test], axis=0)
    
    n_samples, n_features, T = X.shape
    n_classes = dataset.n_classes
    
    # Show up to 5 samples per class
    samples_per_class = 1
    sample_indices = []
    for c in range(n_classes):
        class_idx = np.where(y == c)[0]
        if len(class_idx) > 0:
            selected = class_idx[:samples_per_class]
            sample_indices.extend(selected)
    
    n_show = len(sample_indices)
    
    fig, axes = plt.subplots(n_show, 1, figsize=(12, 2.5 * n_show), squeeze=False)
    
    # Title with metrics if available
    title_str = f"{title}\nShape: {n_samples} samples × {n_features} channels × {T} timesteps | {n_classes} classes\n"
    title_str += f"Mode: {dataset.sample_mode} | Target offset: {dataset.target_offset}"
    if result:
        title_str += f"\nAcc: {result['accuracy']:.3f} | AUC: {result['auc']:.3f}" if result.get('auc') else f"\nAcc: {result['accuracy']:.3f}"
    
    fig.suptitle(title_str, fontsize=11)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_features))
    
    for row, sample_idx in enumerate(sample_indices):
        ax = axes[row, 0]
        sample_class = y[sample_idx]
        
        for f in range(n_features):
            ax.plot(X[sample_idx, f, :], color=colors[f], label=f'F{f}', alpha=0.8)
        
        ax.set_ylabel('Value')
        ax.legend(loc='upper right', fontsize=8)
        
        # Add sample info box
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8)
        ax.text(0.02, 0.95, f'Sample {sample_idx} | Class: {sample_class}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=bbox_props)
    
    axes[-1, 0].set_xlabel('Time Step')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()


def flatten_dataset(dataset: GeneratedDataset):
    """Flatten 3D dataset to 2D for TabPFN."""
    # X_train: (n_samples, n_features, t) -> (n_samples, n_features * t)
    X_train_flat = dataset.X_train.reshape(dataset.X_train.shape[0], -1)
    X_test_flat = dataset.X_test.reshape(dataset.X_test.shape[0], -1)
    return X_train_flat, dataset.y_train, X_test_flat, dataset.y_test


def run_benchmark(n_datasets: int = 64, output_dir: str = None):
    """Run TabPFN benchmark on synthetic datasets."""
    
    if output_dir is None:
        output_dir = "/Users/franco/Documents/TabPFN3D/06_generator_experiments/benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Config - same as dataset_generator
    nn_config = RandomNNConfig(
        memory_dim_range=(1, 8),
        memory_init='uniform',
        stochastic_input_dim_range=(0, 0),
        n_time_transforms_range=(1, 5),
        n_hidden_layers_range=(2, 5),  # shallow networks
        n_nodes_per_layer_range=(4, 16),
        activation_choices=(
            'identity', 'log', 'sigmoid', 'abs', 'sin', 'tanh',
            'rank', 'square', 'power', 'softplus', 'step', 'modulo',
        ),
        weight_init_choices=('xavier_normal',),
        weight_scale_range=(0.9, 1.1),
        bias_std_range=(0.0, 0.1),
        node_noise_prob_range=(0.05, 0.30),
        node_noise_std_range=(0.01, 0.1),
        noise_dist_choices=('normal',),
        per_layer_activation=True,
        quantization_node_prob=0.0,  # no quantization in network
        quantization_n_classes_range=(2, 8),
        seq_length=200,
    )
    ds_config = DatasetConfig()
    
    # Results
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{output_dir}/results_{timestamp}.json"
    
    print("=" * 60)
    print("TabPFN Benchmark on Synthetic Datasets")
    print("=" * 60)
    print(f"\nGenerating and evaluating {n_datasets} datasets...")
    print(f"Output: {output_dir}\n")
    
    # Initialize TabPFN
    tabpfn = TabPFNClassifier(n_estimators=4, ignore_pretraining_limits=True)
    
    successful = 0
    for i in range(n_datasets):
        # Try multiple seeds
        for attempt in range(10):
            try:
                seed = 1000 + i * 100 + attempt  # Different seed base
                dataset = generate_dataset(nn_config, ds_config, seed=seed)
                
                # Flatten for TabPFN
                X_train, y_train, X_test, y_test = flatten_dataset(dataset)
                
                # Check dimensions
                n_train, n_flat_features = X_train.shape
                n_test = X_test.shape[0]
                n_classes = dataset.n_classes
                
                # Skip if too many features for TabPFN (limit ~500)
                if n_flat_features > 500:
                    continue
                
                # Train and predict
                tabpfn.fit(X_train, y_train)
                y_pred = tabpfn.predict(X_test)
                y_proba = tabpfn.predict_proba(X_test)
                
                # Metrics
                acc = accuracy_score(y_test, y_pred)
                
                # AUC (handle multiclass)
                if n_classes == 2:
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    try:
                        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                    except:
                        auc = None
                
                result = {
                    'dataset_id': int(i),
                    'seed': int(seed),
                    'n_train': int(n_train),
                    'n_test': int(n_test),
                    'n_features': int(dataset.X_train.shape[1]),
                    't_subseq': int(dataset.X_train.shape[2]),
                    'n_flat_features': int(n_flat_features),
                    'n_classes': int(n_classes),
                    'sample_mode': dataset.sample_mode,
                    'target_offset': int(dataset.target_offset),
                    'accuracy': float(acc),
                    'auc': float(auc) if auc is not None else None,
                }
                results.append(result)
                
                # Save JSON incrementally
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Generate visualization
                vis_path = f"{output_dir}/dataset{i:02d}.png"
                visualize_dataset(dataset, vis_path, f"Synthetic Dataset {i}", result)
                
                auc_str = f"{auc:.3f}" if auc else "N/A"
                print(f"  [{i+1:2d}/{n_datasets}] {n_train}×{dataset.X_train.shape[1]}×{dataset.X_train.shape[2]} | "
                      f"{n_classes} cls | {dataset.sample_mode:15s} | Acc: {acc:.3f} | AUC: {auc_str}")
                
                successful += 1
                break
                
            except ValueError as e:
                if "boring" in str(e).lower():
                    continue
                else:
                    print(f"  [{i+1:2d}/{n_datasets}] Error: {e}")
                    break
            except Exception as e:
                print(f"  [{i+1:2d}/{n_datasets}] Error: {e}")
                break
        else:
            print(f"  [{i+1:2d}/{n_datasets}] Failed after 10 attempts")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if results:
        accs = [r['accuracy'] for r in results]
        aucs = [r['auc'] for r in results if r['auc'] is not None]
        
        print(f"Successful datasets: {successful}/{n_datasets}")
        print(f"Accuracy: mean={np.mean(accs):.3f}, std={np.std(accs):.3f}, min={np.min(accs):.3f}, max={np.max(accs):.3f}")
        if aucs:
            print(f"AUC:      mean={np.mean(aucs):.3f}, std={np.std(aucs):.3f}, min={np.min(aucs):.3f}, max={np.max(aucs):.3f}")
        
        # By sample mode
        for mode in ['iid', 'sliding_window', 'mixed']:
            mode_results = [r for r in results if r['sample_mode'] == mode]
            if mode_results:
                mode_accs = [r['accuracy'] for r in mode_results]
                print(f"\n{mode:15s}: n={len(mode_results):2d}, Acc mean={np.mean(mode_accs):.3f}")
    
    # Final save
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    run_benchmark(n_datasets=64)
