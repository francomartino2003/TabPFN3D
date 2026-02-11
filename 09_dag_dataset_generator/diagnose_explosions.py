#!/usr/bin/env python3
"""
Diagnose which datasets have extreme values and why.
Checks per-node value ranges after propagation to identify
which activations / DAG depths cause explosions.
"""

import numpy as np
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from dag_dataset_generator import (
    sample_hyperparams, build_dag, build_network, assign_roles,
    positional_encoding_matrix, sample_roots, propagate_dag,
    pool_over_d, HYPER_CONFIG,
)


def diagnose_one(rng, idx):
    hyper = sample_hyperparams(rng)
    dag = build_dag(hyper.V, hyper.M, rng, max_parents=HYPER_CONFIG["max_parents"])
    net = build_network(dag, hyper.d, hyper.kernel_size, hyper.gain, rng)
    feature_nodes, target_node = assign_roles(dag, hyper.n_features, rng)
    pe_matrix = positional_encoding_matrix(hyper.T, hyper.d)

    # Just propagate ONE sample to check ranges
    root_values = sample_roots(
        hyper.M, hyper.T, hyper.d,
        hyper.init_type, hyper.root_std, hyper.root_a,
        pe_matrix, rng,
    )
    all_values = propagate_dag(root_values, dag, net)

    # Check per-node ranges
    max_abs = 0.0
    has_inf = False
    has_nan = False
    node_info = []

    for node_id in dag.topo_order:
        v = all_values[node_id]
        finite = v[np.isfinite(v)]
        n_inf = np.isinf(v).sum()
        n_nan = np.isnan(v).sum()

        if n_inf > 0:
            has_inf = True
        if n_nan > 0:
            has_nan = True

        if len(finite) > 0:
            abs_max = np.abs(finite).max()
            max_abs = max(max_abs, abs_max)
        else:
            abs_max = float('inf')

        # Get activation for this node
        if node_id in dag.roots:
            act = "root"
            n_parents = 0
        else:
            act = net.node_params[node_id].activation
            n_parents = len(dag.parents[node_id])

        node_info.append({
            "id": node_id,
            "act": act,
            "n_parents": n_parents,
            "abs_max": abs_max,
            "inf": n_inf,
            "nan": n_nan,
            "mean": finite.mean() if len(finite) > 0 else float('nan'),
            "std": finite.std() if len(finite) > 0 else float('nan'),
        })

    # Classify
    is_exploding = has_inf or has_nan or max_abs > 1e10
    status = "EXPLODING" if is_exploding else ("LARGE" if max_abs > 1e4 else "OK")

    # Activation distribution
    acts = [ni["act"] for ni in node_info if ni["act"] != "root"]

    return {
        "idx": idx,
        "status": status,
        "max_abs": max_abs,
        "has_inf": has_inf,
        "has_nan": has_nan,
        "V": hyper.V,
        "M": hyper.M,
        "d": hyper.d,
        "T": hyper.T,
        "K": hyper.kernel_size,
        "gain": hyper.gain,
        "init": hyper.init_type,
        "root_std": hyper.root_std,
        "root_a": hyper.root_a,
        "activations": acts,
        "nodes": node_info,
    }


def main():
    n = 200
    seed = 42
    rng = np.random.default_rng(seed)

    results = []
    for i in range(n):
        r = diagnose_one(rng, i)
        results.append(r)

    # Summary
    exploding = [r for r in results if r["status"] == "EXPLODING"]
    large = [r for r in results if r["status"] == "LARGE"]
    ok = [r for r in results if r["status"] == "OK"]

    print(f"Total: {n} datasets")
    print(f"  OK:        {len(ok)}")
    print(f"  LARGE:     {len(large)}  (max_abs > 1e4)")
    print(f"  EXPLODING: {len(exploding)}  (inf/nan or max_abs > 1e10)")

    # Which activations appear in exploding datasets?
    print(f"\n--- Activation frequency in EXPLODING datasets ---")
    act_explode = Counter()
    for r in exploding:
        for a in r["activations"]:
            act_explode[a] += 1
    for a, c in act_explode.most_common():
        print(f"  {a:15s}  {c}")

    print(f"\n--- Activation frequency in OK datasets ---")
    act_ok = Counter()
    for r in ok:
        for a in r["activations"]:
            act_ok[a] += 1
    for a, c in act_ok.most_common():
        print(f"  {a:15s}  {c}")

    # Show which nodes explode in detail for first few
    print(f"\n--- Detail of first 10 EXPLODING datasets ---")
    for r in exploding[:10]:
        print(f"\n  Dataset {r['idx']}: V={r['V']} M={r['M']} d={r['d']} "
              f"T={r['T']} K={r['K']} gain={r['gain']:.2f} "
              f"init={r['init']} std={r['root_std']:.3f} a={r['root_a']:.3f}")
        print(f"  max_abs={r['max_abs']:.2e}  inf={r['has_inf']}  nan={r['has_nan']}")
        print(f"  Activations: {Counter(r['activations'])}")
        print(f"  Per-node:")
        for ni in r["nodes"]:
            flag = ""
            if ni["inf"] > 0 or ni["nan"] > 0 or ni["abs_max"] > 1e10:
                flag = " *** EXPLODED ***"
            print(f"    node {ni['id']:3d}  act={ni['act']:12s}  "
                  f"parents={ni['n_parents']}  "
                  f"|max|={ni['abs_max']:.2e}  "
                  f"mean={ni['mean']:.2e}  std={ni['std']:.2e}  "
                  f"inf={ni['inf']} nan={ni['nan']}{flag}")

    # Distribution of max_abs for OK datasets
    print(f"\n--- max_abs distribution for OK datasets ---")
    ok_maxes = sorted([r["max_abs"] for r in ok])
    if ok_maxes:
        print(f"  min={ok_maxes[0]:.2e}  median={ok_maxes[len(ok_maxes)//2]:.2e}  "
              f"max={ok_maxes[-1]:.2e}")
        # Histogram bins
        for threshold in [1, 10, 100, 1e3, 1e4]:
            count = sum(1 for m in ok_maxes if m <= threshold)
            print(f"  <= {threshold:.0e}: {count}/{len(ok_maxes)}")

    # DAG depth distribution: exploding vs OK
    print(f"\n--- DAG depth (V-M = non-root layers) ---")
    print(f"  EXPLODING: mean V-M = {np.mean([r['V']-r['M'] for r in exploding]):.1f}" if exploding else "")
    print(f"  OK:        mean V-M = {np.mean([r['V']-r['M'] for r in ok]):.1f}" if ok else "")


if __name__ == "__main__":
    main()
