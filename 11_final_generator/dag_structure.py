"""
DAG structure builder for the final dataset generator.

Builds a layered causal DAG:
  - Layer 0: single root node (latent, dimension d).
  - Layers 1…L: hidden nodes, each "series", "tabular" (continuous), or "discrete".
  - Fully connected between consecutive layers, with random connection drops.
  - Role assignment: some series nodes → features, one discrete node → target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from hyperparameters import DAGHyperparameters, RoleHyperparameters


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Node:
    """A single node in the DAG."""
    id: int
    layer: int
    node_type: str          # "root", "series", "tabular", or "discrete"
    parents: List[int] = field(default_factory=list)
    role: Optional[str] = None  # "feature", "target", or None

    def __repr__(self):
        parents_str = ','.join(str(p) for p in self.parents)
        role_str = f'  role={self.role}' if self.role else ''
        return (f'Node(id={self.id}, L{self.layer}, {self.node_type}, '
                f'parents=[{parents_str}]{role_str})')


@dataclass
class DAGStructure:
    """Complete DAG topology + role assignment."""
    nodes: List[Node]
    layers: List[List[int]]
    root_d: int
    connection_drop_prob: float
    series_node_prob: float
    discrete_node_prob: float

    # Convenience accessors
    @property
    def root(self) -> Node:
        return self.nodes[0]

    @property
    def feature_nodes(self) -> List[Node]:
        return [n for n in self.nodes if n.role == 'feature']

    @property
    def target_node(self) -> Optional[Node]:
        return next((n for n in self.nodes if n.role == 'target'), None)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def summary(self) -> str:
        lines = [
            f'DAG: {self.n_layers} layers, root_d={self.root_d}, '
            f'drop_p={self.connection_drop_prob:.2f}, '
            f'series_p={self.series_node_prob:.2f}, '
            f'discrete_p={self.discrete_node_prob:.2f}',
            f'Layers: {[len(l) for l in self.layers]} nodes',
        ]
        for l_idx, layer_ids in enumerate(self.layers):
            nodes_str = '  '.join(repr(self.nodes[nid]) for nid in layer_ids)
            lines.append(f'  L{l_idx}: {nodes_str}')
        feat = self.feature_nodes
        tgt = self.target_node
        lines.append(f'Features: {[n.id for n in feat]}  |  '
                      f'Target: {tgt.id if tgt else None}')
        return '\n'.join(lines)


# ── Sampling helpers ───────────────────────────────────────────────────────────

def _log_uniform_int(rng: np.random.Generator, lo: int, hi: int) -> int:
    if lo == hi:
        return lo
    log_val = rng.uniform(np.log(lo), np.log(hi))
    return int(np.clip(np.round(np.exp(log_val)), lo, hi))


# ── Builder ────────────────────────────────────────────────────────────────────

def build_dag(
    dag_hp: DAGHyperparameters,
    role_hp: RoleHyperparameters,
    rng: np.random.Generator,
) -> DAGStructure:
    """
    Sample and build a complete DAG structure.

    Steps:
      1. Sample global params.
      2. Create root node (layer 0).
      3. Create hidden layers: each node is series / discrete / tabular.
      3b. Guarantee ≥1 series and ≥1 discrete.
      4. Wire fully-connected edges with random drops.
      5. Assign roles: features from series, target from discrete.
      6. Prune layers below deepest role node.
      7. Re-index node ids 0…N-1.
    """

    # ── 1. Global parameters ──────────────────────────────────────────────

    root_d = _log_uniform_int(rng, *dag_hp.root_d_range)
    n_layers = _log_uniform_int(rng, *dag_hp.n_layers_range)       # log-uniform → favors smaller
    connection_drop_prob = rng.uniform(*dag_hp.connection_drop_prob_range)
    series_node_prob = rng.uniform(*dag_hp.series_node_prob_range)
    discrete_node_prob = rng.uniform(*dag_hp.discrete_node_prob_range)

    # ── 2. Root ───────────────────────────────────────────────────────────

    node_id = 0
    root = Node(id=node_id, layer=0, node_type='root')
    nodes = [root]
    layers: List[List[int]] = [[0]]
    node_id += 1

    # ── 3. Hidden layers ──────────────────────────────────────────────────

    for l_idx in range(1, n_layers + 1):
        n_nodes = _log_uniform_int(rng, *dag_hp.nodes_per_layer_range)  # favors smaller
        layer_ids = []
        for _ in range(n_nodes):
            if rng.random() < series_node_prob:
                ntype = 'series'
            elif rng.random() < discrete_node_prob:
                ntype = 'discrete'
            else:
                ntype = 'tabular'
            node = Node(id=node_id, layer=l_idx, node_type=ntype)
            nodes.append(node)
            layer_ids.append(node_id)
            node_id += 1
        layers.append(layer_ids)

    # ── 3b. Guarantee ≥1 series + ≥1 discrete ────────────────────────────

    non_root = [n for n in nodes if n.node_type != 'root']

    if not any(n.node_type == 'series' for n in non_root):
        convert = non_root[rng.integers(0, len(non_root))]
        convert.node_type = 'series'

    if not any(n.node_type == 'discrete' for n in non_root):
        # Pick a non-root, non-series node; if none, add one
        candidates = [n for n in non_root if n.node_type not in ('series',)]
        if candidates:
            convert = candidates[rng.integers(0, len(candidates))]
            convert.node_type = 'discrete'
        else:
            extra = Node(id=node_id, layer=len(layers) - 1, node_type='discrete')
            nodes.append(extra)
            layers[-1].append(node_id)
            node_id += 1

    # ── 4. Wire edges ─────────────────────────────────────────────────────

    for l_idx in range(1, len(layers)):
        parent_ids = layers[l_idx - 1]
        for child_id in layers[l_idx]:
            child = nodes[child_id]
            candidate_parents = list(parent_ids)
            if len(candidate_parents) > dag_hp.min_parents:
                kept = [p for p in candidate_parents
                        if rng.random() >= connection_drop_prob]
                if len(kept) < dag_hp.min_parents:
                    pool = [p for p in candidate_parents if p not in kept]
                    missing = dag_hp.min_parents - len(kept)
                    extra = rng.choice(pool, size=min(missing, len(pool)),
                                       replace=False)
                    kept.extend(extra.tolist())
                child.parents = sorted(kept)
            else:
                child.parents = candidate_parents

    # ── 5. Assign roles ───────────────────────────────────────────────────

    series_ids = [n.id for n in nodes if n.node_type == 'series']
    discrete_ids = [n.id for n in nodes if n.node_type == 'discrete']

    # Target: one discrete node (must exist after step 3b)
    assert len(discrete_ids) > 0, 'BUG: no discrete nodes after guarantee step'
    target_id = rng.choice(discrete_ids)
    nodes[target_id].role = 'target'
    assert nodes[target_id].node_type == 'discrete', \
        f'BUG: target node {target_id} is {nodes[target_id].node_type}, expected discrete'

    # Features: geometric number of series nodes
    n_features = min(
        rng.geometric(p=role_hp.n_features_geometric_p),
        role_hp.max_features,
        len(series_ids),
    )
    feature_ids = rng.choice(series_ids, size=n_features, replace=False)
    for fid in feature_ids:
        nodes[fid].role = 'feature'

    # ── 6. Prune below deepest role layer ─────────────────────────────────

    role_nodes = [n for n in nodes if n.role is not None]
    max_role_layer = max(n.layer for n in role_nodes)

    if max_role_layer < len(layers) - 1:
        pruned_ids = set()
        for l_idx in range(max_role_layer + 1, len(layers)):
            pruned_ids.update(layers[l_idx])
        nodes = [n for n in nodes if n.id not in pruned_ids]
        layers = layers[: max_role_layer + 1]
        for n in nodes:
            n.parents = [p for p in n.parents if p not in pruned_ids]

    # ── 7. Re-index 0…N-1 ────────────────────────────────────────────────

    old_to_new = {n.id: i for i, n in enumerate(nodes)}
    for i, n in enumerate(nodes):
        n.parents = sorted(old_to_new[p] for p in n.parents)
        n.id = i
    layers = [[old_to_new[oid] for oid in lids] for lids in layers]

    return DAGStructure(
        nodes=nodes,
        layers=layers,
        root_d=root_d,
        connection_drop_prob=connection_drop_prob,
        series_node_prob=series_node_prob,
        discrete_node_prob=discrete_node_prob,
    )


# ── Visualisation ──────────────────────────────────────────────────────────────

_TYPE_COLORS = {
    'root':     '#6C5CE7',  # purple
    'series':   '#00B894',  # green
    'tabular':  '#FDCB6E',  # yellow
    'discrete': '#74B9FF',  # blue
}

_ROLE_EDGE = {
    'feature': '#E17055',  # orange-red
    'target':  '#D63031',  # red
}

_TYPE_SHORT = {'root': 'R', 'series': 'S', 'tabular': 'T', 'discrete': 'D'}


def visualize_dag(dag: DAGStructure, save_path: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import os

    n_layers = dag.n_layers
    max_nodes = max(len(l) for l in dag.layers)

    fig_w = max(6, 2.2 * max_nodes)
    fig_h = max(4, 1.8 * n_layers)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    pos = {}
    for l_idx, layer_ids in enumerate(dag.layers):
        n = len(layer_ids)
        y = -l_idx
        x_start = -(n - 1) / 2.0
        for j, nid in enumerate(layer_ids):
            pos[nid] = (x_start + j, y)

    # Edges
    for node in dag.nodes:
        cx, cy = pos[node.id]
        for pid in node.parents:
            px, py = pos[pid]
            ax.annotate(
                '', xy=(cx, cy + 0.15), xytext=(px, py - 0.15),
                arrowprops=dict(arrowstyle='->', color='#636E72', lw=0.8,
                                connectionstyle='arc3,rad=0.05'),
            )

    # Nodes
    r = 0.18
    for node in dag.nodes:
        x, y = pos[node.id]
        fc = _TYPE_COLORS.get(node.node_type, '#DFE6E9')
        ec = _ROLE_EDGE.get(node.role, '#2D3436')
        lw = 3.0 if node.role else 1.2
        ax.add_patch(plt.Circle((x, y), r, facecolor=fc, edgecolor=ec,
                                 linewidth=lw, zorder=3))
        fs = 8 if len(dag.nodes) < 25 else 6
        ax.text(x, y, str(node.id), ha='center', va='center',
                fontsize=fs, fontweight='bold', zorder=4)
        if node.role:
            ax.text(x, y - r - 0.08, node.role.upper(), ha='center', va='top',
                    fontsize=6, color=ec, fontweight='bold', zorder=4)
        ax.text(x, y + r + 0.05, _TYPE_SHORT.get(node.node_type, '?'),
                ha='center', va='bottom', fontsize=6, color='#636E72', zorder=4)

    for l_idx in range(n_layers):
        ax.text(-(max_nodes) / 2.0 - 0.6, -l_idx, f'L{l_idx}',
                ha='right', va='center', fontsize=9, color='#2D3436',
                fontstyle='italic')

    legend_handles = [
        mpatches.Patch(facecolor=_TYPE_COLORS['root'],     edgecolor='#2D3436', label='Root'),
        mpatches.Patch(facecolor=_TYPE_COLORS['series'],   edgecolor='#2D3436', label='Series'),
        mpatches.Patch(facecolor=_TYPE_COLORS['tabular'],  edgecolor='#2D3436', label='Tabular'),
        mpatches.Patch(facecolor=_TYPE_COLORS['discrete'], edgecolor='#2D3436', label='Discrete'),
        mpatches.Patch(facecolor='white', edgecolor=_ROLE_EDGE['feature'], linewidth=2, label='Feature'),
        mpatches.Patch(facecolor='white', edgecolor=_ROLE_EDGE['target'],  linewidth=2, label='Target'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7, framealpha=0.9)
    ax.set_title(
        f'DAG  d={dag.root_d}  layers={[len(l) for l in dag.layers]}  '
        f'drop={dag.connection_drop_prob:.2f}  '
        f'ser={dag.series_node_prob:.2f}  disc={dag.discrete_node_prob:.2f}',
        fontsize=10,
    )
    ax.set_xlim(-(max_nodes) / 2.0 - 1, (max_nodes) / 2.0 + 1)
    ax.set_ylim(-n_layers + 0.5, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── CLI demo ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Sample and visualise DAG structures')
    parser.add_argument('--n', type=int, default=5, help='Number of DAGs to sample')
    parser.add_argument('--seed', type=int, default=0, help='Base seed')
    args = parser.parse_args()

    dag_hp = DAGHyperparameters()
    role_hp = RoleHyperparameters()
    out_dir = os.path.join(os.path.dirname(__file__), 'visualizations', 'dag_structure')

    for i in range(args.n):
        rng = np.random.default_rng(args.seed + i)
        dag = build_dag(dag_hp, role_hp, rng)
        print(f'\n{"="*60}')
        print(f'Seed {args.seed + i}')
        print(dag.summary())
        visualize_dag(dag, os.path.join(out_dir, f'dag_seed{args.seed + i}.png'))
