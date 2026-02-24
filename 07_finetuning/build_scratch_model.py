"""
Build a from-scratch TabPFN model (no pretrained weights) with configurable nlayers.

This creates a PerFeatureTransformer with:
  - features_per_group = 8
  - All weights Xavier-initialized (no pretrained checkpoint loaded)
  - Temporal add_embeddings patch (feature_emb + sinusoidal PE)
  - Wrapped in TabPFNClassifier shell for forward_single_dataset compatibility
"""

import types

import torch
from torch import nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / '00_TabPFN' / 'src'))

from tabpfn import TabPFNClassifier
from tabpfn.architectures.base.config import ModelConfig
from tabpfn.architectures.base.transformer import PerFeatureTransformer
from tabpfn.architectures.base import get_encoder
from tabpfn.architectures.base.encoders import LinearInputEncoderStep

from tabpfn_temporal import (
    _reinit_module,
    temporal_add_embeddings,
    set_temporal_info,
    clear_temporal_info,
)


def build_tabpfn_from_scratch(
    nlayers: int = 12,
    emsize: int = 192,
    nhead: int = 6,
    fpg: int = 8,
    device: str = "auto",
) -> tuple:
    """Build a TabPFN from scratch — no pretrained weights.

    Returns (model, clf, all_params) where all_params is a list of every
    trainable parameter (all are fresh).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  Building from-scratch TabPFN: nlayers={nlayers}, emsize={emsize}, "
          f"nhead={nhead}, fpg={fpg}")

    config = ModelConfig(
        max_num_classes=10,
        num_buckets=0,
        emsize=emsize,
        nhead=nhead,
        nlayers=nlayers,
        nhid_factor=4,
        features_per_group=fpg,
        dropout=0.0,
        feature_positional_embedding="subspace",
        encoder_use_bias=False,
        encoder_type="linear",
        recompute_layer=True,
        recompute_attn=False,
    )

    encoder = get_encoder(
        num_features=fpg,
        embedding_size=emsize,
        remove_empty_features=True,
        remove_duplicate_features=False,
        nan_handling_enabled=True,
        normalize_on_train_only=True,
        normalize_to_ranking=False,
        normalize_x=True,
        remove_outliers=False,
        normalize_by_used_features=True,
        encoder_use_bias=False,
        encoder_type="linear",
    )

    model = PerFeatureTransformer(
        config=config,
        encoder=encoder,
        n_out=10,
        activation="gelu",
        zero_init=False,
    )

    _reinit_module(model)
    print(f"  All weights Xavier-initialized from scratch")

    model.add_embeddings = types.MethodType(temporal_add_embeddings, model)
    print(f"  Patched add_embeddings: feature_emb(j) + sinusoidal_PE(group_idx)")

    model.to(device)

    # Wrap in a real TabPFNClassifier shell, then swap in our model
    clf = TabPFNClassifier(
        device=device,
        n_estimators=1,
        ignore_pretraining_limits=True,
        fit_mode="batched",
        differentiable_input=False,
        inference_config={"FEATURE_SHIFT_METHOD": None},
    )
    clf._initialize_model_variables()
    clf.models_ = [model]

    all_params = list(model.parameters())

    n_params = sum(p.numel() for p in all_params)
    print(f"  Total parameters: {n_params:,}  (all fresh)")

    return model, clf, all_params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlayers", type=int, default=12)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model, clf, params = build_tabpfn_from_scratch(
        nlayers=args.nlayers, device=args.device)
    print(f"\nModel layers: {len(list(model.transformer_encoder.layers))}")
    print(f"Params: {sum(p.numel() for p in params):,}")
