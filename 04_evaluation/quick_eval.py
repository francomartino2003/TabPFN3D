#!/usr/bin/env python3
"""Quick single-run evaluation: finetuned vs vanilla TabPFN, same schedule.

Original schedule with T>96 guard:
  - 8 iterations: shuffle + class perm each
  - Odd iters: squash (IQR + soft clip)
  - Iters 2,3,6,7: global pool K=16 S=8  ** only if T > 96 **
  - Per-channel normalise, softmax temp 0.9

Two phases to avoid OOM: finetuned (MPS) first, then vanilla (CPU).
"""
from __future__ import annotations

import sys, time, gc
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "00_TabPFN" / "src"))
sys.path.insert(0, str(ROOT / "03_finetuning"))

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import wilcoxon
from tabpfn import TabPFNClassifier
from tabpfn.preprocessing.configs import ClassifierEnsembleConfig, PreprocessorConfig

from model import (
    WINDOW, pad_and_expand_overlap, set_temporal_info,
    set_global_input, build_overlap_model, per_channel_normalize,
)

# ── Config ────────────────────────────────────────────────────────────────────
SOFTMAX_TEMPERATURE = 0.9
MAX_M_TIMES_T       = 2000
SUBSAMPLE_N         = 1_000
SUBSAMPLE_SEED      = 0
SEED                = 42
POOL_K, POOL_S      = 16, 8
POOL_T_THRESH       = 96     # pool only if T > 96

CKPT_PATH = ROOT / "03_finetuning/checkpoints/phase2/best.pt"
DATA_ROOT = ROOT / "01_real_data"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _soft_clip(x, B=3.0):
    return x / np.sqrt(1.0 + (x / B) ** 2)

def _squash(X_tr, X_te, m, T):
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for j in range(m):
        c0, c1 = j * T, j * T + T
        vals = X_tr[:, c0:c1].ravel()
        fin  = vals[np.isfinite(vals)]
        if len(fin) == 0:
            X_tr[:, c0:c1] = 0; X_te[:, c0:c1] = 0; continue
        med = np.median(fin)
        q_lo, q_hi = np.percentile(fin, 25), np.percentile(fin, 75)
        s = (1.0/(q_hi-q_lo) if q_hi != q_lo
             else (2.0/(np.max(fin)-np.min(fin)) if np.max(fin) != np.min(fin) else 0))
        if s == 0:
            X_tr[:, c0:c1] = 0; X_te[:, c0:c1] = 0; continue
        X_tr[:, c0:c1] = _soft_clip((X_tr[:, c0:c1] - med) * s)
        X_te[:, c0:c1] = _soft_clip((X_te[:, c0:c1] - med) * s)
    return X_tr.astype(np.float32), X_te.astype(np.float32)

def _shuffle(X, m, T, perm):
    return X.reshape(X.shape[0], m, T)[:, perm, :].reshape(X.shape[0], m * T)

def _global_pool(X_flat, m, T):
    from numpy.lib.stride_tricks import sliding_window_view
    if T < POOL_K: return None
    n  = X_flat.shape[0]
    w  = sliding_window_view(X_flat.reshape(n, m, T), POOL_K, axis=2)[:, :, ::POOL_S, :]
    Tn = w.shape[2]
    out = np.concatenate([w.mean(3), w.max(3), w.min(3)], axis=1).reshape(n, 3*m*Tn).astype(np.float32)
    return out, 3*m, Tn

def _pcn(X_tr_3d, X_te_3d):
    X_tr_3d, X_te_3d = X_tr_3d.copy().astype(np.float64), X_te_3d.copy().astype(np.float64)
    for j in range(X_tr_3d.shape[1]):
        vals = X_tr_3d[:, j, :].ravel()
        fin  = vals[np.isfinite(vals)]
        if len(fin) == 0: continue
        mu, std = fin.mean(), fin.std()
        if std < 1e-8: std = 1.0
        X_tr_3d[:, j, :] = (X_tr_3d[:, j, :] - mu) / std
        X_te_3d[:, j, :] = (X_te_3d[:, j, :] - mu) / std
    return X_tr_3d.astype(np.float32), X_te_3d.astype(np.float32)


def _iter_preprocess(X_tr, X_te, y_tr, n_classes, m, T, rng, it):
    """Original schedule: shuffle, squash odd, pool {2,3,6,7} if T>96, pcn."""
    fp = rng.permutation(m)
    X_tr_p = _shuffle(X_tr, m, T, fp)
    X_te_p = _shuffle(X_te, m, T, fp)
    cp = rng.permutation(n_classes)
    y_tr_p = cp[y_tr]

    if it % 2 == 1:
        X_tr_p, X_te_p = _squash(X_tr_p, X_te_p, m, T)

    m_eff, T_eff = m, T
    if ((it // 2) % 2 == 1) and T > POOL_T_THRESH:
        pooled = _global_pool(X_tr_p, m, T)
        if pooled is not None:
            Xp2, mn, Tn = pooled
            if mn * Tn <= MAX_M_TIMES_T:
                pte = _global_pool(X_te_p, m, T)
                if pte is not None:
                    X_tr_p, X_te_p = Xp2, pte[0]
                    m_eff, T_eff = mn, Tn

    X_tr3, X_te3 = _pcn(
        X_tr_p.reshape(-1, m_eff, T_eff),
        X_te_p.reshape(-1, m_eff, T_eff),
    )
    return (X_tr3.reshape(-1, m_eff*T_eff), X_te3.reshape(-1, m_eff*T_eff),
            X_tr3, X_te3, y_tr_p, cp, m_eff, T_eff)


def _metrics(y_true, proba, n_classes):
    acc = float(accuracy_score(y_true, proba.argmax(1)))
    try:
        auc = (float(roc_auc_score(y_true, proba[:, 1]))
               if n_classes == 2
               else float(roc_auc_score(y_true, proba, multi_class="ovr")))
    except Exception:
        auc = float("nan")
    return acc, auc


# ── Finetuned (MPS) ──────────────────────────────────────────────────────────

def run_finetuned(model, ds, dev):
    X_tr, X_te = ds["X_train"], ds["X_test"]
    y_tr, nc, m, T = ds["y_train"], ds["n_classes"], ds["m"], ds["T"]
    rng  = np.random.RandomState(SEED)
    psum = np.zeros((X_te.shape[0], nc), dtype=np.float64)
    ok   = 0
    for it in range(8):
        try:
            Xtrp, Xtep, Xtr3, Xte3, ytrp, cp, me, Te = \
                _iter_preprocess(X_tr, X_te, y_tr, nc, m, T, rng, it)
            X_tr_pad, _, ng = pad_and_expand_overlap(Xtrp, me, Te)
            X_te_pad, _, _  = pad_and_expand_overlap(Xtep, me, Te)
            set_temporal_info(model, me, ng * WINDOW, group_size=WINDOW)
            if hasattr(model, "global_conv_encoder"):
                set_global_input(model, Xtr3, Xte3)
            Xt = torch.as_tensor(X_tr_pad, dtype=torch.float32, device=dev)
            yt = torch.as_tensor(ytrp,     dtype=torch.float32, device=dev)
            Xe = torch.as_tensor(X_te_pad, dtype=torch.float32, device=dev)
            with torch.no_grad():
                out = model(torch.cat([Xt, Xe]).unsqueeze(1), yt.unsqueeze(1),
                            only_return_standard_out=True, categorical_inds=[[]])
            lo = (out.squeeze(1) if out.ndim == 3 else out)[:, :nc][:, cp]
            psum += torch.softmax(lo / SOFTMAX_TEMPERATURE, dim=-1).cpu().numpy()
            ok += 1
            del Xt, yt, Xe, out, lo
        except Exception as e:
            print(f"      [ft it{it}] {type(e).__name__}: {str(e)[:60]}")
    if dev == "mps":
        torch.mps.empty_cache()
    return _metrics(ds["y_test"], psum / ok, nc) if ok > 0 else (float("nan"), float("nan"))


# ── Vanilla (CPU) ────────────────────────────────────────────────────────────

_DUMMY_CFG = ClassifierEnsembleConfig(
    preprocess_config=PreprocessorConfig("none", categorical_name="numeric"),
    feature_shift_count=0, class_permutation=None,
    add_fingerprint_feature=False, polynomial_features="no",
    feature_shift_decoder=None, subsample_ix=None, _model_index=0,
)

def run_vanilla(clf, ds):
    X_tr, X_te = ds["X_train"], ds["X_test"]
    y_tr, nc, m, T = ds["y_train"], ds["n_classes"], ds["m"], ds["T"]
    rng  = np.random.RandomState(SEED)
    psum = np.zeros((X_te.shape[0], nc), dtype=np.float64)
    ok   = 0
    for it in range(8):
        try:
            Xtrp, Xtep, _, _, ytrp, cp, me, Te = \
                _iter_preprocess(X_tr, X_te, y_tr, nc, m, T, rng, it)
            X_tr_t = torch.as_tensor(Xtrp, dtype=torch.float32, device="cpu").unsqueeze(0)
            y_tr_t = torch.as_tensor(ytrp.astype(np.float32), device="cpu").unsqueeze(0)
            X_te_t = torch.as_tensor(Xtep, dtype=torch.float32, device="cpu").unsqueeze(0)
            clf.n_classes_ = nc
            clf.fit_from_preprocessed(
                [X_tr_t], [y_tr_t], cat_ix=[[[]]], configs=[[_DUMMY_CFG]],
            )
            with torch.no_grad():
                logits = clf.forward([X_te_t], return_raw_logits=True)
            if   logits.ndim == 2: lo = logits
            elif logits.ndim == 3: lo = logits.squeeze(1)
            elif logits.ndim == 4: lo = logits.mean(dim=(1, 2))
            else: continue
            lo = lo[:, :nc][:, cp]
            psum += torch.softmax(lo / SOFTMAX_TEMPERATURE, dim=-1).cpu().numpy()
            ok += 1
            del X_tr_t, y_tr_t, X_te_t, logits, lo
        except Exception as e:
            print(f"      [vn it{it}] {type(e).__name__}: {str(e)[:60]}")
    return _metrics(ds["y_test"], psum / ok, nc) if ok > 0 else (float("nan"), float("nan"))


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_datasets():
    summary  = pd.read_csv(DATA_ROOT / "datasets_summary.csv")
    eligible = summary[summary["passes_pfn_filters"]].copy()
    rows, skipped = [], []
    for _, row in eligible.iterrows():
        name = row["dataset"]
        col  = row["collection"].lower()
        trp  = DATA_ROOT / "data" / col / f"{name}_train.npz"
        tep  = DATA_ROOT / "data" / col / f"{name}_test.npz"
        if not trp.exists() or not tep.exists():
            skipped.append(name); continue
        try:
            tr = np.load(trp, allow_pickle=False)
            te = np.load(tep, allow_pickle=False)
            X_tr_3d, y_tr_raw = tr["X"].astype(np.float32), tr["y"]
            X_te_3d, y_te_raw = te["X"].astype(np.float32), te["y"]
            n_tr, m, T = X_tr_3d.shape
            nc = int(row["n_classes"])
            if n_tr >= SUBSAMPLE_N:
                rng = np.random.RandomState(SUBSAMPLE_SEED)
                idx = rng.choice(n_tr, SUBSAMPLE_N, replace=False); idx.sort()
                X_tr_3d, y_tr_raw = X_tr_3d[idx], y_tr_raw[idx]
            le = LabelEncoder(); le.fit(y_tr_raw)
            y_tr = le.transform(y_tr_raw).astype(np.int64)
            y_te = le.transform(y_te_raw).astype(np.int64)
            X_tr = X_tr_3d.reshape(-1, m*T).astype(np.float32)
            X_te = X_te_3d.reshape(-1, m*T).astype(np.float32)
            np.putmask(X_tr, ~np.isfinite(X_tr), np.nan)
            np.putmask(X_te, ~np.isfinite(X_te), np.nan)
            rows.append(dict(name=name, collection=col.upper(),
                             X_train=X_tr, X_test=X_te,
                             y_train=y_tr, y_test=y_te,
                             n_classes=nc, m=m, T=T))
        except Exception:
            skipped.append(name)
    if skipped:
        print(f"  Skipped {len(skipped)}: {skipped[:5]}...")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("="*78)
    print("  Quick eval — finetuned vs vanilla TabPFN (same schedule)")
    print("="*78)
    print(f"  Schedule: squash odd, pool K={POOL_K} S={POOL_S} on iters {{2,3,6,7}} if T>{POOL_T_THRESH}")
    print(f"  1 run per dataset (seed={SEED})")

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
           else "cpu")
    print(f"  Finetuned device: {dev}  |  Vanilla: cpu")

    print("\nLoading datasets...")
    datasets = load_datasets()
    n_pool = sum(1 for d in datasets if d["T"] > POOL_T_THRESH)
    print(f"  {len(datasets)} datasets  ({n_pool} with T>{POOL_T_THRESH} → pool active)")

    # ── Phase 1: Finetuned (MPS) ─────────────────────────────────────────────
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
           else "cpu")
    print(f"\n── Phase 1: Finetuned model ({dev}) ──")
    ft_model, _, _, _ = build_overlap_model(device=dev)
    ckpt = torch.load(CKPT_PATH, map_location=dev, weights_only=False)
    ft_model.load_state_dict(ckpt["model_state_dict"])
    ft_model.eval()
    print(f"  step={ckpt.get('step','?')}  version={ckpt.get('version','?')}")

    ft_results = {}
    t0 = time.time()
    for i, ds in enumerate(datasets):
        name, m, T = ds["name"], ds["m"], ds["T"]
        pool = "pool" if T > POOL_T_THRESH else "plain"
        t1 = time.time()
        acc, auc = run_finetuned(ft_model, ds, dev)
        ft_results[name] = (acc, auc, pool)
        print(f"  [{i+1:3d}/{len(datasets)}] {name:<35s} [{pool:<5s}] "
              f"acc={acc:.4f} auc={auc:.4f}  ({time.time()-t1:.0f}s)")
    print(f"  Phase 1 done: {time.time()-t0:.0f}s")

    del ft_model, ckpt
    gc.collect()
    if dev == "mps":
        torch.mps.empty_cache()

    # ── Phase 2: Vanilla TabPFN (CPU) ────────────────────────────────────────
    print(f"\n── Phase 2: Vanilla TabPFN (CPU) ──")
    clf = TabPFNClassifier(
        device="cpu", n_estimators=1,
        ignore_pretraining_limits=True, fit_mode="batched",
        differentiable_input=False,
        inference_config={"FEATURE_SHIFT_METHOD": None},
    )
    clf._initialize_model_variables()

    vn_results = {}
    t0 = time.time()
    for i, ds in enumerate(datasets):
        name = ds["name"]
        pool = ft_results[name][2]
        t1 = time.time()
        acc, auc = run_vanilla(clf, ds)
        vn_results[name] = (acc, auc)
        print(f"  [{i+1:3d}/{len(datasets)}] {name:<35s} [{pool:<5s}] "
              f"acc={acc:.4f} auc={auc:.4f}  ({time.time()-t1:.0f}s)")
    print(f"  Phase 2 done: {time.time()-t0:.0f}s")

    del clf; gc.collect()

    # ── Results ──────────────────────────────────────────────────────────────
    rows = []
    for ds in datasets:
        name = ds["name"]
        ft_acc, ft_auc, pool = ft_results[name]
        vn_acc, vn_auc = vn_results[name]
        rows.append(dict(name=name, m=ds["m"], T=ds["T"], n_classes=ds["n_classes"],
                         pool=pool, ft_acc=ft_acc, ft_auc=ft_auc,
                         vn_acc=vn_acc, vn_auc=vn_auc))

    df = pd.DataFrame(rows)
    df_ok = df.dropna(subset=["ft_acc", "vn_acc"])

    print(f"\n{'='*78}")
    print(f"  SUMMARY  ({len(df_ok)} datasets)")
    print(f"{'='*78}")

    for metric, fc, vc in [("ACC", "ft_acc", "vn_acc"), ("AUC", "ft_auc", "vn_auc")]:
        fv = df_ok[fc].values
        vv = df_ok[vc].values
        d  = fv - vv
        w, t, l = (d > 0).sum(), (d == 0).sum(), (d < 0).sum()
        try:
            _, p = wilcoxon(fv, vv, alternative="greater")
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            p_str = f"p={p:.6f} {stars}"
        except Exception:
            p_str = "n/a"
        print(f"\n  {metric}:  finetuned={fv.mean():.4f}  vanilla={vv.mean():.4f}  "
              f"Δ={d.mean():+.4f} (median {np.median(d):+.4f})")
        print(f"         Win/Tie/Loss: {w}/{t}/{l}   Wilcoxon: {p_str}")

    # Per-tag
    print(f"\n  Per-tag ACC breakdown:")
    for tag in ["pool", "plain"]:
        sub = df_ok[df_ok["pool"] == tag]
        if len(sub) == 0: continue
        d = sub["ft_acc"] - sub["vn_acc"]
        print(f"    {tag:<6} n={len(sub):3d}  ft={sub['ft_acc'].mean():.4f}  "
              f"vn={sub['vn_acc'].mean():.4f}  Δ={d.mean():+.4f}  "
              f"W/T/L={( d>0).sum()}/{(d==0).sum()}/{(d<0).sum()}")

    out = ROOT / "04_evaluation/results/quick_eval.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
