"""
Worker: Evaluator for the overlap model (GPU 2).

Polls last.pt written by worker_trainer_v2, evaluates on fixed synthetic
batches and on real datasets, saves best.pt (lowest synthetic loss) and
appends to history.json.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "02_synthetic_data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "00_TabPFN" / "src"))

from model import build_overlap_model, pad_and_expand_overlap, set_temporal_info, set_global_input, per_channel_normalize, WINDOW, STRIDE
from data_utils import FinetuneConfig, SyntheticDataGenerator, load_all_pfn_datasets
from augmentation import augment_dataset
from inference import forward_single_dataset, deserialize_batch

SOFTMAX_TEMPERATURE = 0.9


def evaluate_synthetic(model, clf, device, synth_batches):
    """Evaluate on fixed synthetic batches (no_grad). Returns loss/acc metrics."""
    model.eval()
    total_loss, total_acc, n_valid = 0.0, 0.0, 0

    for batch in synth_batches:
        for data in batch:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    out = forward_single_dataset(model, clf, device, data)
                    if out is None:
                        continue
                    total_loss += out["loss"].item()
                    total_acc += out["accuracy"].item()
                    n_valid += 1
                    del out
            except Exception:
                continue

    n = max(1, n_valid)
    return {"loss": total_loss / n, "accuracy": total_acc / n, "n_valid": n_valid}


def evaluate_real(model, clf, device, real_datasets):
    """Evaluate on real datasets with patch extraction."""
    model.eval()
    results = []

    for data in real_datasets:
        res = {"name": data["name"], "n_classes": data["n_classes"]}
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            m, T = data["n_features_orig"], data["T"]

            X_tr_3d = data["X_train"].reshape(-1, m, T)
            X_te_3d = data["X_test"].reshape(-1, m, T)
            X_tr_3d_n, X_te_3d_n = per_channel_normalize(X_tr_3d, X_te_3d)

            X_tr_flat = X_tr_3d_n.reshape(-1, m * T)
            X_te_flat = X_te_3d_n.reshape(-1, m * T)

            X_tr_p, T_pad, n_groups = pad_and_expand_overlap(X_tr_flat, m, T)
            X_te_p, _, _ = pad_and_expand_overlap(X_te_flat, m, T)
            T_eff = n_groups * WINDOW
            set_temporal_info(model, m, T_eff, group_size=WINDOW)

            set_global_input(model, X_tr_3d_n, X_te_3d_n)

            with torch.no_grad():
                X_tr_t = torch.as_tensor(X_tr_p, dtype=torch.float32, device=device)
                y_tr_t = torch.as_tensor(data["y_train"], dtype=torch.float32, device=device)
                X_te_t = torch.as_tensor(X_te_p, dtype=torch.float32, device=device)

                X_full = torch.cat([X_tr_t, X_te_t], dim=0).unsqueeze(1)
                y_in = y_tr_t.unsqueeze(1)

                output = model(X_full, y_in, only_return_standard_out=True, categorical_inds=[[]])
                logits = output.squeeze(1) if output.ndim == 3 else output
                logits = logits[:, : data["n_classes"]]
                proba = torch.softmax(logits / SOFTMAX_TEMPERATURE, dim=-1)
                proba_np = proba.cpu().numpy()

            del X_tr_t, y_tr_t, X_te_t, X_full, y_in, output, logits, proba

            preds = proba_np.argmax(axis=1)
            res["accuracy"] = float(accuracy_score(data["y_test"], preds))
            try:
                if data["n_classes"] == 2:
                    res["auc"] = float(roc_auc_score(data["y_test"], proba_np[:, 1]))
                else:
                    res["auc"] = float(roc_auc_score(data["y_test"], proba_np, multi_class="ovr"))
            except Exception:
                res["auc"] = None
            res["status"] = "success"

        except Exception as e:
            res["accuracy"] = None
            res["auc"] = None
            res["status"] = "failed"
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results.append(res)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Worker: overlap evaluator")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--eval-interval", type=float, default=60,
                        help="Seconds between evaluation polls")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Evaluator] device={device}", flush=True)

    model, clf, _, _ = build_overlap_model(device=device)

    # Fixed synthetic eval batches — same constraints as worker_generator
    cfg = FinetuneConfig(batch_size=args.batch_size)
    synth_gen = SyntheticDataGenerator(cfg, seed=cfg.seed + 5000)
    aug_rng = np.random.RandomState(cfg.seed + 9999)

    synth_eval_batches = []
    for _ in range(2):
        n_unique = args.batch_size // 4
        originals = []
        for _ in range(n_unique):
            try:
                originals.append(synth_gen.generate_one())
            except Exception:
                continue
        batch = list(originals)
        for orig in originals:
            for _ in range(3):
                try:
                    batch.append(
                        augment_dataset(
                            orig, aug_rng,
                            max_T=cfg.max_T, max_m=cfg.max_m,
                            max_m_times_T=cfg.max_m_times_T,
                            group_size=cfg.group_size,
                        )
                    )
                except Exception:
                    batch.append(orig)
        synth_eval_batches.append(batch)

    print(
        f"[Evaluator] Built {len(synth_eval_batches)} synth eval batches "
        f"({sum(len(b) for b in synth_eval_batches)} datasets)",
        flush=True,
    )

    real_datasets = load_all_pfn_datasets(group_size=cfg.group_size)
    print(f"[Evaluator] Loaded {len(real_datasets)} real datasets (all PFN-eligible)", flush=True)

    last_pt = ckpt_dir / "last.pt"
    best_pt = ckpt_dir / "best.pt"
    history_path = log_dir / "history.json"

    last_evaluated_step = -1
    best_synth_loss = float("inf")
    history = []

    print(f"[Evaluator] Polling {last_pt} every {args.eval_interval}s ...", flush=True)

    while True:
        if not last_pt.exists():
            time.sleep(args.eval_interval)
            continue

        try:
            ckpt = torch.load(last_pt, map_location=device, weights_only=False)
        except Exception:
            time.sleep(5)
            continue

        step = ckpt.get("step", 0)
        finished = ckpt.get("finished", False)

        if step <= last_evaluated_step and not finished:
            time.sleep(args.eval_interval)
            continue

        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\n[Evaluator] Evaluating step {step} ...", flush=True)

        t0 = time.time()
        synth_res = evaluate_synthetic(model, clf, device, synth_eval_batches)
        dt_synth = time.time() - t0

        t0 = time.time()
        real_res = evaluate_real(model, clf, device, real_datasets)
        dt_real = time.time() - t0

        real_accs = [r["accuracy"] for r in real_res if r.get("accuracy") is not None]
        real_aucs = [r["auc"] for r in real_res if r.get("auc") is not None]
        mean_real_acc = float(np.mean(real_accs)) if real_accs else 0.0
        mean_real_auc = float(np.mean(real_aucs)) if real_aucs else 0.0

        entry = {
            "step": step,
            "synth_loss": synth_res["loss"],
            "synth_acc": synth_res["accuracy"],
            "real_acc": mean_real_acc,
            "real_auc": mean_real_auc,
            "n_real": len(real_accs),
        }
        history.append(entry)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if synth_res["loss"] < best_synth_loss:
            best_synth_loss = synth_res["loss"]
            shutil.copy2(str(last_pt), str(best_pt))
            print(f"  >> New best synth loss: {best_synth_loss:.4f} @ step {step}", flush=True)

        print(
            f"  [Eval] step={step} synth_loss={synth_res['loss']:.4f} "
            f"synth_acc={synth_res['accuracy']:.4f} "
            f"real_acc={mean_real_acc:.4f} real_auc={mean_real_auc:.4f} "
            f"(synth {dt_synth:.0f}s, real {dt_real:.0f}s)",
            flush=True,
        )

        last_evaluated_step = step
        if finished:
            print("[Evaluator] Trainer finished. Final eval done.", flush=True)
            break

        time.sleep(args.eval_interval)

    print(f"[Evaluator] DONE. History saved to {history_path}", flush=True)


if __name__ == "__main__":
    main()
