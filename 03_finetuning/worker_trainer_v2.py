"""
Worker: Trainer for the overlap model (GPU 1).

Reads .npz batches written by worker_generator, trains the overlap TabPFN,
saves last.pt atomically after each step, and signals the evaluator which
promotes last.pt → best.pt whenever eval improves.

Three training modes (--mode):
---------------------------------
  phase1  Encoder warm-up.  Freeze everything except:
            • model.encoder  (MLP 32→96→96→192)
            • model.feature_positional_embedding_embeddings  (Linear 48→192)
            • model.temporal_pe_projection                   (Linear 48→192)
            • model.transformer_encoder.layers[0]            (first PFN layer)
          Schedule: constant LR for --warmup-const-steps, then cosine to --lr-min.
          Default: 1500 constant + 3000 cosine (1e-4 → 1e-5), total 4500 steps.
          → saves to checkpoints/phase1/  (last.pt + best.pt)

  phase2  Full fine-tune.  All params unfrozen.
          Schedule: constant LR for --warmup-const-steps, then cosine to --lr-min.
          Default: 1500 constant + 6000 cosine (1e-5 → 1e-7), total 7500 steps.
          Must be started with --resume pointing to phase1/best.pt.
          → saves to checkpoints/phase2/  (last.pt + best.pt)

  full    Two-group cosine schedule (original behaviour).
          Fresh params at --lr-fresh, pretrained at --lr-pretrained,
          both annealed to --lr-min over --n-steps with --warmup-steps warmup.
"""

import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent / "02_synthetic_data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "00_TabPFN" / "src"))

from model import build_overlap_model
from inference import forward_single_dataset, deserialize_batch


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def _lr_cosine(step: int, warmup: int, total: int, lr: float, lr_min: float) -> float:
    """Linear warmup then cosine decay (used by 'full' mode)."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min / lr + (1.0 - lr_min / lr) * cosine


def _lr_const_then_cosine(step: int, warmup_const: int, total: int,
                          lr: float, lr_min: float) -> float:
    """Constant LR for warmup_const steps, then cosine decay to lr_min."""
    if step < warmup_const:
        return 1.0
    progress = min(1.0, (step - warmup_const) / max(1, total - warmup_const))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min / lr + (1.0 - lr_min / lr) * cosine


def _make_phase_schedule(warmup_const: int, total: int, lr: float, lr_min: float):
    """Return a LambdaLR-compatible schedule function (captures args by value)."""
    def _fn(step):
        return _lr_const_then_cosine(step, warmup_const, total, lr, lr_min)
    return _fn


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

def train_step(model, clf, optimizer, scheduler, batch, device, grad_clip=1.0):
    """One training step: per-dataset grad accumulation + clipping."""
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    grad_accum = [torch.zeros_like(p) for p in params]
    total_loss, total_acc, n_valid = 0.0, 0.0, 0

    for i, data in enumerate(batch):
        try:
            optimizer.zero_grad()
            out = forward_single_dataset(model, clf, device, data)
            if out is None:
                continue
            if out["loss"].requires_grad:
                out["loss"].backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                for j, p in enumerate(params):
                    if p.grad is not None:
                        grad_accum[j] += p.grad
                total_loss += out["loss"].item()
                total_acc += out["accuracy"].item()
                n_valid += 1
            del out
        except Exception:
            continue
        if torch.cuda.is_available() and (i + 1) % 8 == 0:
            torch.cuda.empty_cache()

    if n_valid > 0:
        for j, p in enumerate(params):
            p.grad = grad_accum[j] / n_valid
        optimizer.step()
        scheduler.step()

    del grad_accum
    return {
        "loss": total_loss / max(1, n_valid),
        "accuracy": total_acc / max(1, n_valid),
        "n_valid": n_valid,
        "lr": scheduler.get_last_lr()[0],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint_atomic(path, model, optimizer, scheduler, step,
                           train_losses, train_accs, mode, finished=False):
    """Atomic save: write to .pt.tmp then rename."""
    ckpt = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_losses": train_losses,
        "train_accs": train_accs,
        "finished": finished,
        "mode": mode,
        "version": "overlap_v2",
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(ckpt, tmp)
    os.replace(str(tmp), str(path))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Worker: overlap trainer")

    # I/O
    parser.add_argument("--batch-dir",      type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)

    # Training mode
    parser.add_argument(
        "--mode", choices=["full", "phase1", "phase2"], default="full",
        help=(
            "full:   two-group cosine schedule (lr-fresh / lr-pretrained). "
            "phase1: freeze all except encoder+embeddings+layer0, constant lr. "
            "phase2: all params, constant lr (resume from phase1/best.pt)."
        ),
    )

    # LR for phase1 / phase2  (single value)
    parser.add_argument("--lr", type=float, default=None,
                        help="Base learning rate for phase1/phase2 modes.")
    parser.add_argument("--lr-min", type=float, default=None,
                        help="Minimum LR at end of cosine annealing (phase1/phase2/full).")
    parser.add_argument("--warmup-const-steps", type=int, default=0,
                        help="Steps of constant LR before cosine begins (phase1/phase2).")

    # LR for full mode (two groups)
    parser.add_argument("--lr-fresh",      type=float, default=1e-4)
    parser.add_argument("--lr-pretrained", type=float, default=5e-5)
    parser.add_argument("--warmup-steps",  type=int,   default=200)

    # Common
    parser.add_argument("--n-steps",      type=int,   default=3_000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip",    type=float, default=1.0)
    parser.add_argument("--device",       type=str,   default="auto")

    # Resume
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--resume-weights-only", action="store_true",
                        help="Load only model weights; reset step/optimizer/scheduler.")

    args = parser.parse_args()

    # ── Device ──
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Paths ──
    batch_dir = Path(args.batch_dir)
    ckpt_dir  = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Build model ──
    model, clf, fresh_params, pretrained_params = build_overlap_model(device=device)

    # ── Mode-specific setup ──
    mode = args.mode

    if mode == "phase1":
        # ── Phase 1: encoder warm-up ──
        # Trainable: fresh params (encoder + emb projections) + first transformer layer
        lr           = args.lr      if args.lr      is not None else 1e-4
        lr_min       = args.lr_min  if args.lr_min  is not None else 1e-5
        warmup_const = args.warmup_const_steps  # 0 → purely constant

        for p in model.parameters():
            p.requires_grad = False

        first_layer = list(model.transformer_encoder.layers)[0]
        trainable_modules = [
            ("encoder",                          model.encoder),
            ("feature_positional_emb_projection", model.feature_positional_embedding_embeddings),
            ("temporal_pe_projection",            model.temporal_pe_projection),
            ("transformer_encoder.layers[0]",     first_layer),
        ]
        seen_ids: set = set()
        trainable: list = []
        for name, mod in trainable_modules:
            n_before = len(trainable)
            for p in mod.parameters():
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    p.requires_grad = True
                    trainable.append(p)
            n_added = len(trainable) - n_before
            print(f"  [phase1] trainable: {name}  ({n_added:,} params)", flush=True)

        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"  [phase1] total trainable: {n_trainable:,} / {n_total:,}", flush=True)
        print(f"  [phase1] lr={lr}  warmup_const={warmup_const}  lr_min={lr_min}"
              f"  n_steps={args.n_steps}", flush=True)

        optimizer = AdamW([{"params": trainable, "lr": lr}], weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, [_make_phase_schedule(warmup_const, args.n_steps, lr, lr_min)]
        )

    elif mode == "phase2":
        # ── Phase 2: full fine-tune ──
        lr           = args.lr      if args.lr      is not None else 1e-5
        lr_min       = args.lr_min  if args.lr_min  is not None else 1e-7
        warmup_const = args.warmup_const_steps

        for p in model.parameters():
            p.requires_grad = True

        n_total = sum(p.numel() for p in model.parameters())
        print(f"  [phase2] all params trainable: {n_total:,}", flush=True)
        print(f"  [phase2] lr={lr}  warmup_const={warmup_const}  lr_min={lr_min}"
              f"  n_steps={args.n_steps}", flush=True)
        print(f"  [phase2] NOTE: must resume from phase1/best.pt via --resume --resume-weights-only",
              flush=True)

        optimizer = AdamW(
            [{"params": list(model.parameters()), "lr": lr}],
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, [_make_phase_schedule(warmup_const, args.n_steps, lr, lr_min)]
        )

    else:
        # ── Full mode: two-group cosine ──
        lr_f   = args.lr_fresh
        lr_p   = args.lr_pretrained
        lr_min = args.lr_min if args.lr_min is not None else 1e-7
        warmup = args.warmup_steps
        total  = args.n_steps

        n_fresh = sum(p.numel() for p in fresh_params)
        n_pre   = sum(p.numel() for p in pretrained_params)
        print(f"  [full] lr_fresh={lr_f}  lr_pretrained={lr_p}  lr_min={lr_min}", flush=True)
        print(f"  [full] warmup={warmup}  n_steps={total}", flush=True)
        print(f"  [full] fresh params: {n_fresh:,}  pretrained params: {n_pre:,}", flush=True)

        optimizer = AdamW(
            [
                {"params": fresh_params,      "lr": lr_f},
                {"params": pretrained_params, "lr": lr_p},
            ],
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            [
                lambda s: _lr_cosine(s, warmup, total, lr_f, lr_min),
                lambda s: _lr_cosine(s, warmup, total, lr_p, lr_min),
            ],
        )

    model.train()

    # ── Resume ──
    step = 0
    train_losses: list = []
    train_accs:   list = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if args.resume_weights_only:
            print(f"[Trainer] Loaded weights from {args.resume}  (step reset to 0)", flush=True)
        else:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            step         = ckpt["step"]
            train_losses = ckpt.get("train_losses", [])
            train_accs   = ckpt.get("train_accs",   [])
            print(f"[Trainer] Resumed from step {step} ({args.resume})", flush=True)

    print(f"[Trainer] mode={mode}  device={device}  batch_dir={batch_dir}", flush=True)
    print(f"[Trainer] Waiting for batches ...", flush=True)

    # ── Training loop ──
    while True:
        files = sorted(glob.glob(str(batch_dir / "batch_*.npz")))
        if not files:
            if (batch_dir / "DONE").exists():
                print("[Trainer] Generator DONE and no files left — stopping.", flush=True)
                break
            time.sleep(2)
            continue

        npz_path = files[0]
        try:
            batch = deserialize_batch(npz_path)
        except Exception as e:
            print(f"[Trainer] Error loading {npz_path}: {e}", flush=True)
            time.sleep(1)
            continue

        try:
            os.remove(npz_path)
        except OSError:
            pass

        t0     = time.time()
        result = train_step(model, clf, optimizer, scheduler, batch, device,
                            grad_clip=args.grad_clip)
        dt     = time.time() - t0

        step += 1
        train_losses.append(result["loss"])
        train_accs.append(result["accuracy"])

        save_checkpoint_atomic(
            ckpt_dir / "last.pt", model, optimizer, scheduler,
            step, train_losses, train_accs, mode,
        )

        if step >= args.n_steps:
            print(f"[Trainer] Reached n_steps={args.n_steps} — stopping.", flush=True)
            break

        if step % 5 == 0 or step <= 3:
            lrs    = scheduler.get_last_lr()
            lr_str = f"lr={lrs[0]:.2e}" if len(lrs) == 1 else f"lr_f={lrs[0]:.2e} lr_p={lrs[1]:.2e}"
            print(
                f"  [Train] step={step}/{args.n_steps} loss={result['loss']:.4f} "
                f"acc={result['accuracy']:.4f} {lr_str} "
                f"valid={result['n_valid']}/{len(batch)} dt={dt:.1f}s",
                flush=True,
            )

    save_checkpoint_atomic(
        ckpt_dir / "last.pt", model, optimizer, scheduler,
        step, train_losses, train_accs, mode, finished=True,
    )
    print(f"[Trainer] DONE after {step} steps.  Checkpoint: {ckpt_dir}/last.pt", flush=True)


if __name__ == "__main__":
    main()
