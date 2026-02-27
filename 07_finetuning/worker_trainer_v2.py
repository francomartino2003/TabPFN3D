"""
Worker: Trainer for overlap model (GPU 1).

Reads .npz batches from disk, trains the overlap TabPFN (1 fresh + 24 pretrained),
saves last.pt. Deletes consumed .npz files to free disk.

Key difference from worker_trainer.py:
  - Uses build_overlap_model (pretrained backbone + 1 new layer)
  - Expands overlapping windows (16, stride 8) before each forward pass
  - Two parameter groups: fresh (higher LR) and pretrained (lower LR)
"""

import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent.parent / '12_kernel_dag_generator'))
sys.path.insert(0, str(Path(__file__).parent.parent / '00_TabPFN' / 'src'))

from tabpfn.preprocessing.configs import ClassifierEnsembleConfig, PreprocessorConfig

from build_overlap_model import (
    build_overlap_model,
    expand_overlap_flat,
    pad_and_expand_overlap,
    WINDOW, STRIDE,
)
from tabpfn_temporal import set_temporal_info

SOFTMAX_TEMPERATURE = 0.9


def deserialize_batch(path: str) -> list:
    """Load a .npz file back into a list of dataset dicts."""
    data = np.load(path, allow_pickle=False)
    n = int(data['__n__'])
    batch = []
    for i in range(n):
        ds = {
            'X_train':        data[f'{i}_X_train'],
            'X_test':         data[f'{i}_X_test'],
            'y_train':        data[f'{i}_y_train'],
            'y_test':         data[f'{i}_y_test'],
            'n_classes':      int(data[f'{i}_n_classes']),
            'n_features':     int(data[f'{i}_n_features']),
            'n_features_orig': int(data[f'{i}_n_features_orig']),
            'T':              int(data[f'{i}_T']),
            'n_samples':      int(data[f'{i}_n_samples']),
        }
        batch.append(ds)
    return batch


def _make_dummy_ensemble_config():
    return ClassifierEnsembleConfig(
        preprocess_config=PreprocessorConfig("none", categorical_name="numeric"),
        feature_shift_count=0,
        class_permutation=None,
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_decoder=None,
        subsample_ix=None,
        _model_index=0,
    )


def forward_single_dataset(model, clf, device, data):
    """Forward pass with overlapping group expansion."""
    try:
        m = data['n_features_orig']
        T = data['T']

        X_train_exp, T_pad, n_groups = pad_and_expand_overlap(
            data['X_train'], m, T, WINDOW, STRIDE)
        X_test_exp, _, _ = pad_and_expand_overlap(
            data['X_test'], m, T, WINDOW, STRIDE)

        T_eff = n_groups * WINDOW
        set_temporal_info(model, m, T_eff, group_size=WINDOW)

        y_test_t = torch.tensor(data['y_test'], dtype=torch.long, device=device)

        X_train_t = torch.as_tensor(
            X_train_exp, dtype=torch.float32, device=device).unsqueeze(0)
        y_train_t = torch.as_tensor(
            data['y_train'], dtype=torch.float32, device=device).unsqueeze(0)
        X_test_t = torch.as_tensor(
            X_test_exp, dtype=torch.float32, device=device).unsqueeze(0)

        dummy_cfg = _make_dummy_ensemble_config()
        clf.n_classes_ = data['n_classes']
        clf.fit_from_preprocessed(
            [X_train_t], [y_train_t],
            cat_ix=[[[]]],
            configs=[[dummy_cfg]],
        )
        logits = clf.forward([X_test_t], return_raw_logits=True)

        if logits.ndim == 2:
            logits_QL = logits
        elif logits.ndim == 3:
            logits_QL = logits.squeeze(1)
        elif logits.ndim == 4:
            logits_QL = logits.mean(dim=(1, 2))
        else:
            return None

        if y_test_t.min() < 0 or y_test_t.max() >= logits_QL.shape[-1]:
            return None

        loss = F.cross_entropy(logits_QL, y_test_t)
        with torch.no_grad():
            acc = (logits_QL.argmax(dim=-1) == y_test_t).float().mean()

        return {'loss': loss, 'accuracy': acc}
    except Exception:
        return None


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
            if out['loss'].requires_grad:
                out['loss'].backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                for j, p in enumerate(params):
                    if p.grad is not None:
                        grad_accum[j] += p.grad
                total_loss += out['loss'].item()
                total_acc += out['accuracy'].item()
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
        'loss': total_loss / max(1, n_valid),
        'accuracy': total_acc / max(1, n_valid),
        'n_valid': n_valid,
        'lr': scheduler.get_last_lr()[0],
    }


def save_checkpoint_atomic(path: Path, model, optimizer, scheduler, step,
                           train_losses, train_accs, finished=False):
    """Atomic save: write to .tmp then rename."""
    ckpt = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'train_accs': train_accs,
        'finished': finished,
        'version': 'overlap_v1',
    }
    tmp = path.with_suffix('.pt.tmp')
    torch.save(ckpt, tmp)
    os.replace(str(tmp), str(path))


def main():
    parser = argparse.ArgumentParser(description='Worker: overlap trainer')
    parser.add_argument('--batch-dir', type=str, required=True)
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--lr-fresh', type=float, default=1e-4,
                        help='LR for fresh params (encoder, new layer)')
    parser.add_argument('--lr-pretrained', type=float, default=1e-5,
                        help='LR for pretrained params (24 layers + decoder)')
    parser.add_argument('--lr-min', type=float, default=1e-7)
    parser.add_argument('--warmup-steps', type=int, default=200)
    parser.add_argument('--n-steps', type=int, default=100000,
                        help='Total expected training steps for LR schedule')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    ckpt_dir  = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"[Trainer] lr_fresh={args.lr_fresh} lr_pretrained={args.lr_pretrained} "
          f"device={device}", flush=True)

    model, clf, fresh_params, pretrained_params = build_overlap_model(device=device)
    model.train()

    optimizer = AdamW([
        {'params': fresh_params, 'lr': args.lr_fresh},
        {'params': pretrained_params, 'lr': args.lr_pretrained},
    ], weight_decay=args.weight_decay)

    warmup = args.warmup_steps
    total  = args.n_steps
    lr_fresh = args.lr_fresh
    lr_pretrained = args.lr_pretrained
    lr_min = args.lr_min

    def _lr_lambda_fresh(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min / lr_fresh + (1.0 - lr_min / lr_fresh) * cosine

    def _lr_lambda_pretrained(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min / lr_pretrained + (1.0 - lr_min / lr_pretrained) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, [_lr_lambda_fresh, _lr_lambda_pretrained])

    step = 0
    train_losses = []
    train_accs = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        step = ckpt['step']
        train_losses = ckpt.get('train_losses', [])
        train_accs = ckpt.get('train_accs', [])
        print(f"[Trainer] Resumed from step {step}", flush=True)

    print(f"[Trainer] Waiting for batches in {batch_dir} ...", flush=True)

    while True:
        files = sorted(glob.glob(str(batch_dir / 'batch_*.npz')))
        if not files:
            done_file = batch_dir / 'DONE'
            if done_file.exists():
                print(f"[Trainer] Generator DONE and no files left — stopping.",
                      flush=True)
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

        t0 = time.time()
        result = train_step(model, clf, optimizer, scheduler, batch, device,
                            grad_clip=args.grad_clip)
        dt = time.time() - t0

        step += 1
        train_losses.append(result['loss'])
        train_accs.append(result['accuracy'])

        save_checkpoint_atomic(
            ckpt_dir / 'last.pt', model, optimizer, scheduler,
            step, train_losses, train_accs)

        if step % 5 == 0 or step <= 3:
            lrs = scheduler.get_last_lr()
            print(f"  [Train] step={step} loss={result['loss']:.4f} "
                  f"acc={result['accuracy']:.4f} lr_f={lrs[0]:.2e} "
                  f"lr_p={lrs[1]:.2e} "
                  f"valid={result['n_valid']}/{len(batch)} dt={dt:.1f}s",
                  flush=True)

    save_checkpoint_atomic(
        ckpt_dir / 'last.pt', model, optimizer, scheduler,
        step, train_losses, train_accs, finished=True)
    print(f"[Trainer] DONE after {step} steps.", flush=True)


if __name__ == '__main__':
    main()
