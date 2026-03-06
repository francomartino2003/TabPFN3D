#!/usr/bin/env python3
"""Plot training loss and accuracy with 30-step moving average."""
import re
import sys
import argparse
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    plt = None


def parse_log(log_text: str):
    """Extract step, loss, acc from trainer log lines."""
    pattern = r'\[Train\]\s+step=(\d+)\s+loss=([\d.]+)\s+acc=([\d.]+)'
    matches = re.findall(pattern, log_text)
    steps = [int(m[0]) for m in matches]
    losses = [float(m[1]) for m in matches]
    accs = [float(m[2]) for m in matches]
    return steps, losses, accs


def moving_average(arr, window=30):
    """Moving average with window size, step 1."""
    result = []
    for i in range(window - 1, len(arr)):
        result.append(sum(arr[i - window + 1 : i + 1]) / window)
    return result


def main():
    parser = argparse.ArgumentParser(description='Plot loss/acc with MA-30')
    parser.add_argument('log_file', nargs='?', default=None,
                        help='Path to trainer.log, or "-" for stdin')
    parser.add_argument('--window', type=int, default=30)
    parser.add_argument('--out', type=str, default='train_curves_ma30.png')
    args = parser.parse_args()

    if args.log_file == '-':
        log_text = sys.stdin.read()
    elif args.log_file and Path(args.log_file).exists():
        with open(args.log_file) as f:
            log_text = f.read()
    elif args.log_file:
        print(f"File not found: {args.log_file}")
        return 1
    else:
        # Default: try common log locations
        candidates = [
            Path(__file__).parent / 'logs_overlap' / 'ov1' / 'trainer.log',
            Path(__file__).parent.parent / 'logs' / 'overlap_v1_*.out',
        ]
        log_text = None
        for p in candidates:
            if p.exists():
                with open(p) as f:
                    log_text = f.read()
                break
            # glob for *.out
            if '*' in str(p):
                import glob
                files = glob.glob(str(p))
                if files:
                    with open(files[0]) as f:
                        log_text = f.read()
                    break
        if log_text is None:
            print("No log file found. Usage: python plot_train_curves.py path/to/trainer.log")
            return 1

    steps, losses, accs = parse_log(log_text)
    if not steps:
        print("No [Train] lines found in log.")
        return 1

    if not HAS_MPL:
        print("matplotlib not installed. pip install matplotlib")
        return 1

    w = args.window
    ma_loss = moving_average(losses, w)
    ma_acc = moving_average(accs, w)
    steps_ma = steps[w - 1:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Loss
    ax1.plot(steps, losses, 'o', alpha=0.25, ms=1.5, color='gray', label='Raw')
    ax1.plot(steps_ma, ma_loss, 'b-', lw=2, label=f'MA-{w}')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(steps, accs, 'o', alpha=0.25, ms=1.5, color='gray', label='Raw')
    ax2.plot(steps_ma, ma_acc, 'g-', lw=2, label=f'MA-{w}')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='lower right')
    ax2.set_title('Training Accuracy')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).parent / args.out
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    print(f"Steps: {steps[0]} - {steps[-1]}, n={len(steps)}")
    print(f"Loss: min={min(losses):.4f}, max={max(losses):.4f} | MA min={min(ma_loss):.4f}")
    print(f"Acc:  min={min(accs):.4f}, max={max(accs):.4f} | MA max={max(ma_acc):.4f}")
    return 0


if __name__ == '__main__':
    exit(main())
