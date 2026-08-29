"""Collect E3 checkpoint metrics from evaluation logs and plot mIoU by epoch."""
import argparse
import csv
import re
from pathlib import Path


MIou_RE = re.compile(r"mIoU:\s*([0-9.]+)")
PA_RE = re.compile(r"PA:\s*([0-9.]+)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("runs/eval_logs"))
    parser.add_argument("--out-csv", type=Path, default=Path("runs/eval_miou.csv"))
    parser.add_argument("--out-png", type=Path, default=Path("runs/eval_miou.png"))
    args = parser.parse_args()

    rows = []
    for epoch in (20, 25, 30, 35):
        log = args.log_dir / f"E3_epoch_{epoch:04d}_eval.log"
        text = log.read_text() if log.exists() else ""
        miou = MIou_RE.findall(text)
        pa = PA_RE.findall(text)
        if not miou:
            raise SystemExit(f"missing mIoU in {log}; finish that evaluation first")
        rows.append({"epoch": epoch, "mIoU": float(miou[-1]),
                     "PA": float(pa[-1]) if pa else ""})

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("epoch", "mIoU", "PA"))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt
    epochs = [row["epoch"] for row in rows]
    mious = [row["mIoU"] for row in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, mious, marker="o", linewidth=2)
    for epoch, miou in zip(epochs, mious):
        plt.annotate(f"{miou:.4f}", (epoch, miou), xytext=(0, 8),
                     textcoords="offset points", ha="center")
    plt.xlabel("Training epoch")
    plt.ylabel("Validation mIoU")
    plt.title("E3 validation mIoU by checkpoint")
    plt.xticks(epochs)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=160)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()