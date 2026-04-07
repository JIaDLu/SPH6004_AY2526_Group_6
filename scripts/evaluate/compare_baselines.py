"""
scripts/evaluate/compare_baselines.py – Aggregate Stage 1 results.

Scans outputs/*/metrics.json for stage=="ts_only" runs, produces:
  results/ts_baseline_comparison.csv
  results/plots/ts_baseline_bar.png      grouped bar: MAE / RMSE / MedAE
  results/plots/ts_baseline_curves.png   training-loss + val-MAE curves

Usage
-----
python -m scripts.evaluate.compare_baselines
python -m scripts.evaluate.compare_baselines --output_root outputs --result_dir results
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", default=str(ROOT / "outputs"))
    p.add_argument("--result_dir",  default=str(ROOT / "results"))
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_all_metrics(output_root: Path) -> pd.DataFrame:
    records = []
    for mf in sorted(output_root.glob("*/metrics.json")):
        with open(mf) as f:
            d = json.load(f)
        if d.get("stage") != "ts_only":
            continue
        d["run_dir"] = str(mf.parent.name)
        records.append(d)

    if not records:
        print("[compare] No stage=ts_only metrics.json found in", output_root)
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values("test_MAE").reset_index(drop=True)
    return df


def load_history(output_root: Path, arch: str) -> dict | None:
    """Load the latest history.json for a given arch."""
    candidates = sorted(output_root.glob(f"{arch}_*/history.json"))
    if not candidates:
        return None
    with open(candidates[-1]) as f:
        return json.load(f)


# ── Plots ─────────────────────────────────────────────────────────────────────

ARCH_COLORS = {"lstm": "#4C72B0", "gru": "#DD8452", "mha": "#55A868"}
METRICS     = ["test_MAE", "test_RMSE", "test_MedAE"]
METRIC_LABELS = ["MAE (h)", "RMSE (h)", "MedAE (h)"]


def plot_bar(df: pd.DataFrame, save_path: Path) -> None:
    """Grouped bar chart comparing MAE / RMSE / MedAE across architectures."""
    arches = df["arch"].tolist()
    x      = np.arange(len(METRICS))
    width  = 0.8 / len(arches)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (_, row) in enumerate(df.iterrows()):
        arch   = row["arch"]
        values = [row[m] for m in METRICS]
        offset = (i - len(arches) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, values, width,
                        label=arch.upper(),
                        color=ARCH_COLORS.get(arch, f"C{i}"),
                        edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylabel("Hours")
    ax.set_title("Stage 1 TS-Only Baseline Comparison")
    ax.legend(title="Architecture", framealpha=0.9)
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[compare] Saved → {save_path}")


def plot_curves(df: pd.DataFrame, output_root: Path, save_path: Path) -> None:
    """Training-loss + val-MAE curves for each architecture."""
    arches_in_df = df["arch"].tolist()
    histories    = {a: load_history(output_root, a) for a in arches_in_df}
    valid        = {a: h for a, h in histories.items() if h is not None}

    if not valid:
        print("[compare] No history.json found; skipping curves plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for arch, h in valid.items():
        color = ARCH_COLORS.get(arch, None)
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["train_loss"], label=arch.upper(), color=color)
        axes[1].plot(epochs, h["val_mae"],    label=arch.upper(), color=color)

    for ax, title, ylabel in zip(
        axes,
        ["Training Loss (log-MSE)", "Validation MAE (hours, original space)"],
        ["Log-MSE", "MAE (h)"],
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title="Architecture", framealpha=0.9)
        ax.grid(alpha=0.35, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[compare] Saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args       = parse_args()
    out_root   = Path(args.output_root)
    result_dir = Path(args.result_dir)
    plot_dir   = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compare] Scanning {out_root} …")
    df = load_all_metrics(out_root)

    if df.empty:
        return

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_cols = ["arch", "best_val_mae",
                "test_MAE", "test_RMSE", "test_MedAE", "test_R2", "run_dir"]
    csv_path = result_dir / "ts_baseline_comparison.csv"
    df[csv_cols].to_csv(csv_path, index=False)
    print(f"[compare] Saved → {csv_path}")
    print("\n" + df[csv_cols].to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_bar(df, plot_dir / "ts_baseline_bar.png")
    plot_curves(df, out_root, plot_dir / "ts_baseline_curves.png")

    # ── Best model summary ────────────────────────────────────────────────────
    best = df.iloc[0]
    print(f"\n{'─'*55}")
    print(f"  Best Stage 1 arch : {best['arch'].upper()}")
    print(f"  Test MAE          : {best['test_MAE']:.4f} h")
    print(f"  Test RMSE         : {best['test_RMSE']:.4f} h")
    print(f"  Canonical ckpt    : checkpoints/ts_{best['arch']}_best.pt")
    print(f"  → Use this arch + ckpt for Stage 2 multimodal experiments.")
    print(f"{'─'*55}")


if __name__ == "__main__":
    main()