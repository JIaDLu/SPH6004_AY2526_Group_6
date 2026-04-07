"""
scripts/evaluate/compare_multimodal.py – Aggregate and compare Stage 1 + Stage 2 results.

Scans outputs/*/metrics.json for ALL runs (ts_only + multimodal), produces:
  results/multimodal_comparison.csv
  results/plots/multimodal_bar.png      MAE / RMSE / MedAE per variant
  results/plots/multimodal_curves.png   val-MAE learning curves

Usage
-----
python -m scripts.evaluate.compare_multimodal
python -m scripts.evaluate.compare_multimodal --ts_arch lstm
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
    p.add_argument("--ts_arch",     default=None,
                   help="Filter to runs using this TS arch (e.g. 'lstm')")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

STAGE_ORDER   = {"ts_only": 0, "multimodal": 1}
VARIANT_ORDER = {
    # Stage 1
    "lstm_baseline":       0,
    "gru_baseline":        1,
    "mha_baseline":        2,
    # Stage 2
    "ts_static":           3,
    "ts_text":             4,
    "ts_static_text":      5,
}
VARIANT_LABELS = {
    "ts_only":         "TS Only",
    "ts_static":       "TS + Static",
    "ts_text":         "TS + Text",
    "ts_static_text":  "TS + Static + Text",
}
VARIANT_COLORS = {
    "ts_only":        "#4C72B0",
    "ts_static":      "#55A868",
    "ts_text":        "#DD8452",
    "ts_static_text": "#C44E52",
}


def load_all_metrics(output_root: Path, ts_arch: str | None) -> pd.DataFrame:
    records = []
    for mf in sorted(output_root.glob("*/metrics.json")):
        with open(mf) as f:
            d = json.load(f)
        stage = d.get("stage", "")
        if stage not in ("ts_only", "multimodal"):
            continue

        # arch filter
        arch = d.get("arch") or d.get("ts_arch", "")
        if ts_arch and arch != ts_arch:
            continue

        # Normalise: ts_only rows get variant = "ts_only"
        d.setdefault("variant",  "ts_only")
        d.setdefault("ts_arch",  arch)
        d["run_dir"] = str(mf.parent.name)
        records.append(d)

    if not records:
        print("[compare_mm] No matching metrics.json found in", output_root)
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values(["stage", "test_MAE"]).reset_index(drop=True)
    return df


def load_history(output_root: Path, run_dir: str) -> dict | None:
    hf = output_root / run_dir / "history.json"
    if not hf.exists():
        return None
    with open(hf) as f:
        return json.load(f)


# ── Plots ─────────────────────────────────────────────────────────────────────

METRICS       = ["test_MAE", "test_RMSE", "test_MedAE"]
METRIC_LABELS = ["MAE (h)", "RMSE (h)", "MedAE (h)"]


def _run_label(row: pd.Series) -> str:
    """Human-readable label combining variant + arch."""
    variant = row.get("variant", "ts_only")
    arch    = row.get("ts_arch") or row.get("arch", "")
    vl      = VARIANT_LABELS.get(variant, variant)
    return f"{vl}\n({arch.upper()})" if arch else vl   #type:ignore


def plot_bar(df: pd.DataFrame, save_path: Path) -> None:
    """Grouped bar chart: one group per metric, one bar per run."""
    labels   = [_run_label(r) for _, r in df.iterrows()]
    n_runs   = len(df)
    x        = np.arange(len(METRICS))
    width    = min(0.8 / n_runs, 0.25)

    fig, ax = plt.subplots(figsize=(max(9, n_runs * 2), 5))

    for i, (_, row) in enumerate(df.iterrows()):
        variant = row.get("variant", "ts_only")
        color   = VARIANT_COLORS.get(variant, f"C{i}")
        values  = [row[m] for m in METRICS]
        offset  = (i - n_runs / 2 + 0.5) * width
        bars    = ax.bar(x + offset, values, width,
                         label=labels[i], color=color,
                         edgecolor="white", linewidth=0.5, alpha=0.9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylabel("Hours")
    ax.set_title("Stage 1 vs Stage 2 – Multimodal Comparison")
    ax.legend(title="Model", framealpha=0.9, fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[compare_mm] Saved → {save_path}")


def plot_curves(df: pd.DataFrame, output_root: Path, save_path: Path) -> None:
    """Val-MAE learning curves for every run."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for _, row in df.iterrows():
        h = load_history(output_root, row["run_dir"])
        if h is None or "val_mae" not in h:
            continue
        variant = row.get("variant", "ts_only")
        color   = VARIANT_COLORS.get(variant, None)
        label   = _run_label(row).replace("\n", " ")
        epochs  = range(1, len(h["val_mae"]) + 1)
        ax.plot(epochs, h["val_mae"], label=label, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val MAE (h, original space)")
    ax.set_title("Validation MAE – All Runs")
    ax.legend(title="Model", framealpha=0.9, fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[compare_mm] Saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args       = parse_args()
    out_root   = Path(args.output_root)
    result_dir = Path(args.result_dir)
    plot_dir   = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compare_mm] Scanning {out_root} …")
    df = load_all_metrics(out_root, args.ts_arch)

    if df.empty:
        return

    # ── CSV ──────────────────────────────────────────────────────────────────
    cols     = ["stage", "variant", "ts_arch", "best_val_mae",
                "test_MAE", "test_RMSE", "test_MedAE", "test_R2", "run_dir"]
    csv_path = result_dir / "multimodal_comparison.csv"
    df[cols].to_csv(csv_path, index=False)
    print(f"[compare_mm] Saved → {csv_path}")
    print("\n" + df[cols].to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_bar(df,    plot_dir / "multimodal_bar.png")
    plot_curves(df, out_root, plot_dir / "multimodal_curves.png")

    # ── Best model summary ────────────────────────────────────────────────────
    best = df.sort_values("test_MAE").iloc[0]
    print(f"\n{'─'*55}")
    print(f"  Best overall model : {_run_label(best).replace(chr(10), ' ')}")
    print(f"  Test MAE           : {best['test_MAE']:.4f} h")
    print(f"  Test RMSE          : {best['test_RMSE']:.4f} h")
    print(f"  Stage              : {best['stage']}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    main()