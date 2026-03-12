"""
Calibration Analysis — Oscar Prediction Models
================================================

Evaluates how well-calibrated each model's confidence scores are:
  - Reliability diagrams (confidence vs actual accuracy)
  - Expected Calibration Error (ECE)
  - Brier scores

Usage:
  python calibration.py

Outputs:
  - plots/10_calibration_reliability.png
  - plots/11_calibration_per_model.png
  - Calibration metrics table printed to console
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from helpers import SCRIPT_DIR

PLOT_DIR = SCRIPT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

# Confidence bins: [0, 0.3), [0.3, 0.5), [0.5, 0.7), [0.7, 0.9), [0.9, 1.0]
BIN_EDGES = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
BIN_LABELS = ["0–30%", "30–50%", "50–70%", "70–90%", "90–100%"]


def _to_bool(val):
    """Convert Correct column to boolean, handling string 'True'/'False'."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return bool(val)


def compute_calibration(bt_df, model=None):
    """Bucket predictions by confidence and compute actual accuracy per bin.

    Parameters
    ----------
    bt_df : DataFrame
        Backtest results with columns: Year, Category, Predicted, Actual,
        Correct, Confidence, Model.
    model : str, optional
        If provided, filter to this model only.

    Returns
    -------
    DataFrame with columns: bin_label, mean_confidence, actual_accuracy, count
    """
    df = bt_df.copy()
    df["Correct"] = df["Correct"].apply(_to_bool).astype(int)

    if model is not None:
        df = df[df["Model"] == model]

    df["bin"] = pd.cut(df["Confidence"], bins=BIN_EDGES, labels=BIN_LABELS,
                       include_lowest=True, right=False)

    grouped = df.groupby("bin", observed=False).agg(
        mean_confidence=("Confidence", "mean"),
        actual_accuracy=("Correct", "mean"),
        count=("Correct", "count"),
    ).reset_index().rename(columns={"bin": "bin_label"})

    # Replace NaN mean_confidence (empty bins) with bin midpoint
    midpoints = [0.15, 0.4, 0.6, 0.8, 0.95]
    for i, row in grouped.iterrows():
        if pd.isna(row["mean_confidence"]):
            grouped.at[i, "mean_confidence"] = midpoints[i]
        if pd.isna(row["actual_accuracy"]):
            grouped.at[i, "actual_accuracy"] = 0.0

    return grouped


def _compute_ece(cal_df, total):
    """Compute Expected Calibration Error from a calibration DataFrame."""
    ece = 0.0
    for _, row in cal_df.iterrows():
        if row["count"] > 0:
            ece += (row["count"] / total) * abs(row["mean_confidence"] - row["actual_accuracy"])
    return ece


def plot_calibration_reliability(bt_df):
    """Reliability diagram with per-model lines and count distribution.

    Saves to plots/10_calibration_reliability.png.
    """
    df = bt_df.copy()
    df["Correct"] = df["Correct"].apply(_to_bool).astype(int)
    models = sorted(df["Model"].unique())

    fig, (ax_rel, ax_bar) = plt.subplots(
        2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )

    # Diagonal reference
    ax_rel.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.5, label="Perfect calibration")

    palette = sns.color_palette("husl", len(models))

    # Per-model lines
    for color, model in zip(palette, models):
        cal = compute_calibration(bt_df, model=model)
        mask = cal["count"] > 0
        ax_rel.plot(cal.loc[mask, "mean_confidence"], cal.loc[mask, "actual_accuracy"],
                    marker="o", ms=5, lw=1.2, color=color, alpha=0.6, label=model)

    # Aggregate line (all models)
    cal_all = compute_calibration(bt_df)
    mask = cal_all["count"] > 0
    ax_rel.plot(cal_all.loc[mask, "mean_confidence"], cal_all.loc[mask, "actual_accuracy"],
                marker="s", ms=7, lw=2.5, color="black", label="All models", zorder=10)

    ax_rel.set_ylabel("Actual Accuracy", fontsize=12)
    ax_rel.set_title("Model Calibration: Reliability Diagram",
                     fontsize=13, fontweight="bold")
    ax_rel.legend(fontsize=8, loc="upper left", ncol=2)
    ax_rel.set_xlim(-0.02, 1.02)
    ax_rel.set_ylim(-0.02, 1.02)

    # Bar subplot — count distribution
    x_pos = np.arange(len(BIN_LABELS))
    ax_bar.bar(x_pos, cal_all["count"], color="steelblue", edgecolor="white", width=0.7)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(BIN_LABELS, fontsize=9)
    ax_bar.set_ylabel("Count", fontsize=11)
    ax_bar.set_xlabel("Confidence Bin", fontsize=11)

    fig.tight_layout()
    out = PLOT_DIR / "10_calibration_reliability.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_calibration_per_model(bt_df):
    """3x3 grid of per-model reliability curves with ECE annotation.

    Saves to plots/11_calibration_per_model.png.
    """
    df = bt_df.copy()
    df["Correct"] = df["Correct"].apply(_to_bool).astype(int)
    models = sorted(df["Model"].unique())[:9]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes_flat = axes.flatten()

    for idx, ax in enumerate(axes_flat):
        if idx < len(models):
            model = models[idx]
            cal = compute_calibration(bt_df, model=model)
            total = cal["count"].sum()
            ece = _compute_ece(cal, total) if total > 0 else 0.0

            mask = cal["count"] > 0
            ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
            ax.plot(cal.loc[mask, "mean_confidence"], cal.loc[mask, "actual_accuracy"],
                    marker="o", ms=5, lw=1.8, color="steelblue")
            ax.set_title(model, fontsize=11, fontweight="bold")
            ax.annotate(f"ECE = {ece:.3f}", xy=(0.05, 0.90), xycoords="axes fraction",
                        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
        else:
            ax.set_visible(False)

    fig.suptitle("Calibration by Model", fontsize=13, fontweight="bold", y=1.01)
    fig.supxlabel("Mean Confidence", fontsize=12)
    fig.supylabel("Actual Accuracy", fontsize=12)
    fig.tight_layout()
    out = PLOT_DIR / "11_calibration_per_model.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def compute_calibration_metrics(bt_df):
    """Compute ECE and Brier score per model.

    Returns
    -------
    DataFrame with columns: Model, ECE, Brier_Score — sorted by ECE ascending.
    """
    df = bt_df.copy()
    df["Correct"] = df["Correct"].apply(_to_bool).astype(int)
    models = sorted(df["Model"].unique())

    rows = []
    for model in models:
        mdf = df[df["Model"] == model]
        cal = compute_calibration(bt_df, model=model)
        total = cal["count"].sum()
        ece = _compute_ece(cal, total) if total > 0 else 0.0
        brier = ((mdf["Confidence"] - mdf["Correct"]) ** 2).mean()
        rows.append({"Model": model, "ECE": round(ece, 4), "Brier_Score": round(brier, 4)})

    return pd.DataFrame(rows).sort_values("ECE").reset_index(drop=True)


def main():
    bt_df = pd.read_csv(SCRIPT_DIR / "backtest_results.csv")
    print(f"Loaded {len(bt_df)} backtest rows, {bt_df['Model'].nunique()} models.\n")

    plot_calibration_reliability(bt_df)
    plot_calibration_per_model(bt_df)

    metrics = compute_calibration_metrics(bt_df)
    print("\nCalibration Metrics (sorted by ECE):")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
