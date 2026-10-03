"""
visualization.py
Publication-quality maps (susceptibility, uncertainty, classified risk) and
diagnostic charts (correlation heatmap, feature importance, ROC/PR curves,
observed-vs-predicted, sensitivity analysis). All figures are saved to
outputs/figures rather than only displayed, so they persist for reports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .utils import get_logger

logger = get_logger(__name__)


def _save(fig, out_path: str | Path, dpi: int = 300):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved figure: %s", out_path)


def plot_raster_map(arr: np.ndarray, title: str, out_path: str | Path,
                     cmap: str = "YlGnBu", label: str = "Score",
                     vmin: Optional[float] = None, vmax: Optional[float] = None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(label)
    _save(fig, out_path)
    plt.close(fig)


def plot_classified_map(classes: np.ndarray, legend: dict, title: str, out_path: str | Path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches

    n = len(legend)
    colors = ["#2166ac", "#67a9cf", "#fee090", "#fc8d59", "#b2182b"][:n]
    cmap = ListedColormap(colors)
    bounds = np.arange(0.5, n + 1.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(classes, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    patches = [mpatches.Patch(color=colors[i], label=legend[i + 1]) for i in range(n)]
    ax.legend(handles=patches, loc="lower left", bbox_to_anchor=(1.01, 0), fontsize=9)
    _save(fig, out_path)
    plt.close(fig)


def plot_correlation_heatmap(corr_df, title: str, out_path: str | Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr_df.columns)))
    ax.set_yticklabels(corr_df.columns, fontsize=8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.045, label="Correlation coefficient")
    _save(fig, out_path)
    plt.close(fig)


def plot_feature_importance(weights: dict, title: str, out_path: str | Path):
    import matplotlib.pyplot as plt
    items = sorted(weights.items(), key=lambda kv: kv[1])
    names, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(names))))
    ax.barh(names, vals, color="#2166ac")
    ax.set_xlabel("Weight")
    ax.set_title(title, fontsize=13, fontweight="bold")
    _save(fig, out_path)
    plt.close(fig)


def plot_observed_vs_predicted(observed: np.ndarray, predicted: np.ndarray,
                                title: str, out_path: str | Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(observed, predicted, alpha=0.6, edgecolor="k", linewidth=0.3)
    lims = [min(np.nanmin(observed), np.nanmin(predicted)),
            max(np.nanmax(observed), np.nanmax(predicted))]
    ax.plot(lims, lims, "r--", label="1:1 line")
    ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    _save(fig, out_path)
    plt.close(fig)


def plot_roc_curve(observed_event: np.ndarray, predicted_score: np.ndarray,
                    out_path: str | Path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    mask = np.isfinite(observed_event) & np.isfinite(predicted_score)
    fpr, tpr, _ = roc_curve(observed_event[mask], predicted_score[mask])
    auc = roc_auc_score(observed_event[mask], predicted_score[mask])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})", color="#2166ac")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Heavy Rainfall Event Classification — ROC Curve")
    ax.legend()
    _save(fig, out_path)
    plt.close(fig)


def plot_sensitivity_bar(report_df, out_path: str | Path):
    import matplotlib.pyplot as plt
    top = report_df.groupby("factor")["mean_abs_change"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(top))))
    ax.barh(top.index, top.values, color="#b2182b")
    ax.set_xlabel("Mean |change| in susceptibility score")
    ax.set_title("Weight Sensitivity by Factor", fontsize=13, fontweight="bold")
    _save(fig, out_path)
    plt.close(fig)
