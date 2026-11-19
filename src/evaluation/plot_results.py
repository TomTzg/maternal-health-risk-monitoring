"""Create figures directly from in-memory training and evaluation results."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.utilities import FIGURES


def _wilson_interval(errors, total, z=1.96):
    """Return a two-sided 95% Wilson score interval for a binomial proportion."""
    if total == 0:
        return np.nan, np.nan
    proportion = errors / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def plot_model_comparison(comparison):
    comparison = comparison.sort_values("validation_macro_f1")
    display_names = {
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "decision_tree": "Decision Tree",
        "rbf_svm": "RBF SVM",
        "gaussian_nb": "Gaussian NB",
        "logistic_regression": "Logistic Regression",
        "majority_baseline": "Majority Baseline",
    }
    names = comparison["model"].map(display_names).fillna(comparison["model"])
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.barh(names, comparison["validation_macro_f1"], color="#3977a8", edgecolor="white")
    ax.errorbar(comparison["validation_macro_f1"], np.arange(len(comparison)),
                xerr=comparison["validation_std"], fmt="none", ecolor="#253746",
                capsize=3, linewidth=1.2)
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("Validation macro-F1")
    ax.set_title("Model comparison on cross-validation")
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    for bar, score, std in zip(bars, comparison["validation_macro_f1"], comparison["validation_std"]):
        ax.text(score + std + 0.012, bar.get_y() + bar.get_height() / 2, f"{score:.3f}",
                va="center", fontsize=9)
    fig.text(0.5, 0.015, "Error bars show ±1 standard deviation across 5 CV folds (not a confidence interval).",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURES / "model_comparison_macro_f1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_age_group_errors(predictions):
    groups = pd.cut(predictions["Age"], [-np.inf, 19, 29, 39, np.inf],
                    labels=["<20", "20–29", "30–39", "40+"])
    rows = []
    for group in groups.cat.categories:
        mask = groups == group
        total = int(mask.sum())
        errors = int((predictions.loc[mask, "RiskLevel"] != predictions.loc[mask, "PredictedRisk"]).sum())
        lower, upper = _wilson_interval(errors, total)
        rows.append((str(group), total, errors, errors / total if total else np.nan, lower, upper))

    labels, counts, error_counts, rates, lowers, uppers = zip(*rows)
    positions = np.arange(len(labels))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(positions, rates, color="#d47a45", width=0.62, edgecolor="white")
    ax.errorbar(positions, rates,
                yerr=[np.array(rates) - np.array(lowers), np.array(uppers) - np.array(rates)],
                fmt="none", ecolor="#253746", capsize=4, linewidth=1.3)
    ax.set_xticks(positions, [f"{label}\n(n={count})" for label, count in zip(labels, counts)])
    ax.set_ylim(0, max(0.25, max(uppers) + 0.06))
    ax.set_ylabel("Held-out error rate")
    ax.set_xlabel("Age group")
    ax.set_title("Held-out errors by age group")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    for x, rate, err_count in zip(positions, rates, error_counts):
        ax.text(x, rate + 0.012, f"{err_count}/{counts[x]} errors", ha="center", va="bottom", fontsize=9)
    fig.text(0.5, 0.015, "Error bars show 95% Wilson intervals; small age groups have greater uncertainty.",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURES / "age_group_error_rate.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
