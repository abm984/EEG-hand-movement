"""
src/evaluate.py
---------------
Evaluation metrics and publication-quality visualisations.

Functions:
  - plot_confusion_matrix
  - plot_roc_curves
  - plot_accuracy_bars
  - plot_topomap_band_power
  - plot_csp_patterns
  - plot_erp
  - plot_learning_curve
  - save_all_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

PALETTE = {
    "LogisticRegression": "#4C72B0",
    "SVM_RBF":            "#DD8452",
    "RandomForest":       "#55A868",
    "EEGNet":             "#C44E52",
    "Chance":             "#8172B2",
}


# ── 1. Confusion matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    labels: list[str] = ("Left fist", "Right fist"),
    save_path: str | None = None,
) -> plt.Figure:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ── 2. ROC curves ─────────────────────────────────────────────────────────────

def plot_roc_curves(
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    save_path: str | None = None,
) -> plt.Figure:
    """
    roc_data = {'ModelName': (fpr, tpr, auc), ...}
    """
    from sklearn.metrics import auc as sklearn_auc

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance (AUC = 0.50)")

    for name, (fpr, tpr, auc_val) in roc_data.items():
        color = PALETTE.get(name, "#555")
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {auc_val:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Left vs Right Motor Imagery", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ── 3. Accuracy comparison bar chart ─────────────────────────────────────────

def plot_accuracy_bars(
    results: list[dict],
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart comparing model accuracies with error bars."""
    models = [r["model"] for r in results]
    accs = [r.get("accuracy_mean", 0) * 100 for r in results]
    stds = [r.get("accuracy_std", 0) * 100 for r in results]
    colors = [PALETTE.get(m, "#888") for m in models]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(models, accs, color=colors, alpha=0.85, edgecolor="white",
                  linewidth=0.8, yerr=stds, capsize=5, error_kw={"elinewidth": 1.5})

    # Chance line
    ax.axhline(50, color="#888", ls="--", lw=1.2, label="Chance (50%)")
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Model Comparison — EEG Motor Imagery (Left vs Right)",
                 fontsize=12, pad=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)

    for bar, val, std in zip(bars, accs, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ── 4. Scalp topomap for band power ──────────────────────────────────────────

def plot_topomap_band_power(
    epochs,                   # mne.Epochs
    band: str = "alpha",
    classes: list[str] = ("left_fist", "right_fist"),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot scalp topomap of band power difference between two classes.
    Requires mne.Epochs with proper channel info.
    """
    try:
        import mne
        from mne.time_frequency import psd_array_welch

        BANDS = {
            "delta":  (1, 4),   "theta": (4, 8),
            "alpha":  (8, 13),  "beta": (13, 30),
            "gamma": (30, 40),
        }
        lo, hi = BANDS.get(band, (8, 13))
        sfreq = epochs.info["sfreq"]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        class_data = {}
        for cls in classes:
            X = epochs[cls].get_data(units="uV")
            psd, freqs = psd_array_welch(X, sfreq=sfreq, fmin=lo, fmax=hi,
                                          n_per_seg=int(sfreq), verbose=False)
            class_data[cls] = np.mean(psd, axis=(0, 2))  # mean over epochs and freqs

        for idx, (cls, power) in enumerate(class_data.items()):
            im, _ = mne.viz.plot_topomap(
                np.log10(power + 1e-12), epochs.info,
                axes=axes[idx], show=False,
                cmap="RdBu_r", contours=6,
            )
            axes[idx].set_title(f"{cls.replace('_', ' ').title()}\n{band.title()} power",
                                 fontsize=10)

        # Difference map
        diff = class_data[classes[0]] - class_data[classes[1]]
        im, _ = mne.viz.plot_topomap(
            diff, epochs.info, axes=axes[2], show=False,
            cmap="RdBu_r", contours=6,
        )
        axes[2].set_title(f"Difference\n({classes[0]} − {classes[1]})", fontsize=10)
        plt.colorbar(im, ax=axes[2], shrink=0.8, label="Log power diff (µV²/Hz)")
        plt.suptitle(f"{band.title()}-band Scalp Topography", fontsize=13, y=1.02)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path)
        return fig
    except Exception as e:
        print(f"[topomap] Could not plot: {e}")
        return plt.figure()


# ── 5. ERP time-series ────────────────────────────────────────────────────────

def plot_erp(
    epochs,
    channels: list[str] = ("C3", "Cz", "C4"),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot event-related potentials for left vs right imagery."""
    fig, axes = plt.subplots(1, len(channels), figsize=(14, 3.5), sharey=True)

    colors = {"left_fist": PALETTE["LogisticRegression"],
              "right_fist": PALETTE["SVM_RBF"]}

    for ax, ch in zip(axes, channels):
        if ch not in epochs.ch_names:
            ch = epochs.ch_names[0]
        times = epochs.times
        for cls, color in colors.items():
            try:
                data = epochs[cls].get_data(picks=[ch], units="uV")
                mean = data[:, 0, :].mean(axis=0)
                sem = data[:, 0, :].std(axis=0) / np.sqrt(len(data))
                ax.plot(times, mean, color=color, label=cls.replace("_", " "), lw=2)
                ax.fill_between(times, mean - sem, mean + sem,
                                color=color, alpha=0.2)
            except Exception:
                pass
        ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.axhline(0, color="gray", ls="-", lw=0.5, alpha=0.4)
        ax.set_title(f"Channel {ch}", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=9)

    axes[0].set_ylabel("Amplitude (µV)", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper right")
    plt.suptitle("Event-Related Potentials — Motor Imagery Onset", fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ── 6. Learning curve ─────────────────────────────────────────────────────────

def plot_learning_curve(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y,
        cv=5, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=1,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-",
            color="#4C72B0", label="Training accuracy")
    ax.fill_between(train_sizes,
                    train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1),
                    alpha=0.2, color="#4C72B0")
    ax.plot(train_sizes, val_scores.mean(axis=1), "s-",
            color="#C44E52", label="Validation accuracy")
    ax.fill_between(train_sizes,
                    val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1),
                    alpha=0.2, color="#C44E52")
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="Chance")
    ax.set_xlabel("Training set size", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(f"Learning Curve — {title}", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0.3, 1.0)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ── 7. EEGNet training loss curve ─────────────────────────────────────────────

def plot_training_loss(
    train_losses: list[float],
    model_name: str = "EEGNet",
    save_path: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(train_losses, color=PALETTE.get(model_name, "#C44E52"), lw=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax.set_title(f"{model_name} Training Loss", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ── Save everything ───────────────────────────────────────────────────────────

def save_all_figures(
    results: list[dict],
    epochs=None,
    output_dir: str = "results/",
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Accuracy bar chart
    plot_accuracy_bars(results, save_path=str(out / "accuracy_comparison.png"))
    print(f"✅ Saved accuracy_comparison.png")

    if epochs is not None:
        for band in ["alpha", "beta"]:
            plot_topomap_band_power(
                epochs, band=band,
                save_path=str(out / f"topomap_{band}.png"),
            )
            print(f"✅ Saved topomap_{band}.png")

        plot_erp(epochs, save_path=str(out / "erp.png"))
        print(f"✅ Saved erp.png")

    print(f"\n📂 All figures saved to {out.resolve()}")
