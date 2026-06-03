"""
src/train.py
------------
Training + cross-validation for all EEG motor imagery models.

Cross-validation strategy:
  - Subject-independent (Leave-One-Subject-Out, LOSO)
    Best for BCI generalisation — never train and test on same subject.
  - Also supports stratified 5-fold for quick benchmarking.

Usage:
    python src/train.py --data-dir data/ --subjects 10 --cv loso --output results/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from preprocessing import preprocess_cohort, get_pooled_arrays
from features import make_csp_pipeline, make_psd_pipeline
from models import make_lr, make_svm, make_rf, EEGNetClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

SFREQ = 160.0
ALL_SUBJECTS = [f"sub-{i:03d}" for i in range(1, 110)]


# ── Metrics helper ────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray | None = None) -> dict:
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }
    if y_proba is not None:
        try:
            metrics["auc_roc"] = round(float(roc_auc_score(y_true, y_proba[:, 1])), 4)
        except Exception:
            metrics["auc_roc"] = None
    return metrics


# ── Build full pipeline: features + model ─────────────────────────────────────

def build_pipeline(model_name: str, feature_type: str, sfreq: float) -> Pipeline:
    """
    Combine a feature extractor with a classifier into one Pipeline.
    EEGNet receives raw epoch arrays (no hand-crafted features).
    """
    if model_name == "EEGNet":
        # EEGNet handles its own normalisation internally
        return EEGNetClassifier(epochs=80, batch_size=32)

    feat_pipe = (
        make_csp_pipeline(sfreq=sfreq)
        if feature_type == "csp"
        else make_psd_pipeline(sfreq=sfreq)
    )
    if model_name == "LogisticRegression":
        clf = make_lr()
    elif model_name == "SVM_RBF":
        clf = make_svm()
    elif model_name == "RandomForest":
        clf = make_rf()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Combine: features → StandardScaler is inside clf
    return Pipeline([
        ("features", feat_pipe),
        ("clf", clf["clf"] if hasattr(clf, "steps") else clf),
    ])


# ── LOSO cross-validation ─────────────────────────────────────────────────────

def run_loso_cv(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    feature_type: str = "csp",
    sfreq: float = SFREQ,
) -> dict:
    """
    Leave-One-Subject-Out cross-validation.

    Returns per-fold and aggregate metrics.
    """
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(groups=subject_ids)
    logger.info(f"\n{'='*50}")
    logger.info(f"Model: {model_name} | CV: LOSO ({n_splits} folds)")
    logger.info(f"{'='*50}")

    fold_metrics = []
    t0 = time.time()

    for fold, (train_idx, test_idx) in enumerate(
        tqdm(logo.split(X, y, groups=subject_ids),
             total=n_splits, desc=f"  {model_name}")
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline = build_pipeline(model_name, feature_type, sfreq)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = (
            pipeline.predict_proba(X_test)
            if hasattr(pipeline, "predict_proba")
            else None
        )
        metrics = compute_metrics(y_test, y_pred, y_proba)
        fold_metrics.append(metrics)

    elapsed = time.time() - t0

    # Aggregate
    results = {
        "model": model_name,
        "cv": "LOSO",
        "n_subjects": n_splits,
        "feature_type": feature_type,
        "elapsed_s": round(elapsed, 1),
    }
    for k in fold_metrics[0]:
        vals = [m[k] for m in fold_metrics if m[k] is not None]
        results[f"{k}_mean"] = round(float(np.mean(vals)), 4)
        results[f"{k}_std"] = round(float(np.std(vals)), 4)

    logger.info(
        f"  Accuracy: {results['accuracy_mean']:.2%} ± {results['accuracy_std']:.2%}"
    )
    return results


# ── K-fold cross-validation ───────────────────────────────────────────────────

def run_kfold_cv(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    feature_type: str = "csp",
    sfreq: float = SFREQ,
) -> dict:
    """Stratified K-fold CV — quicker than LOSO, less realistic."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    logger.info(f"\nModel: {model_name} | CV: {n_splits}-fold")

    fold_metrics = []
    t0 = time.time()

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline = build_pipeline(model_name, feature_type, sfreq)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = (
            pipeline.predict_proba(X_test)
            if hasattr(pipeline, "predict_proba")
            else None
        )
        fold_metrics.append(compute_metrics(y_test, y_pred, y_proba))

    elapsed = time.time() - t0
    results = {
        "model": model_name,
        "cv": f"StratifiedKFold-{n_splits}",
        "feature_type": feature_type,
        "elapsed_s": round(elapsed, 1),
    }
    for k in fold_metrics[0]:
        vals = [m[k] for m in fold_metrics if m[k] is not None]
        results[f"{k}_mean"] = round(float(np.mean(vals)), 4)
        results[f"{k}_std"] = round(float(np.std(vals)), 4)

    logger.info(
        f"  Accuracy: {results['accuracy_mean']:.2%} ± {results['accuracy_std']:.2%}"
    )
    return results


# ── Main training entry point ─────────────────────────────────────────────────

def train_all(
    data_dir: str,
    subject_ids: list[str],
    cv_strategy: str = "loso",
    feature_type: str = "csp",
    output_dir: str = "results/",
    models: list[str] | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & preprocess ──
    logger.info(f"\n📂 Loading {len(subject_ids)} subjects from {data_dir}")
    all_epochs, all_info = preprocess_cohort(subject_ids, data_dir)
    X, y, subject_ids_arr = get_pooled_arrays(all_epochs)

    logger.info(f"   X shape : {X.shape}")
    logger.info(f"   y shape : {y.shape} | balance: {np.bincount(y)}")

    # ── Train models ──
    model_names = models or ["LogisticRegression", "SVM_RBF", "RandomForest", "EEGNet"]
    all_results = []

    for model_name in model_names:
        if cv_strategy == "loso":
            result = run_loso_cv(
                model_name, X, y, subject_ids_arr,
                feature_type=feature_type
            )
        else:
            result = run_kfold_cv(
                model_name, X, y,
                feature_type=feature_type
            )
        all_results.append(result)

    # ── Save results ──
    results_path = output_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n✅ Results saved to {results_path}")

    # ── Save plots ──
    try:
        from evaluate import save_all_figures
        if all_epochs:
            logger.info(f"Generating and saving figures to {output_dir}...")
            save_all_figures(all_results, epochs=all_epochs[0], output_dir=str(output_dir))
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")

    # ── Print summary table ──
    print("\n" + "=" * 65)
    print(f"{'Model':<22} {'Accuracy':>10} {'F1 (macro)':>12} {'AUC-ROC':>10}")
    print("-" * 65)
    for r in all_results:
        acc = f"{r.get('accuracy_mean', 0):.2%} ±{r.get('accuracy_std', 0):.2%}"
        f1 = f"{r.get('f1_macro_mean', 0):.3f}"
        auc = f"{r.get('auc_roc_mean', 0):.3f}" if r.get('auc_roc_mean') else "  N/A"
        print(f"{r['model']:<22} {acc:>12} {f1:>10} {auc:>8}")
    print("=" * 65)

    return all_results


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train EEG Motor Imagery models")
    parser.add_argument("--data-dir", default="data/", help="Dataset root directory")
    parser.add_argument("--subjects", type=int, default=10,
                        help="Number of subjects to use (default: 10)")
    parser.add_argument("--cv", choices=["loso", "kfold"], default="kfold",
                        help="Cross-validation strategy")
    parser.add_argument("--features", choices=["csp", "psd"], default="csp",
                        help="Feature extraction method")
    parser.add_argument("--models", nargs="+",
                        default=["LogisticRegression", "SVM_RBF", "RandomForest", "EEGNet"],
                        help="Models to train")
    parser.add_argument("--output", default="results/", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n = min(args.subjects, 109)
    subjects = ALL_SUBJECTS[:n]

    train_all(
        data_dir=args.data_dir,
        subject_ids=subjects,
        cv_strategy=args.cv,
        feature_type=args.features,
        output_dir=args.output,
        models=args.models,
    )
