"""
evaluate.py
===========
Model evaluation on the held-out test set (Volume 5 metrics).

Because churn is imbalanced, we report precision / recall / F1 / ROC-AUC, not
just accuracy. We also plot the confusion matrix, the ROC curve, and the top
feature importances -- the three charts an interviewer expects to see.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


def _save(fig, name: str) -> None:
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] saved {path.name}")


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series,
             cv_results: dict | None = None, best_name: str | None = None) -> dict:
    """Score the fitted pipeline on the test set and write metrics + plots."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": best_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    if cv_results is not None:
        metrics["cross_validation"] = cv_results

    print("\n[eval] TEST-SET PERFORMANCE")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {k:<10}: {metrics[k]:.3f}")
    print("\n[eval] classification report:")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

    # Confusion matrix ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Stayed", "Churned"], cmap="Blues", ax=ax
    )
    ax.set_title("Confusion matrix (test set)")
    _save(fig, "06_confusion_matrix.png")

    # ROC curve -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=best_name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="random")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    _save(fig, "07_roc_curve.png")

    # Feature importances / coefficients ------------------------------------
    _plot_importances(model)

    # Persist metrics -------------------------------------------------------
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] saved metrics -> {config.METRICS_PATH}")
    return metrics


def _plot_importances(model) -> None:
    """Plot the 15 most influential features from the fitted pipeline."""
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]
    names = pre.get_feature_names_out()

    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
        title = "Top feature importances"
    elif hasattr(clf, "coef_"):
        vals = np.abs(clf.coef_[0])
        title = "Top features (|logistic coefficient|)"
    else:
        return

    imp = (pd.Series(vals, index=names)
           .sort_values(ascending=False).head(15).iloc[::-1])
    # tidy names: strip the "num__" / "cat__" ColumnTransformer prefixes
    imp.index = [n.split("__", 1)[-1] for n in imp.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel("importance")
    _save(fig, "08_feature_importance.png")


if __name__ == "__main__":
    import joblib
    from .train import load_split
    Xtr, Xte, ytr, yte = load_split()
    mdl = joblib.load(config.MODEL_PATH)
    evaluate(mdl, Xte, yte)
