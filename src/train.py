"""
train.py
========
Model training (Volume 4 Ch 5 pipelines + a preview of Volume 5 classifiers).

What it does:
  1. Loads clean data, adds engineered features.
  2. Makes a STRATIFIED train/test split (preserves the 27% churn rate on both
     sides -- essential for imbalanced data).
  3. Builds three candidate models, each as a full Pipeline
     (preprocessing + classifier) so every learned step fits on train only.
  4. Compares them with 5-fold stratified cross-validation on the TRAINING set
     using ROC-AUC and F1 (accuracy is misleading on imbalanced data).
  5. Refits the best model on the full training set and saves it with joblib.

The whole preprocessing + model lives in one object, so there is no data
leakage and the saved file can be loaded and used to predict directly.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .data import get_clean_data
from .features import add_features, build_preprocessor, split_columns


def _candidate_models() -> dict[str, object]:
    """Return the classifiers we compare. class_weight handles imbalance."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            random_state=config.RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=config.RANDOM_STATE
        ),
    }


def load_split():
    """Load engineered data and return a stratified train/test split."""
    df = add_features(get_clean_data(save=True))
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        stratify=y, random_state=config.RANDOM_STATE,
    )
    print(f"[train] train={len(X_train)}  test={len(X_test)}  "
          f"(churn rate train={y_train.mean():.1%}, test={y_test.mean():.1%})")
    return X_train, X_test, y_train, y_test


def compare_models(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[str, dict]:
    """Cross-validate every candidate; return (best_name, results dict)."""
    numeric_cols, categorical_cols = split_columns(
        pd.concat([X_train, y_train.rename(config.TARGET)], axis=1), config.TARGET
    )
    pre = build_preprocessor(numeric_cols, categorical_cols)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    results: dict[str, dict] = {}
    print("\n[train] 5-fold cross-validation on the training set:")
    print(f"  {'model':<22}{'ROC-AUC':>10}{'F1':>10}{'Recall':>10}")
    for name, clf in _candidate_models().items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring=["roc_auc", "f1", "recall"], n_jobs=-1,
        )
        results[name] = {
            "roc_auc": float(np.mean(scores["test_roc_auc"])),
            "roc_auc_std": float(np.std(scores["test_roc_auc"])),
            "f1": float(np.mean(scores["test_f1"])),
            "recall": float(np.mean(scores["test_recall"])),
        }
        r = results[name]
        print(f"  {name:<22}{r['roc_auc']:>10.3f}{r['f1']:>10.3f}{r['recall']:>10.3f}")

    # Select by F1, not ROC-AUC or accuracy. On an imbalanced churn problem the
    # business goal is to CATCH churners (recall) without too many false alarms
    # (precision); F1 balances those. Here it also favours the interpretable
    # model, which is easier to explain and act on.
    best = max(results, key=lambda k: results[k]["f1"])
    print(f"[train] best by F1: {best}")
    return best, results


def train_best(X_train, y_train, best_name: str) -> Pipeline:
    """Fit the chosen model on the full training set and persist it."""
    numeric_cols, categorical_cols = split_columns(
        pd.concat([X_train, y_train.rename(config.TARGET)], axis=1), config.TARGET
    )
    pre = build_preprocessor(numeric_cols, categorical_cols)
    clf = _candidate_models()[best_name]
    model = Pipeline([("pre", pre), ("clf", clf)])
    model.fit(X_train, y_train)
    joblib.dump(model, config.MODEL_PATH)
    print(f"[train] saved fitted pipeline -> {config.MODEL_PATH}")
    return model


if __name__ == "__main__":
    Xtr, Xte, ytr, yte = load_split()
    best_name, _ = compare_models(Xtr, ytr)
    train_best(Xtr, ytr, best_name)
