"""
features.py
===========
Feature engineering (Volume 4, Chapter 4) + the leakage-safe preprocessor
(Volume 4, Chapters 2 & 5).

Two parts:

  add_features(df)      -> deterministic, row-wise engineered columns. These
                           are pure functions of a single row, so they are safe
                           to compute on the whole dataset (no target used, no
                           parameters learned from data).

  build_preprocessor() -> a ColumnTransformer that imputes + scales numeric
                           columns and one-hot encodes categoricals. It is a
                           *transformer only*; it is fitted inside the Pipeline
                           on the training split, so nothing leaks.

Engineered features and the reasoning behind each (this is what you explain in
an interview):
  - tenure_group        binning tenure into loyalty bands can expose a
                        non-linear relationship with churn.
  - num_services        an aggregation: how many paid services the customer has.
                        More services usually means "stickier" customers.
  - avg_charges_per_mo  a ratio (TotalCharges / tenure): the customer's typical
                        monthly spend over their whole lifetime.
  - is_new_customer     a flag: churn is heavily concentrated in the first few
                        months, so "newness" is a strong, simple signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Service columns used to build the "number of services" aggregation feature.
_SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

# Engineered numeric columns (added by add_features).
_ENGINEERED_NUMERIC = ["num_services", "avg_charges_per_mo"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns. Deterministic and target-free (no leakage)."""
    df = df.copy()

    # --- Binning: tenure -> loyalty bands (0-1yr, 1-2yr, 2-4yr, 4-5yr, 5-6yr).
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, np.inf],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"],
    ).astype(str)

    # --- Aggregation: count how many real services the customer subscribes to.
    #     A value counts as a service if it is not "No" and not "No ... service".
    def _is_active(value: object) -> bool:
        s = str(value)
        return s not in ("No", "No phone service", "No internet service")

    df["num_services"] = df[_SERVICE_COLS].map(_is_active).sum(axis=1)

    # --- Ratio: average monthly spend over the customer's lifetime.
    #     Guard tenure == 0 (new customers) to avoid divide-by-zero; for them
    #     the best estimate of lifetime monthly spend is the current charge.
    df["avg_charges_per_mo"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"].clip(lower=1),
        df["MonthlyCharges"],
    )

    # --- Flag: is this a brand-new customer (<= 6 months)?
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int).map({0: "No", 1: "Yes"})

    return df


def split_columns(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) for everything except target."""
    features = df.drop(columns=[target])
    numeric_cols = features.select_dtypes(include="number").columns.tolist()
    categorical_cols = features.select_dtypes(exclude="number").columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """
    Build the leakage-safe preprocessing step.

    Numeric   -> median impute (robust) + standard scale (needed by logistic
                 regression; harmless for trees).
    Categorical -> most-frequent impute + one-hot encode. handle_unknown="ignore"
                 keeps the pipeline from crashing on categories unseen in training.
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
