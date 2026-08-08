"""
data.py
=======
Data collection + cleaning (Volume 4, Chapters 1-2).

Responsibilities:
  1. Load the raw Telco CSV.
  2. Fix the well-known data-quality problems in this dataset:
       - `TotalCharges` is stored as text and has 11 blank strings.
       - the target `Churn` is text ("Yes"/"No") and must become 1/0.
       - `customerID` is an identifier with no predictive value.
  3. Save a clean, typed copy as Parquet (smaller + preserves dtypes).

Design choice: row-wise fixes (type parsing, dropping an ID column, encoding
the target) are safe to do on the whole dataset. Anything *learned* from the
data (scaling, imputation values, encoders) is deliberately left for the
scikit-learn Pipeline in train.py so that it fits on the training split only
and never leaks test information.
"""

from __future__ import annotations

import pandas as pd

from . import config


def load_raw() -> pd.DataFrame:
    """Read the raw CSV exactly as delivered."""
    df = pd.read_csv(config.DATA_RAW)
    print(f"[data] loaded raw data: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned, typed DataFrame ready for feature engineering."""
    df = df.copy()

    # 1) customerID is a unique identifier -> not a feature. Drop it.
    if config.ID_COLUMN in df.columns:
        df = df.drop(columns=[config.ID_COLUMN])

    # 2) TotalCharges is text with 11 blank strings. Coerce to numeric so the
    #    blanks become NaN, then treat them. These 11 rows are all tenure == 0
    #    (brand-new customers who have not been billed yet), so a TotalCharges
    #    of 0 is the correct, meaningful fill -- not an arbitrary guess.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    new_customers = df["tenure"] == 0
    df.loc[new_customers, "TotalCharges"] = df.loc[
        new_customers, "TotalCharges"
    ].fillna(0.0)
    # Safety net: if any NaN remains for some other reason, fill with median.
    if df["TotalCharges"].isna().any():
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # 3) Encode the target: Churn "Yes"/"No" -> 1/0.
    df[config.TARGET] = (df[config.TARGET] == "Yes").astype(int)

    # 4) SeniorCitizen arrives as 0/1 int; make it a clear categorical string so
    #    it is one-hot encoded consistently with the other yes/no columns.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    print(
        f"[data] cleaned data: {df.shape[0]} rows x {df.shape[1]} cols, "
        f"{df.isna().sum().sum()} missing values remaining"
    )
    return df


def get_clean_data(save: bool = True) -> pd.DataFrame:
    """Full load -> clean, optionally caching the result.

    Prefers Parquet (smaller, keeps dtypes) but falls back to CSV if no Parquet
    engine (pyarrow/fastparquet) is installed, so the project always runs.
    """
    df = clean(load_raw())
    if save:
        try:
            df.to_parquet(config.DATA_PROCESSED, index=False)
            print(f"[data] saved clean data -> {config.DATA_PROCESSED}")
        except ImportError:
            csv_path = config.DATA_PROCESSED.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            print(f"[data] pyarrow not found; saved CSV instead -> {csv_path}")
    return df


if __name__ == "__main__":
    get_clean_data()
