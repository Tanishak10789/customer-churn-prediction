"""
eda.py
======
Exploratory Data Analysis (Volume 4, Chapter 3).

Generates the charts that let you understand the data BEFORE modelling and
saves them to reports/figures/. EDA ends in written findings, not just plots --
the key conclusions are printed at the end and summarised in the README.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend so it runs anywhere (no display)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config

sns.set_theme(style="whitegrid")
_PALETTE = {"Stayed": "#4C72B0", "Churned": "#DD8452"}


def _save(fig: plt.Figure, name: str) -> None:
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] saved {path.name}")


def run_eda(df: pd.DataFrame) -> None:
    """Create the standard EDA plots for the churn dataset."""
    d = df.copy()
    d["ChurnLabel"] = d[config.TARGET].map({0: "Stayed", 1: "Churned"})

    # 1) Target class balance -- the first thing to check (imbalance!).
    fig, ax = plt.subplots(figsize=(5, 4))
    order = ["Stayed", "Churned"]
    counts = d["ChurnLabel"].value_counts().reindex(order)
    sns.barplot(x=order, y=counts.values, hue=order, palette=_PALETTE,
                legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 40, f"{v}\n({v / len(d):.0%})", ha="center", va="bottom")
    ax.set(title="Class balance: churn is imbalanced (~27%)",
           ylabel="customers", xlabel="")
    ax.set_ylim(0, counts.max() * 1.18)
    _save(fig, "01_class_balance.png")

    # 2) Churn rate by contract type -- the single strongest categorical driver.
    fig, ax = plt.subplots(figsize=(6, 4))
    rate = d.groupby("Contract")[config.TARGET].mean().sort_values(ascending=False)
    sns.barplot(x=rate.index, y=rate.values, hue=rate.index,
                palette="flare", legend=False, ax=ax)
    for i, v in enumerate(rate.values):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center", va="bottom")
    ax.set(title="Churn rate by contract type", ylabel="churn rate", xlabel="")
    ax.set_ylim(0, rate.max() * 1.18)
    _save(fig, "02_churn_by_contract.png")

    # 3) Tenure distribution split by churn -- new customers churn far more.
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=d, x="tenure", hue="ChurnLabel", palette=_PALETTE,
                 bins=30, element="step", stat="density", common_norm=False, ax=ax)
    ax.set(title="Tenure distribution by churn", xlabel="tenure (months)")
    _save(fig, "03_tenure_by_churn.png")

    # 4) Monthly charges distribution by churn -- higher charges -> more churn.
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(data=d, x="MonthlyCharges", hue="ChurnLabel", palette=_PALETTE,
                fill=True, common_norm=False, alpha=0.4, ax=ax)
    ax.set(title="Monthly charges by churn", xlabel="monthly charges ($)")
    _save(fig, "04_monthlycharges_by_churn.png")

    # 5) Correlation heatmap of numeric features (spot redundancy + drivers).
    num = d.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(num.corr(numeric_only=True), annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set(title="Correlation heatmap (numeric features)")
    _save(fig, "05_correlation_heatmap.png")

    # ---- Written findings (the real output of EDA) ---------------------------
    print("\n[eda] KEY FINDINGS")
    overall = d[config.TARGET].mean()
    print(f"  - Overall churn rate: {overall:.1%} (imbalanced -> use F1/ROC-AUC, not accuracy)")
    by_contract = d.groupby("Contract")[config.TARGET].mean()
    print(f"  - Month-to-month churn: {by_contract['Month-to-month']:.0%} vs "
          f"two-year: {by_contract['Two year']:.0%} (contract is a top driver)")
    lowt = d[d["tenure"] <= 6][config.TARGET].mean()
    print(f"  - Customers with tenure <= 6 months churn at {lowt:.0%} (newness matters)")


if __name__ == "__main__":
    from .data import get_clean_data
    from .features import add_features
    run_eda(add_features(get_clean_data(save=False)))
