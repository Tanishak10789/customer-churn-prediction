"""
predict.py
==========
Inference: load the saved pipeline and score new customers.

This demonstrates the payoff of a Pipeline -- the exact same cleaning-adjacent
feature engineering + preprocessing + model is applied to new rows with one
call, so training and serving cannot drift apart.

Note: add_features() is applied here too, because the saved pipeline expects the
engineered columns (tenure_group, num_services, avg_charges_per_mo,
is_new_customer). The heavy learned steps (impute/scale/encode) live inside the
pipeline; only the deterministic row-wise features are added first.
"""

from __future__ import annotations

import joblib
import pandas as pd

from . import config
from .features import add_features


def load_model():
    """Load the fitted pipeline saved by train.py."""
    return joblib.load(config.MODEL_PATH)


def predict(customers: pd.DataFrame) -> pd.DataFrame:
    """Return churn prediction + probability for each input customer row."""
    model = load_model()
    prepared = add_features(customers.copy())
    proba = model.predict_proba(prepared)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = customers.copy()
    out["churn_probability"] = proba.round(3)
    out["churn_prediction"] = pd.Series(pred).map({0: "Stayed", 1: "Churned"}).values
    return out


def _demo_customer() -> pd.DataFrame:
    """One example customer for a quick smoke test."""
    return pd.DataFrame([{
        "gender": "Female", "SeniorCitizen": "No", "Partner": "No",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.7,
        "TotalCharges": 191.4,
    }])


if __name__ == "__main__":
    result = predict(_demo_customer())
    cols = ["tenure", "Contract", "MonthlyCharges",
            "churn_probability", "churn_prediction"]
    print(result[cols].to_string(index=False))
