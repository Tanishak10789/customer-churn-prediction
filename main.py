"""
main.py
=======
Run the whole project end to end:

    clean data -> EDA figures -> train + compare models -> evaluate -> demo predict

Usage:
    python main.py

Every artefact (clean parquet, figures, saved model, metrics.json) is written
under data/, reports/, and models/.
"""

from src.data import get_clean_data
from src.eda import run_eda
from src.evaluate import evaluate
from src.features import add_features
from src.predict import predict, _demo_customer
from src.train import compare_models, load_split, train_best


def run() -> None:
    print("=" * 70)
    print("TELCO CUSTOMER CHURN PREDICTION -- full pipeline")
    print("=" * 70)

    # 1) EDA on the cleaned + engineered data.
    run_eda(add_features(get_clean_data(save=True)))

    # 2) Train / compare / select.
    X_train, X_test, y_train, y_test = load_split()
    best_name, cv_results = compare_models(X_train, y_train)
    model = train_best(X_train, y_train, best_name)

    # 3) Evaluate on the held-out test set.
    evaluate(model, X_test, y_test, cv_results=cv_results, best_name=best_name)

    # 4) Demonstrate inference on a new customer.
    print("\n[main] example prediction on a new customer:")
    demo = predict(_demo_customer())
    print(demo[["tenure", "Contract", "MonthlyCharges",
                "churn_probability", "churn_prediction"]].to_string(index=False))

    print("\nDone. See reports/figures/, reports/metrics.json, and models/.")


if __name__ == "__main__":
    run()
