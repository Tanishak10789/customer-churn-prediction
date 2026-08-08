"""
config.py
=========
Central place for file paths, column groups, and constants so that every
module reads from one source of truth. Keeping configuration in one file is a
small habit that makes a project reproducible and easy to change.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all resolved relative to the project root, so the code runs from
# anywhere without hard-coded absolute paths).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "telco.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "telco_clean.parquet"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"

# ---------------------------------------------------------------------------
# Modelling constants
# ---------------------------------------------------------------------------
TARGET = "Churn"                 # what we are predicting (Yes/No -> 1/0)
ID_COLUMN = "customerID"         # identifier, must be dropped before modelling
RANDOM_STATE = 42                # fixed seed everywhere -> reproducible runs
TEST_SIZE = 0.20                 # 80/20 train/test split

# Columns that are numeric after cleaning.
NUMERIC_BASE = ["tenure", "MonthlyCharges", "TotalCharges"]

# Ensure output folders exist on import.
for _d in (DATA_PROCESSED.parent, MODELS_DIR, FIGURES_DIR, METRICS_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)
