from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"

TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_FEATURES = ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
CATEGORICAL_FEATURES = [
    "gender",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]


def load_raw(path: str | Path = RAW_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # TotalCharges is text and is blank only for tenure == 0 customers
    # (not billed yet), so blanks become 0 instead of being imputed.
    out = df.copy()

    total = pd.to_numeric(out["TotalCharges"].astype(str).str.strip(), errors="coerce")
    blank = total.isna()
    if not (out.loc[blank, "tenure"] == 0).all():
        raise ValueError("Found blank TotalCharges for customers with tenure > 0")
    out["TotalCharges"] = total.fillna(0.0)

    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling", TARGET]:
        out[col] = out[col].map({"Yes": 1, "No": 0}).astype(int)
    out["SeniorCitizen"] = out["SeniorCitizen"].astype(int)

    return out


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    return df[features], df[TARGET]


def load_clean(path: str | Path = RAW_CSV) -> pd.DataFrame:
    return clean(load_raw(path))
