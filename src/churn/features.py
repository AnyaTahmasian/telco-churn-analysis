from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
ENGINEERED_FEATURES = ["num_addon_services", "avg_monthly_spend"]


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["num_addon_services"] = (X[ADDON_SERVICES] == "Yes").sum(axis=1)
    months = X["tenure"].where(X["tenure"] > 0)
    X["avg_monthly_spend"] = (X["TotalCharges"] / months).fillna(X["MonthlyCharges"])
    return X


def _engineered_names(transformer, input_features):
    return list(input_features) + ENGINEERED_FEATURES


def build_preprocessor(scale_numeric: bool = True) -> Pipeline:
    # fitted inside the pipeline so CV folds never leak into each other
    numeric = NUMERIC_FEATURES + ENGINEERED_FEATURES
    columns = ColumnTransformer(
        transformers=[
            ("num", StandardScaler() if scale_numeric else "passthrough", numeric),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("engineer", FunctionTransformer(add_engineered_features, feature_names_out=_engineered_names)),
            ("columns", columns),
        ]
    )
