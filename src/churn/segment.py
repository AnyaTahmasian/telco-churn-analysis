from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .features import add_engineered_features

SEGMENT_FEATURES = ["tenure", "MonthlyCharges", "num_addon_services"]


def segment_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return add_engineered_features(df)[SEGMENT_FEATURES]


def silhouette_by_k(df: pd.DataFrame, ks=range(2, 9), random_state: int = 42) -> pd.DataFrame:
    Xs = StandardScaler().fit_transform(segment_matrix(df))
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Xs)
        rows.append({"k": k, "silhouette": silhouette_score(Xs, km.labels_, sample_size=3000,
                                                           random_state=random_state),
                     "inertia": km.inertia_})
    return pd.DataFrame(rows)


def fit_segments(df: pd.DataFrame, k: int, random_state: int = 42) -> pd.Series:
    Xs = StandardScaler().fit_transform(segment_matrix(df))
    labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(Xs)
    return pd.Series(labels, index=df.index, name="segment")


def describe_segments(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    # churn is only used here to describe segments, never to build them
    frame = segment_matrix(df).assign(segment=labels.values, Churn=df["Churn"].values)
    summary = frame.groupby("segment").agg(
        customers=("Churn", "size"),
        avg_tenure=("tenure", "mean"),
        avg_monthly_charges=("MonthlyCharges", "mean"),
        avg_addon_services=("num_addon_services", "mean"),
        churn_rate=("Churn", "mean"),
    )
    return summary.round(2).sort_values("churn_rate", ascending=False)
