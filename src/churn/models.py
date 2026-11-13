from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, validation_curve
from sklearn.pipeline import Pipeline

from .features import build_preprocessor

RANDOM_STATE = 42
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"}


def make_cv(n_splits: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def candidate_models() -> dict[str, Pipeline]:
    return {
        "Logistic regression": Pipeline(
            [
                ("prep", build_preprocessor(scale_numeric=True)),
                ("model", LogisticRegression(C=1.0, max_iter=2000)),
            ]
        ),
        "Random forest": Pipeline(
            [
                ("prep", build_preprocessor(scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "Gradient boosting": Pipeline(
            [
                ("prep", build_preprocessor(scale_numeric=False)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05, max_iter=300, max_depth=3, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
    }


def compare_models(X: pd.DataFrame, y: pd.Series, models: dict[str, Pipeline] | None = None) -> pd.DataFrame:
    models = models or candidate_models()
    rows = []
    for name, pipe in models.items():
        res = cross_validate(clone(pipe), X, y, cv=make_cv(), scoring=SCORING, return_train_score=True)
        row = {"model": name}
        for metric in SCORING:
            row[f"val_{metric}"] = res[f"test_{metric}"].mean()
            row[f"val_{metric}_std"] = res[f"test_{metric}"].std()
            row[f"train_{metric}"] = res[f"train_{metric}"].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").sort_values("val_roc_auc", ascending=False)


def depth_curve(X: pd.DataFrame, y: pd.Series, depths=(2, 4, 6, 8, 12, 16, 24, None)) -> pd.DataFrame:
    pipe = Pipeline(
        [
            ("prep", build_preprocessor(scale_numeric=False)),
            ("model", RandomForestClassifier(n_estimators=150, min_samples_leaf=1, n_jobs=-1,
                                             random_state=RANDOM_STATE)),
        ]
    )
    train, val = validation_curve(
        pipe, X, y, param_name="model__max_depth", param_range=list(depths),
        cv=make_cv(), scoring="roc_auc", n_jobs=1,
    )
    labels = ["none" if d is None else str(d) for d in depths]
    return pd.DataFrame(
        {"max_depth": labels, "train_roc_auc": train.mean(axis=1), "val_roc_auc": val.mean(axis=1),
         "val_std": val.std(axis=1)}
    )


def logistic_coefficients(fitted_pipe: Pipeline) -> pd.Series:
    names = fitted_pipe.named_steps["prep"].get_feature_names_out()
    coefs = fitted_pipe.named_steps["model"].coef_.ravel()
    return pd.Series(coefs, index=names).sort_values()


def top_decile_lift(y_true, proba) -> float:
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(proba))
    top = order[: max(1, len(order) // 10)]
    return y_true[top].mean() / y_true.mean()
