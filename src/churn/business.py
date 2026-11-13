from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CampaignAssumptions:
    # illustrative assumptions, not real company numbers
    offer_cost: float = 50.0
    success_rate: float = 0.30
    months_retained: int = 12


def expected_profit(y_true, proba, monthly_charges, threshold: float,
                    a: CampaignAssumptions = CampaignAssumptions()) -> float:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    monthly = np.asarray(monthly_charges, dtype=float)
    contacted = proba >= threshold
    saved_value = a.success_rate * a.months_retained * monthly[contacted & (y_true == 1)].sum()
    cost = a.offer_cost * contacted.sum()
    return float(saved_value - cost)


def profit_curve(y_true, proba, monthly_charges, a: CampaignAssumptions = CampaignAssumptions(),
                 thresholds=None) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2) if thresholds is None else thresholds
    proba = np.asarray(proba)
    rows = [
        {
            "threshold": t,
            "profit": expected_profit(y_true, proba, monthly_charges, t, a),
            "customers_contacted": int((proba >= t).sum()),
        }
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def best_threshold(curve: pd.DataFrame) -> pd.Series:
    return curve.loc[curve["profit"].idxmax()]
