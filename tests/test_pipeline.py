import numpy as np
import pandas as pd
import pytest

from churn.business import CampaignAssumptions, best_threshold, expected_profit, profit_curve
from churn.data import clean, load_clean, load_raw, split_xy
from churn.db import build_database, load_named_queries, run_query
from churn.features import add_engineered_features, build_preprocessor


@pytest.fixture(scope="module")
def df():
    return load_clean()


def test_clean_types_and_no_missing(df):
    assert df["TotalCharges"].dtype == float
    assert df["Churn"].isin([0, 1]).all()
    assert not df.isna().any().any()
    assert len(df) == 7043


def test_blank_total_charges_only_for_new_customers():
    raw = load_raw()
    blank = raw["TotalCharges"].str.strip() == ""
    assert (raw.loc[blank, "tenure"] == 0).all()


def test_clean_rejects_blank_charges_for_existing_customer():
    raw = load_raw().head(5).copy()
    raw.loc[0, "TotalCharges"] = " "
    raw.loc[0, "tenure"] = 10
    with pytest.raises(ValueError):
        clean(raw)


def test_engineered_features(df):
    X, _ = split_xy(df)
    out = add_engineered_features(X)
    assert out["num_addon_services"].between(0, 6).all()
    new = out["tenure"] == 0
    assert np.allclose(out.loc[new, "avg_monthly_spend"], out.loc[new, "MonthlyCharges"])


def test_preprocessor_output_shape(df):
    X, _ = split_xy(df)
    Xt = build_preprocessor().fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert not np.isnan(np.asarray(Xt, dtype=float)).any()


def test_sql_matches_pandas(df):
    conn = build_database(df)
    queries = load_named_queries()
    assert "churn_by_contract" in queries
    by_contract = run_query(conn, queries["churn_by_contract"]).set_index("Contract")
    expected = (df.groupby("Contract")["Churn"].mean() * 100).round(1)
    assert by_contract["churn_rate_pct"].sort_index().tolist() == expected.sort_index().tolist()


def test_expected_profit_hand_computed():
    y = np.array([1, 0, 1, 0])
    proba = np.array([0.9, 0.8, 0.2, 0.1])
    monthly = np.array([100.0, 50.0, 80.0, 30.0])
    a = CampaignAssumptions(offer_cost=10, success_rate=0.5, months_retained=12)
    assert expected_profit(y, proba, monthly, 0.5, a) == pytest.approx(0.5 * 12 * 100 - 2 * 10)


def test_best_threshold_picks_max():
    curve = pd.DataFrame({"threshold": [0.1, 0.2, 0.3], "profit": [5, 9, 1], "customers_contacted": [3, 2, 1]})
    assert best_threshold(curve)["threshold"] == 0.2
    y = np.array([1, 0]); p = np.array([0.6, 0.4]); m = np.array([10.0, 10.0])
    assert len(profit_curve(y, p, m)) > 0
