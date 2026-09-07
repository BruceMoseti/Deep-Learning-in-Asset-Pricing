"""Check predictor formulas against direct computation.

The point-in-time tests in ``test_no_lookahead.py`` prove that no predictor sees
the future.  They would still pass if a predictor computed the wrong thing from
the right data, so the formulas are checked here separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from xsap.config import Config
from xsap.features import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    build_features,
    make_target,
    rank_normalise,
)

CFG = Config(min_assets_per_month=5)


@pytest.fixture
def built(toy_panel, toy_factors):
    data = build_features(toy_panel, toy_factors, CFG)
    wide = toy_panel.pivot(index="month", columns="asset", values="ret")
    excess = wide.sub(toy_factors["rf"].reindex(wide.index), axis=0)
    return data, excess


def test_every_declared_predictor_is_produced(built):
    data, _ = built
    assert set(FEATURE_NAMES) <= set(data.columns)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    # Groups partition the predictor list, which the ablation relies on.
    grouped = [f for names in FEATURE_GROUPS.values() for f in names]
    assert sorted(grouped) == sorted(FEATURE_NAMES)


def test_one_month_return_is_this_months_excess_return(built):
    data, excess = built
    month, asset = data.index[500]
    assert data.loc[(month, asset), "ret_1m"] == pytest.approx(
        excess.loc[month, asset], abs=1e-12
    )


@pytest.mark.parametrize(
    "name,start_lag,end_lag",
    [("mom_2_6", 5, 1), ("mom_2_12", 11, 1), ("mom_7_12", 11, 6), ("mom_13_36", 35, 12)],
)
def test_momentum_compounds_the_right_window(built, name, start_lag, end_lag):
    """Each momentum predictor must skip the months its definition skips."""
    data, excess = built
    month, asset = data.index[3000]
    window = excess.loc[month - start_lag : month - end_lag, asset]
    assert len(window) == start_lag - end_lag + 1
    assert data.loc[(month, asset), name] == pytest.approx(
        float(np.prod(1.0 + window) - 1.0), rel=1e-9
    )


@pytest.mark.parametrize("window", [6, 12, 36])
def test_volatility_is_the_trailing_standard_deviation(built, window):
    data, excess = built
    month, asset = data.index[3000]
    trailing = excess.loc[month - window + 1 : month, asset]
    assert len(trailing) == window
    assert data.loc[(month, asset), f"vol_{window}"] == pytest.approx(
        float(trailing.std(ddof=1)), rel=1e-9
    )


def test_downside_volatility_only_counts_losses(built):
    data, excess = built
    month, asset = data.index[3000]
    trailing = excess.loc[month - 35 : month, asset].clip(upper=0.0)
    assert data.loc[(month, asset), "dnvol_36"] == pytest.approx(
        float(np.sqrt((trailing**2).mean())), rel=1e-9
    )


def test_beta_matches_an_explicit_regression(built, toy_factors):
    data, excess = built
    month, asset = data.index[4000]
    window = slice(month - 35, month)
    y = excess.loc[window, asset].to_numpy()
    x = toy_factors.loc[window, "mktrf"].to_numpy()
    slope = stats.linregress(x, y).slope
    assert data.loc[(month, asset), "beta_36"] == pytest.approx(slope, rel=1e-8)


def test_idiosyncratic_volatility_matches_the_regression_residual(built, toy_factors):
    data, excess = built
    month, asset = data.index[4000]
    window = slice(month - 35, month)
    y = excess.loc[window, asset].to_numpy()
    x = toy_factors.loc[window, "mktrf"].to_numpy()
    fit = stats.linregress(x, y)
    resid = y - (fit.intercept + fit.slope * x)
    assert data.loc[(month, asset), "idiovol_36"] == pytest.approx(
        float(resid.std(ddof=0)), rel=1e-6
    )


def test_market_correlation_matches_numpy(built, toy_factors):
    data, excess = built
    month, asset = data.index[4000]
    window = slice(month - 35, month)
    expected = np.corrcoef(
        excess.loc[window, asset].to_numpy(), toy_factors.loc[window, "mktrf"].to_numpy()
    )[0, 1]
    assert data.loc[(month, asset), "corr_36"] == pytest.approx(expected, rel=1e-8)


def test_drawdown_features_match_a_direct_path_computation(built):
    data, excess = built
    month, asset = data.index[4000]
    path = excess.loc[month - 35 : month, asset].to_numpy()
    level = np.cumprod(1.0 + path)
    drawdown = level / np.maximum.accumulate(level) - 1.0
    assert data.loc[(month, asset), "maxdd_36"] == pytest.approx(
        float(drawdown.min()), rel=1e-9
    )
    assert data.loc[(month, asset), "dist_high_36"] == pytest.approx(
        float(drawdown[-1]), rel=1e-9
    )


def test_distance_from_high_is_never_positive(built):
    data, _ = built
    assert data["dist_high_36"].max() <= 1e-12
    assert data["maxdd_36"].max() <= 1e-12
    assert (data["maxdd_36"] <= data["dist_high_36"] + 1e-12).all()


def test_seasonality_averages_the_same_calendar_month(built):
    """Lags must be multiples of twelve, and must exclude the current month."""
    data, excess = built
    month, asset = data.index[4000]
    lags = [month - 12 * k for k in range(1, 21)]
    values = [
        excess.loc[lag, asset]
        for lag in lags
        if lag in excess.index and not np.isnan(excess.loc[lag, asset])
    ]
    assert data.loc[(month, asset), "seas_20y"] == pytest.approx(
        float(np.mean(values)), rel=1e-9
    )
    assert all(lag.month == month.month for lag in lags)


def test_rank_normalisation_spans_the_unit_interval(built):
    data, _ = built
    ranked = rank_normalise(data)
    assert ranked[list(FEATURE_NAMES)].abs().max().max() <= 1.0

    # With no ties the extremes land exactly on the endpoints and the
    # cross-section is centred.  ``rank(pct=True)`` would put the mean at +1/n.
    # ``dist_high_36`` is excluded because it is capped at zero, so assets
    # sitting at a trailing high tie and averaged ranks cannot reach the
    # endpoint.
    month = data.index.get_level_values("month").unique()[30]
    for name in ("vol_12", "mom_2_12", "beta_36"):
        block = ranked.loc[month, name]
        assert data.loc[month, name].nunique() == len(block)
        assert block.min() == pytest.approx(-1.0)
        assert block.max() == pytest.approx(1.0)
        assert block.mean() == pytest.approx(0.0, abs=1e-12)


def test_ranks_are_centred_even_with_a_mass_point(built):
    """A capped predictor still has to come out inside the interval."""
    data, _ = built
    ranked = rank_normalise(data)
    assert ranked["dist_high_36"].between(-1.0, 1.0).all()
    assert (data["dist_high_36"] == 0.0).any(), "expected ties at the trailing high"


def test_rank_normalisation_is_monotone_within_a_month(built):
    data, _ = built
    ranked = rank_normalise(data)
    month = data.index.get_level_values("month").unique()[20]
    before = data.loc[month, "vol_12"].rank()
    after = ranked.loc[month, "vol_12"].rank()
    pd.testing.assert_series_equal(before, after, check_names=False)


def test_target_is_demeaned_within_each_month(built):
    data, _ = built
    y = make_target(data, "zscore")
    by_month = y.groupby("month")
    assert by_month.mean().abs().max() < 1e-10
    assert by_month.std().sub(1.0).abs().max() < 1e-10


def test_rank_target_is_uniform_within_each_month(built):
    data, _ = built
    y = make_target(data, "rank")
    assert y.between(-1.0, 1.0).all()
    assert y.groupby("month").mean().abs().max() < 1e-10


def test_unknown_target_mode_is_rejected(built):
    data, _ = built
    with pytest.raises(ValueError, match="unknown target mode"):
        make_target(data, "something-else")


def test_assets_without_enough_history_are_excluded(built):
    """The longest lookback is five years, so nothing appears before then."""
    data, excess = built
    first_predictor_month = data.index.get_level_values("month").min()
    assert first_predictor_month >= excess.index.min() + 59
    assert data[list(FEATURE_NAMES)].drop(columns=list(FEATURE_GROUPS["static"])).notna().all().all()
