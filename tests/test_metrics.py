from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xsap.metrics import (
    diebold_mariano,
    ic_difference_test,
    monthly_ic,
    r2_oos,
    summarise_forecasts,
)


@pytest.fixture
def paired_series():
    rng = np.random.default_rng(5)
    months = pd.period_range("2000-01", periods=60, freq="M")
    index = pd.MultiIndex.from_product([months, [f"a{i}" for i in range(50)]], names=["month", "asset"])
    y = pd.Series(rng.normal(size=len(index)), index=index)
    return y


def test_perfect_forecast_scores_one(paired_series):
    y = paired_series
    assert monthly_ic(y, y).mean() == pytest.approx(1.0)
    assert r2_oos(y, y) == pytest.approx(1.0)


def test_zero_forecast_scores_zero(paired_series):
    y = paired_series
    zero = pd.Series(0.0, index=y.index)
    assert r2_oos(zero, y) == pytest.approx(0.0)


def test_r2_oos_is_negative_for_a_forecast_worse_than_no_view(paired_series):
    """A forecast that overshoots is worse than abstaining, and must be marked so."""
    y = paired_series
    assert r2_oos(3.0 * y, y) < 0.0


def test_ic_is_invariant_to_monotone_rescaling(paired_series):
    y = paired_series
    rng = np.random.default_rng(6)
    prediction = pd.Series(rng.normal(size=len(y)), index=y.index) + 0.3 * y
    base = monthly_ic(prediction, y)
    scaled = monthly_ic(prediction * 100.0 + 7.0, y)
    pd.testing.assert_series_equal(base, scaled)


def test_ic_sign_flips_with_the_forecast(paired_series):
    y = paired_series
    rng = np.random.default_rng(8)
    prediction = pd.Series(rng.normal(size=len(y)), index=y.index) + 0.3 * y
    assert monthly_ic(prediction, y).mean() == pytest.approx(
        -monthly_ic(-prediction, y).mean(), abs=1e-12
    )


def test_summary_reports_positive_skill_for_an_informative_forecast(paired_series):
    y = paired_series
    rng = np.random.default_rng(9)
    prediction = 0.4 * y + pd.Series(rng.normal(size=len(y)), index=y.index)
    stats = summarise_forecasts(prediction, y)
    assert stats["rank_ic"] > 0.2
    assert stats["rank_ic_tstat"] > 5
    assert 0 < stats["hit_rate"] <= 1
    assert stats["n_months"] == 60


def test_a_constant_forecast_scores_zero_rather_than_being_dropped(paired_series):
    """Declining to rank the cross-section is worth zero, not worth omitting.

    If these months were dropped, a model's average IC would be taken over only
    the months in which it chose to have a view -- selection on the model's own
    confidence.
    """
    y = paired_series
    flat = pd.Series(0.0, index=y.index)
    ic = monthly_ic(flat, y)
    assert ic.notna().all()
    assert ic.abs().max() == pytest.approx(0.0)

    rng = np.random.default_rng(20)
    informative = pd.Series(rng.normal(size=len(y)), index=y.index) + y
    months = y.index.get_level_values("month").unique()
    partial = informative.copy()
    partial.loc[months[:30]] = 0.0

    diluted = monthly_ic(partial, y)
    assert diluted.notna().all()
    assert diluted.loc[months[:30]].abs().max() == pytest.approx(0.0)
    # Half the months carry no view, so the average must be roughly halved
    # rather than unchanged.
    assert diluted.mean() < 0.6 * monthly_ic(informative, y).mean()


def test_diebold_mariano_prefers_the_better_forecast(paired_series):
    y = paired_series
    rng = np.random.default_rng(10)
    noise = pd.Series(rng.normal(size=len(y)), index=y.index)
    good, bad = 0.5 * y + 0.2 * noise, 0.5 * noise
    result = diebold_mariano(good, bad, y)
    assert result["favours"] == "a"
    assert result["tstat"] < -2

    flipped = diebold_mariano(bad, good, y)
    assert flipped["favours"] == "b"
    assert flipped["loss_diff"] == pytest.approx(-result["loss_diff"])


def test_ic_difference_test_detects_a_real_gap(paired_series):
    y = paired_series
    rng = np.random.default_rng(30)
    noise = pd.Series(rng.normal(size=len(y)), index=y.index)
    better = 0.6 * y + 0.4 * noise
    worse = 0.1 * y + 0.9 * noise

    result = ic_difference_test(better, worse, y)
    assert result["ic_a"] > result["ic_b"]
    assert result["ic_difference"] > 0
    assert result["tstat"] > 3
    assert result["pvalue"] < 0.01


def test_ic_difference_test_finds_nothing_between_identical_forecasts(paired_series):
    """The property that makes a null result trustworthy."""
    y = paired_series
    rng = np.random.default_rng(31)
    prediction = pd.Series(rng.normal(size=len(y)), index=y.index) + 0.2 * y
    result = ic_difference_test(prediction, prediction, y)
    assert result["ic_difference"] == pytest.approx(0.0)
    assert not np.isfinite(result["tstat"]) or abs(result["tstat"]) < 1e-6


def test_ic_difference_test_is_antisymmetric(paired_series):
    y = paired_series
    rng = np.random.default_rng(32)
    noise = pd.Series(rng.normal(size=len(y)), index=y.index)
    a, b = 0.5 * y + noise, 0.2 * y + noise
    forward = ic_difference_test(a, b, y)
    backward = ic_difference_test(b, a, y)
    assert forward["ic_difference"] == pytest.approx(-backward["ic_difference"])
    assert forward["tstat"] == pytest.approx(-backward["tstat"])


def test_pairing_is_tighter_than_comparing_two_standard_errors(paired_series):
    """Why the test is paired: both models face the same cross-section monthly.

    The common component cancels in the difference, so the paired standard
    error is much smaller than either series' own.
    """
    y = paired_series
    rng = np.random.default_rng(33)
    common = pd.Series(rng.normal(size=len(y)), index=y.index)
    a = 0.5 * y + common
    b = 0.45 * y + common

    paired_se = abs(ic_difference_test(a, b, y)["ic_difference"]) / abs(
        ic_difference_test(a, b, y)["tstat"]
    )
    unpaired_se = monthly_ic(a, y).std() / np.sqrt(len(monthly_ic(a, y)))
    assert paired_se < 0.5 * unpaired_se


def test_diebold_mariano_finds_no_difference_between_identical_forecasts(paired_series):
    y = paired_series
    rng = np.random.default_rng(11)
    prediction = pd.Series(rng.normal(size=len(y)), index=y.index)
    result = diebold_mariano(prediction, prediction, y)
    assert result["loss_diff"] == pytest.approx(0.0)
