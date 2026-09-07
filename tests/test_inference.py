from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xsap.inference import (
    benjamini_hochberg,
    bootstrap_statistic,
    factor_alpha,
    grs_test,
    mean_tstat,
    multiple_testing_summary,
    newey_west_variance,
    ols_newey_west,
)


def test_newey_west_reduces_to_the_sample_variance_without_lags():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    assert newey_west_variance(x, 0) == pytest.approx(x.var(ddof=0))


def test_newey_west_inflates_the_variance_of_a_persistent_series():
    """Positive autocorrelation must widen the standard error, not narrow it."""
    rng = np.random.default_rng(2)
    innovations = rng.normal(size=4000)
    persistent = np.zeros(4000)
    for t in range(1, 4000):
        persistent[t] = 0.7 * persistent[t - 1] + innovations[t]
    assert newey_west_variance(persistent, 12) > 2.0 * persistent.var(ddof=0)


def test_mean_tstat_has_approximately_correct_size_under_the_null():
    """Reject at 5% about 5% of the time when the mean really is zero."""
    rng = np.random.default_rng(3)
    rejections = [
        abs(mean_tstat(rng.normal(size=400), lags=6).tstat) > 1.96 for _ in range(400)
    ]
    assert 0.02 < np.mean(rejections) < 0.11


def test_mean_tstat_detects_a_real_mean():
    rng = np.random.default_rng(4)
    x = rng.normal(loc=0.5, scale=1.0, size=400)
    result = mean_tstat(x, lags=6)
    assert result.tstat > 5
    assert result.pvalue < 1e-6


def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(2000, 2))
    y = 0.3 + 1.5 * x[:, 0] - 0.8 * x[:, 1] + rng.normal(scale=0.1, size=2000)
    fit = ols_newey_west(y, x, lags=4)
    assert fit["params"] == pytest.approx([0.3, 1.5, -0.8], abs=0.02)
    assert fit["r2"] > 0.99


def test_factor_alpha_is_zero_for_a_pure_factor_combination():
    """A strategy that is only repackaged factors has no alpha, by construction."""
    rng = np.random.default_rng(6)
    months = pd.period_range("1990-01", periods=400, freq="M")
    factors = pd.DataFrame(
        {name: rng.normal(0.004, 0.03, len(months)) for name in
         ("mktrf", "smb", "hml", "rmw", "cma", "mom")},
        index=months,
    )
    strategy = 0.7 * factors["mktrf"] + 0.4 * factors["hml"] - 0.2 * factors["mom"]
    result = factor_alpha(strategy, factors)
    assert result["alpha"] == pytest.approx(0.0, abs=1e-9)
    assert result["betas"]["hml"] == pytest.approx(0.4, abs=1e-6)


def test_factor_alpha_survives_when_returns_are_orthogonal_to_the_factors():
    rng = np.random.default_rng(7)
    months = pd.period_range("1990-01", periods=500, freq="M")
    factors = pd.DataFrame(
        {name: rng.normal(0.004, 0.03, len(months)) for name in
         ("mktrf", "smb", "hml", "rmw", "cma", "mom")},
        index=months,
    )
    strategy = pd.Series(rng.normal(0.008, 0.02, len(months)), index=months)
    result = factor_alpha(strategy, factors)
    assert result["alpha"] == pytest.approx(0.008, abs=0.003)
    assert result["alpha_tstat"] > 4


def test_grs_has_approximately_correct_size_when_n_is_small_relative_to_t():
    """The reference case for Experiment 4: with N/T small, GRS behaves."""
    rng = np.random.default_rng(8)
    rejections = []
    for _ in range(200):
        n_obs, n_assets = 480, 10
        factor = rng.normal(0.005, 0.045, n_obs)
        beta = rng.uniform(0.7, 1.3, n_assets)
        resid = rng.normal(0.0, 0.05, (n_obs, n_assets))
        returns = pd.DataFrame(factor[:, None] * beta[None, :] + resid)
        factors = pd.DataFrame({"mktrf": factor})
        rejections.append(grs_test(returns, factors)["pvalue"] < 0.05)
    assert 0.02 < np.mean(rejections) < 0.10


def test_grs_is_undefined_when_assets_outnumber_observations():
    rng = np.random.default_rng(9)
    returns = pd.DataFrame(rng.normal(size=(50, 60)))
    factors = pd.DataFrame({"mktrf": rng.normal(size=50)})
    assert not np.isfinite(grs_test(returns, factors)["statistic"])


def test_bootstrap_interval_has_approximately_nominal_coverage():
    """The real property to check: a 95% interval covers the truth ~95% of the time.

    Testing that one interval happens to bracket the truth would fail 5% of the
    time by construction, which is a test of the seed rather than of the method.
    """
    truth = 0.01
    covered = []
    for trial in range(120):
        x = np.random.default_rng(1000 + trial).normal(truth, 0.05, 400)
        result = bootstrap_statistic(x, np.mean, n_draws=400, mean_block=6, seed=trial)
        covered.append(result["ci_lower"] < truth < result["ci_upper"])
    assert 0.88 < np.mean(covered) < 1.0


def test_bootstrap_point_estimate_is_the_sample_statistic():
    rng = np.random.default_rng(10)
    x = rng.normal(0.01, 0.05, 600)
    result = bootstrap_statistic(x, np.mean, n_draws=400, mean_block=6, seed=1)
    assert result["point"] == pytest.approx(x.mean())
    assert result["mean"] == pytest.approx(x.mean(), abs=4.0 * result["std"])


def test_block_bootstrap_gives_wider_intervals_for_persistent_data():
    """Serial dependence means less information per observation; the interval must widen."""
    rng = np.random.default_rng(21)
    innovations = rng.normal(0.0, 0.03, 600)
    persistent = np.zeros(600)
    for t in range(1, 600):
        persistent[t] = 0.6 * persistent[t - 1] + innovations[t]

    blocked = bootstrap_statistic(persistent, np.mean, n_draws=600, mean_block=24, seed=5)
    iid = bootstrap_statistic(persistent, np.mean, n_draws=600, mean_block=1, seed=5)
    assert blocked["std"] > 1.5 * iid["std"]


def test_bootstrap_flags_a_fragile_result():
    """A statistic indistinguishable from zero should often flip sign."""
    rng = np.random.default_rng(11)
    x = rng.normal(0.001, 0.06, 120)
    result = bootstrap_statistic(x, np.mean, n_draws=800, mean_block=6, seed=2)
    assert result["p_wrong_sign"] > 0.2


def test_benjamini_hochberg_is_less_conservative_than_bonferroni():
    """Both control an error rate; BH controls a weaker one, so finds more."""
    rng = np.random.default_rng(12)
    null_p = rng.uniform(size=180)
    signal_p = rng.uniform(0, 0.002, size=20)
    p = np.concatenate([null_p, signal_p])

    summary = multiple_testing_summary(p, [f"t{i}" for i in range(len(p))])
    n_uncorrected = int(summary["uncorrected"].sum())
    n_bh = int(summary["benjamini_hochberg"].sum())
    n_bonf = int(summary["bonferroni"].sum())
    assert n_bonf <= n_bh <= n_uncorrected
    assert n_bh >= 15


def test_benjamini_hochberg_rejects_nothing_when_all_nulls_are_true():
    rng = np.random.default_rng(13)
    p = rng.uniform(size=500)
    assert benjamini_hochberg(p, 0.05)["n_rejected"] == 0
