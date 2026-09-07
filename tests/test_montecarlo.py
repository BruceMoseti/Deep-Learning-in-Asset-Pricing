from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from xsap.montecarlo import (
    ERROR_MODELS,
    Calibration,
    alpha_tests,
    calibrate,
    cell_seed,
    ledoit_wolf_shrinkage,
    pesaran_yamagata_pvalue,
    simulate_panel,
)


def _calibration(n_assets=150, correlation=0.0, n_pool=600, seed=0):
    rng = np.random.default_rng(seed)
    corr = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    pool = rng.standard_normal((n_pool, n_assets)) @ chol.T
    return Calibration(
        factor_mean=np.array([0.005]),
        factor_cov=np.array([[0.045**2]]),
        betas=rng.uniform(0.7, 1.3, (n_assets, 1)),
        resid_vol=rng.uniform(0.03, 0.07, n_assets),
        resid_corr=corr,
        residuals=pool - pool.mean(axis=0),
        student_df=6.0,
        vol_persistence=0.66,
        vol_innovation_sd=0.22,
        factor_names=("mktrf",),
    )


def _size(calibration, n_assets, n_obs, test, reps=250, seed=1, **kwargs):
    rng = np.random.default_rng(seed)
    pvalues = [
        alpha_tests(
            *simulate_panel(calibration, n_assets, n_obs, rng, **kwargs)
        )[test]
        for _ in range(reps)
    ]
    pvalues = np.asarray(pvalues, dtype=float)
    return float(np.mean(pvalues[np.isfinite(pvalues)] < 0.05))


def test_calibrate_recovers_planted_betas_and_tail_thickness():
    rng = np.random.default_rng(1)
    months = pd.period_range("1960-01", periods=1200, freq="M")
    factor = pd.DataFrame({"mktrf": rng.normal(0.005, 0.045, len(months))}, index=months)
    betas = np.array([0.5, 1.0, 1.5])
    df = 8.0
    shocks = stats.t.rvs(df, size=(len(months), 3), random_state=2) / np.sqrt(
        df / (df - 2)
    )
    returns = pd.DataFrame(
        factor["mktrf"].to_numpy()[:, None] * betas[None, :] + 0.02 * shocks,
        index=months,
        columns=["a", "b", "c"],
    )
    fitted = calibrate(returns, factor)
    assert fitted.betas.ravel() == pytest.approx(betas, abs=0.05)
    assert fitted.factor_mean[0] == pytest.approx(factor["mktrf"].mean())
    assert 5.0 < fitted.student_df < 15.0
    assert fitted.residuals.std(axis=0) == pytest.approx(np.ones(3), abs=0.05)
    # OLS residuals are mean zero by construction, and the resampler relies on
    # the pool inheriting that exactly.
    assert fitted.residuals.mean(axis=0) == pytest.approx(np.zeros(3), abs=1e-12)


@pytest.mark.parametrize("error_model", ERROR_MODELS)
def test_every_error_model_produces_a_clean_null_panel(error_model):
    """No error model may accidentally inject an intercept.

    An earlier version of the empirical resampler did exactly that: the pool of
    residual vectors was not demeaned, so resampling planted an alpha of about
    two percent a year and every test dutifully rejected a null that was in
    fact false.  This is the guard against that class of mistake.
    """
    calibration = _calibration(n_assets=60, correlation=0.3)
    rng = np.random.default_rng(3)
    returns, factors = simulate_panel(
        calibration, 30, 400, rng, error_model=error_model
    )
    assert returns.shape == (400, 30)
    design = np.column_stack([np.ones(400), factors])
    intercepts = np.linalg.lstsq(design, returns, rcond=None)[0][0]
    assert abs(intercepts.mean()) < 0.005


@pytest.mark.parametrize("error_model", ERROR_MODELS)
def test_no_error_model_biases_the_size_of_the_exact_test(error_model):
    """Every error model must leave GRS at its nominal level at low N/T.

    Both earlier resampling schemes failed this: sampling months with
    replacement pushed size to 14%, and sampling without replacement pushed it
    to 0%.  Either would have been reported as a finding about the data.
    """
    calibration = _calibration(n_assets=120, correlation=0.3, n_pool=650, seed=17)
    size = _size(calibration, 10, 240, "grs", error_model=error_model, reps=400, seed=2)
    assert 0.01 < size < 0.11, f"{error_model} gave size {size:.3f}"


def test_error_models_have_the_intended_shape():
    """Heavy tails must be heavy, clustering must cluster, independence independent."""
    calibration = _calibration(n_assets=60, correlation=0.4, n_pool=3000, seed=4)
    rng = np.random.default_rng(5)

    def resid(error_model, n_obs=6000):
        returns, factors = simulate_panel(
            calibration, 25, n_obs, rng, error_model=error_model
        )
        design = np.column_stack([np.ones(n_obs), factors])
        coef = np.linalg.lstsq(design, returns, rcond=None)[0]
        return returns - design @ coef

    gaussian = resid("gaussian")
    heavy = resid("student_t")
    clustered = resid("common_vol")
    independent = resid("gaussian_independent")

    def kurtosis(x):
        return float(np.median(stats.kurtosis(x, axis=0)))

    assert kurtosis(heavy) > kurtosis(gaussian) + 1.0
    assert abs(kurtosis(gaussian)) < 0.6

    def off_diagonal_mean(x):
        corr = np.corrcoef(x, rowvar=False)
        return float(corr[~np.eye(len(corr), dtype=bool)].mean())

    assert off_diagonal_mean(gaussian) == pytest.approx(0.4, abs=0.06)
    assert abs(off_diagonal_mean(independent)) < 0.05

    # Volatility clustering shows up as autocorrelation in squared residuals.
    def squared_autocorrelation(x):
        squared = (x**2).mean(axis=1)
        return float(np.corrcoef(squared[1:], squared[:-1])[0, 1])

    # 0.66 log-volatility persistence, the value measured in the panel, implies
    # a squared-residual autocorrelation of roughly a quarter.
    assert squared_autocorrelation(clustered) > 0.15
    assert abs(squared_autocorrelation(gaussian)) < 0.1


@pytest.mark.parametrize(
    "error_model", ["gaussian", "student_t", "common_vol", "empirical_wild"]
)
def test_grs_holds_its_size_at_high_n_over_t(error_model):
    """The finding that reframes Experiment 4.

    GRS is exact in finite samples under normal, homoskedastic, serially
    independent errors with *any* cross-sectional covariance, so N/T alone does
    not distort it.  What is more surprising, and what this checks, is that its
    size also survives heavy tails, a persistent common volatility factor, and
    residuals taken straight from the data -- at N/T = 0.77.  Whatever goes
    wrong with inference in high dimensions, it is not this.
    """
    calibration = _calibration(n_assets=150, correlation=0.3, n_pool=650, seed=6)
    size = _size(calibration, 100, 130, "grs", error_model=error_model, reps=400, seed=3)
    assert 0.01 < size < 0.12, f"{error_model} gave size {size:.3f}"


def test_wald_chi2_over_rejects_as_n_approaches_t():
    """What actually breaks: using the asymptotic reference instead of the exact one.

    Same quadratic form as GRS, same data, only the reference distribution
    differs.  This is the headline size distortion of Experiment 4.
    """
    calibration = _calibration(n_assets=150, seed=7)
    small = _size(calibration, 10, 200, "wald_chi2", error_model="gaussian")
    large = _size(calibration, 100, 130, "wald_chi2", error_model="gaussian")
    assert small < 0.20
    assert large > 0.60
    assert large > 3.0 * small


def test_shrinkage_without_recalibration_destroys_the_test():
    """Shrinkage fixes the conditioning and breaks the reference distribution.

    A better-conditioned covariance shrinks the quadratic form, but the F
    critical value is unchanged, so the test stops rejecting anything.  Better
    estimation does not by itself give better inference.
    """
    calibration = _calibration(n_assets=150, seed=8)
    size = _size(calibration, 100, 130, "grs_shrunk", error_model="gaussian", reps=300)
    assert size < 0.01


def test_pesaran_yamagata_is_correctly_sized_under_its_own_assumptions():
    calibration = _calibration(n_assets=150, correlation=0.0, seed=9)
    size = _size(
        calibration, 80, 200, "pesaran_yamagata", error_model="gaussian_independent"
    )
    assert 0.01 < size < 0.12


def test_pesaran_yamagata_breaks_under_cross_sectional_dependence():
    """Its advantage is dimension, not robustness; the report has to say so.

    And unlike a small-sample problem, this does not go away with more data:
    the assumption being violated is about the cross-section, not the length of
    the sample.
    """
    calibration = _calibration(n_assets=150, correlation=0.4, seed=10)
    short = _size(calibration, 80, 200, "pesaran_yamagata", error_model="gaussian")
    long = _size(calibration, 80, 600, "pesaran_yamagata", error_model="gaussian")
    assert short > 0.20
    assert long > 0.20


def test_pesaran_yamagata_stays_defined_when_grs_does_not():
    calibration = _calibration(n_assets=150, seed=11)
    rng = np.random.default_rng(12)
    returns, factors = simulate_panel(calibration, 110, 100, rng)
    results = alpha_tests(returns, factors)
    assert not np.isfinite(results["grs"])
    assert np.isfinite(results["pesaran_yamagata"])


def test_shrinkage_improves_conditioning():
    rng = np.random.default_rng(13)
    residuals = rng.normal(size=(120, 100))
    sample = residuals.T @ residuals / 120
    shrunk = ledoit_wolf_shrinkage(residuals)
    assert np.linalg.cond(shrunk) < np.linalg.cond(sample)
    assert np.trace(shrunk) == pytest.approx(np.trace(sample), rel=0.05)


def test_tests_have_power_against_real_mispricing():
    calibration = _calibration(n_assets=150, seed=14)
    rng = np.random.default_rng(15)
    collected = {"grs": [], "pesaran_yamagata": []}
    for _ in range(200):
        returns, factors = simulate_panel(
            calibration,
            20,
            480,
            rng,
            alpha=rng.normal(0.0, 0.004, 20),
            error_model="gaussian_independent",
        )
        results = alpha_tests(returns, factors)
        for name in collected:
            collected[name].append(results[name])
    for name, pvalues in collected.items():
        power = float(np.mean(np.asarray(pvalues) < 0.05))
        assert power > 0.5, f"{name} power was {power:.3f}"


def test_the_study_is_reproducible_across_interpreter_processes():
    """Same seed must give the same numbers in a fresh process.

    An earlier version derived each cell's seed from ``hash(error_model)``.
    String hashing is salted per process, so every number in Experiment 4
    changed between runs while the rest of the pipeline reproduced exactly.
    Running twice under different ``PYTHONHASHSEED`` values is the only way to
    catch that, since within a single process ``hash`` is stable.
    """
    program = textwrap.dedent(
        """
        from xsap.montecarlo import ERROR_MODELS, cell_seed
        print([
            cell_seed(20240601, t, n, m)
            for t in (120, 360)
            for n in (10, 300)
            for m in ERROR_MODELS
        ])
        """
    )
    outputs = []
    for hash_seed in ("0", "1", "999983"):
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            check=True,
        )
        outputs.append(result.stdout.strip())
    assert len(set(outputs)) == 1, "cell seeds depend on the interpreter hash seed"


def test_every_grid_cell_gets_a_distinct_seed():
    """Reusing a seed across cells would correlate their sampling error."""
    seeds = [
        cell_seed(20240601, n_obs, n_assets, model)
        for n_obs in (120, 240, 360)
        for n_assets in (10, 25, 50, 100, 200, 300)
        for model in ERROR_MODELS
    ]
    assert len(set(seeds)) == len(seeds)


def test_pesaran_yamagata_matches_a_direct_computation():
    """Guard the closed-form moments against a transcription error."""
    rng = np.random.default_rng(16)
    n_obs, n_assets = 300, 40
    factors = rng.normal(0.005, 0.04, (n_obs, 1))
    returns = factors @ np.ones((1, n_assets)) + rng.normal(0, 0.05, (n_obs, n_assets))

    design = np.column_stack([np.ones(n_obs), factors])
    t_squared = []
    for i in range(n_assets):
        coef, *_ = np.linalg.lstsq(design, returns[:, i], rcond=None)
        resid = returns[:, i] - design @ coef
        dof = n_obs - 2
        s2 = (resid**2).sum() / dof
        se = np.sqrt(s2 * np.linalg.pinv(design.T @ design)[0, 0])
        t_squared.append((coef[0] / se) ** 2)

    dof = n_obs - 2
    expected = dof / (dof - 2)
    variance = 2 * dof**2 * (dof - 1) / ((dof - 2) ** 2 * (dof - 4))
    statistic = np.sqrt(n_assets) * (np.mean(t_squared) - expected) / np.sqrt(variance)
    assert pesaran_yamagata_pvalue(returns, factors) == pytest.approx(
        float(stats.norm.sf(statistic)), rel=1e-9
    )
