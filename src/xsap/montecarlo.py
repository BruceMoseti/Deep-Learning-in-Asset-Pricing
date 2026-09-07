"""Do standard asset-pricing tests still work when there are many assets?

The result that shapes this experiment
--------------------------------------
It is tempting to assume the Gibbons-Ross-Shanken test degrades as ``N/T``
rises simply because it inverts an ``N x N`` covariance matrix estimated from
``T`` observations.  It does not.  Under its own assumptions -- residuals that
are jointly normal, homoskedastic and independent over time, with *any*
cross-sectional covariance -- GRS is exact in finite samples for every
``N <= T - K - 1``.  Simulating that case and finding a rejection rate of 5%
even at ``N/T = 0.77`` is a check on this code, and it is also the answer to
the question as usually posed.

The real question is therefore not about dimension alone.  It is: real returns
are not normal, not homoskedastic and not serially independent, so how quickly
does the reliability of these tests deteriorate as ``N/T`` grows once the
errors are allowed to look like the errors in the data?  Dimension is the
amplifier, not the cause.

Error structures, in increasing order of realism
------------------------------------------------
``gaussian``
    Normal, homoskedastic, with the cross-sectional correlation measured in the
    real panel.  GRS's assumptions hold exactly.
``gaussian_independent``
    As above but with residuals cross-sectionally independent, which is what
    Pesaran-Yamagata assumes.
``student_t``
    Normal copula with Student-t margins, degrees of freedom calibrated to the
    kurtosis of real residuals.
``common_vol``
    Gaussian shocks scaled by a persistent volatility factor shared across
    assets, which is how volatility clustering appears in real cross-sections.
``empirical_wild``
    Real residual *vectors* with randomly flipped signs.  Makes no
    distributional assumption at all: each month keeps its own magnitude and
    its own cross-sectional pattern, so real non-normality, real
    heteroskedasticity across months and real cross-sectional dependence all
    survive, while the null is imposed exactly.
``empirical_wild_block``
    The same, with one sign per year rather than per month, so serial
    dependence within a year survives too.  The closest thing available to the
    null hypothesis as the data would actually present it.

Tests compared
--------------
``grs``
    Gibbons, Ross and Shanken (1989), referred to ``F(N, T-N-K)``.
``wald_chi2``
    The same quadratic form referred to its asymptotic ``chi2(N)`` -- what
    dropping the finite-sample correction costs.
``grs_shrunk``
    GRS with a Ledoit-Wolf shrinkage covariance: better conditioned, but the
    ``F`` reference becomes approximate.
``pesaran_yamagata``
    Pesaran and Yamagata (2012): a standardised average of squared intercept
    t-statistics referred to ``N(0, 1)``.  Never inverts an ``N x N`` matrix,
    so it stays defined when ``N`` exceeds ``T``, but it assumes only weak
    cross-sectional dependence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

ERROR_MODELS = (
    "gaussian",
    "gaussian_independent",
    "student_t",
    "common_vol",
    "empirical_wild",
    "empirical_wild_block",
)

TESTS = ("grs", "wald_chi2", "grs_shrunk", "pesaran_yamagata")


@dataclass(frozen=True)
class Calibration:
    """Moments and residuals taken from the real panel, to drive the simulation."""

    factor_mean: np.ndarray
    factor_cov: np.ndarray
    betas: np.ndarray  # (n_assets, n_factors)
    resid_vol: np.ndarray  # (n_assets,)
    resid_corr: np.ndarray  # (n_assets, n_assets)
    residuals: np.ndarray  # (n_obs, n_assets), standardised to unit variance
    student_df: float
    vol_persistence: float
    vol_innovation_sd: float
    factor_names: tuple[str, ...]

    @property
    def n_assets(self) -> int:
        return len(self.resid_vol)


def calibrate(returns: pd.DataFrame, factors: pd.DataFrame) -> Calibration:
    """Estimate the simulation inputs from real excess returns.

    Calibrating rather than inventing the error structure is the whole point:
    the conclusion depends on how far real residuals are from the textbook
    assumptions, so those departures have to be measured, not assumed.
    """
    aligned = pd.concat([returns, factors], axis=1).dropna()
    r = aligned[returns.columns].to_numpy(dtype=float)
    f = aligned[factors.columns].to_numpy(dtype=float)

    design = np.column_stack([np.ones(len(f)), f])
    coef = np.linalg.lstsq(design, r, rcond=None)[0]
    resid = r - design @ coef

    vol = resid.std(axis=0, ddof=1)
    corr = np.nan_to_num(np.corrcoef(resid, rowvar=False), nan=0.0)
    np.fill_diagonal(corr, 1.0)

    # For a Student-t with v degrees of freedom the excess kurtosis is
    # 6 / (v - 4), so the median residual kurtosis pins down v.
    excess_kurtosis = float(np.median(stats.kurtosis(resid, axis=0, fisher=True)))
    student_df = 4.0 + 6.0 / excess_kurtosis if excess_kurtosis > 0 else 30.0

    # An AR(1) in the log of the cross-sectional average squared residual, which
    # is the common volatility factor the ``common_vol`` error model reproduces.
    common_variance = (resid / vol) ** 2
    log_vol = 0.5 * np.log(common_variance.mean(axis=1))
    log_vol = log_vol - log_vol.mean()
    persistence = float(np.corrcoef(log_vol[1:], log_vol[:-1])[0, 1])
    innovation_sd = float(np.std(log_vol[1:] - persistence * log_vol[:-1], ddof=1))

    standardised = resid / vol
    return Calibration(
        factor_mean=f.mean(axis=0),
        factor_cov=np.cov(f, rowvar=False).reshape(f.shape[1], f.shape[1]),
        betas=coef[1:].T,
        resid_vol=vol,
        resid_corr=corr,
        # Exactly mean zero, so that resampling these vectors reproduces a true
        # null.  A pool with non-zero column means would inject a spurious
        # intercept and every test would rightly reject it.
        residuals=standardised - standardised.mean(axis=0),
        student_df=float(np.clip(student_df, 4.5, 30.0)),
        vol_persistence=float(np.clip(persistence, 0.0, 0.99)),
        vol_innovation_sd=innovation_sd,
        factor_names=tuple(factors.columns),
    )


def _nearest_psd_cholesky(matrix: np.ndarray) -> np.ndarray:
    """Cholesky factor, repairing the matrix if it is not positive definite.

    Sub-matrices of an estimated correlation matrix need not be positive
    definite, so this is required rather than defensive.
    """
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(matrix)
        repaired = vectors @ np.diag(np.clip(values, 1e-10, None)) @ vectors.T
        scale = np.sqrt(np.diag(repaired))
        repaired = repaired / np.outer(scale, scale)
        return np.linalg.cholesky(repaired + 1e-8 * np.eye(len(matrix)))


def _standardised_shocks(
    calibration: Calibration,
    picks: np.ndarray,
    n_obs: int,
    rng: np.random.Generator,
    error_model: str,
) -> np.ndarray:
    """Unit-variance residual shocks of shape ``(n_obs, len(picks))``."""
    n_assets = len(picks)

    if error_model == "gaussian_independent":
        return rng.standard_normal((n_obs, n_assets))

    if error_model in {"gaussian", "student_t", "common_vol"}:
        chol = _nearest_psd_cholesky(calibration.resid_corr[np.ix_(picks, picks)])
        normal = rng.standard_normal((n_obs, n_assets)) @ chol.T
        if error_model == "gaussian":
            return normal
        if error_model == "common_vol":
            # One persistent log-volatility factor shared by every asset, with
            # persistence and innovation size taken from the data, rescaled so
            # the unconditional variance is still one.
            log_vol = np.zeros(n_obs)
            innovation = rng.normal(0.0, calibration.vol_innovation_sd, n_obs)
            for t in range(1, n_obs):
                log_vol[t] = calibration.vol_persistence * log_vol[t - 1] + innovation[t]
            vol = np.exp(log_vol)
            return normal * (vol / np.sqrt(np.mean(vol**2)))[:, None]
        # Student-t margins joined by a Gaussian copula, so the cross-sectional
        # dependence is unchanged and only the tails get heavier.
        df = calibration.student_df
        uniform = stats.norm.cdf(normal / np.sqrt(np.diag(chol @ chol.T))[None, :])
        heavy = stats.t.ppf(np.clip(uniform, 1e-12, 1 - 1e-12), df)
        return heavy / np.sqrt(df / (df - 2.0))

    if error_model in {"empirical_wild", "empirical_wild_block"}:
        # Wild bootstrap: take real residual vectors and flip their signs.
        #
        # Two simpler schemes were tried first and both turned out to measure
        # the scheme rather than the data.  Resampling months *with* replacement
        # repeats months, so the covariance is estimated from fewer distinct
        # observations than the nominal T and the test over-rejects.  Sampling
        # *without* replacement from a finite pool shrinks the variance of the
        # estimated intercept by the finite-population correction, and the test
        # under-rejects just as badly.  Sign flips avoid both: every month
        # appears at most once, so nothing is duplicated, and the variance of
        # the intercept estimate is exactly the one the test assumes, because
        # flipping signs leaves the residual second moments untouched.  The
        # null holds exactly since each flip is mean zero, while each month's
        # own magnitude and its cross-sectional pattern survive intact.
        pool = calibration.residuals[:, picks]
        pool = pool - pool.mean(axis=0)
        n_pool = len(pool)
        if n_obs > n_pool:
            raise ValueError(
                f"empirical resampling needs n_obs <= {n_pool}, got {n_obs}"
            )
        rows = rng.choice(n_pool, size=n_obs, replace=False)
        if error_model == "empirical_wild":
            signs = rng.choice((-1.0, 1.0), size=n_obs)
        else:
            # One sign per year, so serial dependence within a block survives.
            block = 12
            per_block = rng.choice((-1.0, 1.0), size=int(np.ceil(n_obs / block)))
            signs = np.repeat(per_block, block)[:n_obs]
            rows = np.sort(rows)
        return pool[rows] * signs[:, None]

    raise ValueError(f"unknown error model: {error_model}")


def simulate_panel(
    calibration: Calibration,
    n_assets: int,
    n_obs: int,
    rng: np.random.Generator,
    *,
    alpha: np.ndarray | None = None,
    error_model: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``(returns, factors)`` from the calibrated factor model."""
    picks = rng.choice(calibration.n_assets, size=n_assets, replace=False)
    betas = calibration.betas[picks]
    vols = calibration.resid_vol[picks]

    n_factors = len(calibration.factor_mean)
    factors = rng.multivariate_normal(
        calibration.factor_mean, calibration.factor_cov, size=n_obs
    ).reshape(n_obs, n_factors)

    residuals = _standardised_shocks(calibration, picks, n_obs, rng, error_model) * vols
    intercept = np.zeros(n_assets) if alpha is None else np.asarray(alpha, dtype=float)
    return intercept + factors @ betas.T + residuals, factors


def ledoit_wolf_shrinkage(residuals: np.ndarray) -> np.ndarray:
    """Linear shrinkage of the sample covariance toward a scaled identity."""
    n_obs, n_assets = residuals.shape
    sample = residuals.T @ residuals / n_obs
    mean_variance = np.trace(sample) / n_assets
    target = mean_variance * np.eye(n_assets)

    dispersion = float(np.sum((sample - target) ** 2)) / n_assets
    if dispersion <= 0:
        return sample
    squared = residuals**2
    noise = float(
        np.sum(squared.T @ squared) / n_obs - np.sum(sample**2)
    ) / (n_assets * n_obs)
    intensity = float(np.clip(min(max(noise, 0.0), dispersion) / dispersion, 0.0, 1.0))
    return intensity * target + (1.0 - intensity) * sample


def _quadratic_form(alpha: np.ndarray, covariance: np.ndarray) -> float:
    try:
        solved = np.linalg.solve(covariance, alpha)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(covariance) @ alpha
    return float(alpha @ solved)


def pesaran_yamagata_pvalue(returns: np.ndarray, factors: np.ndarray) -> float:
    """Standardised average of squared intercept t-statistics.

    Each asset is regressed on the factors separately, so nothing of size
    ``N x N`` is inverted and the statistic stays defined however large ``N``
    becomes.  Under the null with normal, cross-sectionally independent errors
    each squared t-statistic is ``F(1, T-K-1)``, whose mean and variance are
    known in closed form; averaging ``N`` of them and standardising gives a
    statistic that is standard normal for large ``N``.
    """
    n_obs, _ = returns.shape
    n_factors = factors.shape[1]
    dof = n_obs - n_factors - 1
    if dof <= 4:
        return float("nan")

    design = np.column_stack([np.ones(n_obs), factors])
    xtx_inv = np.linalg.pinv(design.T @ design)
    coef = xtx_inv @ (design.T @ returns)
    resid = returns - design @ coef
    variance_alpha = xtx_inv[0, 0] * (resid**2).sum(axis=0) / dof
    with np.errstate(divide="ignore", invalid="ignore"):
        t_squared = coef[0] ** 2 / variance_alpha
    t_squared = t_squared[np.isfinite(t_squared)]
    if len(t_squared) < 2:
        return float("nan")

    expected = dof / (dof - 2.0)
    variance = 2.0 * dof**2 * (dof - 1.0) / ((dof - 2.0) ** 2 * (dof - 4.0))
    statistic = (
        np.sqrt(len(t_squared)) * (t_squared.mean() - expected) / np.sqrt(variance)
    )
    # One sided: non-zero intercepts can only push the average upward.
    return float(stats.norm.sf(statistic))


def alpha_tests(returns: np.ndarray, factors: np.ndarray) -> dict[str, float]:
    """p-values from each test of the joint null that all intercepts are zero."""
    n_obs, n_assets = returns.shape
    n_factors = factors.shape[1]

    design = np.column_stack([np.ones(n_obs), factors])
    coef = np.linalg.lstsq(design, returns, rcond=None)[0]
    alpha, resid = coef[0], returns - design @ coef

    mean_f = factors.mean(axis=0)
    cov_f = np.cov(factors, rowvar=False).reshape(n_factors, n_factors)
    sharpe_sq = float(mean_f @ np.linalg.pinv(cov_f) @ mean_f)

    dof = n_obs - n_assets - n_factors
    out: dict[str, float] = {}

    if dof > 0:
        quad = _quadratic_form(alpha, resid.T @ resid / n_obs)
        grs = (dof / n_assets) * quad / (1.0 + sharpe_sq)
        out["grs"] = float(stats.f.sf(grs, n_assets, dof))
        out["wald_chi2"] = float(
            stats.chi2.sf(n_obs * quad / (1.0 + sharpe_sq), n_assets)
        )
        shrunk = _quadratic_form(alpha, ledoit_wolf_shrinkage(resid))
        out["grs_shrunk"] = float(
            stats.f.sf((dof / n_assets) * shrunk / (1.0 + sharpe_sq), n_assets, dof)
        )
    else:
        out["grs"] = out["wald_chi2"] = out["grs_shrunk"] = np.nan

    out["pesaran_yamagata"] = pesaran_yamagata_pvalue(returns, factors)
    return out


def simulate_pvalues(
    calibration: Calibration,
    n_assets: int,
    n_obs: int,
    n_replications: int,
    *,
    error_model: str = "gaussian",
    alpha_sd: float = 0.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """p-values from every test over repeated draws of one ``(N, T)`` design."""
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {name: [] for name in TESTS}
    for _ in range(n_replications):
        intercepts = None if alpha_sd == 0.0 else rng.normal(0.0, alpha_sd, n_assets)
        returns, factors = simulate_panel(
            calibration, n_assets, n_obs, rng, alpha=intercepts, error_model=error_model
        )
        for name, pvalue in alpha_tests(returns, factors).items():
            collected[name].append(pvalue)
    return {name: np.asarray(values, dtype=float) for name, values in collected.items()}


def size_power_study(
    calibration: Calibration,
    asset_grid: tuple[int, ...],
    obs_grid: tuple[int, ...],
    n_replications: int = 400,
    *,
    error_models: tuple[str, ...] = ERROR_MODELS,
    alpha_sd: float = 0.003,
    nominal: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Size, raw power and size-adjusted power for every test over an ``N x T`` grid.

    Raw power cannot be compared across tests whose sizes differ -- a test that
    rejects a true null 96% of the time will also "detect" mispricing 96% of
    the time while knowing nothing.  So power is also reported at an
    empirically calibrated critical value, taken from this cell's own null
    distribution, which is the only comparison that means anything.
    """
    rows = []
    for n_obs in obs_grid:
        for n_assets in asset_grid:
            if n_assets > calibration.n_assets:
                continue
            for error_model in error_models:
                offset = seed + 7919 * n_obs + 31 * n_assets + 7 * hash(error_model) % 997
                under_null = simulate_pvalues(
                    calibration,
                    n_assets,
                    n_obs,
                    n_replications,
                    error_model=error_model,
                    alpha_sd=0.0,
                    seed=offset,
                )
                under_alternative = simulate_pvalues(
                    calibration,
                    n_assets,
                    n_obs,
                    n_replications,
                    error_model=error_model,
                    alpha_sd=alpha_sd,
                    seed=offset + 555_555,
                )
                row = {
                    "error_model": error_model,
                    "n_assets": n_assets,
                    "n_obs": n_obs,
                    "ratio_n_over_t": n_assets / n_obs,
                }
                for name in TESTS:
                    null = under_null[name][np.isfinite(under_null[name])]
                    alt = under_alternative[name][np.isfinite(under_alternative[name])]
                    row[f"{name}_defined"] = float(np.isfinite(under_null[name]).mean())
                    if len(null) < 20 or len(alt) < 20:
                        row[f"{name}_size"] = np.nan
                        row[f"{name}_power"] = np.nan
                        row[f"{name}_power_adj"] = np.nan
                        continue
                    row[f"{name}_size"] = float(np.mean(null < nominal))
                    row[f"{name}_power"] = float(np.mean(alt < nominal))
                    critical = float(np.quantile(null, nominal))
                    row[f"{name}_power_adj"] = float(np.mean(alt <= critical))
                rows.append(row)
    return pd.DataFrame(rows)


def real_data_tests(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    asset_grid: tuple[int, ...],
    obs_windows: tuple[int, ...],
    n_draws: int = 40,
    seed: int = 0,
) -> pd.DataFrame:
    """Apply the tests to random subsets of the real panel.

    Included to show where the constraint bites in practice rather than in
    simulation.  Note what this cannot do: a rejection here mixes genuine
    mispricing with whatever size distortion the test has, which is exactly
    why the calibrated simulation above is necessary to interpret it.
    """
    aligned = pd.concat([returns, factors], axis=1).dropna()
    r = aligned[returns.columns]
    f = aligned[factors.columns]

    rows = []
    for n_obs in obs_windows:
        if n_obs > len(r):
            continue
        window = slice(len(r) - n_obs, len(r))
        for n_assets in asset_grid:
            if n_assets > r.shape[1]:
                continue
            rng = np.random.default_rng(seed + 31 * n_assets + n_obs)
            collected = {name: [] for name in TESTS}
            for _ in range(n_draws):
                picks = rng.choice(r.shape[1], size=n_assets, replace=False)
                subset = r.iloc[window, picks].to_numpy(dtype=float)
                for name, pvalue in alpha_tests(
                    subset, f.iloc[window].to_numpy(dtype=float)
                ).items():
                    collected[name].append(pvalue)
            row = {
                "n_assets": n_assets,
                "n_obs": n_obs,
                "ratio_n_over_t": n_assets / n_obs,
            }
            for name in TESTS:
                pvalues = np.asarray(collected[name], dtype=float)
                usable = np.isfinite(pvalues)
                row[f"{name}_reject_rate"] = (
                    float(np.mean(pvalues[usable] < 0.05)) if usable.any() else np.nan
                )
                row[f"{name}_defined"] = float(usable.mean())
            rows.append(row)
    return pd.DataFrame(rows)
