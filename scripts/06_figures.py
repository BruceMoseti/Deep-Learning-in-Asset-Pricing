#!/usr/bin/env python3
"""Build every figure in the report from the saved result files."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xsap.artifacts import factors, load_series, read_json
from xsap.config import FIGURES, RESULTS, Config, ensure_dirs

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    }
)

LADDER = ["single-signal", "ols", "ridge", "lasso", "enet", "xgboost", "neural-net"]
HIGHLIGHT = {"xgboost": "#c0392b", "ridge": "#2c3e50", "single-signal": "#7f8c8d"}


def _table(name: str, index=None) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / f"{name}.csv")
    return frame.set_index(index) if index else frame


def _save(fig, name: str) -> None:
    path = FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


def _order(names) -> list[str]:
    return [m for m in LADDER if m in set(names)]


def figure_forecast_accuracy() -> None:
    accuracy = _table("exp1_forecast_accuracy", "model")
    models = _order(accuracy.index)
    accuracy = accuracy.loc[models]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    colours = [HIGHLIGHT.get(m, "#95a5a6") for m in models]

    axes[0].bar(models, accuracy["rank_ic"], color=colours)
    axes[0].set_title("Mean rank information coefficient")
    axes[0].set_ylabel("Spearman IC")

    axes[1].bar(models, accuracy["rank_ic_tstat"], color=colours)
    axes[1].axhline(1.96, color="black", lw=0.8, ls="--")
    axes[1].axhline(3.0, color="#c0392b", lw=0.8, ls=":")
    axes[1].set_title("Newey-West t-statistic of mean IC")
    axes[1].text(-0.4, 2.04, "1.96", fontsize=7)
    axes[1].text(-0.4, 3.08, "3.0", fontsize=7, color="#c0392b")

    # The single-characteristic baseline is a ranking, not a calibrated return
    # forecast, so its squared error is not comparable and its -30% would
    # compress the axis to the point of hiding the differences that matter.
    fitted = [m for m in models if m != "single-signal"]
    axes[2].bar(
        fitted,
        accuracy.loc[fitted, "r2_oos"] * 100.0,
        color=[HIGHLIGHT.get(m, "#95a5a6") for m in fitted],
    )
    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[2].set_title("Out-of-sample $R^2$ vs a zero forecast")
    axes[2].set_ylabel("percent")
    axes[2].text(
        0.5,
        0.03,
        "baseline omitted: a ranking, not a\ncalibrated forecast ($R^2$ = "
        f"{accuracy.loc['single-signal', 'r2_oos'] * 100:.0f}%)",
        transform=axes[2].transAxes,
        fontsize=6.5,
        ha="center",
        style="italic",
    )

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    fig.suptitle(
        "Experiment 1: forecast accuracy out of sample, 1990-2023 (407 months)",
        y=1.06,
        fontsize=10,
    )
    _save(fig, "fig1_forecast_accuracy")


def figure_model_comparisons() -> None:
    """The headline of Experiment 1: which differences are actually detectable."""
    path = RESULTS / "exp1_model_comparisons.csv"
    if not path.exists():
        print("  skipping comparison figure (exp1_model_comparisons.csv not found)")
        return
    comparisons = _table("exp1_model_comparisons")
    order = [m for m in LADDER if m in set(comparisons["model"])]
    adjacent = [(a, b) for a, b in zip(order[1:], order[:-1])]

    best_linear = read_json("exp1_run").get("best_linear_model", "enet")
    against_linear = [
        (m, best_linear) for m in order if m != best_linear
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, pairs, title in (
        (axes[0], adjacent, "Each rung against the one below it"),
        (axes[1], against_linear, f"Each model against {best_linear}"),
    ):
        labels, diffs, errors, colours = [], [], [], []
        for model, benchmark in pairs:
            row = comparisons[
                (comparisons["model"] == model) & (comparisons["benchmark"] == benchmark)
            ]
            if row.empty:
                continue
            diff = float(row["ic_difference"].iloc[0])
            tstat = float(row["ic_diff_tstat"].iloc[0])
            labels.append(f"{model}\nvs {benchmark}")
            diffs.append(diff)
            # Back out the standard error from the point estimate and t.
            errors.append(1.96 * abs(diff / tstat) if tstat else np.nan)
            colours.append("#c0392b" if abs(tstat) > 1.96 else "#95a5a6")

        positions = np.arange(len(labels))
        ax.barh(positions, diffs, xerr=errors, color=colours, capsize=3, height=0.6)
        ax.axvline(0.0, color="black", lw=1.0)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel("difference in mean rank IC (95% interval)")
        ax.set_title(title, fontsize=9)
        ax.invert_yaxis()

    fig.suptitle(
        "Red = the 95% interval excludes zero.  Almost nothing does.",
        y=1.05,
        fontsize=10,
    )
    _save(fig, "fig13_model_comparisons")


def figure_calibration_gap() -> None:
    """Why a positive IC can sit next to a negative out-of-sample R-squared."""
    accuracy = _table("exp1_forecast_accuracy", "model")
    models = _order(accuracy.index)
    accuracy = accuracy.loc[models]

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.scatter(
        accuracy["calibration_slope"],
        accuracy["r2_oos"] * 100.0,
        s=60,
        c=[HIGHLIGHT.get(m, "#95a5a6") for m in models],
        zorder=3,
    )
    for model in models:
        ax.annotate(
            model,
            (accuracy.loc[model, "calibration_slope"], accuracy.loc[model, "r2_oos"] * 100.0),
            textcoords="offset points",
            xytext=(6, 3),
            fontsize=7.5,
        )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axvline(1.0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("calibration slope (1.0 = correctly scaled forecast)")
    ax.set_ylabel("out-of-sample $R^2$, percent")
    ax.set_title("A forecast can rank well and still be badly scaled")
    _save(fig, "fig2_calibration_gap")


def figure_cumulative_performance() -> None:
    gross = load_series("strategy_returns_gross")
    net = load_series("strategy_returns_net_10bps")
    gross.index = pd.PeriodIndex(gross.index, freq="M").to_timestamp()
    net.index = pd.PeriodIndex(net.index, freq="M").to_timestamp()

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for name in _order(gross.columns):
        style = {"color": HIGHLIGHT.get(name, "#bdc3c7"), "lw": 1.6 if name in HIGHLIGHT else 1.0}
        axes[0].plot(gross.index, np.exp(np.log1p(gross[name]).cumsum()), label=name, **style)
        axes[1].plot(net.index, np.exp(np.log1p(net[name]).cumsum()), label=name, **style)
    axes[0].set_title("Gross of costs")
    axes[1].set_title("Net of 10 bps one-way")
    axes[0].set_ylabel("cumulative value of $1 long-short")
    for ax in axes:
        ax.set_yscale("log")
        ax.axhline(1.0, color="black", lw=0.8)
    axes[1].legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle(
        "Experiment 2: decile long-short portfolios, formed monthly out of sample",
        y=1.04,
        fontsize=10,
    )
    _save(fig, "fig3_cumulative_performance")


def figure_cost_erosion() -> None:
    performance = _table("exp2_portfolio_performance", "model")
    models = _order(performance.index)
    levels = [0.0, 5.0, 10.0, 20.0]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for name in models:
        sharpes = [performance.loc[name, "sharpe_gross"]] + [
            performance.loc[name, f"sharpe_net_{b:g}bps"] for b in levels[1:]
        ]
        axes[0].plot(
            levels,
            sharpes,
            marker="o",
            ms=4,
            color=HIGHLIGHT.get(name, "#bdc3c7"),
            lw=1.6 if name in HIGHLIGHT else 1.0,
            label=name,
        )
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_xlabel("one-way transaction cost, basis points")
    axes[0].set_ylabel("annualised Sharpe ratio")
    axes[0].set_title("Sharpe ratio against trading cost")
    axes[0].legend(fontsize=7, frameon=False, ncol=2)

    order = performance.loc[models, "breakeven_cost_bps"].sort_values()
    axes[1].barh(
        order.index,
        order.to_numpy(),
        color=[HIGHLIGHT.get(m, "#95a5a6") for m in order.index],
    )
    for level, style in ((10.0, "--"), (20.0, ":")):
        axes[1].axvline(level, color="black", lw=0.8, ls=style)
    axes[1].set_xlabel("break-even one-way cost, basis points")
    axes[1].set_title("Cost at which the average gross return is consumed")
    fig.suptitle(
        "The ranking of models by accuracy is not the ranking by tradability",
        y=1.04,
        fontsize=10,
    )
    _save(fig, "fig4_cost_erosion")


def figure_quantile_profile() -> None:
    profile = _table("exp2_quantile_profile", "predicted_decile")
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    width = 0.26
    shown = [m for m in ("single-signal", "ridge", "xgboost") if m in profile.columns]
    for offset, name in enumerate(shown):
        ax.bar(
            profile.index + (offset - 1) * width,
            profile[name] * 100.0,
            width=width,
            label=name,
            color=HIGHLIGHT.get(name, "#95a5a6"),
        )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("predicted decile (0 = lowest forecast)")
    ax.set_ylabel("realised annualised excess return, percent")
    ax.set_title("Realised return by predicted decile")
    ax.legend(fontsize=7.5, frameon=False)
    _save(fig, "fig5_quantile_profile")


def figure_rolling_accuracy() -> None:
    ic = load_series("ic_monthly")
    ic.index = pd.PeriodIndex(ic.index, freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    for name in ("ridge", "xgboost"):
        if name in ic.columns:
            ax.plot(
                ic.index,
                ic[name].rolling(36, min_periods=24).mean(),
                color=HIGHLIGHT[name],
                lw=1.4,
                label=f"{name} (36-month mean)",
            )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel("rank IC")
    ax.set_title("Accuracy is positive on average and far from constant")
    ax.legend(fontsize=7.5, frameon=False)
    _save(fig, "fig6_rolling_accuracy")


def figure_multiple_testing() -> None:
    published = _table("exp3_predictors_all", "predictor")
    own = _table("exp3_own_strategy_vs_published", "model")
    n_tests = len(published)
    from scipy import stats

    bonferroni_t = float(stats.norm.isf(0.025 / n_tests))

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.hist(
        published["tstat_nw"].abs().clip(upper=12),
        bins=40,
        color="#bdc3c7",
        edgecolor="white",
        label=f"{n_tests} published predictors",
    )
    ax.axvline(1.96, color="black", lw=1.0, ls="--", label="uncorrected 5%: |t| = 1.96")
    ax.axvline(3.0, color="#e67e22", lw=1.0, ls="-.", label="Harvey-Liu-Zhu hurdle: |t| = 3.0")
    ax.axvline(
        bonferroni_t,
        color="#8e44ad",
        lw=1.0,
        ls=":",
        label=f"Bonferroni over {n_tests}: |t| = {bonferroni_t:.2f}",
    )
    for name in ("xgboost",):
        if name in own.index:
            ax.axvline(
                abs(own.loc[name, "tstat_nw"]),
                color=HIGHLIGHT[name],
                lw=2.0,
                label=f"this project ({name}): |t| = {abs(own.loc[name, 'tstat_nw']):.2f}",
            )
    ax.set_xlabel("|Newey-West t-statistic| of the mean long-short return")
    ax.set_ylabel("number of predictors")
    ax.set_title(
        "Experiment 3: this project's strategy against the published cross-section"
    )
    ax.legend(fontsize=7, frameon=False)
    _save(fig, "fig7_multiple_testing")


def figure_test_size() -> None:
    size = _table("exp4_size")
    tests = ["grs", "wald_chi2", "grs_shrunk", "pesaran_yamagata"]
    shown = ["gaussian_independent", "gaussian", "empirical_wild_block"]
    titles = {
        "gaussian_independent": "Normal, cross-sectionally independent",
        "gaussian": "Normal, real cross-sectional correlation",
        "empirical_wild_block": "Residuals resampled from the data",
    }
    colours = {
        "grs": "#2c3e50",
        "wald_chi2": "#c0392b",
        "grs_shrunk": "#16a085",
        "pesaran_yamagata": "#e67e22",
    }

    fig, axes = plt.subplots(1, len(shown), figsize=(12, 3.5), sharey=True)
    for ax, model in zip(axes, shown):
        block = size[(size["error_model"] == model) & (size["n_obs"] == 360)]
        block = block.sort_values("n_assets")
        for test in tests:
            ax.plot(
                block["n_assets"] / block["n_obs"],
                block[test],
                marker="o",
                ms=4,
                color=colours[test],
                label=test,
            )
        ax.axhline(0.05, color="black", lw=1.0, ls="--")
        ax.set_title(titles[model], fontsize=9)
        ax.set_xlabel("N / T")
    axes[0].set_ylabel("rejection rate under a true null")
    axes[0].text(0.02, 0.062, "nominal 5%", fontsize=7)
    axes[-1].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Experiment 4: empirical size at T = 360.  GRS holds; its asymptotic "
        "counterpart does not.",
        y=1.05,
        fontsize=10,
    )
    _save(fig, "fig8_test_size")


def figure_test_power() -> None:
    power = _table("exp4_size_adjusted_power")
    colours = {"grs": "#2c3e50", "pesaran_yamagata": "#e67e22"}
    shown = ["gaussian_independent", "empirical_wild_block"]
    titles = {
        "gaussian_independent": "Weak cross-sectional dependence\n(Pesaran-Yamagata's assumption holds)",
        "empirical_wild_block": "Real cross-sectional dependence\n(assumption violated)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, model in zip(axes, shown):
        block = power[(power["error_model"] == model) & (power["n_obs"] == 360)]
        block = block.sort_values("n_assets")
        for test, colour in colours.items():
            ax.plot(
                block["n_assets"],
                block[test],
                marker="o",
                ms=4,
                color=colour,
                label=test,
            )
        ax.set_title(titles[model], fontsize=9)
        ax.set_xlabel("number of assets N")
        ax.set_xscale("log")
    axes[0].set_ylabel("size-adjusted power")
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle(
        "Which test to use depends on the cross-sectional dependence, not on N alone",
        y=1.06,
        fontsize=10,
    )
    _save(fig, "fig9_test_power")


def figure_ablation() -> None:
    path = RESULTS / "exp5_feature_ablation.csv"
    if not path.exists():
        print("  skipping ablation figure (exp5_feature_ablation.csv not found)")
        return
    ablation = _table("exp5_feature_ablation", "predictor_set")
    baseline = ablation.loc["all_predictors", "rank_ic"]

    without = ablation[ablation.index.str.startswith("without_")].copy()
    only = ablation[ablation.index.str.startswith("only_")].copy()
    without.index = without.index.str.replace("without_", "", regex=False)
    only.index = only.index.str.replace("only_", "", regex=False)
    order = without["rank_ic"].sort_values().index

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    axes[0].barh(order, without.loc[order, "rank_ic"], color="#2c3e50")
    axes[0].axvline(baseline, color="#c0392b", lw=1.2, ls="--", label="all predictors")
    axes[0].set_title("IC with one group removed")
    axes[0].set_xlabel("rank IC")
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")

    axes[1].barh(order, only.loc[order, "rank_ic"], color="#16a085")
    axes[1].axvline(baseline, color="#c0392b", lw=1.2, ls="--")
    axes[1].axvline(0.0, color="black", lw=0.8)
    axes[1].set_title("IC with that group alone")
    axes[1].set_xlabel("rank IC")
    fig.suptitle(
        "Experiment 5: removing a group and using it alone answer different questions",
        y=1.05,
        fontsize=10,
    )
    _save(fig, "fig10_feature_ablation")


def figure_regimes() -> None:
    path = RESULTS / "exp5_regimes.csv"
    if not path.exists():
        print("  skipping regime figure (exp5_regimes.csv not found)")
        return
    regimes = _table("exp5_regimes")
    shown = regimes[regimes["model"].isin(["ridge", "xgboost", "single-signal"])]

    dimensions = list(dict.fromkeys(shown["dimension"]))
    fig, axes = plt.subplots(1, len(dimensions), figsize=(13, 3.4), sharey=True)
    for ax, dimension in zip(np.atleast_1d(axes), dimensions):
        block = shown[shown["dimension"] == dimension]
        states = list(dict.fromkeys(block["state"]))
        models = _order(block["model"].unique())
        width = 0.8 / max(len(models), 1)
        for offset, model in enumerate(models):
            values = [
                block[(block["state"] == s) & (block["model"] == model)]["rank_ic"].mean()
                for s in states
            ]
            ax.bar(
                np.arange(len(states)) + offset * width,
                values,
                width=width,
                label=model,
                color=HIGHLIGHT.get(model, "#95a5a6"),
            )
        ax.set_xticks(np.arange(len(states)) + width)
        ax.set_xticklabels(states, fontsize=7.5)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_title(dimension, fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("rank IC")
    np.atleast_1d(axes)[-1].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Accuracy by market state, using only information available beforehand",
        y=1.05,
        fontsize=10,
    )
    _save(fig, "fig11_regimes")


def figure_universe() -> None:
    cfg = Config()
    ff = factors(cfg)
    ic = load_series("ic_monthly")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))

    market = ff["mktrf"]
    level = np.exp(np.log1p(market).cumsum())
    axes[0].plot(market.index.to_timestamp(), level, color="#2c3e50", lw=1.2)
    axes[0].set_yscale("log")
    axes[0].set_title("Market factor, cumulative (context for the sample)")
    axes[0].set_ylabel("cumulative value")

    axes[1].hist(ic["xgboost"].dropna(), bins=40, color="#c0392b", edgecolor="white")
    axes[1].axvline(0.0, color="black", lw=1.0)
    axes[1].axvline(
        ic["xgboost"].mean(), color="#2c3e50", lw=1.4, ls="--", label="mean IC"
    )
    axes[1].set_title("Monthly rank IC, xgboost")
    axes[1].set_xlabel("rank IC in a single month")
    axes[1].legend(fontsize=7.5, frameon=False)
    _save(fig, "fig12_context")


def main() -> None:
    ensure_dirs()
    print("building figures")
    for builder in (
        figure_forecast_accuracy,
        figure_model_comparisons,
        figure_calibration_gap,
        figure_cumulative_performance,
        figure_cost_erosion,
        figure_quantile_profile,
        figure_rolling_accuracy,
        figure_multiple_testing,
        figure_test_size,
        figure_test_power,
        figure_ablation,
        figure_regimes,
        figure_universe,
    ):
        builder()


if __name__ == "__main__":
    main()
