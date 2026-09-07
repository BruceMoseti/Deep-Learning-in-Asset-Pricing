#!/usr/bin/env python3
"""Experiment 4 -- can conventional asset-pricing tests be trusted with many assets?

The question as usually posed assumes the answer.  Testing it properly means
simulating under a null that is true by construction, with errors calibrated to
the real panel, and checking how often each test rejects anyway.

Four tests of the joint null that all intercepts are zero are compared over a
grid of ``N`` and ``T``, under six error structures ranging from the textbook
Gaussian case to residual vectors taken straight from the data.  Because tests
with different sizes cannot be compared on raw power, power is also reported at
critical values calibrated from each cell's own null distribution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from xsap.artifacts import factors, save_json, save_table
from xsap.config import Config
from xsap.data import load_portfolio_panel
from xsap.inference import FF6
from xsap.montecarlo import (
    ERROR_MODELS,
    TESTS,
    calibrate,
    real_data_tests,
    size_power_study,
)

ASSET_GRID = (10, 25, 50, 100, 200, 300)
OBS_GRID = (120, 360)
# Cross-sectional dispersion of true alpha under the alternative: 0.05% per
# month, about 0.6% a year.  Chosen deliberately small -- roughly a third of the
# standard error of a single asset's alpha at T = 240 -- so that no test can see
# it one asset at a time and the joint test has to earn its power by
# aggregating.  Larger values saturate every test at power one and make the
# comparison uninformative.
ALPHA_SD = 0.0005


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=300)
    parser.add_argument("--quick", action="store_true", help="small grid, few draws")
    args = parser.parse_args()

    cfg = Config()
    ff = factors(cfg)
    panel = load_portfolio_panel(cfg)
    excess = panel.pivot(index="month", columns="asset", values="ret").sub(
        ff["rf"], axis=0
    )
    calibration = calibrate(excess, ff[list(FF6)])

    off_diagonal = calibration.resid_corr[
        ~np.eye(calibration.n_assets, dtype=bool)
    ]
    save_json(
        {
            "n_assets": calibration.n_assets,
            "n_months": int(len(calibration.residuals)),
            "factors": list(FF6),
            "residual_correlation_mean": float(off_diagonal.mean()),
            "residual_correlation_p90": float(np.quantile(off_diagonal, 0.9)),
            "residual_vol_mean_monthly": float(calibration.resid_vol.mean()),
            "implied_student_df": calibration.student_df,
            "log_vol_persistence": calibration.vol_persistence,
            "log_vol_innovation_sd": calibration.vol_innovation_sd,
            "alpha_sd_under_alternative": ALPHA_SD,
            "replications": args.replications,
        },
        "exp4_calibration",
    )
    print(
        f"calibrated on {calibration.n_assets} assets, "
        f"{len(calibration.residuals)} months; "
        f"residual correlation {off_diagonal.mean():.3f}, "
        f"implied t degrees of freedom {calibration.student_df:.1f}, "
        f"log-volatility persistence {calibration.vol_persistence:.2f}"
    )

    asset_grid = (10, 50, 150) if args.quick else ASSET_GRID
    obs_grid = (240,) if args.quick else OBS_GRID
    replications = 60 if args.quick else args.replications

    print("\nsimulating size and power")
    study = size_power_study(
        calibration,
        asset_grid,
        obs_grid,
        replications,
        error_models=ERROR_MODELS,
        alpha_sd=ALPHA_SD,
        seed=cfg.seed,
    )
    save_table(
        study.set_index(["error_model", "n_obs", "n_assets"]).sort_index(),
        "exp4_size_power_full",
    )

    size = study.pivot_table(
        index=["error_model", "n_obs", "n_assets"],
        values=[f"{t}_size" for t in TESTS],
    ).sort_index()
    size.columns = [c.replace("_size", "") for c in size.columns]
    save_table(size, "exp4_size")

    power = study.pivot_table(
        index=["error_model", "n_obs", "n_assets"],
        values=[f"{t}_power_adj" for t in TESTS],
    ).sort_index()
    power.columns = [c.replace("_power_adj", "") for c in power.columns]
    save_table(power, "exp4_size_adjusted_power")

    # How often each test is even computable.  For GRS this is not a size
    # question but an existence one: the statistic requires T > N + K.
    defined = study.pivot_table(
        index=["n_obs", "n_assets"], values=[f"{t}_defined" for t in TESTS]
    ).sort_index()
    defined.columns = [c.replace("_defined", "") for c in defined.columns]
    save_table(defined, "exp4_test_defined")

    # The one-line summary: worst size distortion each test suffers anywhere in
    # the grid, and where.
    worst = []
    for test in TESTS:
        column = f"{test}_size"
        usable = study[study[column].notna()]
        if usable.empty:
            continue
        row = usable.loc[usable[column].idxmax()]
        worst.append(
            {
                "test": test,
                "nominal_size": 0.05,
                "worst_size": row[column],
                "at_error_model": row["error_model"],
                "at_n_assets": int(row["n_assets"]),
                "at_n_obs": int(row["n_obs"]),
                "at_ratio_n_over_t": row["ratio_n_over_t"],
                "median_size_over_grid": float(usable[column].median()),
                "median_size_adjusted_power": float(
                    usable[f"{test}_power_adj"].median()
                ),
            }
        )
    save_table(pd.DataFrame(worst).set_index("test"), "exp4_worst_case_size")

    print("\napplying the same tests to the real panel")
    real = real_data_tests(
        excess,
        ff[list(FF6)],
        asset_grid=(10, 25, 50, 100, 200, 300, calibration.n_assets),
        obs_windows=(120, 240, 480, len(excess.dropna())),
        n_draws=20 if args.quick else 40,
        seed=cfg.seed,
    )
    save_table(real.set_index(["n_obs", "n_assets"]).sort_index(), "exp4_real_data")

    print("\nempirical size, nominal 5 percent:")
    print(size.to_string(float_format=lambda v: f"{v: .3f}"))
    print("\nworst size distortion per test:")
    print(pd.DataFrame(worst).set_index("test").to_string(float_format=lambda v: f"{v: .3f}"))


if __name__ == "__main__":
    main()
