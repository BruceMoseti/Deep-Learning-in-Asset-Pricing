import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def toy_panel():
    """A small synthetic panel with the same shape as the real one.

    120 months, 40 assets, balanced, with a planted dependence of next-month
    return on this month's return so that tests can distinguish "found the
    signal" from "found nothing".
    """
    rng = np.random.default_rng(11)
    months = pd.period_range("1990-01", periods=180, freq="M")
    assets = [f"a{i:02d}" for i in range(40)]

    market = rng.normal(0.006, 0.045, len(months))
    beta = rng.uniform(0.6, 1.4, len(assets))
    noise = rng.normal(0.0, 0.05, (len(months), len(assets)))
    returns = market[:, None] * beta[None, :] + noise

    # Plant a one-month reversal: next month's idiosyncratic return loads
    # negatively on this month's.
    returns[1:] -= 0.15 * noise[:-1]

    wide = pd.DataFrame(returns, index=months, columns=assets)
    long = wide.stack().rename("ret").reset_index()
    long.columns = ["month", "asset", "ret"]
    long["family"] = "toy"
    long["sort1"] = np.tile(np.linspace(0, 1, 5), 8 * len(months))[: len(long)]
    long["sort2"] = np.tile(np.linspace(0, 1, 8), 5 * len(months))[: len(long)]
    long["is_industry"] = 0.0
    long["ret"] = long["ret"] * 100.0  # parsers hand back percent
    long["ret"] = long["ret"] / 100.0
    return long.sort_values(["month", "asset"], ignore_index=True)


@pytest.fixture
def toy_factors(toy_panel):
    months = pd.PeriodIndex(toy_panel["month"].unique(), freq="M")
    rng = np.random.default_rng(12)
    wide = toy_panel.pivot(index="month", columns="asset", values="ret")
    return pd.DataFrame(
        {
            "mktrf": wide.mean(axis=1) - 0.003,
            "smb": rng.normal(0.002, 0.03, len(months)),
            "hml": rng.normal(0.003, 0.03, len(months)),
            "rmw": rng.normal(0.002, 0.02, len(months)),
            "cma": rng.normal(0.002, 0.02, len(months)),
            "mom": rng.normal(0.005, 0.04, len(months)),
            "rf": 0.003,
        },
        index=months,
    )
