"""Central configuration.

Every number that affects a reported result lives here so that the research
report and the code cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"

SEED = 20240601


@dataclass(frozen=True)
class SplitConfig:
    """Expanding-window walk-forward schedule.

    The test block is one calendar year.  ``validation_years`` immediately
    precede it and are used only for hyper-parameter selection and early
    stopping; the training block is everything before that and grows by one
    year per step.  Nothing after the end of the test year is ever touched.
    """

    min_train_years: int = 15
    validation_years: int = 5
    test_years: int = 1
    first_test_year: int = 1990
    last_test_year: int = 2023


@dataclass(frozen=True)
class PortfolioConfig:
    n_quantiles: int = 10
    weighting: str = "equal"  # "equal" or "rank"
    cost_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)


@dataclass(frozen=True)
class Config:
    seed: int = SEED
    # Sample start.  1963-07 is the standard post-Compustat start used by the
    # Fama-French portfolio sets that sort on accounting variables.
    sample_start: str = "1963-07"
    sample_end: str = "2023-12"
    min_history_months: int = 60
    min_assets_per_month: int = 30
    target: str = "zscore"  # "zscore" or "rank"
    split: SplitConfig = field(default_factory=SplitConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    newey_west_lags: int = 6
    bootstrap_draws: int = 5000
    block_length: int = 12


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROC, RESULTS, FIGURES, TABLES):
        d.mkdir(parents=True, exist_ok=True)
