"""The audit that matters: prove the pipeline cannot see the future.

These are the checks that would catch the failure mode which makes most
backtests worthless.  They test behaviour, not implementation, so they keep
working if the feature code is rewritten.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xsap.config import Config
from xsap.features import FEATURE_GROUPS, FEATURE_NAMES, build_features, make_target, rank_normalise
from xsap.metrics import monthly_ic
from xsap.walkforward import expanding_splits

DYNAMIC = [f for f in FEATURE_NAMES if f not in FEATURE_GROUPS["static"]]


def test_features_do_not_change_when_the_future_changes(toy_panel, toy_factors):
    """Corrupting returns after month t must leave every predictor at t alone.

    This is the definition of point-in-time, tested directly rather than
    asserted in a comment.
    """
    baseline = build_features(toy_panel, toy_factors, Config(min_assets_per_month=5))
    cutoff = pd.Period("2000-06", freq="M")

    corrupted = toy_panel.copy()
    future = corrupted["month"] > cutoff
    rng = np.random.default_rng(99)
    corrupted.loc[future, "ret"] = rng.uniform(-0.5, 2.0, int(future.sum()))

    perturbed = build_features(corrupted, toy_factors, Config(min_assets_per_month=5))

    index = baseline.index[baseline.index.get_level_values("month") <= cutoff]
    left = baseline.loc[index, DYNAMIC]
    right = perturbed.loc[index, DYNAMIC]
    pd.testing.assert_frame_equal(left, right, check_exact=False, atol=1e-12)


def test_label_is_next_month_excess_return(toy_panel, toy_factors):
    """``fwd_excess_ret`` at (t, i) must equal asset i's excess return in t+1."""
    data = build_features(toy_panel, toy_factors, Config(min_assets_per_month=5))
    wide = toy_panel.pivot(index="month", columns="asset", values="ret")
    excess = wide.sub(toy_factors["rf"].reindex(wide.index), axis=0)

    sample = data.sample(200, random_state=3)
    for (month, asset), row in sample.iterrows():
        assert row["fwd_excess_ret"] == pytest.approx(
            excess.loc[month + 1, asset], abs=1e-12
        )


def test_splits_are_ordered_disjoint_and_embargoed(toy_panel, toy_factors):
    """Train precedes validation precedes test, and labels never cross a block."""
    data = build_features(toy_panel, toy_factors, Config(min_assets_per_month=5))
    months = pd.PeriodIndex(data.index.get_level_values("month").unique(), freq="M")
    cfg = Config(
        min_assets_per_month=5,
    )
    object.__setattr__(cfg.split, "min_train_years", 5)
    object.__setattr__(cfg.split, "validation_years", 2)
    object.__setattr__(cfg.split, "first_test_year", 2000)
    object.__setattr__(cfg.split, "last_test_year", 2004)

    splits = expanding_splits(months, cfg)
    assert splits, "expected at least one split"

    for split in splits:
        assert set(split.train).isdisjoint(split.valid)
        assert set(split.valid).isdisjoint(split.test)
        assert set(split.train).isdisjoint(split.test)
        assert split.train.max() < split.valid.min()
        assert split.valid.max() < split.test.min()
        # A row dated m carries the return of m+1, so the last label in each
        # block must still land strictly before the next block starts.
        assert split.train.max() + 1 < split.valid.min()
        assert split.valid.max() + 1 < split.test.min()

    # Training windows grow, which is what makes it an expanding scheme.
    sizes = [len(s.train) for s in splits]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_rolling_window_caps_training_length(toy_panel, toy_factors):
    data = build_features(toy_panel, toy_factors, Config(min_assets_per_month=5))
    months = pd.PeriodIndex(data.index.get_level_values("month").unique(), freq="M")
    cfg = Config(min_assets_per_month=5)
    object.__setattr__(cfg.split, "min_train_years", 5)
    object.__setattr__(cfg.split, "validation_years", 2)
    object.__setattr__(cfg.split, "first_test_year", 2001)
    object.__setattr__(cfg.split, "last_test_year", 2004)

    rolling = expanding_splits(months, cfg, rolling_years=6)
    assert all(len(s.train) <= 6 * 12 for s in rolling)


def test_shuffled_labels_destroy_measured_skill(toy_panel, toy_factors):
    """Permuting the label inside each month must drive the IC to zero.

    If a bug let information flow from label to prediction, this control would
    still show skill.  It is the cheapest end-to-end guard available.
    """
    data = build_features(toy_panel, toy_factors, Config(min_assets_per_month=5))
    data = rank_normalise(data)
    target = make_target(data)

    honest = monthly_ic(-data["ret_1m"], target)
    assert honest.mean() > 0.05, "planted reversal should be detectable"

    rng = np.random.default_rng(7)
    shuffled = target.groupby("month").transform(
        lambda block: rng.permutation(block.to_numpy())
    )
    scrambled = monthly_ic(-data["ret_1m"], shuffled)
    # Mean IC on shuffled labels must be inside its own sampling error.
    stderr = scrambled.std() / np.sqrt(scrambled.notna().sum())
    assert abs(scrambled.mean()) < 3.0 * stderr


def test_using_future_features_would_show_up(toy_panel, toy_factors):
    """A deliberate leak must produce an obviously larger IC than the real run.

    Establishes that the accuracy measure is actually sensitive to leakage, so
    that a near-zero control result means something.
    """
    data = build_features(toy_panel, toy_factors, Config(min_assets_per_month=5))
    data = rank_normalise(data)
    target = make_target(data)

    legitimate = monthly_ic(-data["ret_1m"], target).mean()
    leaked = monthly_ic(target, target).mean()
    assert leaked > 0.9
    assert leaked > 5 * abs(legitimate)


def test_rank_normalisation_uses_only_within_month_information(toy_panel, toy_factors):
    """Changing one month's cross-section must not move any other month's ranks."""
    data = build_features(toy_panel, toy_factors, Config(min_assets_per_month=5))
    baseline = rank_normalise(data)

    tampered = data.copy()
    month = tampered.index.get_level_values("month").unique()[10]
    tampered.loc[month, "vol_12"] = tampered.loc[month, "vol_12"] * 100.0
    after = rank_normalise(tampered)

    other = baseline.index.get_level_values("month") != month
    pd.testing.assert_series_equal(
        baseline.loc[other, "vol_12"], after.loc[other, "vol_12"]
    )
