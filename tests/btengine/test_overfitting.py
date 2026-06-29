"""Tests for the overfitting-diagnostics module."""

import math

import numpy as np
import pytest

from bbstrader.btengine import overfitting as of


def test_psr_increases_with_sharpe():
    low = of.probabilistic_sharpe_ratio(0.05, n_obs=500)
    high = of.probabilistic_sharpe_ratio(0.20, n_obs=500)
    assert 0.0 <= low <= high <= 1.0


def test_psr_more_data_more_confidence():
    short = of.probabilistic_sharpe_ratio(0.1, n_obs=50)
    long = of.probabilistic_sharpe_ratio(0.1, n_obs=2000)
    assert long > short


def test_expected_max_sharpe_grows_with_trials():
    few = of.expected_max_sharpe(n_trials=5, sharpe_variance=0.01)
    many = of.expected_max_sharpe(n_trials=500, sharpe_variance=0.01)
    assert many > few > 0


def test_deflated_sharpe_below_undeflated():
    sr, n_obs, var = 0.15, 1000, 0.01
    psr = of.probabilistic_sharpe_ratio(sr, n_obs)
    dsr = of.deflated_sharpe_ratio(sr, n_obs, n_trials=100, sharpe_variance=var)
    # Deflating for 100 trials lowers confidence vs. the raw PSR(0).
    assert dsr < psr
    assert 0.0 <= dsr <= 1.0


def test_pbo_low_when_one_config_truly_dominant():
    rng = np.random.default_rng(0)
    T, N = 400, 10
    noise = rng.normal(0, 0.01, size=(T, N))
    # Config 0 has a genuine positive drift across all observations.
    noise[:, 0] += 0.01
    pbo = of.cscv_pbo(noise, n_splits=8)
    assert pbo < 0.4


def test_pbo_high_for_demeaned_noise():
    # Demeaning each column over the full sample forces IS gains to be OOS
    # losses the textbook overfitting scenario, so PBO should be high.
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, size=(400, 12))
    noise = noise - noise.mean(axis=0, keepdims=True)
    pbo = of.cscv_pbo(noise, n_splits=8)
    assert pbo > 0.6


def test_pbo_requires_even_splits():
    with pytest.raises(ValueError):
        of.cscv_pbo(np.zeros((100, 4)), n_splits=7)


def test_combinatorial_splits_count_and_disjoint():
    splits = list(of.combinatorial_splits(120, n_groups=6, n_test_groups=2))
    assert len(splits) == math.comb(6, 2)
    for train_idx, test_idx in splits:
        assert set(train_idx).isdisjoint(set(test_idx))


def test_combinatorial_splits_embargo_shrinks_train():
    no_embargo = next(of.combinatorial_splits(120, 6, 2, embargo=0))
    with_embargo = next(of.combinatorial_splits(120, 6, 2, embargo=5))
    assert len(with_embargo[0]) < len(no_embargo[0])
