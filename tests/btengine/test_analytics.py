"""Tests for the risk-analytics module (VaR/CVaR, Monte Carlo, regimes, factors)."""

import numpy as np
import pandas as pd
import pytest

from bbstrader.btengine import analytics as an


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)
    return rng.normal(0.0005, 0.01, size=1000)


def test_historical_var_is_positive_loss(returns):
    var = an.historical_var(returns, level=0.95)
    assert var > 0
    # ~5% of observations should be worse than -VaR.
    frac = np.mean(returns < -var)
    assert 0.03 < frac < 0.07


def test_cvar_exceeds_var(returns):
    var = an.historical_var(returns, level=0.95)
    cvar = an.historical_cvar(returns, level=0.95)
    # Expected shortfall is at least as large as VaR.
    assert cvar >= var


def test_parametric_matches_historical_for_normal(returns):
    hv = an.historical_var(returns, 0.95)
    pv = an.parametric_var(returns, 0.95)
    # For approximately normal returns the two are close.
    assert pv == pytest.approx(hv, rel=0.25)


def test_parametric_cvar_positive(returns):
    assert an.parametric_cvar(returns, 0.95) > 0


def test_monte_carlo_is_deterministic_and_banded(returns):
    a = an.monte_carlo_bootstrap(returns, n_sims=500, horizon=50, seed=7)
    b = an.monte_carlo_bootstrap(returns, n_sims=500, horizon=50, seed=7)
    assert np.allclose(a.terminal_returns, b.terminal_returns)
    # Bands are ordered q5 <= q50 <= q95 at the final step.
    assert a.bands["q5"][-1] <= a.bands["q50"][-1] <= a.bands["q95"][-1]
    assert a.horizon == 50
    assert 0.0 <= a.prob_loss <= 1.0


def test_cusum_detects_mean_shift():
    series = np.concatenate([np.zeros(50), np.ones(50) * 5.0])
    points = an.cusum_change_points(series, threshold=1.0)
    assert points.size > 0
    # The first detected change is near the shift at index 50.
    assert any(45 <= p <= 60 for p in points)


def test_volatility_regimes_labels_calm_and_turbulent():
    calm = np.random.default_rng(1).normal(0, 0.001, 100)
    wild = np.random.default_rng(2).normal(0, 0.05, 100)
    series = np.concatenate([calm, wild])
    labels = an.volatility_regimes(series, window=10, n_states=2)
    assert set(np.unique(labels)).issubset({0, 1})
    # The turbulent second half should average a higher regime label.
    assert labels[120:].mean() > labels[:80].mean()


def test_factor_exposure_recovers_known_beta():
    rng = np.random.default_rng(0)
    market = rng.normal(0, 0.01, 500)
    true_alpha, true_beta = 0.0002, 1.5
    asset = true_alpha + true_beta * market + rng.normal(0, 0.001, 500)
    res = an.factor_exposure(asset, market)
    assert res["beta_factor_0"] == pytest.approx(true_beta, rel=0.1)
    assert res["alpha"] == pytest.approx(true_alpha, abs=0.001)
    assert res["r_squared"] > 0.9


def test_factor_exposure_dataframe_names():
    rng = np.random.default_rng(0)
    factors = pd.DataFrame(
        {"mkt": rng.normal(0, 0.01, 300), "size": rng.normal(0, 0.01, 300)}
    )
    asset = 1.0 * factors["mkt"] + 0.5 * factors["size"]
    res = an.factor_exposure(asset.to_numpy(), factors)
    assert "beta_mkt" in res and "beta_size" in res


def test_rolling_beta_shape():
    rng = np.random.default_rng(0)
    market = rng.normal(0, 0.01, 200)
    asset = 1.2 * market
    rb = an.rolling_beta(asset, market, window=60)
    assert len(rb) == 200
    assert np.isnan(rb[0])  # warm-up
    assert rb[-1] == pytest.approx(1.2, rel=0.05)
