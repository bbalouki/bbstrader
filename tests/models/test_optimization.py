import pandas as pd
import numpy as np
import pytest
from bbstrader.models.optimization import (
    markowitz_weights,
    hierarchical_risk_parity,
    equal_weighted,
    optimized_weights,
    black_litterman_weights,
)


@pytest.fixture
def sample_prices():
    dates = pd.date_range("2020-01-01", periods=100)
    data = {
        "AAPL": np.linspace(100, 150, 100) + np.random.normal(0, 2, 100),
        "MSFT": np.linspace(200, 250, 100) + np.random.normal(0, 2, 100),
        "GOOG": np.linspace(1000, 1100, 100) + np.random.normal(0, 5, 100),
    }
    return pd.DataFrame(data, index=dates)


def test_markowitz_weights(sample_prices):
    weights = markowitz_weights(sample_prices)
    assert isinstance(weights, dict)
    assert len(weights) == 3
    assert np.isclose(sum(weights.values()), 1.0)


def test_markowitz_min_vol(sample_prices):
    weights = markowitz_weights(sample_prices, min_vol=True)
    assert isinstance(weights, dict)
    assert len(weights) == 3
    assert np.isclose(sum(weights.values()), 1.0)


def test_hrp_weights(sample_prices):
    weights = hierarchical_risk_parity(prices=sample_prices)
    assert isinstance(weights, dict)
    assert len(weights) == 3
    assert np.isclose(sum(weights.values()), 1.0)


def test_equal_weighted(sample_prices):
    weights = equal_weighted(prices=sample_prices)
    assert isinstance(weights, dict)
    assert len(weights) == 3
    assert all(np.isclose(w, 1 / 3) for w in weights.values())


def test_black_litterman_weights(sample_prices):
    views = {"AAPL": 0.05, "MSFT": 0.02}
    weights = black_litterman_weights(sample_prices, views=views)
    assert isinstance(weights, dict)
    assert len(weights) == 3
    assert np.isclose(sum(weights.values()), 1.0)


def test_optimized_weights_methods(sample_prices):
    methods = ["markowitz", "min_vol", "hrp", "equal", "black_litterman"]
    for method in methods:
        kwargs = {}
        if method == "black_litterman":
            kwargs = {"views": {"AAPL": 0.05}}
        weights = optimized_weights(prices=sample_prices, method=method, **kwargs)
        assert isinstance(weights, dict)
        assert np.isclose(sum(weights.values()), 1.0)
