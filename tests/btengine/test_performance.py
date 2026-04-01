import numpy as np
import pandas as pd
import pytest

from bbstrader.btengine.performance import (
    calculate_risk_metrics,
    create_calmar_ratio,
    create_drawdowns,
    create_omega_ratio,
    create_sharpe_ratio,
    create_sortino_ratio,
    create_tail_ratio,
)


@pytest.fixture
def sample_returns():
    dates = pd.date_range("2020-01-01", periods=100)
    returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
    return returns


@pytest.fixture
def sample_benchmark():
    dates = pd.date_range("2020-01-01", periods=100)
    returns = pd.Series(np.random.normal(0.0005, 0.015, 100), index=dates)
    return returns


def test_create_drawdowns(sample_returns):
    drawdown, max_dd, max_duration = create_drawdowns(sample_returns)
    assert isinstance(drawdown, pd.Series)
    assert isinstance(max_dd, float)
    assert isinstance(max_duration, (float, int))


def test_create_drawdowns_empty():
    empty_returns = pd.Series([], dtype=float)
    drawdown, max_dd, max_duration = create_drawdowns(empty_returns)
    assert drawdown.empty
    assert max_dd == 0.0
    assert max_duration == 0.0


def test_ratios(sample_returns):
    assert isinstance(create_sharpe_ratio(sample_returns), (float, np.float64))
    assert isinstance(create_sortino_ratio(sample_returns), (float, np.float64))
    assert isinstance(create_omega_ratio(sample_returns), (float, np.float64))
    assert isinstance(create_calmar_ratio(sample_returns), (float, np.float64))
    assert isinstance(create_tail_ratio(sample_returns), (float, np.float64))


def test_risk_metrics(sample_returns, sample_benchmark):
    metrics = calculate_risk_metrics(sample_returns, sample_benchmark)
    assert isinstance(metrics, dict)
    assert "alpha" in metrics
    assert "beta" in metrics
    assert "volatility" in metrics
