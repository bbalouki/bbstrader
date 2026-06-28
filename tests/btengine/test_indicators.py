"""Tests for the vectorized indicator library.

Indicators are checked against hand-computed or pandas-reference values on
deterministic inputs. Every indicator must return an array the same length as
its input, NaN-padded where there is insufficient history.
"""

import numpy as np
import pandas as pd
import pytest

from bbstrader.core import indicators as ind

PRICES = np.array(
    [10.0, 11.0, 12.0, 11.0, 10.0, 12.0, 13.0, 14.0, 13.0, 15.0],
    dtype=float,
)


def test_sma_matches_pandas_rolling_mean():
    out = ind.sma(PRICES, 3)
    ref = pd.Series(PRICES).rolling(3).mean().to_numpy()
    assert out.shape == PRICES.shape
    np.testing.assert_allclose(out, ref, equal_nan=True)


def test_sma_nan_padding_length():
    out = ind.sma(PRICES, 4)
    assert np.isnan(out[:3]).all()
    assert not np.isnan(out[3:]).any()


def test_window_larger_than_series_is_all_nan():
    out = ind.sma(PRICES, len(PRICES) + 5)
    assert out.shape == PRICES.shape
    assert np.isnan(out).all()


def test_ema_seeded_with_sma_and_recurses():
    window = 3
    out = ind.ema(PRICES, window)
    alpha = 2.0 / (window + 1.0)
    expected_seed = PRICES[:window].mean()
    assert out[window - 1] == pytest.approx(expected_seed)
    expected_next = alpha * PRICES[window] + (1 - alpha) * expected_seed
    assert out[window] == pytest.approx(expected_next)


def test_wma_weights_recent_bar_highest():
    window = 3
    out = ind.wma(PRICES, window)
    weights = np.array([1.0, 2.0, 3.0])
    expected = np.dot(PRICES[:3], weights) / weights.sum()
    assert out[2] == pytest.approx(expected)


def test_rolling_std_matches_pandas():
    out = ind.rolling_std(PRICES, 4, ddof=0)
    ref = pd.Series(PRICES).rolling(4).std(ddof=0).to_numpy()
    np.testing.assert_allclose(out, ref, equal_nan=True)


def test_zscore_zero_variance_is_nan():
    flat = np.array([5.0, 5.0, 5.0, 5.0], dtype=float)
    out = ind.zscore(flat, 3)
    assert np.isnan(out[2:]).all()


def test_rsi_all_gains_is_100():
    rising = np.arange(1.0, 20.0)
    out = ind.rsi(rising, 14)
    assert out[-1] == pytest.approx(100.0)


def test_rsi_bounded_between_0_and_100():
    out = ind.rsi(PRICES, 3)
    valid = out[~np.isnan(out)]
    assert ((valid >= 0) & (valid <= 100)).all()


def test_true_range_first_bar_is_high_minus_low():
    high = np.array([11.0, 12.0, 13.0])
    low = np.array([9.0, 10.0, 11.0])
    close = np.array([10.0, 11.0, 12.0])
    tr = ind.true_range(high, low, close)
    assert tr[0] == pytest.approx(2.0)
    # bar 1: max(12-10, |12-10|, |10-10|) = 2
    assert tr[1] == pytest.approx(2.0)


def test_atr_length_and_padding():
    high = PRICES + 1
    low = PRICES - 1
    out = ind.atr(high, low, PRICES, 3)
    assert out.shape == PRICES.shape
    assert np.isnan(out[:2]).all()
    assert not np.isnan(out[2:]).any()


def test_bollinger_bands_ordering():
    lower, middle, upper = ind.bollinger_bands(PRICES, 3, 2.0)
    valid = ~np.isnan(middle)
    assert (lower[valid] <= middle[valid]).all()
    assert (middle[valid] <= upper[valid]).all()


def test_macd_histogram_is_difference():
    macd_line, signal_line, hist = ind.macd(PRICES, fast=2, slow=4, signal=2)
    valid = ~np.isnan(hist)
    np.testing.assert_allclose(
        hist[valid], (macd_line - signal_line)[valid], equal_nan=False
    )


def test_stochastic_bounded():
    high = PRICES + 1
    low = PRICES - 1
    k, d = ind.stochastic(high, low, PRICES, 3, 2)
    kvalid = k[~np.isnan(k)]
    assert ((kvalid >= 0) & (kvalid <= 100)).all()


def test_donchian_channel_contains_prices():
    high = PRICES + 1
    low = PRICES - 1
    lower, upper = ind.donchian(high, low, 3)
    valid = ~np.isnan(upper)
    assert (upper[valid] >= high[valid] - 1).all()
    assert (lower[valid] <= low[valid] + 1).all()


def test_invalid_window_raises():
    with pytest.raises(ValueError):
        ind.sma(PRICES, 0)


def test_macd_fast_must_be_less_than_slow():
    with pytest.raises(ValueError):
        ind.macd(PRICES, fast=26, slow=12)
