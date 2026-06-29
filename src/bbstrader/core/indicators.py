"""Vectorized technical indicators.

A small, dependency-free indicator library built on NumPy so that it can be used
identically from backtest strategies (`BacktestStrategy`) and live strategies
(`LiveStrategy`). Every function operates on a 1-D array of prices (or OHLC
arrays) and returns an array of the **same length** as the input, left-padded
with ``NaN`` where there is not yet enough history. This lines up with the
output of ``BaseStrategy.get_asset_values(...)`` so an indicator value at index
``-1`` corresponds to the latest bar.

The implementations are plain NumPy (no third-party TA dependency), which keeps
the install lean and leaves the door open to JIT-compiling the hot paths later.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "sma",
    "ema",
    "wma",
    "rolling_std",
    "zscore",
    "roc",
    "rsi",
    "true_range",
    "atr",
    "bollinger_bands",
    "macd",
    "stochastic",
    "donchian",
]

ArrayLike = Union[Sequence[float], NDArray[np.float64]]


def _as_float_array(values: ArrayLike) -> NDArray[np.float64]:
    """Return ``values`` as a contiguous 1-D float64 array."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1-D array, got shape {arr.shape}.")
    return arr


def _check_window(window: int, name: str = "window") -> None:
    """Validate that a rolling-window length is a positive integer.

    Args:
        window (int): The window length to validate.
        name (str): The parameter name used in the error message.

    Raises:
        ValueError: If ``window`` is less than 1.
    """
    if window < 1:
        raise ValueError(f"{name} must be a positive integer, got {window}.")


def sma(values: ArrayLike, window: int) -> NDArray[np.float64]:
    """Simple moving average over ``window`` bars (NaN-padded)."""
    _check_window(window)
    arr = _as_float_array(values)
    out = np.full(arr.shape, np.nan)
    if arr.size < window:
        return out
    # Use a cumulative-sum sliding window for an O(n) average.
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    out[window - 1 :] = (cumsum[window:] - cumsum[:-window]) / window
    return out


def ema(values: ArrayLike, window: int) -> NDArray[np.float64]:
    """Exponential moving average with span ``window`` (NaN until seeded).

    The average is seeded with the SMA of the first ``window`` values, matching
    the common charting convention.
    """
    _check_window(window)
    arr = _as_float_array(values)
    out = np.full(arr.shape, np.nan)
    if arr.size < window:
        return out
    alpha = 2.0 / (window + 1.0)
    prev = float(arr[:window].mean())
    out[window - 1] = prev
    for i in range(window, arr.size):
        prev = alpha * arr[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


def wma(values: ArrayLike, window: int) -> NDArray[np.float64]:
    """Linearly weighted moving average (most recent bar weighted highest)."""
    _check_window(window)
    arr = _as_float_array(values)
    out = np.full(arr.shape, np.nan)
    if arr.size < window:
        return out
    weights = np.arange(1.0, window + 1.0)
    denom = weights.sum()
    for i in range(window - 1, arr.size):
        out[i] = np.dot(arr[i - window + 1 : i + 1], weights) / denom
    return out


def rolling_std(values: ArrayLike, window: int, ddof: int = 0) -> NDArray[np.float64]:
    """Rolling standard deviation over ``window`` bars (NaN-padded)."""
    _check_window(window)
    arr = _as_float_array(values)
    out = np.full(arr.shape, np.nan)
    if arr.size < window:
        return out
    for i in range(window - 1, arr.size):
        out[i] = arr[i - window + 1 : i + 1].std(ddof=ddof)
    return out


def zscore(values: ArrayLike, window: int) -> NDArray[np.float64]:
    """Rolling z-score: ``(price - rolling_mean) / rolling_std``.

    Bars where the rolling standard deviation is zero yield ``NaN`` to avoid a
    divide-by-zero.
    """
    _check_window(window)
    arr = _as_float_array(values)
    mean = sma(arr, window)
    std = rolling_std(arr, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(std > 0, (arr - mean) / std, np.nan)
    return out


def roc(values: ArrayLike, window: int) -> NDArray[np.float64]:
    """Rate of change in percent over ``window`` bars."""
    _check_window(window)
    arr = _as_float_array(values)
    out = np.full(arr.shape, np.nan)
    if arr.size <= window:
        return out
    prior = arr[:-window]
    with np.errstate(invalid="ignore", divide="ignore"):
        out[window:] = np.where(
            prior != 0, (arr[window:] - prior) / prior * 100.0, np.nan
        )
    return out


def rsi(values: ArrayLike, window: int = 14) -> NDArray[np.float64]:
    """Wilder's Relative Strength Index over ``window`` bars (NaN-padded)."""
    _check_window(window)
    arr = _as_float_array(values)
    out = np.full(arr.shape, np.nan)
    if arr.size <= window:
        return out
    delta = np.diff(arr)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    # Wilder's smoothing: seed with the simple average of the first window.
    avg_gain = gains[:window].mean()
    avg_loss = losses[:window].mean()

    def _rsi_from(avg_gain: float, avg_loss: float) -> float:
        """Return the RSI value for a smoothed average gain and loss.

        Args:
            avg_gain (float): The smoothed average up-move over the window.
            avg_loss (float): The smoothed average down-move over the window.

        Returns:
            float: The RSI in [0, 100]; 100 when there are no losses.
        """
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    out[window] = _rsi_from(avg_gain, avg_loss)
    for i in range(window + 1, arr.size):
        avg_gain = (avg_gain * (window - 1) + gains[i - 1]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i - 1]) / window
        out[i] = _rsi_from(avg_gain, avg_loss)
    return out


def true_range(
    high: ArrayLike, low: ArrayLike, close: ArrayLike
) -> NDArray[np.float64]:
    """True range: ``max(high-low, |high-prev_close|, |low-prev_close|)``."""
    h = _as_float_array(high)
    low_arr = _as_float_array(low)
    c = _as_float_array(close)
    if not (h.size == low_arr.size == c.size):
        raise ValueError("high, low and close must have the same length.")
    out = np.full(h.shape, np.nan)
    if h.size == 0:
        return out
    out[0] = h[0] - low_arr[0]
    prev_close = c[:-1]
    hl = h[1:] - low_arr[1:]
    hc = np.abs(h[1:] - prev_close)
    lc = np.abs(low_arr[1:] - prev_close)
    out[1:] = np.maximum.reduce([hl, hc, lc])
    return out


def atr(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, window: int = 14
) -> NDArray[np.float64]:
    """Average True Range using Wilder's smoothing (NaN-padded)."""
    _check_window(window)
    tr = true_range(high, low, close)
    out = np.full(tr.shape, np.nan)
    if tr.size < window:
        return out
    prev = float(np.nanmean(tr[:window]))
    out[window - 1] = prev
    for i in range(window, tr.size):
        prev = (prev * (window - 1) + tr[i]) / window
        out[i] = prev
    return out


def bollinger_bands(
    values: ArrayLike, window: int = 20, num_std: float = 2.0
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Bollinger Bands; returns ``(lower, middle, upper)`` arrays."""
    _check_window(window)
    arr = _as_float_array(values)
    middle = sma(arr, window)
    std = rolling_std(arr, window)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return lower, middle, upper


def macd(
    values: ArrayLike,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """MACD; returns ``(macd_line, signal_line, histogram)`` arrays."""
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be smaller than slow ({slow}).")
    arr = _as_float_array(values)
    macd_line = ema(arr, fast) - ema(arr, slow)
    # The signal line is an EMA of the MACD line over its valid (non-NaN) tail.
    signal_line = np.full(arr.shape, np.nan)
    valid = ~np.isnan(macd_line)
    if valid.any():
        start = int(np.argmax(valid))
        tail = ema(macd_line[start:], signal)
        signal_line[start:] = tail
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    k_window: int = 14,
    d_window: int = 3,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Stochastic oscillator; returns ``(%K, %D)`` arrays."""
    _check_window(k_window, "k_window")
    _check_window(d_window, "d_window")
    h = _as_float_array(high)
    low_arr = _as_float_array(low)
    c = _as_float_array(close)
    if not (h.size == low_arr.size == c.size):
        raise ValueError("high, low and close must have the same length.")
    percent_k = np.full(c.shape, np.nan)
    for i in range(k_window - 1, c.size):
        window_high = h[i - k_window + 1 : i + 1].max()
        window_low = low_arr[i - k_window + 1 : i + 1].min()
        span = window_high - window_low
        if span > 0:
            percent_k[i] = (c[i] - window_low) / span * 100.0
    percent_d = sma(percent_k, d_window)
    return percent_k, percent_d


def donchian(
    high: ArrayLike, low: ArrayLike, window: int = 20
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Donchian channel; returns ``(lower, upper)`` arrays over ``window`` bars.

    The channel at bar ``i`` uses bars ``[i-window+1, i]`` (inclusive), so it is
    safe to compare the *previous* bar's channel against the current price for a
    breakout without look-ahead.
    """
    _check_window(window)
    h = _as_float_array(high)
    low_arr = _as_float_array(low)
    if h.size != low_arr.size:
        raise ValueError("high and low must have the same length.")
    upper = np.full(h.shape, np.nan)
    lower = np.full(h.shape, np.nan)
    for i in range(window - 1, h.size):
        upper[i] = h[i - window + 1 : i + 1].max()
        lower[i] = low_arr[i - window + 1 : i + 1].min()
    return lower, upper
