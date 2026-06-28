"""Tests for multi-timeframe resampling."""

import numpy as np
import pandas as pd
import pytest

from bbstrader.btengine.timeframe import MultiTimeFrame, resample_ohlcv


def _hourly_frame(days: int = 3) -> pd.DataFrame:
    # 24 hourly bars per day, deterministic ramp.
    idx = pd.date_range("2020-01-01", periods=days * 24, freq="h")
    close = np.arange(len(idx), dtype=float) + 1.0
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones(len(idx)),
        },
        index=idx,
    )


def test_resample_hourly_to_daily_aggregation():
    df = _hourly_frame(days=2)
    daily = resample_ohlcv(df, "D1", label="left", closed="left")
    assert len(daily) == 2
    first = daily.iloc[0]
    # Day 1 spans hourly closes 1..24.
    assert first["open"] == pytest.approx(1.0)
    assert first["close"] == pytest.approx(24.0)
    assert first["high"] == pytest.approx(24.5)
    assert first["low"] == pytest.approx(0.5)
    assert first["volume"] == pytest.approx(24.0)


def test_resample_accepts_timeframe_code_and_raw_rule():
    df = _hourly_frame(days=1)
    by_code = resample_ohlcv(df, "4h")
    by_rule = resample_ohlcv(df, "4h")
    assert len(by_code) == len(by_rule)


def test_resample_requires_datetime_index():
    df = pd.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(TypeError):
        resample_ohlcv(df, "D1")


class _StubData:
    """Minimal DataHandler-like object exposing get_latest_bars."""

    def __init__(self, frame):
        self._frame = frame

    def get_latest_bars(self, symbol, N=1, df=True):
        return self._frame.tail(N)


def test_multitimeframe_drops_partial_bucket():
    # 2 full days + 6 extra hours -> the 3rd (partial) daily bar is dropped.
    df = _hourly_frame(days=2)
    extra = _hourly_frame(days=1).iloc[:6]
    extra.index = pd.date_range("2020-01-03", periods=6, freq="h")
    frame = pd.concat([df, extra])
    mtf = MultiTimeFrame(_StubData(frame), lookback=1000)
    completed = mtf.htf_bars("X", "D1", drop_partial=True)
    with_partial = mtf.htf_bars("X", "D1", drop_partial=False)
    assert len(with_partial) == len(completed) + 1


def test_multitimeframe_htf_value():
    df = _hourly_frame(days=3)
    mtf = MultiTimeFrame(_StubData(df))
    val = mtf.htf_value("X", "D1", "close", drop_partial=True)
    # Last completed daily close among the first two full days is 48.
    assert val == pytest.approx(48.0)


def test_htf_value_none_when_insufficient():
    df = _hourly_frame(days=1).iloc[:3]
    mtf = MultiTimeFrame(_StubData(df))
    # Only a single (partial) bucket -> dropped -> nothing completed.
    assert mtf.htf_value("X", "D1") is None
