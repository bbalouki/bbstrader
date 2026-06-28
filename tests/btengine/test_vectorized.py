"""Tests for the vectorized research fast-path backtester."""

import numpy as np
import pytest

from bbstrader.btengine.vectorized import vectorized_backtest


def test_buy_and_hold_matches_price_return():
    # Enter on bar 0, never exit -> equity tracks the price return.
    close = np.array([100.0, 110.0, 121.0, 133.1])
    entries = np.array([True, False, False, False])
    exits = np.zeros(4, dtype=bool)
    res = vectorized_backtest(close, entries, exits, init_cash=1000.0)
    # Held from bar 1 onward; total return == close[-1]/close[0] - 1.
    assert res.total_return == pytest.approx(close[-1] / close[0] - 1.0)
    assert res.num_trades == 1


def test_long_only_never_shorts():
    close = np.array([100.0, 90.0, 80.0, 70.0])
    entries = np.array([True, False, False, False])
    exits = np.array([False, True, False, False])
    res = vectorized_backtest(close, entries, exits)
    assert (res.position >= 0).all()
    # Position closed on bar 1, so it is flat for the decline afterward.
    assert res.position[2] == 0.0


def test_exit_stops_further_losses():
    close = np.array([100.0, 110.0, 90.0, 80.0])
    entries = np.array([True, False, False, False])
    exits = np.array([False, True, False, False])
    res = vectorized_backtest(close, entries, exits, init_cash=1000.0)
    # Captured the +10% to bar 1, then flat -> final equity 1100.
    assert res.equity[-1] == pytest.approx(1100.0)


def test_fees_reduce_return():
    close = np.array([100.0, 105.0, 110.0, 115.0])
    entries = np.array([True, False, False, True])
    exits = np.array([False, False, True, False])
    clean = vectorized_backtest(close, entries, exits, fees=0.0)
    charged = vectorized_backtest(close, entries, exits, fees=0.01)
    assert charged.total_return < clean.total_return


def test_short_side():
    close = np.array([100.0, 90.0, 80.0, 70.0])
    short_entries = np.array([True, False, False, False])
    short_exits = np.array([False, False, False, True])
    res = vectorized_backtest(
        close,
        entries=np.zeros(4, dtype=bool),
        exits=np.zeros(4, dtype=bool),
        short_entries=short_entries,
        short_exits=short_exits,
        allow_short=True,
        init_cash=1000.0,
    )
    # Short a falling market -> profit.
    assert res.total_return > 0
    assert (res.position <= 0).all()


def test_short_signals_require_allow_short():
    close = np.array([100.0, 90.0])
    with pytest.raises(ValueError):
        vectorized_backtest(
            close,
            entries=np.zeros(2, dtype=bool),
            exits=np.zeros(2, dtype=bool),
            short_entries=np.array([True, False]),
            short_exits=np.zeros(2, dtype=bool),
            allow_short=False,
        )


def test_metrics_and_frame():
    close = np.array([100.0, 102.0, 101.0, 105.0, 107.0])
    entries = np.array([True, False, False, False, False])
    exits = np.array([False, False, False, False, True])
    res = vectorized_backtest(close, entries, exits)
    summary = res.summary()
    assert set(summary) == {
        "total_return",
        "sharpe",
        "max_drawdown",
        "num_trades",
        "win_rate",
        "exposure",
    }
    assert 0.0 <= res.exposure <= 1.0
    df = res.to_frame()
    assert list(df.columns) == ["Position", "Returns", "Equity"]
    assert len(df) == len(close)


def test_signal_length_validation():
    with pytest.raises(ValueError):
        vectorized_backtest(
            np.array([1.0, 2.0, 3.0]), np.array([True]), np.array([False])
        )
