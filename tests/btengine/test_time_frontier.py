"""Tests for opt-in time-frontier (next-bar) fills.

With ``time_frontier=True`` an order generated from bar ``t`` must not fill at
bar ``t``; it fills at bar ``t+1`` at the configured price (open by default).
This removes same-bar look-ahead. The default path (no flag) is unchanged.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from bbstrader.btengine.backtest import run_backtest
from bbstrader.btengine.data import CSVDataHandler
from bbstrader.btengine.event import Events, MarketEvent
from bbstrader.btengine.execution import SimExecutionHandler
from bbstrader.btengine.strategy import BacktestStrategy

# Distinct open vs close so we can tell which price a fill used.
N_BARS = 8
OPENS = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0]
CLOSES = [105.0, 115.0, 125.0, 135.0, 145.0, 155.0, 165.0, 175.0]


def _write_fixture(csv_dir: Path) -> None:
    dates = pd.date_range("2020-01-02", periods=N_BARS, freq="B")
    df = pd.DataFrame(
        {
            "Datetime": dates,
            "Open": OPENS,
            "High": [c + 1 for c in CLOSES],
            "Low": [o - 1 for o in OPENS],
            "Close": CLOSES,
            "Adj Close": CLOSES,
            "Volume": [1000] * N_BARS,
        }
    )
    df.to_csv(csv_dir / "TEST.csv", index=False)


class _BuyOnFirstBar(BacktestStrategy):
    """Issues a single buy on the first bar; never trades again."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bar = 0
        self.fill_prices = []

    def calculate_signals(self, event: MarketEvent) -> None:
        if event.type != Events.MARKET:
            return
        self._bar += 1
        if self._bar == 1:
            symbol = self.symbols[0]
            price = self.data.get_latest_bar_value(symbol, "close")
            dt = self.data.get_latest_bar_datetime(symbol)
            self.buy_mkt(1, symbol, price, 10, dtime=dt)

    def update_trades_from_fill(self, event) -> None:
        super().update_trades_from_fill(event)
        self.fill_prices.append((event.fill_cost, event.quantity))


def _run(csv_dir, strategy, **kwargs):
    from bbstrader.btengine.backtest import BacktestEngine

    engine = BacktestEngine(
        ["TEST"],
        100000.0,
        0.0,
        datetime(2020, 1, 2),
        CSVDataHandler,
        SimExecutionHandler,
        strategy,
        csv_dir=str(csv_dir),
        print_stats=False,
        **kwargs,
    )
    engine._run_backtest()
    engine.portfolio.create_equity_curve_dataframe()
    return engine


def test_time_frontier_fills_at_next_bar_open(tmp_path):
    _write_fixture(tmp_path)
    engine = _run(tmp_path, _BuyOnFirstBar, time_frontier=True, fill_on="open")
    fills = engine.strategy.fill_prices
    assert len(fills) == 1
    fill_price, qty = fills[0]
    # Signal fired on bar 1 (close=105); fill must be at bar 2 open = 110.
    assert fill_price == pytest.approx(OPENS[1])
    assert qty == pytest.approx(10)


def test_default_path_unchanged_without_flag(tmp_path):
    _write_fixture(tmp_path)
    engine = _run(tmp_path, _BuyOnFirstBar)
    fills = engine.strategy.fill_prices
    # Default: same-bar fill, fill_cost left as None (portfolio uses bar price).
    assert len(fills) == 1
    assert fills[0][0] is None


def test_latency_fills_after_n_bars(tmp_path):
    _write_fixture(tmp_path)
    # Signal on bar 1; with latency=2 the fill lands on bar 3 (open=120).
    engine = _run(tmp_path, _BuyOnFirstBar, latency=2, fill_on="open")
    fills = engine.strategy.fill_prices
    assert len(fills) == 1
    assert fills[0][0] == pytest.approx(OPENS[2])


def test_time_frontier_changes_equity_vs_default(tmp_path):
    _write_fixture(tmp_path)
    default_engine = _run(tmp_path, _BuyOnFirstBar)
    _write_fixture(tmp_path)
    tf_engine = _run(tmp_path, _BuyOnFirstBar, time_frontier=True, fill_on="open")
    default_total = default_engine.portfolio.equity_curve["Total"].iloc[-1]
    tf_total = tf_engine.portfolio.equity_curve["Total"].iloc[-1]
    # Filling at the next open instead of the same-bar close changes the P&L.
    assert default_total != pytest.approx(tf_total)
