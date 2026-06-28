"""Tests for opt-in intrabar (OHLC) stop/limit evaluation.

With ``intrabar_fills=True`` a pending stop/limit triggers when the bar's
high/low reaches the order price, even if the close does not. The default
(close-only) behavior is preserved.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from bbstrader.btengine.backtest import BacktestEngine
from bbstrader.btengine.data import CSVDataHandler
from bbstrader.btengine.event import Events, MarketEvent
from bbstrader.btengine.execution import SimExecutionHandler
from bbstrader.btengine.strategy import BacktestStrategy

# Bar 2 spikes up to a high of 130 but closes back at 105 -- a buy stop at 120
# is touched intrabar but not on the close.
DATA = {
    "Datetime": pd.date_range("2020-01-02", periods=4, freq="B"),
    "Open": [100.0, 104.0, 104.0, 104.0],
    "High": [101.0, 130.0, 106.0, 106.0],
    "Low": [99.0, 103.0, 103.0, 103.0],
    "Close": [100.0, 105.0, 105.0, 105.0],
    "Adj Close": [100.0, 105.0, 105.0, 105.0],
    "Volume": [1000, 1000, 1000, 1000],
}


def _write_fixture(csv_dir: Path) -> None:
    pd.DataFrame(DATA).to_csv(csv_dir / "TEST.csv", index=False)


class _BuyStopStrategy(BacktestStrategy):
    """Places a buy-stop at 120 on the first bar and records fills."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._placed = False
        self.fills = []

    def calculate_signals(self, event: MarketEvent) -> None:
        if event.type != Events.MARKET:
            return
        symbol = self.symbols[0]
        dt = self.data.get_latest_bar_datetime(symbol)
        if not self._placed:
            self.buy_stop(1, symbol, 120.0, 10, dtime=dt)
            self._placed = True

    def update_trades_from_fill(self, event) -> None:
        super().update_trades_from_fill(event)
        self.fills.append(event.symbol)


def _run(csv_dir, **kwargs):
    engine = BacktestEngine(
        ["TEST"],
        100000.0,
        0.0,
        datetime(2020, 1, 2),
        CSVDataHandler,
        SimExecutionHandler,
        _BuyStopStrategy,
        csv_dir=str(csv_dir),
        print_stats=False,
        **kwargs,
    )
    engine._run_backtest()
    return engine


def test_intrabar_triggers_on_high(tmp_path):
    _write_fixture(tmp_path)
    engine = _run(tmp_path, intrabar_fills=True)
    # The bar-2 high (130) crosses the 120 stop -> order fills.
    assert len(engine.strategy.fills) == 1


def test_close_only_does_not_trigger(tmp_path):
    _write_fixture(tmp_path)
    engine = _run(tmp_path)  # default close-only
    # No close ever reaches 120, so the stop never fills.
    assert len(engine.strategy.fills) == 0
