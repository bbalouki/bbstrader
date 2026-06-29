"""Tests for multi-strategy simulation sharing one portfolio and clock."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from bbstrader.btengine.backtest import BacktestEngine
from bbstrader.btengine.data import CSVDataHandler
from bbstrader.btengine.event import Events, MarketEvent
from bbstrader.btengine.execution import SimExecutionHandler
from bbstrader.btengine.strategy import BacktestStrategy, MultiStrategy

N_BARS = 6


def _write_symbol(csv_dir: Path, name: str, base: float) -> None:
    dates = pd.date_range("2020-01-02", periods=N_BARS, freq="B")
    closes = [base + i for i in range(N_BARS)]
    pd.DataFrame(
        {
            "Datetime": dates,
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Adj Close": closes,
            "Volume": [1000] * N_BARS,
        }
    ).to_csv(csv_dir / f"{name}.csv", index=False)


class _BuyOneSymbol(BacktestStrategy):
    """Buys a single configured symbol on the first bar and holds."""

    TARGET = None  # overridden per subclass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._done = False

    def calculate_signals(self, event: MarketEvent) -> None:
        if event.type != Events.MARKET or self._done:
            return
        symbol = self.TARGET
        price = self.data.get_latest_bar_value(symbol, "close")
        dt = self.data.get_latest_bar_datetime(symbol)
        self.buy_mkt(1, symbol, price, 10, dtime=dt)
        self._done = True


class _BuyAAA(_BuyOneSymbol):
    TARGET = "AAA"


class _BuyBBB(_BuyOneSymbol):
    TARGET = "BBB"


def _build_engine(csv_dir):
    _write_symbol(csv_dir, "AAA", 100.0)
    _write_symbol(csv_dir, "BBB", 200.0)
    return BacktestEngine(
        ["AAA", "BBB"],
        100000.0,
        0.0,
        datetime(2020, 1, 2),
        CSVDataHandler,
        SimExecutionHandler,
        [_BuyAAA, _BuyBBB],
        csv_dir=str(csv_dir),
        print_stats=False,
    )


def test_engine_wraps_multiple_strategies(tmp_path):
    engine = _build_engine(tmp_path)
    assert len(engine.strategies) == 2
    assert isinstance(engine.strategy, MultiStrategy)


def test_both_strategies_trade_into_shared_portfolio(tmp_path):
    engine = _build_engine(tmp_path)
    engine._run_backtest()
    # Both children bought into the single shared portfolio.
    assert engine.portfolio.current_positions["AAA"] == pytest.approx(10)
    assert engine.portfolio.current_positions["BBB"] == pytest.approx(10)


def test_single_strategy_not_wrapped(tmp_path):
    _write_symbol(tmp_path, "AAA", 100.0)
    engine = BacktestEngine(
        ["AAA"],
        100000.0,
        0.0,
        datetime(2020, 1, 2),
        CSVDataHandler,
        SimExecutionHandler,
        _BuyAAA,
        csv_dir=str(tmp_path),
        print_stats=False,
    )
    assert not isinstance(engine.strategy, MultiStrategy)
    assert len(engine.strategies) == 1
