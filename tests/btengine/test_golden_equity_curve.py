"""Golden-master tests that lock the backtest engine's numeric output.

These run a real ``BacktestEngine`` end-to-end on a small deterministic fixture
(no mocking of the engine, per the project's testing conventions) so that the
replayable-data and preallocated-portfolio refactors -- and any future
vectorization -- cannot silently change results.
"""

from datetime import datetime
from pathlib import Path
from queue import Queue

import pandas as pd
import pytest

from bbstrader.btengine.backtest import BacktestEngine
from bbstrader.btengine.data import CSVDataHandler
from bbstrader.btengine.event import Events, MarketEvent
from bbstrader.btengine.execution import SimExecutionHandler
from bbstrader.btengine.strategy import BacktestStrategy

# Deterministic 6-bar close series; the strategy buys on bar 1 and exits on
# bar 4, giving a fully reproducible equity path.
CLOSES = [100.0, 101.0, 103.0, 102.0, 105.0, 104.0]

# Golden equity curve captured from the engine. The duplicated first/last rows
# come from the seed row and the trailing exhaustion MarketEvent, which the
# engine has always emitted; they are part of the locked behavior.
EXPECTED_TOTAL = [
    100000.0,
    100000.0,
    100008.7,
    100028.7,
    100018.7,
    100017.4,
    100017.4,
    100017.4,
]


def _write_fixture(csv_dir: Path) -> None:
    dates = pd.date_range("2023-01-02", periods=len(CLOSES), freq="B")
    df = pd.DataFrame(
        {
            "Datetime": dates,
            "Open": CLOSES,
            "High": [c + 1 for c in CLOSES],
            "Low": [c - 1 for c in CLOSES],
            "Close": CLOSES,
            "Adj Close": CLOSES,
            "Volume": [1000] * len(CLOSES),
        }
    )
    df.to_csv(csv_dir / "TEST.csv", index=False)


class _GoldenStrategy(BacktestStrategy):
    """Buys 10 shares on the first bar, exits on the fourth."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._bar = 0

    def calculate_signals(self, event: MarketEvent) -> None:
        if event.type != Events.MARKET:
            return
        self._bar += 1
        symbol = self.symbols[0]
        price = self.data.get_latest_bar_value(symbol, "close")
        dt = self.data.get_latest_bar_datetime(symbol)
        if self._bar == 1:
            self.buy_mkt(0, symbol, price, 10, dtime=dt)
        elif self._bar == 4:
            self.close_positions(0, symbol, price, 10, dtime=dt)


def _run(csv_dir: Path) -> pd.DataFrame:
    engine = BacktestEngine(
        ["TEST"],
        100000.0,
        0.0,
        datetime(2023, 1, 2),
        CSVDataHandler,
        SimExecutionHandler,
        _GoldenStrategy,
        csv_dir=str(csv_dir),
        print_stats=False,
    )
    engine._run_backtest()
    engine.portfolio.create_equity_curve_dataframe()
    return engine.portfolio.equity_curve


def test_golden_equity_curve(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    curve = _run(tmp_path)

    assert curve["Total"].tolist() == pytest.approx(EXPECTED_TOTAL)
    # Equity Curve is the cumulative product of returns; first value is NaN.
    expected_equity = [pytest.approx(t / EXPECTED_TOTAL[1]) for t in EXPECTED_TOTAL[1:]]
    assert curve["Equity Curve"].tolist()[1:] == expected_equity


def test_data_handler_is_replayable(tmp_path: Path) -> None:
    # The CSVDataHandler rewrites the fixture on load (adds returns/adj_close),
    # so build it once and replay the *same* handler via reset().
    _write_fixture(tmp_path)
    events: "Queue[MarketEvent]" = Queue()
    handler = CSVDataHandler(events, ["TEST"], csv_dir=str(tmp_path))

    def drain() -> list:
        seen = []
        while handler.continue_backtest:
            handler.update_bars()
            if handler.continue_backtest:
                dt = handler.get_latest_bar_datetime("TEST")
                close = handler.get_latest_bar_value("TEST", "close")
                seen.append((dt, close))
        return seen

    first_pass = drain()
    assert len(first_pass) == len(CLOSES)

    handler.reset()
    assert handler._cursor == 0
    assert handler.continue_backtest is True

    second_pass = drain()
    assert second_pass == first_pass
