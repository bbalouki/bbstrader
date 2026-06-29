"""Tests for opt-in per-bar funding/swap (overnight carry) costs.

A ``funding_model`` charges the carrying cost of every open position once per
bar. The default path (no model, or ``NoFunding``) leaves equity untouched and
is guarded by the golden equity-curve test elsewhere. A ``FixedRateFunding``
debits longs and credits shorts at an annual rate on notional.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from bbstrader.btengine.data import CSVDataHandler
from bbstrader.btengine.event import Events, MarketEvent
from bbstrader.btengine.execution import SimExecutionHandler
from bbstrader.btengine.friction import FixedRateFunding, NoFunding
from bbstrader.btengine.strategy import BacktestStrategy

N_BARS = 8
OPENS = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0]
CLOSES = [105.0, 115.0, 125.0, 135.0, 145.0, 155.0, 165.0, 175.0]
QUANTITY = 10

# Daily rate chosen so per-bar rate is exactly 0.001 (= 0.252 / 252).
ANNUAL_RATE = 0.252
PERIODS = 252
PER_BAR_RATE = ANNUAL_RATE / PERIODS


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


class _BuyAndHold(BacktestStrategy):
    """Opens a single long on the first bar and holds it to the end."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bar = 0

    def calculate_signals(self, event: MarketEvent) -> None:
        if event.type != Events.MARKET:
            return
        self._bar += 1
        if self._bar == 1:
            symbol = self.symbols[0]
            price = self.data.get_latest_bar_value(symbol, "close")
            dt = self.data.get_latest_bar_datetime(symbol)
            self.buy_mkt(1, symbol, price, QUANTITY, dtime=dt)


class _SellAndHold(BacktestStrategy):
    """Opens a single short on the first bar and holds it to the end."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bar = 0

    def calculate_signals(self, event: MarketEvent) -> None:
        if event.type != Events.MARKET:
            return
        self._bar += 1
        if self._bar == 1:
            symbol = self.symbols[0]
            price = self.data.get_latest_bar_value(symbol, "close")
            dt = self.data.get_latest_bar_datetime(symbol)
            self.sell_mkt(1, symbol, price, QUANTITY, dtime=dt)


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


def _final_total(engine) -> float:
    return float(engine.portfolio.equity_curve["Total"].iloc[-1])


def _expected_carry(engine) -> float:
    """The carry the funding model accrues over a run.

    Each bar's carry base is the position's mark-to-market value, which the
    portfolio records as the symbol's holdings column. Summing that column and
    scaling by the per-bar rate gives the total long carry (negative for a held
    short, since the column is negative).
    """
    market_value = engine.portfolio.equity_curve["TEST"].fillna(0.0)
    return PER_BAR_RATE * float(market_value.sum())


def test_no_funding_matches_default(tmp_path):
    _write_fixture(tmp_path)
    default_total = _final_total(_run(tmp_path, _BuyAndHold))
    _write_fixture(tmp_path)
    nofunding_total = _final_total(_run(tmp_path, _BuyAndHold, funding_model=NoFunding()))
    assert nofunding_total == pytest.approx(default_total)


def test_long_funding_debits_expected_carry(tmp_path):
    _write_fixture(tmp_path)
    default_engine = _run(tmp_path, _BuyAndHold)
    default_total = _final_total(default_engine)
    expected_carry = _expected_carry(default_engine)
    assert expected_carry > 0  # A held long has positive notional.
    _write_fixture(tmp_path)
    funded_total = _final_total(
        _run(
            tmp_path,
            _BuyAndHold,
            funding_model=FixedRateFunding(ANNUAL_RATE, periods=PERIODS),
        )
    )
    # A long pays carry, so equity is lower by exactly the accrued cost.
    assert funded_total == pytest.approx(default_total - expected_carry)


def test_short_funding_credits_opposite_sign(tmp_path):
    _write_fixture(tmp_path)
    default_engine = _run(tmp_path, _SellAndHold)
    default_total = _final_total(default_engine)
    expected_carry = _expected_carry(default_engine)
    assert expected_carry < 0  # A held short has negative notional.
    _write_fixture(tmp_path)
    funded_total = _final_total(
        _run(
            tmp_path,
            _SellAndHold,
            funding_model=FixedRateFunding(ANNUAL_RATE, periods=PERIODS),
        )
    )
    # A short earns the symmetric credit, so equity is higher by the carry.
    assert funded_total == pytest.approx(default_total - expected_carry)


def test_fixed_rate_funding_rejects_nonpositive_periods():
    with pytest.raises(ValueError):
        FixedRateFunding(0.05, periods=0)
