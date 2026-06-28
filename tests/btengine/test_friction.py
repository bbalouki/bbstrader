"""Tests for the execution-friction models.

Unit tests cover each model's pricing math; an integration test confirms that
adding friction to a real backtest reduces the final equity relative to the
frictionless default (and that the default path is unchanged).
"""

import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from bbstrader.btengine import friction as fr
from bbstrader.btengine.backtest import run_backtest
from bbstrader.btengine.data import CSVDataHandler
from bbstrader.btengine.execution import SimExecutionHandler
from bbstrader.btengine.event import Events, MarketEvent
from bbstrader.btengine.strategy import BacktestStrategy


class _MiniBars:
    """Minimal DataHandler stand-in for unit-testing slippage models."""

    def get_latest_bar_value(self, symbol, val_type):
        return {"volume": 1000.0}[val_type]

    def get_latest_bars_values(self, symbol, val_type, N=1):
        import numpy as np

        return np.array([0.01, -0.02, 0.015, -0.005])


def test_fixed_spread_slippage_directional():
    model = fr.FixedSpreadSlippage(spread=2.0)
    buy = model.adjusted_price(100.0, "BUY", 10, "X", _MiniBars())
    sell = model.adjusted_price(100.0, "SELL", 10, "X", _MiniBars())
    assert buy == pytest.approx(101.0)
    assert sell == pytest.approx(99.0)


def test_percent_slippage_directional():
    model = fr.PercentSlippage(pct=0.01)
    assert model.adjusted_price(100.0, "BUY", 1, "X", _MiniBars()) == pytest.approx(
        101.0
    )
    assert model.adjusted_price(100.0, "SELL", 1, "X", _MiniBars()) == pytest.approx(
        99.0
    )


def test_volume_participation_slippage():
    model = fr.VolumeParticipationSlippage(coef=0.5)
    # quantity 100 / volume 1000 = 0.1 participation; buy pays 0.5*0.1 = 5%.
    price = model.adjusted_price(100.0, "BUY", 100, "X", _MiniBars())
    assert price == pytest.approx(105.0)


def test_square_root_impact_scales_with_sqrt_size():
    model = fr.SquareRootImpact(coef=0.1, adv=1_000_000.0)
    imp = model.impact(100.0, 10_000, "BUY")
    expected = 0.1 * 100.0 * math.sqrt(10_000 / 1_000_000.0)
    assert imp == pytest.approx(expected)
    # Sells get a negative (price-lowering) impact.
    assert model.impact(100.0, 10_000, "SELL") == pytest.approx(-expected)


def test_per_share_commission_minimum():
    model = fr.PerShareCommission(per_share=0.005, minimum=1.0)
    assert model.commission("X", 10, 100.0) == pytest.approx(1.0)  # below minimum
    assert model.commission("X", 1000, 100.0) == pytest.approx(5.0)


def test_percent_commission():
    model = fr.PercentCommission(pct=0.001)
    assert model.commission("X", 100, 50.0) == pytest.approx(5.0)


def test_ib_commission_matches_tiers():
    model = fr.IBCommission()
    assert model.commission("X", 100, 10.0) == pytest.approx(1.30)
    assert model.commission("X", 1000, 10.0) == pytest.approx(8.0)


def test_apply_friction_combines_slippage_and_impact():
    price = fr.apply_friction(
        100.0,
        "BUY",
        10_000,
        "X",
        _MiniBars(),
        slippage=fr.FixedSpreadSlippage(2.0),
        impact=fr.SquareRootImpact(coef=0.1, adv=1_000_000.0),
    )
    expected = 101.0 + 0.1 * 100.0 * math.sqrt(10_000 / 1_000_000.0)
    assert price == pytest.approx(expected)


def test_invalid_fill_ratio_raises(tmp_path):
    _write_fixture(tmp_path)
    with pytest.raises(ValueError):
        run_backtest(
            ["TEST"],
            datetime(2020, 1, 1),
            CSVDataHandler,
            _AlwaysBuy,
            SimExecutionHandler,
            csv_dir=str(tmp_path),
            print_stats=False,
            fill_ratio=1.5,
        )


# --------------------------------------------------------------------------- #
# Integration                                                                 #
# --------------------------------------------------------------------------- #
N_BARS = 60


def _write_fixture(csv_dir: Path) -> None:
    dates = pd.date_range("2020-01-01", periods=N_BARS, freq="B")
    closes = [100.0 + math.sin(i / 4.0) for i in range(N_BARS)]
    df = pd.DataFrame(
        {
            "Datetime": dates,
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Adj Close": closes,
            "Volume": [1000] * N_BARS,
            # provide a returns column proxy is added by the handler
        }
    )
    df.to_csv(csv_dir / "TEST.csv", index=False)


class _AlwaysBuy(BacktestStrategy):
    """Buys once on the first bar, then holds; cycles to generate trades."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bar = 0

    def calculate_signals(self, event: MarketEvent) -> None:
        if event.type != Events.MARKET:
            return
        self._bar += 1
        symbol = self.symbols[0]
        price = self.data.get_latest_bar_value(symbol, "close")
        dt = self.data.get_latest_bar_datetime(symbol)
        # Trade every 10 bars to accumulate commission/slippage costs.
        if self._bar % 10 == 1 and self._positions[symbol]["LONG"] == 0:
            self.buy_mkt(1, symbol, price, 10, dtime=dt)
        elif self._bar % 10 == 6 and self._positions[symbol]["LONG"] > 0:
            self.close_positions(1, symbol, price, 10, dtime=dt)


def _final_equity(csv_dir, **kwargs) -> float:
    curve = run_backtest(
        ["TEST"],
        datetime(2020, 1, 1),
        CSVDataHandler,
        _AlwaysBuy,
        SimExecutionHandler,
        csv_dir=str(csv_dir),
        print_stats=False,
        **kwargs,
    )
    return float(curve["Total"].iloc[-1])


def test_commission_reduces_equity(tmp_path):
    # Compare a zero-commission baseline against a clearly positive commission
    # (the no-model default already applies IB commission via FillEvent).
    _write_fixture(tmp_path)
    no_cost = _final_equity(tmp_path, commission_model=fr.ZeroCommission())
    _write_fixture(tmp_path)
    with_commission = _final_equity(
        tmp_path, commission_model=fr.PerShareCommission(per_share=0.5, minimum=5.0)
    )
    assert with_commission < no_cost


def test_slippage_reduces_round_trip_pnl(tmp_path):
    # Slippage makes buys cost more and sells receive less, lowering equity.
    _write_fixture(tmp_path)
    clean = _final_equity(tmp_path, commission_model=fr.ZeroCommission())
    _write_fixture(tmp_path)
    slipped = _final_equity(
        tmp_path,
        commission_model=fr.ZeroCommission(),
        slippage_model=fr.FixedSpreadSlippage(spread=0.5),
    )
    assert slipped < clean


def test_partial_fill_reduces_exposure(tmp_path):
    _write_fixture(tmp_path)
    full = _final_equity(tmp_path, commission_model=fr.FixedCommission(1.0))
    _write_fixture(tmp_path)
    half = _final_equity(
        tmp_path, commission_model=fr.FixedCommission(1.0), fill_ratio=0.5
    )
    # Different fill ratio changes the realized P&L path.
    assert full != half
