"""Smoke + determinism tests for the strategy templates.

Each template is run end-to-end through a real ``run_backtest`` on a synthetic,
deterministic OHLCV dataset (real implementations, no mocking of the engine).
We assert the run completes and yields a well-formed, reproducible equity curve.
"""

import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from bbstrader.btengine.backtest import run_backtest
from bbstrader.btengine.data import CSVDataHandler
from bbstrader.btengine.execution import SimExecutionHandler
from bbstrader.btengine.templates import (
    DonchianBreakoutStrategy,
    RSIMeanReversionStrategy,
    SMACrossoverStrategy,
)

N_BARS = 120


def _write_fixture(csv_dir: Path, symbol: str = "TEST") -> None:
    # Deterministic trend + oscillation so every template sees entries and exits.
    dates = pd.date_range("2020-01-01", periods=N_BARS, freq="B")
    closes = [100.0 + 0.2 * i + 5.0 * math.sin(i / 6.0) for i in range(N_BARS)]
    df = pd.DataFrame(
        {
            "Datetime": dates,
            "Open": closes,
            "High": [c + 1.0 for c in closes],
            "Low": [c - 1.0 for c in closes],
            "Close": closes,
            "Adj Close": closes,
            "Volume": [1000] * N_BARS,
        }
    )
    df.to_csv(csv_dir / f"{symbol}.csv", index=False)


def _run(csv_dir: Path, strategy, **kwargs) -> pd.DataFrame:
    return run_backtest(
        ["TEST"],
        datetime(2020, 1, 1),
        CSVDataHandler,
        strategy,
        SimExecutionHandler,
        csv_dir=str(csv_dir),
        print_stats=False,
        **kwargs,
    )


TEMPLATE_PARAMS = [
    (SMACrossoverStrategy, {"fast": 5, "slow": 15, "quantity": 10}),
    (
        RSIMeanReversionStrategy,
        {"period": 10, "oversold": 35, "exit_level": 55, "quantity": 10},
    ),
    (DonchianBreakoutStrategy, {"window": 10, "quantity": 10}),
]


@pytest.mark.parametrize("strategy_cls,params", TEMPLATE_PARAMS)
def test_template_produces_wellformed_equity_curve(tmp_path, strategy_cls, params):
    _write_fixture(tmp_path)
    curve = _run(tmp_path, strategy_cls, **params)

    assert isinstance(curve, pd.DataFrame)
    assert {"Total", "Returns", "Equity Curve"}.issubset(curve.columns)
    assert len(curve) >= N_BARS
    # Equity must stay finite and start at the initial capital.
    assert curve["Total"].notna().all()
    assert curve["Total"].iloc[0] == pytest.approx(100000.0)


@pytest.mark.parametrize("strategy_cls,params", TEMPLATE_PARAMS)
def test_template_is_deterministic(tmp_path, strategy_cls, params):
    _write_fixture(tmp_path)
    first = _run(tmp_path, strategy_cls, **params)["Total"].tolist()
    _write_fixture(tmp_path)
    second = _run(tmp_path, strategy_cls, **params)["Total"].tolist()
    assert first == pytest.approx(second)


def test_sma_crossover_rejects_bad_windows(tmp_path):
    _write_fixture(tmp_path)
    with pytest.raises(ValueError):
        _run(tmp_path, SMACrossoverStrategy, fast=20, slow=10)
