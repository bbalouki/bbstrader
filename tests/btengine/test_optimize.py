"""Tests for parameter optimization and walk-forward validation.

These run real backtests over a synthetic deterministic dataset. The in-process
path (``n_jobs=1``) is asserted for determinism; a small ``n_jobs=2`` run
exercises the process pool.
"""

import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from bbstrader.btengine.data import CSVDataHandler
from bbstrader.btengine.optimize import expand_param_grid, optimize, walk_forward
from bbstrader.btengine.templates import SMACrossoverStrategy

N_BARS = 160


def _write_fixture(csv_dir: Path) -> None:
    dates = pd.date_range("2020-01-01", periods=N_BARS, freq="B")
    closes = [100.0 + 0.15 * i + 6.0 * math.sin(i / 5.0) for i in range(N_BARS)]
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
    df.to_csv(csv_dir / "TEST.csv", index=False)


PARAM_GRID = {"fast": [5, 10], "slow": [20, 30], "quantity": [10]}


def _optimize(csv_dir: Path, **over):
    kwargs = dict(
        symbol_list=["TEST"],
        start_date=datetime(2020, 1, 1),
        data_handler=CSVDataHandler,
        strategy=SMACrossoverStrategy,
        param_grid=PARAM_GRID,
        metric="sharpe",
        csv_dir=str(csv_dir),
        print_stats=False,
    )
    kwargs.update(over)
    return optimize(**kwargs)


def test_expand_param_grid_cartesian_product():
    combos = expand_param_grid({"a": [1, 2], "b": [3, 4]})
    assert len(combos) == 4
    assert {"a": 1, "b": 3} in combos


def test_expand_param_grid_random_is_seeded():
    grid = {"a": list(range(10)), "b": list(range(10))}
    first = expand_param_grid(grid, search="random", n_iter=5, seed=42)
    second = expand_param_grid(grid, search="random", n_iter=5, seed=42)
    assert first == second
    assert len(first) == 5


def test_optimize_returns_ranked_table(tmp_path):
    _write_fixture(tmp_path)
    results = _optimize(tmp_path)
    # One row per parameter combination (2 fast x 2 slow x 1 quantity).
    assert len(results) == 4
    for col in ["fast", "slow", "quantity", "sharpe", "total_return", "max_drawdown"]:
        assert col in results.columns
    # Ranked best-first by sharpe (descending, NaNs last).
    sharpe = results["sharpe"].dropna()
    assert sharpe.is_monotonic_decreasing


def test_optimize_is_deterministic(tmp_path):
    _write_fixture(tmp_path)
    first = _optimize(tmp_path)
    second = _optimize(tmp_path)
    pd.testing.assert_frame_equal(first, second)


def test_optimize_parallel_matches_serial(tmp_path):
    _write_fixture(tmp_path)
    serial = (
        _optimize(tmp_path, n_jobs=1)
        .sort_values(["fast", "slow"])
        .reset_index(drop=True)
    )
    parallel = (
        _optimize(tmp_path, n_jobs=2)
        .sort_values(["fast", "slow"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(serial, parallel)


def test_optimize_unknown_metric_raises(tmp_path):
    _write_fixture(tmp_path)
    with pytest.raises(ValueError):
        _optimize(tmp_path, metric="nonexistent")


def test_walk_forward_one_row_per_fold(tmp_path):
    _write_fixture(tmp_path)
    folds = walk_forward(
        ["TEST"],
        datetime(2020, 1, 1),
        CSVDataHandler,
        SMACrossoverStrategy,
        {"fast": [5, 10], "slow": [20, 30], "quantity": [10]},
        n_splits=2,
        metric="sharpe",
        csv_dir=str(tmp_path),
        print_stats=False,
    )
    assert list(folds["fold"]) == [0, 1]
    for col in ["fast", "slow", "sharpe", "total_return"]:
        assert col in folds.columns


def test_walk_forward_is_deterministic(tmp_path):
    _write_fixture(tmp_path)
    grid = {"fast": [5, 10], "slow": [20, 30], "quantity": [10]}
    common = dict(
        symbol_list=["TEST"],
        start_date=datetime(2020, 1, 1),
        data_handler=CSVDataHandler,
        strategy=SMACrossoverStrategy,
        param_grid=grid,
        n_splits=2,
        csv_dir=str(tmp_path),
        print_stats=False,
    )
    first = walk_forward(**common)
    second = walk_forward(**common)
    pd.testing.assert_frame_equal(first, second)
