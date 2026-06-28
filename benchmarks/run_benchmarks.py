"""Reproducible engine benchmarks.

Times the event-driven backtest engine on synthetic data so the README's
performance positioning is verifiable and regressions are caught. Reports
throughput (bars/second) and wall time across dataset sizes, and can emit JSON
for tracking over time.

Usage:
    python benchmarks/run_benchmarks.py --bars 1000 10000 100000
    python benchmarks/run_benchmarks.py --bars 10000 --out results.json

No network access and no extra runtime dependency (uses time.perf_counter).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import tempfile
import time
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List

import pandas as pd

from bbstrader.btengine.backtest import BacktestEngine
from bbstrader.btengine.data import CSVDataHandler
from bbstrader.btengine.event import Events, MarketEvent
from bbstrader.btengine.execution import SimExecutionHandler
from bbstrader.btengine.strategy import BacktestStrategy


class _NoopStrategy(BacktestStrategy):
    """A strategy that emits no orders, isolating raw engine throughput."""

    def calculate_signals(self, event: MarketEvent) -> None:
        return None


def _write_dataset(csv_dir: Path, n_bars: int, symbol: str = "BENCH") -> None:
    dates = pd.date_range("2000-01-01", periods=n_bars, freq="min")
    closes = [100.0 + 0.001 * i + 5.0 * math.sin(i / 50.0) for i in range(n_bars)]
    df = pd.DataFrame(
        {
            "Datetime": dates,
            "Open": closes,
            "High": [c + 0.5 for c in closes],
            "Low": [c - 0.5 for c in closes],
            "Close": closes,
            "Adj Close": closes,
            "Volume": [1000] * n_bars,
        }
    )
    df.to_csv(csv_dir / f"{symbol}.csv", index=False)


def _time_full_backtest(csv_dir: Path, n_bars: int) -> float:
    """Return wall-clock seconds for one full event-driven backtest."""
    with contextlib.redirect_stdout(io.StringIO()):
        start = time.perf_counter()
        engine = BacktestEngine(
            ["BENCH"],
            100000.0,
            0.0,
            datetime(2000, 1, 1),
            CSVDataHandler,
            SimExecutionHandler,
            _NoopStrategy,
            csv_dir=str(csv_dir),
            print_stats=False,
        )
        engine._run_backtest()
        engine.portfolio.create_equity_curve_dataframe()
        elapsed = time.perf_counter() - start
    return elapsed


def _time_data_replay(csv_dir: Path) -> float:
    """Return wall-clock seconds to replay the data feed twice via reset()."""
    handler = CSVDataHandler(Queue(), ["BENCH"], csv_dir=str(csv_dir))
    start = time.perf_counter()
    for _ in range(2):
        handler.reset()
        while handler.continue_backtest:
            handler.update_bars()
    return time.perf_counter() - start


def run(bar_sizes: List[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for n_bars in bar_sizes:
        with tempfile.TemporaryDirectory() as d:
            csv_dir = Path(d)
            _write_dataset(csv_dir, n_bars)
            # The handler rewrites the CSV on first load; build once to warm it.
            bt_seconds = _time_full_backtest(csv_dir, n_bars)
            replay_seconds = _time_data_replay(csv_dir)
        rows.append(
            {
                "bars": n_bars,
                "backtest_seconds": round(bt_seconds, 4),
                "backtest_bars_per_sec": round(n_bars / bt_seconds)
                if bt_seconds
                else None,
                "replay_seconds": round(replay_seconds, 4),
            }
        )
    return rows


def _print_table(rows: List[Dict[str, Any]]) -> None:
    header = f"{'bars':>10} | {'backtest (s)':>13} | {'bars/sec':>12} | {'2x replay (s)':>13}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['bars']:>10} | {r['backtest_seconds']:>13} | "
            f"{str(r['backtest_bars_per_sec']):>12} | {r['replay_seconds']:>13}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="bbstrader engine benchmarks")
    parser.add_argument(
        "--bars",
        type=int,
        nargs="+",
        default=[1000, 10000, 100000],
        help="Dataset sizes to benchmark (number of bars).",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Optional path to write JSON results."
    )
    args = parser.parse_args()

    rows = run(args.bars)
    _print_table(rows)
    if args.out:
        payload = {"generated_at": datetime.now().isoformat(), "results": rows}
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
