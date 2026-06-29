"""Parameter optimization and walk-forward validation for the backtest engine.

This module turns the replayable data feed (``DataHandler.reset()`` /
``n_bars`` / ``_records``, added when the engine was hardened) into practical
research tooling:

- :func:`optimize` runs a grid or random search over strategy parameters,
  optionally across processes, and returns a ranked results table. Each worker
  loads its data **once** and replays it across every parameter combination via
  ``reset()`` no re-reading or re-downloading per run.
- :func:`walk_forward` performs anchored or rolling walk-forward validation by
  slicing the in-memory columnar ``_records``, fitting parameters in-sample and
  scoring them out-of-sample.

Both consume the same ``BaseStrategy`` API as live trading, so a strategy is
written once and optimized without modification.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Type

import pandas as pd

from bbstrader.btengine.backtest import BacktestEngine
from bbstrader.btengine.data import DataHandler
from bbstrader.btengine.execution import ExecutionHandler, SimExecutionHandler
from bbstrader.btengine.performance import create_drawdowns, create_sharpe_ratio
from bbstrader.core.strategy import Strategy

__all__ = ["optimize", "walk_forward", "expand_param_grid"]

# Metrics where a *smaller* value is better (everything else: larger is better).
_LOWER_IS_BETTER = frozenset({"max_drawdown"})


def expand_param_grid(
    param_grid: Dict[str, Sequence[Any]],
    search: str = "grid",
    n_iter: Optional[int] = None,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Expand a parameter grid into a list of concrete parameter dicts.

    Args:
        param_grid: Mapping of parameter name to the sequence of values to try.
        search: ``"grid"`` for the full Cartesian product, ``"random"`` to draw
            ``n_iter`` random combinations.
        n_iter: Number of combinations to sample when ``search == "random"``.
        seed: Seed for the random sampler (deterministic by default).
    """
    keys = list(param_grid.keys())
    value_lists = [list(param_grid[k]) for k in keys]
    combos = [dict(zip(keys, values)) for values in itertools.product(*value_lists)]
    if search == "grid":
        return combos
    if search == "random":
        if n_iter is None:
            raise ValueError("n_iter is required when search='random'.")
        rng = random.Random(seed)
        if n_iter >= len(combos):
            return combos
        return rng.sample(combos, n_iter)
    raise ValueError(f"Unknown search mode: {search!r} (use 'grid' or 'random').")


def _score_curve(curve: Optional[pd.DataFrame], periods: int) -> Dict[str, float]:
    """Compute summary metrics from an equity-curve DataFrame."""
    if curve is None or curve.empty or "Total" not in curve:
        return {
            "total_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "final_equity": float("nan"),
        }
    total = curve["Total"]
    returns = (
        curve["Returns"].dropna() if "Returns" in curve else total.pct_change().dropna()
    )
    start = float(total.iloc[0])
    final = float(total.iloc[-1])
    total_return = (final / start - 1.0) if start else float("nan")
    try:
        sharpe = float(create_sharpe_ratio(returns, periods=periods))
    except Exception:
        sharpe = float("nan")
    equity = (
        curve["Equity Curve"] if "Equity Curve" in curve else (1.0 + returns).cumprod()
    )
    try:
        _, max_dd, _ = create_drawdowns(equity.dropna())
        max_drawdown = abs(float(max_dd))
    except Exception:
        max_drawdown = float("nan")
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "final_equity": final,
    }


def _run_engine(
    symbol_list: List[str],
    start_date: datetime,
    data_handler: Any,
    strategy: Type[Strategy],
    execution_handler: Type[ExecutionHandler],
    initial_capital: float,
    run_kwargs: Dict[str, Any],
) -> pd.DataFrame:
    """Run a single backtest and return its equity curve (no file/plot output)."""
    # Suppress the engine's progress banners so a sweep stays quiet.
    with contextlib.redirect_stdout(io.StringIO()):
        engine = BacktestEngine(
            symbol_list,
            initial_capital,
            0.0,
            start_date,
            data_handler,
            execution_handler,
            strategy,
            **run_kwargs,
        )
        engine._run_backtest()
        engine.portfolio.create_equity_curve_dataframe()
    return engine.portfolio.equity_curve


def _evaluate_combos(
    combos: List[Dict[str, Any]],
    symbol_list: List[str],
    start_date: datetime,
    data_handler_cls: Type[DataHandler],
    strategy: Type[Strategy],
    execution_handler: Type[ExecutionHandler],
    initial_capital: float,
    base_kwargs: Dict[str, Any],
    periods: int,
) -> List[Dict[str, Any]]:
    """Evaluate a chunk of parameter combinations, reusing one data handler.

    The handler is built once from its class and replayed across every
    combination via the engine's instance-reuse path (``reset()``).
    """
    from queue import Queue

    handler = data_handler_cls(Queue(), symbol_list, **base_kwargs)
    rows: List[Dict[str, Any]] = []
    for params in combos:
        run_kwargs = {**base_kwargs, **params}
        curve = _run_engine(
            symbol_list,
            start_date,
            handler,
            strategy,
            execution_handler,
            initial_capital,
            run_kwargs,
        )
        rows.append({**params, **_score_curve(curve, periods)})
    return rows


def _chunked(items: List[Any], n_chunks: int) -> List[List[Any]]:
    """Split ``items`` into at most ``n_chunks`` roughly equal chunks."""
    n_chunks = max(1, min(n_chunks, len(items)))
    size, rem = divmod(len(items), n_chunks)
    chunks: List[List[Any]] = []
    start = 0
    for i in range(n_chunks):
        end = start + size + (1 if i < rem else 0)
        chunks.append(items[start:end])
        start = end
    return chunks


def optimize(
    symbol_list: List[str],
    start_date: datetime,
    data_handler: Type[DataHandler],
    strategy: Type[Strategy],
    param_grid: Dict[str, Sequence[Any]],
    exc_handler: Optional[Type[ExecutionHandler]] = None,
    initial_capital: float = 100000.0,
    metric: str = "sharpe",
    periods: int = 252,
    n_jobs: int = 1,
    search: str = "grid",
    n_iter: Optional[int] = None,
    seed: int = 0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Search strategy parameters and return a ranked results table.

    Args:
        symbol_list: Symbols to backtest.
        start_date: Backtest start date.
        data_handler: A ``DataHandler`` subclass (class, not instance).
        strategy: A ``Strategy`` subclass whose ``**kwargs`` accept the swept
            parameters.
        param_grid: Mapping of parameter name to the values to try.
        exc_handler: Execution handler class (defaults to ``SimExecutionHandler``).
        initial_capital: Starting capital for each run.
        metric: Column to rank by. One of ``total_return``, ``sharpe``,
            ``max_drawdown``, ``final_equity``.
        periods: Annualization factor for the Sharpe ratio (252 daily, etc.).
        n_jobs: Number of worker processes. ``1`` runs in-process
            (deterministic); ``>1`` uses a process pool.
        search: ``"grid"`` (full product) or ``"random"`` (sample ``n_iter``).
        n_iter: Sample size for random search.
        seed: Seed for random search.
        **kwargs: Extra keyword args forwarded to every backtest (data handler,
            strategy, portfolio, execution handler).

    Returns:
        A DataFrame with one row per parameter combination plus its metrics,
        sorted best-first by ``metric``.
    """
    execution_handler = exc_handler or SimExecutionHandler
    combos = expand_param_grid(param_grid, search=search, n_iter=n_iter, seed=seed)
    if not combos:
        raise ValueError("param_grid produced no parameter combinations.")

    if n_jobs <= 1:
        rows = _evaluate_combos(
            combos,
            symbol_list,
            start_date,
            data_handler,
            strategy,
            execution_handler,
            initial_capital,
            kwargs,
            periods,
        )
    else:
        rows = []
        chunks = _chunked(combos, n_jobs)
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    _evaluate_combos,
                    chunk,
                    symbol_list,
                    start_date,
                    data_handler,
                    strategy,
                    execution_handler,
                    initial_capital,
                    kwargs,
                    periods,
                )
                for chunk in chunks
            ]
            for future in futures:
                rows.extend(future.result())

    results = pd.DataFrame(rows)
    if metric not in results.columns:
        raise ValueError(
            f"Unknown metric {metric!r}; available: {list(results.columns)}."
        )
    ascending = metric in _LOWER_IS_BETTER
    param_cols = list(combos[0].keys())
    results = results.sort_values(
        by=metric, ascending=ascending, na_position="last"
    ).reset_index(drop=True)
    return results[
        param_cols + ["total_return", "sharpe", "max_drawdown", "final_equity"]
    ]


def _slice_handler(handler: DataHandler, lo: int, hi: int) -> None:
    """Restrict a handler's replayable records to the half-open window [lo, hi)."""
    for s in handler.symbol_list:
        handler._records[s] = handler._full_records[s][lo:hi]  # type: ignore[attr-defined]
    handler.reset()


def walk_forward(
    symbol_list: List[str],
    start_date: datetime,
    data_handler: Type[DataHandler],
    strategy: Type[Strategy],
    param_grid: Dict[str, Sequence[Any]],
    exc_handler: Optional[Type[ExecutionHandler]] = None,
    initial_capital: float = 100000.0,
    metric: str = "sharpe",
    periods: int = 252,
    n_splits: int = 3,
    anchored: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Anchored or rolling walk-forward validation.

    The full history is divided into ``n_splits + 1`` equal segments. For each
    fold the in-sample window is optimized (in-process), and the best parameter
    set is evaluated on the next out-of-sample segment. With ``anchored=True``
    the in-sample window always starts at bar 0 and grows; with ``anchored=False``
    it rolls forward at a fixed length.

    Returns:
        One row per fold: the chosen parameters plus the out-of-sample metrics.
    """
    from queue import Queue

    execution_handler = exc_handler or SimExecutionHandler
    handler = data_handler(Queue(), symbol_list, **kwargs)
    # Keep a pristine copy so slicing per fold is non-destructive.
    handler._full_records = {  # type: ignore[attr-defined]
        s: list(handler._records[s]) for s in symbol_list
    }
    total_bars = len(handler._full_records[symbol_list[0]])  # type: ignore[attr-defined]
    if total_bars < (n_splits + 1) * 2:
        raise ValueError(
            f"Not enough bars ({total_bars}) for {n_splits} walk-forward splits."
        )
    seg = total_bars // (n_splits + 1)
    combos = expand_param_grid(param_grid)

    fold_rows: List[Dict[str, Any]] = []
    for fold in range(n_splits):
        train_hi = seg * (fold + 1)
        train_lo = 0 if anchored else seg * fold
        test_lo, test_hi = train_hi, seg * (fold + 2)

        # In-sample: pick the best parameters on the training window.
        best_params, best_score = None, None
        for params in combos:
            _slice_handler(handler, train_lo, train_hi)
            curve = _run_engine(
                symbol_list,
                start_date,
                handler,
                strategy,
                execution_handler,
                initial_capital,
                {**kwargs, **params},
            )
            score = _score_curve(curve, periods)[metric]
            better = best_score is None or (
                score < best_score if metric in _LOWER_IS_BETTER else score > best_score
            )
            if score == score and better:  # skip NaN scores
                best_params, best_score = params, score

        if best_params is None:
            best_params = combos[0]

        # Out-of-sample: evaluate the chosen parameters on the test window.
        _slice_handler(handler, test_lo, test_hi)
        oos_curve = _run_engine(
            symbol_list,
            start_date,
            handler,
            strategy,
            execution_handler,
            initial_capital,
            {**kwargs, **best_params},
        )
        fold_rows.append(
            {"fold": fold, **best_params, **_score_curve(oos_curve, periods)}
        )

    return pd.DataFrame(fold_rows)
