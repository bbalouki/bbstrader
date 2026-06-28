"""A vectorized research fast-path backtester.

This is the "does this even have alpha?" loop: it evaluates entry/exit signal
arrays across the entire history with NumPy, with no event queue and no
path-dependent order state. It is for fast hypothesis screening over many
parameter combinations -- orders of magnitude faster than the event-driven
engine -- not for faithful order-state simulation (use ``BacktestEngine`` for
that). The two share the same data: feed it the columnar arrays from a
``DataHandler`` (or any price series).

Signals are boolean arrays aligned to the price series; an indicator from
:mod:`bbstrader.core.indicators` plugs in directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = ["vectorized_backtest", "VectorizedResult"]

ArrayLike = Union[Sequence[float], NDArray[np.float64], pd.Series]


def _as_bool(values: Optional[ArrayLike], n: int) -> NDArray[np.bool_]:
    if values is None:
        return np.zeros(n, dtype=bool)
    arr = np.asarray(values, dtype=bool)
    if arr.shape != (n,):
        raise ValueError(f"Signal array must have length {n}, got {arr.shape}.")
    return arr


def _build_positions(
    entries: NDArray[np.bool_],
    exits: NDArray[np.bool_],
    short_entries: NDArray[np.bool_],
    short_exits: NDArray[np.bool_],
    allow_short: bool,
) -> NDArray[np.float64]:
    """Resolve entry/exit signals into a {-1, 0, +1} position per bar.

    A single O(n) pass enforces the state machine (no double entries, exits only
    when in position). The rest of the P&L math is fully vectorized.
    """
    n = entries.size
    pos = np.zeros(n, dtype=np.float64)
    state = 0.0
    for i in range(n):
        if state == 0.0:
            if entries[i]:
                state = 1.0
            elif allow_short and short_entries[i]:
                state = -1.0
        elif state == 1.0:
            if exits[i]:
                state = 0.0
            if state == 0.0 and allow_short and short_entries[i]:
                state = -1.0
        elif state == -1.0:
            if short_exits[i]:
                state = 0.0
            if state == 0.0 and entries[i]:
                state = 1.0
        pos[i] = state
    return pos


def _extract_trades(pos: NDArray[np.float64]) -> List[Tuple[int, int]]:
    """Return (entry_index, exit_index) pairs from a position array."""
    trades: List[Tuple[int, int]] = []
    state = 0.0
    entry_i = 0
    for i in range(pos.size):
        p = pos[i]
        if state == 0.0 and p != 0.0:
            state, entry_i = p, i
        elif state != 0.0 and (p == 0.0 or np.sign(p) != np.sign(state)):
            trades.append((entry_i, i))
            if p != 0.0:
                state, entry_i = p, i
            else:
                state = 0.0
    if state != 0.0:
        trades.append((entry_i, pos.size - 1))
    return trades


@dataclass
class VectorizedResult:
    """Result of a vectorized backtest with lazily computed metrics."""

    equity: NDArray[np.float64]
    returns: NDArray[np.float64]
    position: NDArray[np.float64]
    trades: List[Tuple[int, int]]
    init_cash: float
    periods: int

    @property
    def total_return(self) -> float:
        return (
            float(self.equity[-1] / self.init_cash - 1.0) if self.equity.size else 0.0
        )

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def exposure(self) -> float:
        """Fraction of bars spent in the market."""
        return float(np.mean(self.position != 0.0)) if self.position.size else 0.0

    @property
    def sharpe(self) -> float:
        r = self.returns
        sd = r.std()
        if sd == 0 or np.isnan(sd):
            return 0.0
        return float(r.mean() / sd * np.sqrt(self.periods))

    @property
    def max_drawdown(self) -> float:
        """Largest peak-to-trough drawdown of the equity curve (as a fraction)."""
        if self.equity.size == 0:
            return 0.0
        running_max = np.maximum.accumulate(self.equity)
        drawdown = self.equity / running_max - 1.0
        return float(abs(drawdown.min()))

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = 0
        for a, b in self.trades:
            if self.equity[b] > self.equity[a]:
                wins += 1
        return wins / len(self.trades)

    def to_frame(self, index: Optional[pd.Index] = None) -> pd.DataFrame:
        df = pd.DataFrame(
            {"Position": self.position, "Returns": self.returns, "Equity": self.equity}
        )
        if index is not None:
            df.index = index
        return df

    def summary(self) -> dict:
        return {
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "exposure": self.exposure,
        }


def vectorized_backtest(
    close: ArrayLike,
    entries: ArrayLike,
    exits: ArrayLike,
    *,
    short_entries: Optional[ArrayLike] = None,
    short_exits: Optional[ArrayLike] = None,
    allow_short: bool = False,
    init_cash: float = 100000.0,
    fees: float = 0.0,
    slippage: float = 0.0,
    periods: int = 252,
) -> VectorizedResult:
    """Run a fully vectorized signal backtest.

    Args:
        close: Price series.
        entries: Boolean array; True opens a long position.
        exits: Boolean array; True closes the long position.
        short_entries / short_exits: Optional short-side signals (require
            ``allow_short=True``).
        allow_short: Permit short positions.
        init_cash: Starting capital.
        fees: Per-unit-turnover fee as a fraction of notional (e.g. 0.0005).
        slippage: Per-unit-turnover slippage as a fraction of notional.
        periods: Annualization factor for the Sharpe ratio.

    Returns:
        A :class:`VectorizedResult` with the equity curve and metrics.
    """
    price = np.asarray(close, dtype=np.float64)
    if price.ndim != 1:
        raise ValueError("close must be a 1-D price series.")
    n = price.size
    ent = _as_bool(entries, n)
    ext = _as_bool(exits, n)
    sent = _as_bool(short_entries, n)
    sext = _as_bool(short_exits, n)
    if (sent.any() or sext.any()) and not allow_short:
        raise ValueError("short signals provided but allow_short is False.")

    pos = _build_positions(ent, ext, sent, sext, allow_short)

    # Bar returns; position from the previous bar is held into the current bar.
    bar_ret = np.zeros(n, dtype=np.float64)
    bar_ret[1:] = price[1:] / price[:-1] - 1.0
    prev_pos = np.concatenate([[0.0], pos[:-1]])
    gross = prev_pos * bar_ret

    # Trading cost charged on turnover at the bar the position changes.
    turnover = np.abs(pos - prev_pos)
    cost = (fees + slippage) * turnover
    strat_ret = gross - cost

    equity = init_cash * np.cumprod(1.0 + strat_ret)
    trades = _extract_trades(pos)
    return VectorizedResult(
        equity=equity,
        returns=strat_ret,
        position=pos,
        trades=trades,
        init_cash=init_cash,
        periods=periods,
    )
