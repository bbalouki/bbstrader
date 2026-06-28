"""Multi-timeframe support: derive higher-timeframe bars from a base feed.

A common institutional pattern is to *execute* on a fast timeframe (e.g. 1m)
while computing *signals* on a slower one (e.g. H1 or daily). Rather than
rearchitecting the event loop into a full multi-clock model, this module lets a
strategy resample the base-timeframe bars it already receives into completed
higher-timeframe (HTF) bars on demand -- inside ``calculate_signals`` -- with no
look-ahead.

``MultiTimeFrame`` wraps a ``DataHandler``; ``resample_ohlcv`` is the underlying
aggregation and can be used standalone on any OHLCV frame.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from bbstrader.btengine.data import DataHandler

__all__ = ["resample_ohlcv", "MultiTimeFrame", "PANDAS_RULE_BY_TF"]

# Map bbstrader timeframe codes to pandas resample rules.
PANDAS_RULE_BY_TF = {
    "1m": "1min",
    "5m": "5min",
    "10m": "10min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "D1": "1D",
    "W1": "1W",
    "MN1": "1MS",
}


def _rule(rule: str) -> str:
    """Accept either a bbstrader timeframe code or a raw pandas rule."""
    return PANDAS_RULE_BY_TF.get(rule, rule)


def resample_ohlcv(
    df: pd.DataFrame, rule: str, *, label: str = "left", closed: str = "left"
) -> pd.DataFrame:
    """Aggregate an OHLCV DataFrame up to a higher timeframe.

    open=first, high=max, low=min, close=last, volume=sum (adj_close=last when
    present). Buckets with no data are dropped. ``df`` must have a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("resample_ohlcv requires a DatetimeIndex.")
    agg = {}
    for col, how in (
        ("open", "first"),
        ("high", "max"),
        ("low", "min"),
        ("close", "last"),
        ("adj_close", "last"),
        ("volume", "sum"),
    ):
        if col in df.columns:
            agg[col] = how
    if "close" not in agg:
        raise ValueError("OHLCV frame must contain at least a 'close' column.")
    out = df.resample(_rule(rule), label=label, closed=closed).agg(agg)
    return out.dropna(subset=["close"])


class MultiTimeFrame:
    """Derive completed higher-timeframe bars from a base-timeframe DataHandler.

    Use inside a strategy's ``calculate_signals`` to read slow-timeframe context
    while executing on the fast base feed::

        mtf = MultiTimeFrame(self.data)
        daily_close = mtf.htf_value(symbol, "D1")   # last *completed* daily close
    """

    def __init__(self, data: DataHandler, lookback: int = 1000) -> None:
        self.data = data
        self.lookback = lookback

    def _base_bars(self, symbol: str, lookback: Optional[int]) -> pd.DataFrame:
        bars = self.data.get_latest_bars(symbol, N=lookback or self.lookback)
        if not isinstance(bars, pd.DataFrame):
            bars = pd.DataFrame([b[1] for b in bars])
        return bars

    def htf_bars(
        self,
        symbol: str,
        rule: str,
        n: Optional[int] = None,
        lookback: Optional[int] = None,
        drop_partial: bool = True,
    ) -> pd.DataFrame:
        """Return resampled HTF bars for ``symbol``.

        With ``drop_partial`` (default) the final, possibly still-forming bucket
        is dropped so only completed HTF bars are visible -- preventing
        look-ahead. ``n`` limits the result to the most recent ``n`` bars.
        """
        base = self._base_bars(symbol, lookback)
        res = resample_ohlcv(base, rule)
        if drop_partial and len(res):
            # Always drop the final bucket: it may still be forming, so this
            # guarantees only completed HTF bars are visible (no look-ahead).
            res = res.iloc[:-1]
        return res.tail(n) if n else res

    def htf_value(
        self,
        symbol: str,
        rule: str,
        val_type: str = "close",
        lookback: Optional[int] = None,
        drop_partial: bool = True,
    ) -> Optional[float]:
        """Latest completed HTF value for ``symbol`` (None if not enough data)."""
        res = self.htf_bars(symbol, rule, lookback=lookback, drop_partial=drop_partial)
        if res.empty or val_type not in res.columns:
            return None
        return float(res[val_type].iloc[-1])
