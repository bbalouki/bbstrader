"""Ready-to-use strategy templates (a small cookbook).

These are concrete, parameterized `BacktestStrategy` subclasses for the most
common archetypes: trend following (SMA crossover), mean reversion (RSI), and
breakout (Donchian channel). They are built entirely on the shared strategy API
-- `get_asset_values` for data, the vectorized
:mod:`bbstrader.core.indicators` for signals, and the `buy_mkt`/`close_positions`
order helpers so they are also natural targets for
:func:`bbstrader.btengine.optimize.optimize`.

Each template trades a single long position per symbol and is long-only, which
keeps them simple to read and to optimize. Subclass or copy them as a starting
point for your own ideas.
"""

from typing import Any, List

from bbstrader.btengine.event import Events, MarketEvent
from bbstrader.btengine.strategy import BacktestStrategy
from bbstrader.core import indicators as ind

__all__ = [
    "SMACrossoverStrategy",
    "RSIMeanReversionStrategy",
    "DonchianBreakoutStrategy",
]


class _TemplateBase(BacktestStrategy):
    """Shared plumbing for the long-only templates."""

    def __init__(
        self,
        events: Any,
        symbol_list: List[str],
        bars: Any,
        **kwargs: Any,
    ) -> None:
        """Initialise shared template state (strategy id and per-symbol size).

        Args:
            events (Any): The engine event queue.
            symbol_list (List[str]): The symbols traded by the strategy.
            bars (Any): The DataHandler providing market data.
            kwargs (Any): Template options; ``strategy_id`` (default 1) and
                ``quantity`` (units per trade, default 100).
        """
        super().__init__(events, symbol_list, bars, **kwargs)
        self.strategy_id = int(kwargs.get("strategy_id", 1))
        quantity = kwargs.get("quantity", 100)
        self.qty = {s: int(quantity) for s in self.symbols}

    def _is_long(self, symbol: str) -> bool:
        """Return True if a long position is currently open in ``symbol``.

        Args:
            symbol (str): The instrument to check.

        Returns:
            bool: Whether the long quantity is positive.
        """
        return self._positions[symbol]["LONG"] > 0

    def _closes(self, symbol: str, window: int):
        """Return the last ``window`` close prices, or None if not yet available."""
        values = self.get_asset_values([symbol], window=window, value_type="close")
        if not values or symbol not in values:
            return None
        arr = values[symbol]
        if arr is None or len(arr) < window:
            return None
        return arr


class SMACrossoverStrategy(_TemplateBase):
    """Trend following: go long when the fast SMA crosses above the slow SMA.

    kwargs:
        fast (int, default 10): Fast SMA window.
        slow (int, default 30): Slow SMA window.
        quantity (int, default 100): Units per trade.
    """

    def __init__(self, events, symbol_list, bars, **kwargs) -> None:
        """Initialise the SMA crossover with fast/slow windows.

        Args:
            events (Any): The engine event queue.
            symbol_list (List[str]): The symbols traded by the strategy.
            bars (Any): The DataHandler providing market data.
            kwargs (Any): ``fast`` (default 10) and ``slow`` (default 30) SMA
                windows, plus the shared template options.

        Raises:
            ValueError: If ``fast`` is not strictly less than ``slow``.
        """
        super().__init__(events, symbol_list, bars, **kwargs)
        self.fast = int(kwargs.get("fast", 10))
        self.slow = int(kwargs.get("slow", 30))
        if self.fast >= self.slow:
            raise ValueError(f"fast ({self.fast}) must be < slow ({self.slow}).")

    def calculate_signals(self, event: MarketEvent) -> None:
        """Enter long on an up-cross and exit on a down-cross of the SMAs.

        Args:
            event (MarketEvent): The market event driving the bar; ignored unless
                it is a MARKET event.
        """
        if event.type != Events.MARKET:
            return
        for symbol in self.symbols:
            # Need one extra bar so we can see the cross (current vs previous).
            arr = self._closes(symbol, self.slow + 1)
            if arr is None:
                continue
            fast = ind.sma(arr, self.fast)
            slow = ind.sma(arr, self.slow)
            price = float(arr[-1])
            dt = self.data.get_latest_bar_datetime(symbol)
            crossed_up = fast[-2] <= slow[-2] and fast[-1] > slow[-1]
            crossed_down = fast[-2] >= slow[-2] and fast[-1] < slow[-1]
            if crossed_up and not self._is_long(symbol):
                self.buy_mkt(
                    self.strategy_id, symbol, price, self.qty[symbol], dtime=dt
                )
            elif crossed_down and self._is_long(symbol):
                self.close_positions(
                    self.strategy_id, symbol, price, self.qty[symbol], dtime=dt
                )


class RSIMeanReversionStrategy(_TemplateBase):
    """Mean reversion: buy when RSI is oversold, exit when it recovers.

    kwargs:
        period (int, default 14): RSI lookback.
        oversold (float, default 30): Entry threshold.
        exit_level (float, default 55): Exit threshold.
        quantity (int, default 100): Units per trade.
    """

    def __init__(self, events, symbol_list, bars, **kwargs) -> None:
        """Initialise the RSI mean-reversion thresholds.

        Args:
            events (Any): The engine event queue.
            symbol_list (List[str]): The symbols traded by the strategy.
            bars (Any): The DataHandler providing market data.
            kwargs (Any): ``period`` (RSI lookback, default 14), ``oversold``
                (entry threshold, default 30) and ``exit_level`` (exit threshold,
                default 55), plus the shared template options.
        """
        super().__init__(events, symbol_list, bars, **kwargs)
        self.period = int(kwargs.get("period", 14))
        self.oversold = float(kwargs.get("oversold", 30.0))
        self.exit_level = float(kwargs.get("exit_level", 55.0))

    def calculate_signals(self, event: MarketEvent) -> None:
        """Buy when RSI is oversold and exit when it recovers above the level.

        Args:
            event (MarketEvent): The market event driving the bar; ignored unless
                it is a MARKET event.
        """
        if event.type != Events.MARKET:
            return
        for symbol in self.symbols:
            arr = self._closes(symbol, self.period + 2)
            if arr is None:
                continue
            rsi = ind.rsi(arr, self.period)
            latest = rsi[-1]
            if latest != latest:  # NaN guard
                continue
            price = float(arr[-1])
            dt = self.data.get_latest_bar_datetime(symbol)
            if latest <= self.oversold and not self._is_long(symbol):
                self.buy_mkt(
                    self.strategy_id, symbol, price, self.qty[symbol], dtime=dt
                )
            elif latest >= self.exit_level and self._is_long(symbol):
                self.close_positions(
                    self.strategy_id, symbol, price, self.qty[symbol], dtime=dt
                )


class DonchianBreakoutStrategy(_TemplateBase):
    """Breakout: go long when price closes above the prior N-bar high.

    The channel is taken from the *previous* bar to avoid look-ahead. Exit when
    price closes below the prior N-bar low.

    kwargs:
        window (int, default 20): Donchian channel lookback.
        quantity (int, default 100): Units per trade.
    """

    def __init__(self, events, symbol_list, bars, **kwargs) -> None:
        """Initialise the Donchian breakout channel lookback.

        Args:
            events (Any): The engine event queue.
            symbol_list (List[str]): The symbols traded by the strategy.
            bars (Any): The DataHandler providing market data.
            kwargs (Any): ``window`` (channel lookback, default 20), plus the
                shared template options.
        """
        super().__init__(events, symbol_list, bars, **kwargs)
        self.window = int(kwargs.get("window", 20))

    def calculate_signals(self, event: MarketEvent) -> None:
        """Go long on a close above the prior N-bar high; exit below the low.

        Args:
            event (MarketEvent): The market event driving the bar; ignored unless
                it is a MARKET event.
        """
        if event.type != Events.MARKET:
            return
        for symbol in self.symbols:
            highs = self.get_asset_values(
                [symbol], window=self.window + 1, value_type="high"
            )
            lows = self.get_asset_values(
                [symbol], window=self.window + 1, value_type="low"
            )
            closes = self._closes(symbol, self.window + 1)
            if closes is None or not highs or not lows:
                continue
            high = highs.get(symbol)
            low = lows.get(symbol)
            if high is None or low is None or len(high) < self.window + 1:
                continue
            upper = ind.donchian(high, low, self.window)[1]
            lower = ind.donchian(high, low, self.window)[0]
            # Compare current close against the *previous* bar's channel.
            prior_upper = upper[-2]
            prior_lower = lower[-2]
            price = float(closes[-1])
            dt = self.data.get_latest_bar_datetime(symbol)
            if price > prior_upper and not self._is_long(symbol):
                self.buy_mkt(
                    self.strategy_id, symbol, price, self.qty[symbol], dtime=dt
                )
            elif price < prior_lower and self._is_long(symbol):
                self.close_positions(
                    self.strategy_id, symbol, price, self.qty[symbol], dtime=dt
                )
