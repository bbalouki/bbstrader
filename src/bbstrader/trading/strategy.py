from datetime import datetime
from enum import IntEnum
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Union
from abc import abstractmethod

import numpy as np
import pandas as pd
from loguru import logger

from bbstrader.api.client import TradeOrder  # type: ignore
from bbstrader.btengine.event import FillEvent, SignalEvent
from bbstrader.config import BBSTRADER_DIR
from bbstrader.core.strategy import (
    BaseStrategy,
    TradeAction,
    TradeSignal,
    TradingMode,
    generate_signal,
)
from bbstrader.metatrader.account import Account
from bbstrader.metatrader.rates import Rates

logger.add(
    f"{BBSTRADER_DIR}/logs/strategy.log",
    enqueue=True,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
)

__all__ = [
    "LiveStrategy",
]


class SignalType(IntEnum):
    BUY = 0
    SELL = 1
    EXIT_LONG = 2
    EXIT_SHORT = 3
    EXIT_ALL_POSITIONS = 4
    EXIT_ALL_ORDERS = 5
    EXIT_STOP = 6
    EXIT_LIMIT = 7


class LiveStrategy(BaseStrategy):
    """
    Strategy implementation for Live Trading.
    Relies on the `Account` class for state (orders, positions, cash)
    and `Rates` for data.
    """

    events: "Queue[Union[SignalEvent, FillEvent]]"

    def __init__(
        self,
        symbol_list: List[str],
        **kwargs: Any,
    ) -> None:
        """
        Initialize the `LiveStrategy` object.

        Args:
            symbol_list : The list of symbols for the strategy.
            **kwargs : Additional keyword arguments for other classes (e.g, Portfolio, ExecutionHandler).
                - max_trades : The maximum number of trades allowed per symbol.
                - time_frame : The time frame for the strategy.
                - logger : The logger object for the strategy.
        """
        super().__init__(symbol_list, **kwargs)
        self.mode = TradingMode.LIVE

    @property
    def account(self) -> Account:
        """Create or access the MT5 Account."""
        return Account(**self.kwargs)

    @property
    def cash(self) -> float:
        return self.account.equity

    @property
    def orders(self) -> List[TradeOrder]:
        """Returns active orders from the Broker."""
        return self.account.get_orders() or []

    @property
    def positions(self) -> List[Any]:
        """Returns open positions from the Broker."""
        return self.account.get_positions() or []

    def get_asset_values(
        self,
        symbol_list: List[str],
        window: int,
        value_type: str = "returns",
        array: bool = True,
        **kwargs,
    ) -> Optional[Dict[str, Union[np.ndarray, pd.Series]]]:
        asset_values: Dict[str, Union[np.ndarray, pd.Series]] = {}
        for asset in symbol_list:
            rates = Rates(asset, timeframe=self.tf, count=window + 1, **self.kwargs)
            if array:
                values = getattr(rates, value_type).to_numpy()
                asset_values[asset] = values[~np.isnan(values)]
            else:
                values = getattr(rates, value_type)
                asset_values[asset] = values

        if all(len(values) >= window for values in asset_values.values()):
            return {a: v[-window:] for a, v in asset_values.items()}

        if kwargs.get("error") == "raise":
            raise ValueError("Not enough data to calculate the values.")
        elif kwargs.get("error") == "ignore":
            return asset_values
        return None

    def signal(
        self, signal: int, symbol: str, sl: float = None, tp: float = None
    ) -> TradeSignal:
        """
        Generate a ``TradeSignal`` object based on the signal value.

        Parameters
        ----------
        signal : int
            An integer value representing the signal type:
            * 0: BUY
            * 1: SELL
            * 2: EXIT_LONG
            * 3: EXIT_SHORT
            * 4: EXIT_ALL_POSITIONS
            * 5: EXIT_ALL_ORDERS
            * 6: EXIT_STOP
            * 7: EXIT_LIMIT
        symbol : str
            The symbol for the trade.

        Returns
        -------
        TradeSignal
            A ``TradeSignal`` object representing the trade signal.

        Raises
        ------
        ValueError
            If the signal value is not between 0 and 7.

        Notes
        -----
        This generates only common signals. For more complex signals, use
        ``generate_signal`` directly.
        """
        signal_id = getattr(self, "id", getattr(self, "ID", None))
        if signal_id is None:
            raise ValueError("Strategy ID not set")

        action_map = {
            SignalType.BUY: TradeAction.BUY,
            SignalType.SELL: TradeAction.SELL,
            SignalType.EXIT_LONG: TradeAction.EXIT_LONG,
            SignalType.EXIT_SHORT: TradeAction.EXIT_SHORT,
            SignalType.EXIT_ALL_POSITIONS: TradeAction.EXIT_ALL_POSITIONS,
            SignalType.EXIT_ALL_ORDERS: TradeAction.EXIT_ALL_ORDERS,
            SignalType.EXIT_STOP: TradeAction.EXIT_STOP,
            SignalType.EXIT_LIMIT: TradeAction.EXIT_LIMIT,
        }

        try:
            action = action_map[SignalType(signal)]
        except (ValueError, KeyError):
            raise ValueError(f"Invalid signal value: {signal}")
        kwargs = (
            {"sl": sl, "tp": tp}
            if action in (TradeAction.BUY, TradeAction.SELL)
            else {}
        )

        return generate_signal(signal_id, symbol, action, **kwargs)
    
    @abstractmethod
    def calculate_signals(self, *args: Any, **kwargs: Any) -> List[TradeSignal]: ...

    def ispositions(
        self,
        symbol: str,
        strategy_id: int,
        position: int,
        max_trades: int,
        one_true: bool = False,
    ) -> bool:
        """
        This function is use for live trading to check if there are open positions
        for a given symbol and strategy. It is used to prevent opening more trades
        than the maximum allowed trades per symbol.

        Args:
            symbol : The symbol for the trade.
            strategy_id : The unique identifier for the strategy.
            position : The position type (1: short, 0: long).
            max_trades : The maximum number of trades allowed per symbol.
            one_true : If True, return True if there is at least one open position.
            account : The `bbstrader.metatrader.Account` object for the strategy.

        Returns:
            bool : True if there are open positions, False otherwise
        """
        positions = self.account.get_positions(symbol=symbol)
        if positions is not None:
            open_positions = [
                pos.ticket
                for pos in positions
                if pos.type == position and pos.magic == strategy_id
            ]
            if one_true:
                return len(open_positions) in range(1, max_trades + 1)
            return len(open_positions) >= max_trades
        return False

    def get_positions_prices(
        self,
        symbol: str,
        strategy_id: int,
        position: int,
    ) -> np.ndarray:
        """
        Get the buy or sell prices for open positions of a given symbol and strategy.

        Args:
            symbol : The symbol for the trade.
            strategy_id : The unique identifier for the strategy.
            position : The position type (1: short, 0: long).
            account : The `bbstrader.metatrader.Account` object for the strategy.

        Returns:
            prices : numpy array of buy or sell prices for open positions if any or an empty array.
        """
        positions = self.account.get_positions(symbol=symbol)
        if positions is not None:
            prices = np.array(
                [
                    pos.price_open
                    for pos in positions
                    if pos.type == position and pos.magic == strategy_id
                ]
            )
            return prices
        return np.array([])

    def get_active_orders(
        self, symbol: str, strategy_id: int, order_type: Optional[int] = None
    ) -> List[TradeOrder]:
        """
        Get the active orders for a given symbol and strategy.

        Args:
            symbol : The symbol for the trade.
            strategy_id : The unique identifier for the strategy.
            order_type : The type of order to filter by (optional):
                    "BUY_LIMIT": 2
                    "SELL_LIMIT": 3
                    "BUY_STOP": 4
                    "SELL_STOP": 5
                    "BUY_STOP_LIMIT": 6
                    "SELL_STOP_LIMIT": 7

        Returns:
            List[TradeOrder] : A list of active orders for the given symbol and strategy.
        """
        all_orders = self.orders
        orders = [
            o
            for o in all_orders
            if isinstance(o, TradeOrder)
            and o.symbol == symbol
            and o.magic == strategy_id
        ]
        if order_type is not None and len(orders) > 0:
            orders = [o for o in orders if o.type == order_type]
        return orders

    def exit_positions(
        self, position: int, prices: np.ndarray, asset: str, th: float = 0.01
    ) -> bool:
        """Logic to determine if positions should be exited based on threshold."""
        if len(prices) == 0:
            return False
        tick_info = self.account.get_tick_info(asset)
        if tick_info is None:
            return False
        bid, ask = tick_info.bid, tick_info.ask
        price = None
        if len(prices) == 1:
            price = prices[0]
        elif len(prices) in range(2, self.max_trades[asset] + 1):
            price = np.mean(prices)

        if price is not None:
            if position == 0:  # Long exit check
                return self.calculate_pct_change(ask, price) >= th
            elif position == 1:  # Short exit check
                return self.calculate_pct_change(bid, price) <= -th
        return False

    def send_trade_report(self, perf_analyzer: Callable, **kwargs: Any) -> None:
        """
        Generates and sends a trade report message containing performance metrics for the current strategy.
        This method retrieves the trade history for the current account, filters it by the strategy's ID,
        computes performance metrics using the provided `perf_analyzer` callable, and formats the results
        into a message. The message includes account information, strategy details, a timestamp, and
        performance metrics. The message is then sent via Telegram using the specified bot token and chat ID.

        Args:
            perf_analyzer (Callable): A function or callable object that takes the filtered trade history
                (as a DataFrame) and additional keyword arguments, and returns a DataFrame of performance metrics.
            **kwargs: Additional keyword arguments, which may include
                - Any other param requires by ``perf_analyzer``
        """
        from bbstrader.trading.utils import send_message

        history = self.account.get_trades_history()
        if history is None or history.empty:
            self.logger.warning("No trades found on this account.")
            return

        ID = getattr(self, "id", None) or getattr(self, "ID")
        history = history[history["magic"] == ID]
        performance = perf_analyzer(history, **kwargs)
        if performance.empty:
            self.logger.warning("No trades found for the current strategy.")
            return

        account_name = self.kwargs.get("account", "MT5 Account")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header = (
            f"TRADE REPORT\n\n"
            f"ACCOUNT: {account_name}\n"
            f"STRATEGY: {self.NAME}\n"
            f"ID: {ID}\n"
            f"DESCRIPTION: {self.DESCRIPTION}\n"
            f"TIMESTAMP: {timestamp}\n\n"
            f"📊 PERFORMANCE:\n"
        )
        metrics = performance.iloc[0].to_dict()

        lines = []
        for key, value in metrics.items():
            if isinstance(value, float):
                value = round(value, 4)
            lines.append(f"{key:<15}: {value}")

        performance_str = "\n".join(lines)
        message = f"{header}{performance_str}"

        send_message(
            message=message,
            telegram=True,
            token=self.kwargs.get("bot_token"),
            chat_id=self.kwargs.get("chat_id"),
        )
