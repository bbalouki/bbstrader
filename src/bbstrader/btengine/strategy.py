from abc import abstractmethod
from datetime import datetime
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

from bbstrader.btengine.data import DataHandler
from bbstrader.btengine.event import Events, FillEvent, MarketEvent, SignalEvent
from bbstrader.config import BBSTRADER_DIR
from bbstrader.core.strategy import BaseStrategy, TradingMode

logger.add(
    f"{BBSTRADER_DIR}/logs/strategy.log",
    enqueue=True,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
)

__all__ = ["BacktestStrategy"]


class BacktestStrategy(BaseStrategy):
    """
    Strategy implementation specifically for Backtesting.
    Handles internal state for orders, positions, trades, and cash.
    Simulates order execution and pending orders.
    """

    _orders: Dict[str, Dict[str, List[SignalEvent]]]
    _positions: Dict[str, Dict[str, Union[int, float]]]
    _trades: Dict[str, Dict[str, int]]
    _holdings: Dict[str, float]
    _portfolio_value: Optional[float]
    events: "Queue[Union[SignalEvent, FillEvent]]"
    data: DataHandler

    def __init__(
        self,
        events: "Queue[Union[SignalEvent, FillEvent]]",
        symbol_list: List[str],
        bars: DataHandler,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the `BacktestStrategy` object.

        Args:
            events : The event queue.
            symbol_list : The list of symbols for the strategy.
            bars : The data handler object.
            **kwargs : Additional keyword arguments for other classes (e.g, Portfolio, ExecutionHandler).
                - max_trades : The maximum number of trades allowed per symbol.
                - time_frame : The time frame for the strategy.
                - logger : The logger object for the strategy.
        """
        super().__init__(symbol_list, **kwargs)
        self.events = events
        self.data = bars
        self.mode = TradingMode.BACKTEST
        self._portfolio_value = None
        self._initialize_portfolio()

    def _initialize_portfolio(self) -> None:
        self._orders = {}
        self._positions = {}
        self._trades = {}
        for symbol in self.symbols:
            self._positions[symbol] = {}
            self._orders[symbol] = {}
            self._trades[symbol] = {}
            for position in ["LONG", "SHORT"]:
                self._trades[symbol][position] = 0
                self._positions[symbol][position] = 0.0
            for order in ["BLMT", "BSTP", "BSTPLMT", "SLMT", "SSTP", "SSTPLMT"]:
                self._orders[symbol][order] = []
        self._holdings = {s: 0.0 for s in self.symbols}

    @property
    def cash(self) -> float:
        return self._portfolio_value or 0.0

    @cash.setter
    def cash(self, value: float) -> None:
        self._portfolio_value = value

    @property
    def orders(self) -> Dict[str, Dict[str, List[SignalEvent]]]:
        return self._orders

    @property
    def trades(self) -> Dict[str, Dict[str, int]]:
        return self._trades

    @property
    def positions(self) -> Dict[str, Dict[str, Union[int, float]]]:
        return self._positions

    @property
    def holdings(self) -> Dict[str, float]:
        return self._holdings

    def get_update_from_portfolio(
        self, positions: Dict[str, float], holdings: Dict[str, float]
    ) -> None:
        """
        Update the positions and holdings for the strategy from the portfolio.

        Positions are the number of shares of a security that are owned in long or short.
        Holdings are the value (postions * price) of the security that are owned in long or short.

        Args:
            positions : The positions for the symbols in the strategy.
            holdings : The holdings for the symbols in the strategy.
        """
        for symbol in self.symbols:
            if symbol in positions:
                if positions[symbol] > 0:
                    self._positions[symbol]["LONG"] = positions[symbol]
                elif positions[symbol] < 0:
                    self._positions[symbol]["SHORT"] = positions[symbol]
                else:
                    self._positions[symbol]["LONG"] = 0
                    self._positions[symbol]["SHORT"] = 0
            if symbol in holdings:
                self._holdings[symbol] = holdings[symbol]

    def update_trades_from_fill(self, event: FillEvent) -> None:
        """
        This method updates the trades for the strategy based on the fill event.
        It is used to keep track of the number of trades executed for each order.
        """
        if event.type == Events.FILL:
            if event.order != "EXIT":
                self._trades[event.symbol][event.order] += 1  # type: ignore
            elif event.order == "EXIT" and event.direction == "BUY":
                self._trades[event.symbol]["SHORT"] = 0
            elif event.order == "EXIT" and event.direction == "SELL":
                self._trades[event.symbol]["LONG"] = 0

    def get_asset_values(
        self,
        symbol_list: List[str],
        window: int,
        value_type: str = "returns",
        array: bool = True,
        **kwargs,
    ) -> Optional[Dict[str, Union[np.typing.NDArray, pd.Series]]]:
        asset_values = {}
        for asset in symbol_list:
            if array:
                values = self.data.get_latest_bars_values(asset, value_type, N=window)
                asset_values[asset] = values[~np.isnan(values)]
            else:
                values_df = self.data.get_latest_bars(asset, N=window)
                if isinstance(values_df, pd.DataFrame):
                    asset_values[asset] = values_df[value_type]

        if all(len(values) >= window for values in asset_values.values()):
            return {a: v[-window:] for a, v in asset_values.items()}
        return None

    @abstractmethod
    def calculate_signals(self, event: MarketEvent) -> None: ...

    def _send_order(
        self,
        id: int,
        symbol: str,
        signal: str,
        strength: float,
        price: float,
        quantity: int,
        dtime: Union[datetime, pd.Timestamp],
    ) -> None:
        position = SignalEvent(
            id,
            symbol,
            dtime,
            signal,
            quantity=quantity,
            strength=strength,
            price=price,  # type: ignore
        )
        log = False
        if signal in ["LONG", "SHORT"]:
            if self._trades[symbol][signal] < self.max_trades[symbol] and quantity > 0:
                self.events.put(position)
                log = True
        elif signal == "EXIT":
            if (
                self._positions[symbol]["LONG"] > 0
                or self._positions[symbol]["SHORT"] < 0
            ):
                self.events.put(position)
                log = True
        if log:
            self.logger.info(
                f"{signal} ORDER EXECUTED: SYMBOL={symbol}, QUANTITY={quantity}, PRICE @{round(price, 5)}",
                custom_time=dtime,
            )

    def buy_mkt(
        self,
        id: int,
        symbol: str,
        price: float,
        quantity: int,
        strength: float = 1.0,
        dtime: Optional[Union[datetime, pd.Timestamp]] = None,
    ) -> None:
        """
        Open a long position

        See `bbstrader.btengine.event.SignalEvent` for more details on arguments.
        """
        if dtime is None:
            dtime = self.get_current_dt()
        self._send_order(id, symbol, "LONG", strength, price, quantity, dtime)

    def sell_mkt(
        self,
        id: int,
        symbol: str,
        price: float,
        quantity: int,
        strength: float = 1.0,
        dtime: Optional[Union[datetime, pd.Timestamp]] = None,
    ) -> None:
        """
        Open a short position

        See `bbstrader.btengine.event.SignalEvent` for more details on arguments.
        """
        if dtime is None:
            dtime = self.get_current_dt()
        self._send_order(id, symbol, "SHORT", strength, price, quantity, dtime)

    def close_positions(
        self,
        id: int,
        symbol: str,
        price: float,
        quantity: int,
        strength: float = 1.0,
        dtime: Optional[Union[datetime, pd.Timestamp]] = None,
    ) -> None:
        """
        Close a position or exit all positions

        See `bbstrader.btengine.event.SignalEvent` for more details on arguments.
        """
        if dtime is None:
            dtime = self.get_current_dt()
        self._send_order(id, symbol, "EXIT", strength, price, quantity, dtime)

    def buy_stop(
        self,
        id: int,
        symbol: str,
        price: float,
        quantity: int,
        strength: float = 1.0,
        dtime: Optional[Union[datetime, pd.Timestamp]] = None,
    ) -> None:
        """
        Open a pending order to buy at a stop price

        See `bbstrader.btengine.event.SignalEvent` for more details on arguments.
        """
        current_price = self.data.get_latest_bar_value(symbol, "close")
        if price <= current_price:
            raise ValueError(
                "The buy_stop price must be greater than the current price."
            )
        if dtime is None:
            dtime = self.get_current_dt()
        order = SignalEvent(
            id,
            symbol,
            dtime,
            "LONG",
            quantity=quantity,
            strength=strength,
            price=price,  # type: ignore
        )
        self._orders[symbol]["BSTP"].append(order)

    def sell_stop(
        self,
        id: int,
        symbol: str,
        price: float,
        quantity: int,
        strength: float = 1.0,
        dtime: Optional[Union[datetime, pd.Timestamp]] = None,
    ) -> None:
        """
        Open a pending order to sell at a stop price

        See `bbstrader.btengine.event.SignalEvent` for more details on arguments.
        """
        current_price = self.data.get_latest_bar_value(symbol, "close")
        if price >= current_price:
            raise ValueError("The sell_stop price must be less than the current price.")
        if dtime is None:
            dtime = self.get_current_dt()
        order = SignalEvent(
            id,
            symbol,
            dtime,  # type: ignore
            "SHORT",
            quantity=quantity,
            strength=strength,
            price=price,
        )
        self._orders[symbol]["SSTP"].append(order)

    def buy_limit(
        self,
        id: int,
        symbol: str,
        price: float,
        quantity: int,
        strength: float = 1.0,
        dtime: Optional[Union[datetime, pd.Timestamp]] = None,
    ) -> None:
        """
        Open a pending order to buy at a limit price

        See `bbstrader.btengine.event.SignalEvent` for more details on arguments.
        """
        current_price = self.data.get_latest_bar_value(symbol, "close")
        if price >= current_price:
            raise ValueError("The buy_limit price must be less than the current price.")
        if dtime is None:
            dtime = self.get_current_dt()
        order = SignalEvent(
            id,
            symbol,
            dtime,
            "LONG",
            quantity=quantity,
            strength=strength,
            price=price,  # type: ignore
        )
        self._orders[symbol]["BLMT"].append(order)

    def sell_limit(
        self,
        id: int,
        symbol: str,
        price: float,
        quantity: int,
        strength: float = 1.0,
        dtime: Optional[Union[datetime, pd.Timestamp]] = None,
    ) -> None:
        """
        Open a pending order to sell at a limit price

        See `bbstrader.btengine.event.SignalEvent` for more details on arguments.
        """
        current_price = self.data.get_latest_bar_value(symbol, "close")
        if price <= current_price:
            raise ValueError(
                "The sell_limit price must be greater than the current price."
            )
        if dtime is None:
            dtime = self.get_current_dt()
        order = SignalEvent(
            id,
            symbol,
            dtime,  # type: ignore
            "SHORT",
            quantity=quantity,
            strength=strength,
            price=price,
        )
        self._orders[symbol]["SLMT"].append(order)

    def buy_stop_limit(
        self,
        id: int,
        symbol: str,
        price: float,
        stoplimit: float,
        quantity: int,
        strength: float = 1.0,
        dtime: Optional[Union[datetime, pd.Timestamp]] = None,
    ) -> None:
        """
        Open a pending order to buy at a stop-limit price

        See `bbstrader.btengine.event.SignalEvent` for more details on arguments.
        """
        current_price = self.data.get_latest_bar_value(symbol, "close")
        if price <= current_price:
            raise ValueError(
                f"The stop price {price} must be greater than the current price {current_price}."
            )
        if price >= stoplimit:
            raise ValueError(
                f"The stop-limit price {stoplimit} must be greater than the price {price}."
            )
        if dtime is None:
            dtime = self.get_current_dt()
        order = SignalEvent(
            id,
            symbol,
            dtime,  # type: ignore
            "LONG",
            quantity=quantity,
            strength=strength,
            price=price,
            stoplimit=stoplimit,
        )
        self._orders[symbol]["BSTPLMT"].append(order)

    def sell_stop_limit(
        self,
        id: int,
        symbol: str,
        price: float,
        stoplimit: float,
        quantity: int,
        strength: float = 1.0,
        dtime: Optional[Union[datetime, pd.Timestamp]] = None,
    ) -> None:
        """
        Open a pending order to sell at a stop-limit price

        See `bbstrader.btengine.event.SignalEvent` for more details on arguments.
        """
        current_price = self.data.get_latest_bar_value(symbol, "close")
        if price >= current_price:
            raise ValueError(
                f"The stop price {price} must be less than the current price {current_price}."
            )
        if price <= stoplimit:
            raise ValueError(
                f"The stop-limit price {stoplimit} must be less than the price {price}."
            )
        if dtime is None:
            dtime = self.get_current_dt()
        order = SignalEvent(
            id,
            symbol,
            dtime,  # type: ignore
            "SHORT",
            quantity=quantity,
            strength=strength,
            price=price,
            stoplimit=stoplimit,
        )
        self._orders[symbol]["SSTPLMT"].append(order)

    def check_pending_orders(self) -> None:
        """
        Check for pending orders and handle them accordingly.
        """

        def logmsg(
            order: SignalEvent,
            type: str,
            symbol: str,
            dtime: Union[datetime, pd.Timestamp],
        ) -> None:
            self.logger.info(
                f"{type} ORDER EXECUTED: SYMBOL={symbol}, QUANTITY={order.quantity}, "
                f"PRICE @ {round(order.price, 5)}",  # type: ignore
                custom_time=dtime,
            )

        def process_orders(
            order_type: str,
            condition: Callable[[SignalEvent], bool],
            execute_fn: Callable[[SignalEvent], None],
            log_label: str,
            symbol: str,
            dtime: Union[datetime, pd.Timestamp],
        ) -> None:
            for order in self._orders[symbol][order_type].copy():
                if condition(order):
                    execute_fn(order)
                    try:
                        self._orders[symbol][order_type].remove(order)
                        assert order not in self._orders[symbol][order_type]
                    except AssertionError:
                        self._orders[symbol][order_type] = [
                            o for o in self._orders[symbol][order_type] if o != order
                        ]
                    logmsg(order, log_label, symbol, dtime)

        for symbol in self.symbols:
            dtime = self.data.get_latest_bar_datetime(symbol)
            latest_close = self.data.get_latest_bar_value(symbol, "close")

            process_orders(
                "BLMT",
                lambda o: latest_close <= o.price,  # type: ignore
                lambda o: self.buy_mkt(
                    o.strategy_id,
                    symbol,
                    o.price,
                    o.quantity,
                    dtime=dtime,  # type: ignore
                ),
                "BUY LIMIT",
                symbol,
                dtime,
            )

            process_orders(
                "SLMT",
                lambda o: latest_close >= o.price,  # type: ignore
                lambda o: self.sell_mkt(
                    o.strategy_id,
                    symbol,
                    o.price,
                    o.quantity,
                    dtime=dtime,  # type: ignore
                ),
                "SELL LIMIT",
                symbol,
                dtime,
            )

            process_orders(
                "BSTP",
                lambda o: latest_close >= o.price,  # type: ignore
                lambda o: self.buy_mkt(
                    o.strategy_id,
                    symbol,
                    o.price,
                    o.quantity,
                    dtime=dtime,  # type: ignore
                ),
                "BUY STOP",
                symbol,
                dtime,
            )

            process_orders(
                "SSTP",
                lambda o: latest_close <= o.price,  # type: ignore
                lambda o: self.sell_mkt(
                    o.strategy_id,
                    symbol,
                    o.price,
                    o.quantity,
                    dtime=dtime,  # type: ignore
                ),
                "SELL STOP",
                symbol,
                dtime,
            )

            process_orders(
                "BSTPLMT",
                lambda o: latest_close >= o.price,  # type: ignore
                lambda o: self.buy_limit(
                    o.strategy_id,
                    symbol,
                    o.stoplimit,
                    o.quantity,
                    dtime=dtime,  # type: ignore
                ),
                "BUY STOP LIMIT",
                symbol,
                dtime,
            )

            process_orders(
                "SSTPLMT",
                lambda o: latest_close <= o.price,  # type: ignore
                lambda o: self.sell_limit(
                    o.strategy_id,
                    symbol,
                    o.stoplimit,
                    o.quantity,
                    dtime=dtime,  # type: ignore
                ),
                "SELL STOP LIMIT",
                symbol,
                dtime,
            )
