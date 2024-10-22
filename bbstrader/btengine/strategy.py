from abc import ABCMeta, abstractmethod
import pytz
import pandas as pd
import numpy as np
from queue import Queue
from datetime import datetime
from bbstrader.config import config_logger
from bbstrader.btengine.event import SignalEvent
from bbstrader.btengine.data import DataHandler
from bbstrader.metatrader.account import Account
from bbstrader.metatrader.rates import Rates
from typing import (
    Dict,
    Union,
    Any,
    List,
    Literal
)

__all__ = ['Strategy', 'MT5Strategy']

class Strategy(metaclass=ABCMeta):
    """
    A `Strategy()` object encapsulates all calculation on market data 
    that generate advisory signals to a `Portfolio` object. Thus all of 
    the "strategy logic" resides within this class. We opted to separate 
    out the `Strategy` and `Portfolio` objects for this backtester, 
    since we believe this is more amenable to the situation of multiple 
    strategies feeding "ideas" to a larger `Portfolio`, which then can handle 
    its own risk (such as sector allocation, leverage). In higher frequency trading, 
    the strategy and portfolio concepts will be tightly coupled and extremely 
    hardware dependent.

    At this stage in the event-driven backtester development there is no concept of 
    an indicator or filter, such as those found in technical trading. These are also 
    good candidates for creating a class hierarchy.

    The strategy hierarchy is relatively simple as it consists of an abstract 
    base class with a single pure virtual method for generating `SignalEvent` objects. 
    """

    @abstractmethod
    def calculate_signals(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            "Should implement calculate_signals()"
        )

    def check_pending_orders(self): ...


class MT5Strategy(Strategy):
    """
    A `MT5Strategy()` object is a subclass of `Strategy` that is used to 
    calculate signals for the MetaTrader 5 trading platform. The signals
    are generated by the `MT5Strategy` object and sent to the the `MT5ExecutionEngine`
    for live trading and `MT5BacktestEngine` objects for backtesting.
    """

    def __init__(self, events: Queue=None, symbol_list: List[str]=None,
                 bars: DataHandler=None, mode: str=None, **kwargs):
        self.events = events
        self.data = bars
        self.symbols = symbol_list
        self.mode = mode
        self.logger = kwargs.get("logger", config_logger("mt5_strategy.log"))
        self._construct_positions_and_orders()

    def _construct_positions_and_orders(self):
        self.positions: Dict[str, Dict[str, int]] = {}
        self.orders: Dict[str, Dict[str, List[SignalEvent]]] = {}
        positions = ['LONG', 'SHORT']
        orders = ['BLMT', 'BSTP', 'BSTPLMT', 'SLMT', 'SSTP', 'SSTPLMT']
        for symbol in self.symbols:
            self.positions[symbol] = {position: 0 for position in positions}
            self.orders[symbol] = {order: [] for order in orders}

    def calculate_signals(self, *args, **kwargs
                          ) -> Dict[str, Union[str, dict, None]] | None:
        """
        Provides the mechanisms to calculate signals for the strategy.
        This methods should return a dictionary of symbols and their respective signals.
        The returned signals should be either string or dictionary objects.

        If a string is used, it should be:
        - ``LONG`` , ``BMKT``, ``BLMT``, ``BSTP``, ``BSTPLMT`` for a long signal (market, limit, stop, stop-limit).
        - ``SHORT``, ``SMKT``, ``SLMT``, ``SSTP``, ``SSTPLMT`` for a short signal (market, limit, stop, stop-limit).
        - ``EXIT``, ``EXIT_LONG``, ``EXIT_LONG_STOP``, ``EXIT_LONG_LIMIT``, ``EXIT_LONG_STOP_LIMIT`` for an exit signal (long).
        - ``EXIT_SHORT``, ``EXIT_SHORT_STOP``, ``EXIT_SHORT_LIMIT``, ``EXIT_SHORT_STOP_LIMIT`` for an exit signal (short).
        - ``EXIT_ALL_ORDERS`` for cancelling all orders.
        - ``EXIT_ALL_POSITIONS`` for exiting all positions.
        - ``EXIT_PROFITABLES`` for exiting all profitable positions.
        - ``EXIT_LOSINGS`` for exiting all losing positions.

        The signals could also be ``EXIT_STOP``, ``EXIT_LIMIT``, ``EXIT_STOP_LIMIT`` for exiting a position.

        If a dictionary is used, it should be:
        for each symbol, a dictionary with the following keys
        - ``action``: The action to take for the symbol (LONG, SHORT, EXIT, etc.)
        - ``price``: The price at which to execute the action.
        - ``stoplimit``: The stop-limit price for STOP-LIMIT orders.

        The dictionary can be use for pending orders (limit, stop, stop-limit) where the price is required.
        """
        pass

    def get_quantities(self, quantities: Union[None, dict, int]) -> dict:
        if quantities is None:
            return {symbol: None for symbol in self.symbols}
        if isinstance(quantities, dict):
            return quantities
        elif isinstance(quantities, int):
            return {symbol: quantities for symbol in self.symbols}
    
    def _send_order(self, id,  symbol: str, signal: str, strength: float, price: float,
                    quantity: int, dtime: datetime | pd.Timestamp):

        position = SignalEvent(id, symbol, dtime, signal,
                               quantity=quantity, strength=strength, price=price)
        self.events.put(position)
        self.logger.info(
            f"{signal} ORDER EXECUTED: SYMBOL={symbol}, QUANTITY={quantity}, PRICE @{price}", custom_time=dtime)

    def buy(self, id: int, symbol: str, price: float, quantity: int, 
            strength: float=1.0, dtime: datetime | pd.Timestamp=None):
        """

        """
        self._send_order(id, symbol, 'LONG', strength, price, quantity, dtime)
        self.positions[symbol]['LONG'] += quantity

    def sell(self, id, symbol, price, quantity, strength=1.0, dtime=None):
        """

        """
        self._send_order(id, symbol, 'SHORT', strength,  price, quantity, dtime)
        self.positions[symbol]['SHORT'] += quantity

    def close(self, id, symbol, price, quantity, strength=1.0, dtime=None):
        """

        """
        self._send_order(id, symbol, 'EXIT', strength, price, quantity, dtime)
        self.positions[symbol]['LONG'] -= quantity

    def buy_stop(self, iid, symbol, price, quantity, strength=1.0, dtime=None):
        """

        """
        current_price = self.data.get_latest_bar_value(symbol, 'close')
        if price <= current_price:
            raise ValueError(
                "The buy_stop price must be greater than the current price.")
        order = SignalEvent(id, symbol, dtime, 'LONG',
                            quantity=quantity, strength=strength, price=price)
        self.orders[symbol]['BSTP'].append(order)

    def sell_stop(self, id, symbol, price, quantity, strength=1.0, dtime=None):
        """

        """
        current_price = self.data.get_latest_bar_value(symbol, 'close')
        if price >= current_price:
            raise ValueError(
                "The sell_stop price must be less than the current price.")
        order = SignalEvent(id, symbol, dtime, 'SHORT',
                            quantity=quantity, strength=strength, price=price)
        self.orders[symbol]['SSTP'].append(order)

    def buy_limit(self, id, symbol, price, quantity, strength=1.0, dtime=None):
        """

        """
        current_price = self.data.get_latest_bar_value(symbol, 'close')
        if price >= current_price:
            raise ValueError(
                "The buy_limit price must be less than the current price.")
        order = SignalEvent(id, symbol, dtime, 'LONG',
                            quantity=quantity, strength=strength, price=price)
        self.orders[symbol]['BLMT'].append(order)

    def sell_limit(self, id, symbol, price, quantity, strength=1.0, dtime=None):
        """

        """
        current_price = self.data.get_latest_bar_value(symbol, 'close')
        if price <= current_price:
            raise ValueError(
                "The sell_limit price must be greater than the current price.")
        order = SignalEvent(id, symbol, dtime, 'SHORT',
                            quantity=quantity, strength=strength, price=price)
        self.orders[symbol]['SLMT'].append(order)

    def buy_stop_limit(self, id: int, symbol: str, price: float, stoplimit: float, 
                       quantity: int, strength: float=1.0, dtime: datetime | pd.Timestamp = None):
        """

        """
        current_price = self.data.get_latest_bar_value(symbol, 'close')
        if price <= current_price:
            raise ValueError(
                f"The stop price {price} must be greater than the current price {current_price}.")
        if price >= stoplimit:
            raise ValueError(
                f"The stop-limit price {stoplimit} must be greater than the price {price}.")
        order = SignalEvent(id, symbol, dtime, 'LONG',
                            quantity=quantity, strength=strength, price=price, stoplimit=stoplimit)
        self.orders[symbol]['BSTPLMT'].append(order)

    def sell_stop_limit(self, id, symbol, price, stoplimit, quantity, strength=1.0, dtime=None):
        """

        """
        current_price = self.data.get_latest_bar_value(symbol, 'close')
        if price >= current_price:
            raise ValueError(
                f"The stop price {price} must be less than the current price {current_price}.")
        if price <= stoplimit:
            raise ValueError(
                f"The stop-limit price {stoplimit} must be less than the price {price}.")
        order = SignalEvent(id, symbol, dtime, 'SHORT',
                            quantity=quantity, strength=strength, price=price, stoplimit=stoplimit)
        self.orders[symbol]['SSTPLMT'].append(order)

    def check_pending_orders(self):
        for symbol in self.symbols:
            dtime = self.data.get_latest_bar_datetime(symbol)
            for order in self.orders[symbol]['BLMT'].copy():
                if self.data.get_latest_bar_value(symbol, 'close') <= order.price:
                    self.buy(order.strategy_id, symbol,
                             order.price, order.quantity, dtime)
                    self.logger.info(
                        f"BUY LIMIT ORDER EXECUTED: SYMBOL={symbol}, QUANTITY={order.quantity}, "
                        f"PRICE @ {order.price}", custom_time=dtime)
                    self.orders[symbol]['BLMT'].remove(order)
            for order in self.orders[symbol]['SLMT'].copy():
                if self.data.get_latest_bar_value(symbol, 'close') >= order.price:
                    self.sell(order.strategy_id, symbol,
                              order.price, order.quantity, dtime)
                    self.logger.info(
                        f"SELL LIMIT ORDER EXECUTED: SYMBOL={symbol}, QUANTITY={order.quantity}, "
                        f"PRICE @ {order.price}", custom_time=dtime)
                    self.orders[symbol]['SLMT'].remove(order)
            for order in self.orders[symbol]['BSTP'].copy():
                if self.data.get_latest_bar_value(symbol, 'close') >= order.price:
                    self.buy(order.strategy_id, symbol,
                             order.price, order.quantity, dtime)
                    self.logger.info(
                        f"BUY STOP ORDER EXECUTED: SYMBOL={symbol}, QUANTITY={order.quantity}, "
                        f"PRICE @ {order.price}", custom_time=dtime)
                    self.orders[symbol]['BSTP'].remove(order)
            for order in self.orders[symbol]['SSTP'].copy():
                if self.data.get_latest_bar_value(symbol, 'close') <= order.price:
                    self.sell(order.strategy_id, symbol,
                              order.price, order.quantity, dtime)
                    self.logger.info(
                        f"SELL STOP ORDER EXECUTED: SYMBOL={symbol}, QUANTITY={order.quantity}, "
                        f"PRICE @ {order.price}", custom_time=dtime)
                    self.orders[symbol]['SSTP'].remove(order)
            for order in self.orders[symbol]['BSTPLMT'].copy():
                if self.data.get_latest_bar_value(symbol, 'close') >= order.price:
                    self.buy_limit(order.strategy_id, symbol,
                                   order.stoplimit, order.quantity, dtime)
                    self.logger.info(
                        f"BUY STOP LIMIT ORDER EXECUTED: SYMBOL={symbol}, QUANTITY={order.quantity}, "
                        f"PRICE @ {order.price}", custom_time=dtime)
                    self.orders[symbol]['BSTPLMT'].remove(order)
            for order in self.orders[symbol]['SSTPLMT'].copy():
                if self.data.get_latest_bar_value(symbol, 'close') <= order.price:
                    self.sell_limit(order.strategy_id, symbol,
                                    order.stoplimit, order.quantity, dtime)
                    self.logger.info(
                        f"SELL STOP LIMIT ORDER EXECUTED: SYMBOL={symbol}, QUANTITY={order.quantity}, "
                        f"PRICE @ {order.price}", custom_time=dtime)
                    self.orders[symbol]['SSTPLMT'].remove(order)

    def get_asset_returns(self,
                          symbol_list: List[str],
                          window: int,
                          bars: DataHandler = None,
                          mode: Literal['backtest', 'live'] = 'backtest',
                          tf: str = 'D1'
                          ) -> Dict[str, np.ndarray] | None:
        """
        Get the historical returns of the assets in the symbol list.

        Args:
            bars : DataHandler for market data handling, required for backtest mode.
            symbol_list : List of ticker symbols for the pairs trading strategy.
            mode : Mode of operation for the strategy.
            window : The lookback period for calculating the returns.
            tf : The time frame for the strategy.

        Returns:
            asset_returns : Historical returns of the assets in the symbol list.
        """
        if mode not in ['backtest', 'live']:
            raise ValueError('Mode must be either backtest or live.')
        asset_returns = {}
        if mode == 'backtest':
            if bars is None:
                raise ValueError('DataHandler is required for backtest mode.')
            for asset in symbol_list:
                returns = bars.get_latest_bars_values(
                    asset, 'returns', N=window)
                asset_returns[asset] = returns[~np.isnan(returns)]
        elif mode == 'live':
            for asset in symbol_list:
                rates = Rates(symbol=asset, time_frame=tf, count=window + 1)
                returns = rates.returns.values
                asset_returns[asset] = returns[~np.isnan(returns)]
        if len(asset_returns[symbol_list[0]]) == window:
            return asset_returns
        else:
            return None

    def is_signal_time(self, period_count, signal_inverval) -> bool:
        """
        Check if we can generate a signal based on the current period count.
        We use the signal interval as a form of periodicity or rebalancing period.

        Args:
            period_count : The current period count (e.g., number of bars).
            signal_inverval : The signal interval for generating signals (e.g., every 5 bars).

        Returns:
            bool : True if we can generate a signal, False otherwise
        """
        if period_count == 0 or period_count is None:
            return True
        return period_count % signal_inverval == 0

    def ispositions(self, symbol, strategy_id, position, max_trades, one_true=False, account=None) -> bool:
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
        account = account or Account()
        positions = account.get_positions(symbol=symbol)
        if positions is not None:
            open_positions = [
                pos for pos in positions if pos.type == position
                and pos.magic == strategy_id
            ]
            if one_true:
                return len(open_positions) in range(1, max_trades + 1)
            return len(open_positions) >= max_trades
        return False

    def get_positions_prices(self, symbol, strategy_id, position, account=None):
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
        account = account or Account()
        positions = account.get_positions(symbol=symbol)
        if positions is not None:
            prices = np.array([
                pos.price_open for pos in positions
                if pos.type == position and pos.magic == strategy_id
            ])
            return prices
        return np.array([])

    def get_current_dt(self, time_zone: str = 'US/Eastern') -> datetime:
        return datetime.now(pytz.timezone(time_zone))

    def convert_time_zone(self, dt: datetime | int | pd.Timestamp,
                          from_tz: str = 'UTC',
                          to_tz: str = 'US/Eastern'
                          ) -> pd.Timestamp:
        from_tz = pytz.timezone(from_tz)
        if isinstance(dt, datetime):
            dt = pd.to_datetime(dt, unit='s')
        elif isinstance(dt, int):
            dt = pd.to_datetime(dt, unit='s')
        if dt.tzinfo is None:
            dt = dt.tz_localize(from_tz)
        else:
            dt = dt.tz_convert(from_tz)

        dt_to = dt.tz_convert(pytz.timezone(to_tz))
        return dt_to


class TWSStrategy(Strategy):
    ...
