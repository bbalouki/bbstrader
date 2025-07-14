"""
Strategies module for trading strategies backtesting and execution.

# NOTE
These strategies inherit from the Strategy class, not from MT5Strategy, because we chose to demonstrate the modular approach to building and backtesting strategies using the bbstrader framework.

If these strategies need to be sent to the Mt5ExecutionEngine,
they must return signals as a list of bbstrader.metatrader.trade.TradingSignal objects.

Later, we will implement the Execution Engine for the Interactive Brokers TWS platform.

DISCLAIMER: 
This module is for educational purposes only and should not be
considered as financial advice. Always consult with a qualified financial advisor before making any investment decisions. 
The authors and contributors of this module are not responsible for any financial losses or damages incurred as a result of using 
this module or the information contained herein.
"""

from datetime import datetime
from queue import Queue
from typing import Dict, List, Literal, Optional, Union

from bbstrader.btengine.backtest import BacktestEngine
from bbstrader.btengine.data import DataHandler, MT5DataHandler, YFDataHandler
from bbstrader.btengine.event import Events, SignalEvent
from bbstrader.btengine.execution import MT5ExecutionHandler, SimExecutionHandler
from bbstrader.btengine.strategy import Strategy
from bbstrader.metatrader.account import Account

from bbstrader.metatrader.trade import TradingMode


__all__ = [
    "StockIndexSTBOTrading",
    "test_strategy",
    "get_quantities",
]


def get_quantities(quantities, symbol_list):
    if isinstance(quantities, dict):
        return quantities
    elif isinstance(quantities, int):
        return {symbol: quantities for symbol in symbol_list}



class StockIndexSTBOTrading(Strategy):
    """
    The StockIndexSTBOTrading class implements a stock index Contract for Difference (CFD)
    Buy-Only trading strategy. This strategy is based on the assumption that stock markets
    typically follow a long-term uptrend. The strategy is designed to capitalize on market
    corrections and price dips, where stocks or indices temporarily drop but are expected
    to recover. It operates in two modes: backtest and live, and it is particularly
    tailored to index trading.
    """

    def __init__(
        self,
        bars: DataHandler = None,
        events: Queue = None,
        symbol_list: List[str] = None,
        mode: TradingMode = TradingMode.BACKTEST,
        **kwargs,
    ):
        """
        Args:
            `bars`: `DataHandler` for market data handling.
            `events`: A queue for managing events.
            `symbol_list`: List of ticker symbols for the pairs trading strategy.
            `mode`: Mode of operation for the strategy.
            kwargs : Additional keyword arguments including
                - rr (float, default: 3.0): The risk-reward ratio used to determine exit points.
                - epsilon (float, default: 0.1): The percentage threshold for price changes when considering new highs or lows.
                - expected_returns (dict): Expected return percentages for each symbol in the symbol list.
                - quantities (int, default: 100): The number of units to trade.
                - max_trades (dict): The maximum number of trades allowed per symbol.
                - logger: A logger object for tracking operations.
                - expert_id (int, default: 5134): Unique identifier for trade positions created by this strategy.
        """
        self.bars = bars
        self.events = events
        self.symbol_list = symbol_list or self.bars.symbol_list
        self.mode = mode

        self.account = Account()

        self.rr = kwargs.get("rr", 3.0)
        self.epsilon = kwargs.get("epsilon", 0.1)
        self._initialize(**kwargs)
        self.logger = kwargs.get("logger")
        self.ID = kwargs.get("expert_id", 5134)

    def _initialize(self, **kwargs):
        symbols = self.symbol_list.copy()
        returns = kwargs.get("expected_returns")
        quantities = kwargs.get("quantities", 100)
        max_trades = kwargs.get("max_trades")

        self.expeted_return = {index: returns[index] for index in symbols}
        self.max_trades = {index: max_trades[index] for index in symbols}
        self.last_price = {index: None for index in symbols}
        self.heightest_price = {index: None for index in symbols}
        self.lowerst_price = {index: None for index in symbols}

        if self.mode == TradingMode.BACKTEST:
            self.qty = get_quantities(quantities, symbols)
            self.num_buys = {index: 0 for index in symbols}
            self.buy_prices = {index: [] for index in symbols}

    def _calculate_pct_change(self, current_price, lh_price):
        return ((current_price - lh_price) / lh_price) * 100

    def calculate_live_signals(self):
        signals = {index: None for index in self.symbol_list}
        for index in self.symbol_list:
            current_price = self.account.get_tick_info(index).ask
            if self.last_price[index] is None:
                self.last_price[index] = current_price
                self.heightest_price[index] = current_price
                self.lowerst_price[index] = current_price
                continue
            else:
                if (
                    self._calculate_pct_change(
                        current_price, self.heightest_price[index]
                    )
                    >= self.epsilon
                ):
                    self.heightest_price[index] = current_price
                elif (
                    self._calculate_pct_change(current_price, self.lowerst_price[index])
                    <= -self.epsilon
                ):
                    self.lowerst_price[index] = current_price

                down_change = self._calculate_pct_change(
                    current_price, self.heightest_price[index]
                )

                if down_change <= -(self.expeted_return[index] / self.rr):
                    signals[index] = "LONG"

                positions = self.account.get_positions(symbol=index)
                if positions is not None:
                    buy_prices = [
                        position.price_open
                        for position in positions
                        if position.type == 0 and position.magic == self.ID
                    ]
                    if len(buy_prices) == 0:
                        continue
                    avg_price = sum(buy_prices) / len(buy_prices)
                    if (
                        self._calculate_pct_change(current_price, avg_price)
                        >= (self.expeted_return[index])
                    ):
                        signals[index] = "EXIT"
                self.logger.info(
                    f"SYMBOL={index} - Hp={self.heightest_price[index]} - "
                    f"Lp={self.lowerst_price[index]} - Cp={current_price} - %chg={round(down_change, 2)}"
                )
        return signals

    def calculate_backtest_signals(self):
        for index in self.symbol_list.copy():
            dt = self.bars.get_latest_bar_datetime(index)
            last_price = self.bars.get_latest_bars_values(index, "close", N=1)

            current_price = last_price[-1]
            if self.last_price[index] is None:
                self.last_price[index] = current_price
                self.heightest_price[index] = current_price
                self.lowerst_price[index] = current_price
                continue
            else:
                if (
                    self._calculate_pct_change(
                        current_price, self.heightest_price[index]
                    )
                    >= self.epsilon
                ):
                    self.heightest_price[index] = current_price
                elif (
                    self._calculate_pct_change(current_price, self.lowerst_price[index])
                    <= -self.epsilon
                ):
                    self.lowerst_price[index] = current_price

            down_change = self._calculate_pct_change(
                current_price, self.heightest_price[index]
            )

            if (
                down_change <= -(self.expeted_return[index] / self.rr)
                and self.num_buys[index] <= self.max_trades[index]
            ):
                signal = SignalEvent(
                    100,
                    index,
                    dt,
                    "LONG",
                    quantity=self.qty[index],
                    price=current_price,
                )
                self.events.put(signal)
                self.num_buys[index] += 1
                self.buy_prices[index].append(current_price)

            elif self.num_buys[index] > 0:
                av_price = sum(self.buy_prices[index]) / len(self.buy_prices[index])
                qty = self.qty[index] * self.num_buys[index]
                if (
                    self._calculate_pct_change(current_price, av_price)
                    >= (self.expeted_return[index])
                ):
                    signal = SignalEvent(
                        100, index, dt, "EXIT", quantity=qty, price=current_price
                    )
                    self.events.put(signal)
                    self.num_buys[index] = 0
                    self.buy_prices[index] = []

    def calculate_signals(self, event=None) -> Dict[str, Union[str, None]]:
        if self.mode == TradingMode.BACKTEST and event is not None:
            if event.type == Events.MARKET:
                self.calculate_backtest_signals()
        elif self.mode == TradingMode.LIVE:
            return self.calculate_live_signals()


def _run_backtest(strategy_name: str, capital: float, symbol_list: list, kwargs: dict):
    """
    Executes a backtest of the specified strategy
    integrating a Hidden Markov Model (HMM) for risk management.
    """
    kwargs["strategy_name"] = strategy_name
    engine = BacktestEngine(
        symbol_list,
        capital,
        0.0,
        datetime.strptime(kwargs["yf_start"], "%Y-%m-%d"),
        kwargs.get("data_handler", YFDataHandler),
        kwargs.get("exc_handler", SimExecutionHandler),
        kwargs.pop("backtester_class"),
        **kwargs,
    )
    engine.simulate_trading()


def _run_sistbo_backtest(capital: float = 100000.0, quantity: int = None):
    ndx = "[NQ100]"
    spx = "[SP500]"
    dji = "[DJI30]"
    dax = "GERMANY40"

    symbol_list = [spx, dax, dji, ndx]
    start = datetime(2010, 6, 1, 2, 0, 0)
    quantity = {ndx: 15, spx: 30, dji: 5, dax: 10}
    kwargs = {
        "expected_returns": {ndx: 1.5, spx: 1.5, dji: 1.0, dax: 1.0},
        "quantities": quantity,
        "max_trades": {ndx: 3, spx: 3, dji: 3, dax: 3},
        "mt5_start": start,
        "yf_start": start.strftime("%Y-%m-%d"),
        "time_frame": "15m",
        "backtester_class": StockIndexSTBOTrading,
        "data_handler": MT5DataHandler,
        "exc_handler": MT5ExecutionHandler,
    }
    _run_backtest("Stock Index Short Term Buy Only ", capital, symbol_list, kwargs)


def test_strategy(
    strategy: Literal["sistbo"] = "sistbo",
    quantity: Optional[int] = 100,
):
    """
    Executes a backtest of the specified strategy

    Args:
        strategy : The strategy to use in test mode. Default is `sma`.
            - `sistbo` Execute `StockIndexSTBOTrading`, for more detail see this class documentation.
        quantity : The quantity of assets to be used in the test backtest. Default is 1000.

    """
    if strategy != "sistbo":
        raise ValueError(
            "Only 'sistbo' strategy is available for testing at the moment."
        )
    _run_sistbo_backtest(quantity=quantity)

