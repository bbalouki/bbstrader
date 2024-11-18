import pprint
import queue
import time
import yfinance as yf
from queue import Queue
from datetime import datetime
from bbstrader.btengine.data import *
from bbstrader.btengine.execution import *
from bbstrader.btengine.portfolio import Portfolio
from bbstrader.btengine.event import SignalEvent
from bbstrader.btengine.strategy import Strategy
from typing import Literal, Optional, List
from tabulate import tabulate

__all__ = [
    "Backtest",
    "BacktestEngine",
    "run_backtest"
]

class Backtest(object):
    """
    The `Backtest()` object encapsulates the event-handling logic and essentially 
    ties together all of the other classes.

    The Backtest object is designed to carry out a nested while-loop event-driven system 
    in order to handle the events placed on the `Event` Queue object. 
    The outer while-loop is known as the "heartbeat loop" and decides the temporal resolution of 
    the backtesting system. In a live environment this value will be a positive number, 
    such as 600 seconds (every ten minutes). Thus the market data and positions 
    will only be updated on this timeframe.

    For the backtester described here the "heartbeat" can be set to zero, 
    irrespective of the strategy frequency, since the data is already available by virtue of 
    the fact it is historical! We can run the backtest at whatever speed we like, 
    since the event-driven system is agnostic to when the data became available, 
    so long as it has an associated timestamp. 

    The inner while-loop actually processes the signals and sends them to the correct 
    component depending upon the event type. Thus the Event Queue is continually being 
    populated and depopulated with events. This is what it means for a system to be event-driven.

    The initialisation of the Backtest object requires the full `symbol list` of traded symbols, 
    the `initial capital`, the `heartbeat` time in milliseconds, the `start datetime` stamp 
    of the backtest as well as the `DataHandler`, `ExecutionHandler`, `Strategy` objects
    and additionnal `kwargs` based on the `ExecutionHandler`, the `DataHandler`, and the `Strategy` used.

    A Queue is used to hold the events. The signals, orders and fills are counted.
    For a `MarketEvent`, the `Strategy` object is told to recalculate new signals, 
    while the `Portfolio` object is told to reindex the time. If a `SignalEvent` 
    object is received the `Portfolio` is told to handle the new signal and convert it into a 
    set of `OrderEvents`, if appropriate. If an `OrderEvent` is received the `ExecutionHandler` 
    is sent the order to be transmitted to the broker (if in a real trading setting). 
    Finally, if a `FillEvent` is received, the Portfolio will update itself to be aware of 
    the new positions.

    """
    pass


class BacktestEngine(Backtest):
    __doc__ = Backtest.__doc__
    def __init__(
        self,
        symbol_list: List[str],
        initial_capital: float,
        heartbeat: float,
        start_date: datetime,
        data_handler: DataHandler,
        execution_handler: ExecutionHandler,
        strategy: Strategy,
        /,
        **kwargs
    ):
        """
        Initialises the backtest.

        Args:
            symbol_list (List[str]): The list of symbol strings.
            intial_capital (float): The starting capital for the portfolio.
            heartbeat (float): Backtest "heartbeat" in seconds
            start_date (datetime): The start datetime of the strategy.
            data_handler (DataHandler) : Handles the market data feed.
            execution_handler (ExecutionHandler) : Handles the orders/fills for trades.
            strategy (Strategy): Generates signals based on market data.
            kwargs : Additional parameters based on the `ExecutionHandler`, 
                the `DataHandler`, the `Strategy` used and the `Portfolio`.
                - show_equity (bool): Show the equity curve of the portfolio.
                - stats_file (str): File to save the summary stats.
        """
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date

        self.dh_cls = data_handler
        self.eh_cls = execution_handler
        self.strategy_cls = strategy
        self.kwargs = kwargs

        self.events = queue.Queue()

        self.signals = 0
        self.orders = 0
        self.fills = 0

        self._generate_trading_instances()
        self.show_equity = kwargs.get("show_equity", False)
        self.stats_file = kwargs.get("stats_file", None)

    def _generate_trading_instances(self):
        """
        Generates the trading instance objects from
        their class types.
        """
        print(
            f"\n[=======   STARTING BACKTEST   =======]\n"
            f"START DATE: {self.start_date} \n"
            f"INITIAL CAPITAL: {self.initial_capital}\n"
        )
        self.data_handler: DataHandler = self.dh_cls(
            self.events, self.symbol_list, **self.kwargs
        )
        self.strategy: Strategy = self.strategy_cls(
            bars=self.data_handler, events=self.events, **self.kwargs
        )
        self.portfolio = Portfolio(
            self.data_handler,
            self.events,
            self.start_date,
            self.initial_capital, **self.kwargs
        )
        self.execution_handler: ExecutionHandler = self.eh_cls(
            self.events, self.data_handler,  **self.kwargs)

    def _run_backtest(self):
        """
        Executes the backtest.
        """
        i = 0
        while True:
            i += 1
            value = self.portfolio.all_holdings[-1]['Total']
            if self.data_handler.continue_backtest == True:
                # Update the market bars
                self.data_handler.update_bars()
                self.strategy.check_pending_orders()
                self.strategy.get_update_from_portfolio(
                    self.portfolio.current_positions,
                    self.portfolio.current_holdings
                )
                self.strategy.cash = value
            else:
                print("\n[======= BACKTEST COMPLETED =======]")
                print(f"END DATE: {self.data_handler.get_latest_bar_datetime(self.symbol_list[0])}")
                print(f"TOTAL BARS: {i} ")
                print(f"PORFOLIO VALUE: {round(value, 2)}")
                break

            # Handle the events
            while True:
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    break
                else:
                    if event is not None:
                        if event.type == 'MARKET':
                            self.strategy.calculate_signals(event)
                            self.portfolio.update_timeindex(event)

                        elif event.type == 'SIGNAL':
                            self.signals += 1
                            self.portfolio.update_signal(event)

                        elif event.type == 'ORDER':
                            self.orders += 1
                            self.execution_handler.execute_order(event)

                        elif event.type == 'FILL':
                            self.fills += 1
                            self.portfolio.update_fill(event)
                            self.strategy.update_trades_from_fill(event)

            time.sleep(self.heartbeat)

    def _output_performance(self):
        """
        Outputs the strategy performance from the backtest.
        """
        self.portfolio.create_equity_curve_dataframe()

        print("\nCreating summary stats...")
        stats = self.portfolio.output_summary_stats()
        print("[======= Summary Stats =======]")
        stat2 = {}
        stat2['Signals'] = self.signals
        stat2['Orders'] = self.orders
        stat2['Fills'] = self.fills
        stats.extend(stat2.items())
        tab_stats = tabulate(stats, headers=["Metric", "Value"], tablefmt="outline")
        print(tab_stats, "\n")
        if self.stats_file:
            with open(self.stats_file, 'a') as f:
                f.write("\n[======= Summary Stats =======]\n")
                f.write(tab_stats)
                f.write("\n")

        if self.show_equity:
            print("\nCreating equity curve...")
            print("\n[======= PORTFOLIO SUMMARY =======]")
            print(
                tabulate(
                    self.portfolio.equity_curve.tail(10), 
                    headers="keys", 
                    tablefmt="outline"), 
                    "\n"
                )
    
    def simulate_trading(self):
        """
        Simulates the backtest and outputs portfolio performance.
        
        Returns:
            pd.DataFrame: The portfilio values over time (capital, equity, returns etc.)
        """
        self._run_backtest()
        self._output_performance()
        return self.portfolio.equity_curve


def run_backtest(
    symbol_list: List[str],
    start_date: datetime,
    data_handler: DataHandler,
    strategy: Strategy,
    exc_handler: Optional[ExecutionHandler] = None,
    initial_capital: float = 100000.0,
    heartbeat: float = 0.0,
    **kwargs
):
    """
    Runs a backtest simulation based on a `DataHandler`, `Strategy`, and `ExecutionHandler`.

    Args:
        symbol_list (List[str]): List of symbol strings for the assets to be backtested.
        
        start_date (datetime): Start date of the backtest.
        
        data_handler (DataHandler): An instance of the `DataHandler` class, responsible for managing 
            and processing market data. Available options include `CSVDataHandler`, 
            `MT5DataHandler`, and `YFDataHandler`. Ensure that the `DataHandler` 
            instance is initialized before passing it to the function.

        strategy (Strategy): The trading strategy to be employed during the backtest.
            The strategy must be an instance of `Strategy` and should include the following attributes:
            - `bars` (DataHandler): The `DataHandler` instance for the strategy.
            - `events` (Queue): Queue instance for managing events.
            - `symbol_list` (List[str]): List of symbols to trade.
            - `mode` (str): 'live' or 'backtest'.

            Additional parameters specific to the strategy should be passed in `**kwargs`.
            The strategy class must implement a `calculate_signals` method to generate `SignalEvent`.

        exc_handler (ExecutionHandler, optional): The execution handler for managing order executions. 
            If not provided, a `SimulatedExecutionHandler` will be used by default. This handler must
            implement an `execute_order` method to process `OrderEvent` in the `Backtest` class.

        initial_capital (float, optional): The initial capital for the portfolio in the backtest. 
            Default is 100,000.

        heartbeat (float, optional): Time delay (in seconds) between iterations of the event-driven 
            backtest loop. Default is 0.0, allowing the backtest to run as fast as possible. This could 
            also be used as a time frame in live trading (e.g., 1m, 5m, 15m) with a live `DataHandler`.

        **kwargs: Additional parameters passed to the `Backtest` instance, which may include strategy-specific,
            data handler, portfolio, or execution handler options.
        
    Returns:
        pd.DataFrame: The portfolio values over time (capital, equities, returns etc.).

    Notes:
        This function generates three outputs:
            - A performance summary saved as an HTML file.
            - An equity curve of the portfolio saved as a CSV file.
            - Monthly returns saved as a PNG image.

    Example:
        >>> from bbstrader.trading.strategies import StockIndexSTBOTrading
        >>> from bbstrader.metatrader.utils import config_logger
        >>> from bbstrader.datahandlers import MT5DataHandler
        >>> from bbstrader.execution import MT5ExecutionHandler
        >>> from datetime import datetime
        >>> 
        >>> logger = config_logger('index_trade.log', console_log=True)
        >>> symbol_list = ['[SP500]', 'GERMANY40', '[DJI30]', '[NQ100]']
        >>> start = datetime(2010, 6, 1, 2, 0, 0)
        >>> kwargs = {
        ...     'expected_returns': {'[NQ100]': 1.5, '[SP500]': 1.5, '[DJI30]': 1.0, 'GERMANY40': 1.0},
        ...     'quantities': {'[NQ100]': 15, '[SP500]': 30, '[DJI30]': 5, 'GERMANY40': 10},
        ...     'max_trades': {'[NQ100]': 3, '[SP500]': 3, '[DJI30]': 3, 'GERMANY40': 3},
        ...     'mt5_start': start,
        ...     'time_frame': '15m',
        ...     'strategy_name': 'SISTBO',
        ... }
        >>> run_backtest(
        ...     symbol_list=symbol_list,
        ...     start_date=start,
        ...     data_handler=MT5DataHandler(),
        ...     strategy=StockIndexSTBOTrading(),
        ...     exc_handler=MT5ExecutionHandler(),
        ...     initial_capital=100000.0,
        ...     heartbeat=0.0,
        ...     **kwargs
        ... )
    """
    if exc_handler is None:
        execution_handler = SimExecutionHandler
    else:
        execution_handler = exc_handler
    engine = BacktestEngine(
        symbol_list, initial_capital, heartbeat, start_date,
        data_handler, execution_handler, strategy, **kwargs
    )
    portfolio = engine.simulate_trading()
    return portfolio


class CerebroEngine:...


class ZiplineEngine:...


def run_backtest_with(engine: Literal["bbstrader", "cerebro", "zipline"], **kwargs):
    """
    """
    if engine == "bbstrader":
        return run_backtest(
            symbol_list=kwargs.get("symbol_list"),
            start_date=kwargs.get("start_date"),
            data_handler=kwargs.get("data_handler"),
            strategy=kwargs.get("strategy"),
            exc_handler=kwargs.get("exc_handler"),
            initial_capital=kwargs.get("initial_capital", 100000.0),
            heartbeat=kwargs.get("heartbeat", 0.0),
            **kwargs
        )
    elif engine == "cerebro":
        #TODO: 
        pass
    elif engine == "zipline":
        #TODO: 
        pass