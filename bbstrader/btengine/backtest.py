import pprint
import queue
from queue import Queue
import time
from datetime import datetime
from bbstrader.btengine.data import DataHandler
from bbstrader.btengine.execution import ExecutionHandler
from bbstrader.btengine.portfolio import Portfolio
from bbstrader.btengine.strategy import Strategy


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

    The initialisation of the Backtest object requires the `data directory`, 
    the full `symbol list` of traded symbols, the `initial capital`, the `heartbeat` 
    time in milliseconds, the `start datetime` stamp of the backtest as well as the `DataHandler`, 
    `ExecutionHandler`, `Portfolio` and `Strategy` objects.

    A Queue is used to hold the events. The signals, orders and fills are counted.
    For a `MarketEvent`, the `Strategy` object is told to recalculate new signals, 
    while the `Portfolio` object is told to reindex the time. If a `SignalEvent` 
    object is received the `Portfolio` is told to handle the new signal and convert it into a 
    set of `OrderEvents`, if appropriate. If an `OrderEvent` is received the `ExecutionHandler` 
    is sent the order to be transmitted to the broker (if in a real trading setting). 
    Finally, if a `FillEvent` is received, the Portfolio will update itself to be aware of 
    the new positions..
    """

    def __init__(
        self,
        csv_dir: str,
        symbol_list: list[str],
        initial_capital: float,
        heartbeat: float,
        start_date: datetime,
        data_handler: DataHandler,
        execution_handler: ExecutionHandler,
        portfolio: Portfolio,
        strategy: Strategy,
        /,
        **kwargs
    ):
        """
        Initialises the backtest.

        Args:
            csv_dir (str): The hard root to the CSV data directory.
            symbol_list (lsit[str]): The list of symbol strings.
            intial_capital (float): The starting capital for the portfolio.
            heartbeat (float): Backtest "heartbeat" in seconds
            start_date (datetime): The start datetime of the strategy.
            data_handler (DataHandler) : Handles the market data feed.
            execution_handler (ExecutionHandler) : Handles the orders/fills for trades.
            portfolio (Portfolio) : Keeps track of portfolio current and prior positions.
            strategy (Portfolio): Generates signals based on market data.
            kwargs (dict): Optional parameters (See data_handler, portfolio, strategy classes).
        """
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date

        self.data_handler_cls = data_handler
        self.execution_handler_cls = execution_handler
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy
        self.kwargs = kwargs

        self.events = queue.Queue()

        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1

        self._generate_trading_instances()

    def _generate_trading_instances(self):
        """
        Generates the trading instance objects from
        their class types.
        """
        print(
            f"\nStarting Backtest on {self.symbol_list} "
            f"with ${self.initial_capital} Initial Capital\n"
        )
        self.data_handler = self.data_handler_cls(
            self.events, self.csv_dir, self.symbol_list
        )
        self.strategy = self.strategy_cls(
            self.data_handler, self.events, **self.kwargs
        )
        self.portfolio = self.portfolio_cls(
            self.data_handler,
            self.events,
            self.start_date,
            self.initial_capital, **self.kwargs
        )
        self.execution_handler = self.execution_handler_cls(self.events)

    def _run_backtest(self):
        """
        Executes the backtest.
        """
        i = 0
        while True:
            i += 1
            print(i)
            # Update the market bars
            if self.data_handler.continue_backtest == True:
                self.data_handler.update_bars()
            else:
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

        time.sleep(self.heartbeat)

    def _output_performance(self):
        """
        Outputs the strategy performance from the backtest.
        """
        self.portfolio.create_equity_curve_dataframe()

        print("\nCreating summary stats...")
        stats = self.portfolio.output_summary_stats()

        print("\nCreating equity curve...")
        print(f"{self.portfolio.equity_curve.tail(10)}\n")
        print("==== Summary Stats ====")
        pprint.pprint(stats)
        stat2 = {}
        stat2['Signals'] = self.signals
        stat2['Orders'] = self.orders
        stat2['Fills'] = self.fills
        pprint.pprint(stat2)

    def simulate_trading(self):
        """
        Simulates the backtest and outputs portfolio performance.
        """
        self._run_backtest()
        self._output_performance()
