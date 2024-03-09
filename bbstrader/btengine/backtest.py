import pprint
import queue
import time

class Backtest(object):
    """
    Enscapsulates the settings and components for carrying out
    an event-driven backtest.
    """

    def __init__(
        self, csv_dir, symbol_list, initial_capital,
        heartbeat, start_date, data_handler,
        execution_handler, portfolio, strategy, /, **kwargs
    ):
        """
        Initialises the backtest.

        Parameters
        ==========

        :param csv_dir : The hard root to the CSV data directory.
        :param symbol_list : The list of symbol strings.
        :param intial_capital : The starting capital for the portfolio.
        :param heartbeat : Backtest "heartbeat" in seconds
        :param start_date : The start datetime of the strategy.
        :param data_handler (Class) : Handles the market data feed.
        :param execution_handler (Class) : Handles the orders/fills for trades.
        :param portfolio (Class) : Keeps track of portfolio current
            and prior positions.
        :param strategy (Class): Generates signals based on market data.
        :param kwargs : Optional parameters (See data_handler, portfolio, strategy classes).
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