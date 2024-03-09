import datetime
import pandas as pd
from btengine.strategy import Strategy
from btengine.event import SignalEvent
from btengine.backtest import Backtest
from btengine.data import HistoricCSVDataHandler
from btengine.portfolio import Portfolio
from btengine.execution import SimulatedExecutionHandler
from strategies.ou import OrnsteinUhlenbeck
from risk_models.hmm import HMMRiskManager


class OUBacktester(Strategy):
    """
    The `OUBacktester` class is a specialized trading strategy that implements 
    the Ornstein-Uhlenbeck (OU) process for mean-reverting financial time series. 
    This class extends the generic `Strategy` class provided by a backtesting framework
    allowing it to integrate seamlessly with the ecosystem of data handling, signal generation
    event management, and execution handling components of the framework. 
    The strategy is designed to operate on historical market data, specifically 
    targeting a single financial instrument (or a list of instruments) 
    for which the trading signals are to be generated.
    """

    def __init__(self, bars, events, **kwargs):
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events

        self.ticker = kwargs.get('tiker')
        self.p = kwargs.get('p', 20)
        self.n = kwargs.get('n', 10)
        self.qty = kwargs.get('qty', 1000)
        self.csvfile = kwargs.get('csvfile')
        assert self.csvfile is not None

        self.data = self._read_csv(self.csvfile)
        self.ou = OrnsteinUhlenbeck(self.data["Close"].values)

        self.hmm = kwargs.get('model')
        assert self.hmm is not None
        self.window = kwargs.get('window', 50)

        self.LONG = False
        self.SHORT = False

    def _read_csv(self, csvfile):
        df = pd.read_csv(csvfile, header=0,
                         names=["Date", "Open", "High", "Low",
                                "Close", "Adj Close", "Volume"],
                         index_col="Date", parse_dates=True)
        return df

    def create_signal(self):
        returns = self.bars.get_latest_bars_values(
            self.ticker, "Returns", N=self.p
        )
        hmm_returns = self.bars.get_latest_bars_values(
            self.ticker, "Returns", N=self.window
        )
        dt = self.bars.get_latest_bar_datetime(self.ticker)
        if len(returns) >= self.p and len(hmm_returns) >= self.window:
            action = self.ou.calculate_signals(
                rts=returns, p=self.p, n=self.n, th=1)
            regime = self.hmm.which_trade_allowed(hmm_returns)

            if action == "SHORT" and self.LONG:
                signal = SignalEvent(1, self.ticker, dt, "EXIT")
                self.events.put(signal)
                print(dt, "EXIT LONG")
                self.LONG = False

            elif action == "LONG" and self.SHORT:
                signal = SignalEvent(1, self.ticker, dt, "EXIT")
                self.events.put(signal)
                print(dt, "EXIT SHORT")
                self.SHORT = False

            if regime == "LONG":
                if action == "LONG" and not self.LONG:
                    self.LONG = True
                    signal = SignalEvent(1, self.ticker, dt, "LONG", self.qty)
                    self.events.put(signal)
                    print(dt, "LONG")

            elif regime == 'SHORT':
                if action == "SHORT" and not self.SHORT:
                    self.SHORT = True
                    signal = SignalEvent(1, self.ticker, dt, "SHORT", self.qty)
                    self.events.put(signal)
                    print(dt, "SHORT")

    def calculate_signals(self, event):
        if event.type == "MARKET":
            self.create_signal()


def run_ou_backtest(
    symbol_list: list,
    backtest_csv: str,
    ou_csv: str,
    hmm_csv: str,
    start_date: datetime.datetime,
    initial_capital: float = 100000.0,
    heartbeat: float = 0.0,
    **kwargs
):
    """
    Initiates and executes a backtest of the Ornstein-Uhlenbeck (OU) trading strategy
    incorporating risk management through a Hidden Markov Model (HMM).

    Parameters
    ==========
    :param symbol_list (list): A list of symbol strings representing the financial instruments to be traded.
    :param backtest_csv (str): The file path to the CSV containing historical market data for backtesting.
    :param ou_csv (str): The file path to the CSV containing data specifically formatted for the OU model.
    :param hmm_csv (str): The file path to the CSV containing data for initializing the HMM risk manager.
    :param start_date (datetime.datetime): The starting date of the backtest period.
    :param initial_capital (float, optional): The initial capital amount in currency units. Defaults to 100000.0.
    :param heartbeat (float, optional): The heartbeat of the backtest, specifying the frequency at which market data is processed. Defaults to 0.0.

    Keyword Arguments:
    Various strategy-specific parameters that can be passed through **kwargs, 
        including but not limited to the lookback periods, trading quantities, and model configurations.
        See `HMMRiskManager` class, `Portfolio` class and `OrnsteinUhlenbeck`  class for  Additional keyword arguments.

    This function sets up the backtesting environment by initializing 
        the necessary components such as the data handler, execution handler, 
        portfolio, and the OUBacktester strategy itself with the HMM risk manager. 
        It then proceeds to simulate trading over the specified period using 
        the provided historical data, applying the OU mean-reverting strategy 
        to make trading decisions based on the calculated signals and risk management directives.
    
    Notes
    =====
    It is imperative that the dataset provided through the `hmm_csv` parameter encompasses a time frame
    preceding that of the `backtest_csv` dataset. This requirement ensures the integrity and validity
    of the Hidden Markov Model's (HMM) risk management system by mandating that the data used for
    backtesting represents a forward-testing scenario. Consequently, the `backtest_csv` data must
    remain unseen by the HMM during its training phase to simulate real-world conditions accurately
    and to avoid look-ahead bias. This approach reinforces the robustness of the strategy's predictive
    capabilities and its applicability to future, unobserved market conditions.
    """
    hmm = HMMRiskManager(csv_filepath=hmm_csv, verbose=True, **kwargs)
    kwargs['csvfile'] = ou_csv
    kwargs['model'] = hmm
    backtest = Backtest(
        backtest_csv, symbol_list, initial_capital, heartbeat, start_date,
        HistoricCSVDataHandler, SimulatedExecutionHandler,
        Portfolio, OUBacktester, **kwargs
    )
    backtest.simulate_trading()
