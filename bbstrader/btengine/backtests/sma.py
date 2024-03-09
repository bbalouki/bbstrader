import numpy as np
import datetime
from btengine.strategy import Strategy
from btengine.event import SignalEvent
from btengine.backtest import Backtest
from btengine.data import HistoricCSVDataHandler
from btengine.portfolio import Portfolio
from btengine.execution import SimulatedExecutionHandler
from filterpy.kalman import KalmanFilter
from risk_models.hmm import HMMRiskManager


class SMAStrategyBacktester(Strategy):
    """
    Carries out a basic Moving Average Crossover strategy bactesting with a
    short/long simple weighted moving average. Default short/long
    windows are 50/200 periods respectively and uses Hiden Markov Model 
    as risk Managment system for filteering signals.

    The trading strategy for this class is exceedingly simple and is used to bettter
    understood . The important issue is the risk management aspect ( the Hmm model )

    The Long-term trend following strategy is of the classic moving average crossover type. 
    The rules are simple:
    • At every bar calculate the 50-day and 200-day simple moving averages (SMA)
    • If the 50-day SMA exceeds the 200-day SMA and the strategy is not invested, then go long
    • If the 200-day SMA exceeds the 50-day SMA and the strategy is invested, then close the
        position
    """

    def __init__(
        self, bars, events, /, **kwargs
    ):
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events

        self.short_window = kwargs.get("short_window", 50)
        self.long_window = kwargs.get("long_window", 200)
        self.bought = self._calculate_initial_bought()
        self.hmm_model = kwargs.get("model")
        self.qty = kwargs.get("quantity", 100)

    def _calculate_initial_bought(self):
        bought = {}
        for s in self.symbol_list:
            bought[s] = 'OUT'
        return bought

    def get_data(self):
        for s in self.symbol_list:
            bar_date = self.bars.get_latest_bar_datetime(s)
            bars = self.bars.get_latest_bars_values(
                s, "Adj Close", N=self.long_window
            )
            returns_val = self.bars.get_latest_bars_values(
                s, "Returns", N=self.long_window
            )
            if len(bars) >= self.long_window and len(returns_val) >= self.long_window:
                regime = self.hmm_model.which_trade_allowed(returns_val)

                short_sma = np.mean(bars[-self.short_window:])
                long_sma = np.mean(bars[-self.long_window:])

                return short_sma, long_sma, regime, s, bar_date
            else:
                return None

    def create_signal(self):
        signal = None
        data = self.get_data()
        if data is not None:
            short_sma, long_sma, regime, s, bar_date = data
            dt = bar_date
            if regime == "LONG":
                # Bulliqh regime
                if short_sma < long_sma and self.bought[s] == "LONG":
                    print(f"EXIT: {bar_date}")
                    signal = SignalEvent(1, s, dt, 'EXIT')
                    self.bought[s] = 'OUT'

                elif short_sma > long_sma and self.bought[s] == "OUT":
                    print(f"LONG: {bar_date}")
                    signal = SignalEvent(
                        1, s, dt, 'LONG', quantity=self.qty)
                    self.bought[s] = 'LONG'

            elif regime == "SHORT":
                # Bearish regime
                if short_sma > long_sma and self.bought[s] == "SHORT":
                    print(f"EXIT: {bar_date}")
                    signal = SignalEvent(1, s, dt, 'EXIT')
                    self.bought[s] = 'OUT'

                elif short_sma < long_sma and self.bought[s] == "OUT":
                    print(f"SHORT: {bar_date}")
                    signal = SignalEvent(
                        1, s, dt, 'SHORT', quantity=self.qty)
                    self.bought[s] = 'SHORT'
        return signal

    def calculate_signals(self, event):
        if event.type == 'MARKET':
            signal = self.create_signal()
            if signal is not None:
                self.events.put(signal)


def run_sma_backtest(
    symbol_list: list,
    backtest_csv: str,
    hmm_csv: str,
    start_date: datetime.datetime,
    initial_capital: float = 100000.0,
    heartbeat: float = 0.0,
    **kwargs
):
    """
    Executes a backtest of the Moving Average Cross Strategy, integrating a Hidden Markov Model (HMM)
    for risk management. This function initializes and runs a simulation of trading activities based
    on historical data.

    Parameters
    ==========
    :param symbol_list: List of symbol strings that the backtest will be run on.
    :param backtest_csv: String path to the CSV file containing the backtest data.
    :param hmm_csv: String path to the CSV file containing the HMM model training data.
    :param start_date: A datetime object representing the start date of the backtest.
    :param initial_capital: Float representing the initial capital for the backtest (default 100000.0).
    :param heartbeat: Float representing the backtest's heartbeat in seconds (default 0.0).
    :param kwargs: See `HMMRiskManager` class and `Portfolio` class for  Additional keyword arguments.

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
    kwargs["model"] = hmm
    backtest = Backtest(
        backtest_csv, symbol_list, initial_capital, heartbeat,
        start_date, HistoricCSVDataHandler, SimulatedExecutionHandler,
        Portfolio, SMAStrategyBacktester, **kwargs
    )
    backtest.simulate_trading()