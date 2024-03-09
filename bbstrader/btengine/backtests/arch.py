import datetime
import pandas as pd
from btengine.strategy import Strategy
from btengine.event import SignalEvent
from btengine.backtest import Backtest
from btengine.data import HistoricCSVDataHandler
from btengine.portfolio import Portfolio
from btengine.execution import SimulatedExecutionHandler
from tseries.arch import (
    load_and_prepare_data, get_prediction
)
from risk_models.hmm import HMMRiskManager


class ArimaGarchStrategyBacktester(Strategy):
    """
    Implements a Backtester for `ArimaGarchStrategy` with hidden Markov Model
    as a risk manager.
    """

    def __init__(self, bars, events, **kwargs):
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events

        self.tiker = kwargs.get('tiker')
        assert self.tiker is not None
        self.window_size = kwargs.get('window_size', 252) # arima

        self.long_market = False
        self.short_market = False
        self.qty = kwargs.get('qty', 100)
        self.hmm_window =  kwargs.get("k", 50)
        self.risk_manager = kwargs.get('risk_manager')

    def get_data(self):
        symbol = self.tiker
        M = self.window_size
        N = self.hmm_window
        dt = self.bars.get_latest_bar_datetime(self.tiker)
        bars = self.bars.get_latest_bars_values(
            symbol, "Close", N=self.window_size
        )
        returns = self.bars.get_latest_bars_values(
            symbol, 'Returns', N=self.hmm_window
        )
        df = pd.DataFrame()
        df['Close'] = bars[-M:]
        df = df.dropna() 
        data =  load_and_prepare_data(df)
        if len(data) >= M and len(returns) >= N:
            return data, returns[-N:], dt
        return None, None, None

    def create_signal(self):
        data, returns, dt = self.get_data()
        signal = None
        if data is not None and returns is not None:
            prediction = get_prediction(data['diff_log_return'])
            regime = self.risk_manager.which_trade_allowed(returns)
            
            # If we are short the market, check for an exit
            if prediction > 0 and self.short_market:
                signal = SignalEvent(1, self.tiker, dt, "EXIT")
                print(f"{dt}: EXIT SHORT")
                self.short_market = False
            
            # If we are long the market, check for an exit
            elif prediction < 0 and self.long_market:
                signal = SignalEvent(1, self.tiker, dt, "EXIT")
                print(f"{dt}: EXIT LONG")
                self.long_market = False

            if regime == "LONG":
                # If we are not in the market, go long
                if prediction > 0 and not self.long_market:
                    signal = SignalEvent(1, self.tiker, dt, "LONG", quantity=self.qty)
                    print(f"{dt}: LONG")
                    self.long_market = True
            
            elif regime == "SHORT":
                # If we are not in the market, go short
                if prediction < 0 and not self.short_market:
                    signal = SignalEvent(1, self.tiker, dt, "SHORT", quantity=self.qty)
                    print(f"{dt}: SHORT")
                    self.short_market = True

        return signal

    def calculate_signals(self, event):
        if event.type == 'MARKET':
            signal = self.create_signal()
            if signal is not None:
                self.events.put(signal)

def run_arch_backtest(
    symbol_list: list,
    backtest_csv: str,
    hmm_csv: str,
    start_date: datetime.datetime,
    initial_capital: float = 100000.0,
    heartbeat: float = 0.0,
    **kwargs
    ):
    """
    Initiates and runs a backtest using an ARIMA-GARCH strategy with a Hidden Markov Model (HMM) risk manager.

    This function sets up the backtesting environment by initializing the necessary components including the 
    data handler, execution handler, portfolio, and the strategy itself, with the ARIMA-GARCH model being at the core. 
    The HMM risk manager is used to assess and manage the risks associated with the trading strategy.

    Parameters
    ==========
    :param symbol_list (list): A list of symbols (tickers) to be included in the backtest.
    :param backtest_csv (str): The filepath to the CSV file containing historical data for backtesting.
    :param hmm_csv (str): The filepath to the CSV file used by the HMM risk manager for model training and risk assessment.
    :param start_date (datetime.datetime): The start date of the backtest period.
    :param initial_capital (float, optional): The initial capital in USD to start the backtest with. Defaults to 100000.0.
    :param heartbeat (float, optional): The heartbeat of the backtest in seconds. 
        This simulates the delay between market data feeds. Defaults to 0.0.
    :param **kwargs: See `HMMRiskManager` class and `Portfolio` class for  Additional keyword arguments.
    
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
    kwargs['risk_manager'] = hmm
    backtest = Backtest(
        backtest_csv, symbol_list, initial_capital, heartbeat, start_date,
        HistoricCSVDataHandler, SimulatedExecutionHandler,
        Portfolio, ArimaGarchStrategyBacktester, **kwargs
    )
    backtest.simulate_trading()

