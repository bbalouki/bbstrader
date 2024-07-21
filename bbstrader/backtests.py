import datetime
import pandas as pd
import numpy as np
from .btengine.strategy import Strategy
from .btengine.event import SignalEvent
from .btengine.backtest import Backtest
from .btengine.data import HistoricCSVDataHandler
from .btengine.portfolio import Portfolio
from .btengine.execution import SimulatedExecutionHandler
from .models.hmm import HMMRiskManager
from filterpy.kalman import KalmanFilter
from .strategies import OrnsteinUhlenbeck
from .tseries import (
    load_and_prepare_data, get_prediction
)


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
        self.window_size = kwargs.get('window_size', 252)  # arima

        self.long_market = False
        self.short_market = False
        self.qty = kwargs.get('qty', 100)
        self.hmm_window = kwargs.get("k", 50)
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
        data = load_and_prepare_data(df)
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
                    signal = SignalEvent(
                        1, self.tiker, dt, "LONG", quantity=self.qty)
                    print(f"{dt}: LONG")
                    self.long_market = True

            elif regime == "SHORT":
                # If we are not in the market, go short
                if prediction < 0 and not self.short_market:
                    signal = SignalEvent(
                        1, self.tiker, dt, "SHORT", quantity=self.qty)
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

    Args:
        symbol_list (list): 
            A list of symbols (tickers) to be included in the backtest.

        backtest_csv (str): 
            The filepath to the CSV file containing historical data for backtesting.

        hmm_csv (str): 
            The filepath to the CSV file used by the HMM risk manager for model training and risk assessment.

        start_date (datetime.datetime): 
            The start date of the backtest period.

        initial_capital (float, optional): 
            The initial capital in USD to start the backtest with. Defaults to 100000.0.

        heartbeat (float, optional): 
            The heartbeat of the backtest in seconds. 
            This simulates the delay between market data feeds. Defaults to 0.0.

        kwargs: 
            See `HMMRiskManager` class and `Portfolio` class for  Additional keyword arguments.

    Notes:
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


class KLFStrategyBacktester(Strategy):
    """
    Implements a Backtester strategy for `KLFStrategy` with hidden Markov Model. 
    """

    def __init__(
        self, bars, events_queue, **kwargs
    ):
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events_queue = events_queue

        self.tickers = kwargs.get("tickers")
        assert self.tickers is not None
        self.qty = kwargs.get("quantity", 100)

        self.latest_prices = np.array([-1.0, -1.0])
        self.delta = kwargs.get("delta", 1e-4)
        self.wt = self.delta/(1-self.delta) * np.eye(2)
        self.vt = kwargs.get("vt", 1e-3)
        self.theta = np.zeros(2)
        self.P = np.zeros((2, 2))
        self.R = None
        self.kf = self._init_kalman()

        self.hmm_model = kwargs.get("model")
        self.window = kwargs.get("window", 50)
        self.hmm_tiker = kwargs.get("hmm_tiker")
        assert self.hmm_tiker is not None

        self.long_market = False
        self.short_market = False

    def _init_kalman(self):
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.zeros((2, 1))  # Initial state
        kf.P = self.P  # Initial covariance
        kf.F = np.eye(2)  # State transition matrix
        kf.Q = self.wt  # Process noise covariance
        kf.R = 1.  # Scalar measurement noise covariance

        return kf

    def calc_slope_intercep(self, prices):
        kf = self.kf
        kf.H = np.array([[prices[1], 1.0]])
        kf.predict()
        kf.update(prices[0])
        slope = kf.x.copy().flatten()[0]
        intercept = kf.x.copy().flatten()[1]

        return slope, intercept

    def calculate_xy_signals(self, et, sqrt_Qt, regime, dt):
        # Make sure there is no position open
        if et >= -sqrt_Qt and self.long_market:
            print("CLOSING LONG: %s" % dt)
            y_signal = SignalEvent(1, self.tickers[1], dt, "EXIT")
            x_signal = SignalEvent(1, self.tickers[0], dt, "EXIT")
            self.events_queue.put(y_signal)
            self.events_queue.put(x_signal)
            self.long_market = False

        elif et <= sqrt_Qt and self.short_market:
            print("CLOSING SHORT: %s" % dt)
            y_signal = SignalEvent(1, self.tickers[1], dt, "EXIT")
            x_signal = SignalEvent(1, self.tickers[0], dt, "EXIT")
            self.events_queue.put(y_signal)
            self.events_queue.put(x_signal)
            self.short_market = False

        # Long Entry
        if regime == "LONG":
            if et <= -sqrt_Qt and not self.long_market:
                print("LONG: %s" % dt)
                y_signal = SignalEvent(
                    1, self.tickers[1], dt, "LONG", self.qty, 1.0)
                x_signal = SignalEvent(
                    1, self.tickers[0], dt, "SHORT", self.qty, self.theta[0])
                self.events_queue.put(y_signal)
                self.events_queue.put(x_signal)
                self.long_market = True

        # Short Entry
        elif regime == "SHORT":
            if et >= sqrt_Qt and not self.short_market:
                print("SHORT: %s" % dt)
                y_signal = SignalEvent(
                    1, self.tickers[1], dt, "SHORT", self.qty, 1.0)
                x_signal = SignalEvent(
                    1, self.tickers[0], "LONG", self.qty, self.theta[0])
                self.events_queue.put(y_signal)
                self.events_queue.put(x_signal)
                self.short_market = True

    def calculate_signals_for_pairs(self):
        p0, p1 = self.tickers[0], self.tickers[1]
        dt = self.bars.get_latest_bar_datetime(p0)
        _x = self.bars.get_latest_bars_values(
            p0, "Close", N=1
        )
        _y = self.bars.get_latest_bars_values(
            p1, "Close", N=1
        )
        returns = self.bars.get_latest_bars_values(
            self.hmm_tiker, "Returns", N=self.window
        )
        if len(returns) >= self.window:
            self.latest_prices[0] = _x[-1]
            self.latest_prices[1] = _y[-1]

            if all(self.latest_prices > -1.0):
                slope, intercept = self.calc_slope_intercep(self.latest_prices)

                self.theta[0] = slope
                self.theta[1] = intercept

                # Create the observation matrix of the latest prices
                # of Y and the intercept value (1.0) as well as the
                # scalar value of the latest price from X
                F = np.asarray([self.latest_prices[0], 1.0]).reshape((1, 2))
                y = self.latest_prices[1]

                # The prior value of the states \theta_t is
                # distributed as a multivariate Gaussian with
                # mean a_t and variance-covariance R_t
                if self.R is not None:
                    self.R = self.C + self.wt
                else:
                    self.R = np.zeros((2, 2))

                # Calculate the Kalman Filter update
                # ---------------------------------
                # Calculate prediction of new observation
                # as well as forecast error of that prediction
                yhat = F.dot(self.theta)
                et = y - yhat

                # Q_t is the variance of the prediction of
                # observations and hence sqrt_Qt is the
                # standard deviation of the predictions
                Qt = F.dot(self.R).dot(F.T) + self.vt
                sqrt_Qt = np.sqrt(Qt)

                # The posterior value of the states \theta_t is
                # distributed as a multivariate Gaussian with mean
                # m_t and variance-covariance C_t
                At = self.R.dot(F.T) / Qt
                self.theta = self.theta + At.flatten() * et
                self.C = self.R - At * F.dot(self.R)
                regime = self.hmm_model.which_trade_allowed(returns)

                self.calculate_xy_signals(et, sqrt_Qt, regime, dt)

    def calculate_signals(self, event):
        """
        Calculate the Kalman Filter strategy.
        """
        if event.type == "MARKET":
            self.calculate_signals_for_pairs()


def run_kf_backtest(
    symbol_list: list,
    backtest_csv: str,
    hmm_csv: str,
    start_date: datetime.datetime,
    initial_capital: float = 100000.0,
    heartbeat: float = 0.0,
    **kwargs
):
    """
    Initializes and runs a backtest using a Kalman Filter strategy with a Hidden Markov Model (HMM) risk manager.

    This function sets up the backtesting environment by initializing necessary components such as data handling, 
    execution handling, portfolio management, and the strategy itself, incorporating a Hidden Markov Model for risk management. 
    It simulates trading over a given dataset to evaluate the performance of the Kalman Filter strategy under specified conditions.

    Args:
        symbol_list (list): 
            A list of ticker symbols to be included in the backtest.

        backtest_csv (str): 
            The file path to the CSV containing the historical data for backtesting.

        hmm_csv (str): 
            The file path to the CSV containing the data for initializing the HMM risk manager.

        start_date (datetime.datetime): The start date of the backtest.

        initial_capital (float, optional): 
            The initial capital for the backtest. Default is 100000.0.

        heartbeat (float, optional): 
            The heartbeat of the backtest, representing the frequency of data updates. Default is 0.0.

        kwargs: 
            See `HMMRiskManager` class and `Portfolio` class for  Additional keyword arguments.

    Notes:
        It is imperative that the dataset provided through the `hmm_csv` parameter encompasses a time frame
        preceding that of the `backtest_csv` dataset. This requirement ensures the integrity and validity
        of the Hidden Markov Model's (HMM) risk management system by mandating that the data used for
        backtesting represents a forward-testing scenario. Consequently, the `backtest_csv` data must
        remain unseen by the HMM during its training phase to simulate real-world conditions accurately
        and to avoid look-ahead bias. This approach reinforces the robustness of the strategy's predictive
        capabilities and its applicability to future, unobserved market conditions.
    """
    hmm = HMMRiskManager(csv_filepath=hmm_csv, verbose=True, **kwargs)
    kwargs['model'] = hmm
    backtest = Backtest(
        backtest_csv, symbol_list, initial_capital, heartbeat, start_date,
        HistoricCSVDataHandler, SimulatedExecutionHandler,
        Portfolio, KLFStrategyBacktester, **kwargs
    )
    backtest.simulate_trading()


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

    Args:
        symbol_list (list): 
            A list of symbol strings representing the financial instruments to be traded.

        backtest_csv (str): 
            The file path to the CSV containing historical market data for backtesting.

        ou_csv (str): 
            The file path to the CSV containing data specifically formatted for the OU model.

        hmm_csv (str): 
            The file path to the CSV containing data for initializing the HMM risk manager.

        start_date (datetime.datetime): The starting date of the backtest period.

        initial_capital (float, optional): 
            The initial capital amount in currency units. Defaults to 100000.0.

        heartbeat (float, optional): 
            The heartbeat of the backtest, specifying the frequency at which market data is processed. Defaults to 0.0.

        kwargs:
            Various strategy-specific parameters that can be passed through **kwargs, 
            including but not limited to the lookback periods, trading quantities, and model configurations.
            See `HMMRiskManager` class, `Portfolio` class and `OrnsteinUhlenbeck`  class for  Additional keyword arguments.

    This function sets up the backtesting environment by initializing 
        the necessary components such as the data handler, execution handler, 
        portfolio, and the OUBacktester strategy itself with the HMM risk manager. 
        It then proceeds to simulate trading over the specified period using 
        the provided historical data, applying the OU mean-reverting strategy 
        to make trading decisions based on the calculated signals and risk management directives.

    Notes:
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
    - At every bar calculate the 50-day and 200-day simple moving averages (SMA)
    - f the 50-day SMA exceeds the 200-day SMA and the strategy is not invested, then go long
    - If the 200-day SMA exceeds the 50-day SMA and the strategy is invested, then close the
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

    Args:
        symbol_list: 
            List of symbol strings that the backtest will be run on.

        backtest_csv: 
            String path to the CSV file containing the backtest data.

        hmm_csv: 
            String path to the CSV file containing the HMM model training data.

        start_date: 
            A datetime object representing the start date of the backtest.

        initial_capital: 
            Float representing the initial capital for the backtest (default 100000.0).

        heartbeat: 
            Float representing the backtest's heartbeat in seconds (default 0.0).

        kwargs: 
            See `HMMRiskManager` class and `Portfolio` class for  Additional keyword arguments.

    Notes:
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
