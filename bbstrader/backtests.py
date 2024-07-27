import datetime
import pandas as pd
import numpy as np
from bbstrader.btengine.strategy import Strategy
from bbstrader.btengine.event import SignalEvent
from bbstrader.btengine.backtest import Backtest
from bbstrader.btengine.data import HistoricCSVDataHandler
from bbstrader.btengine.portfolio import Portfolio
from bbstrader.btengine.execution import SimulatedExecutionHandler
from bbstrader.models.hmm import HMMRiskManager
from filterpy.kalman import KalmanFilter
from bbstrader.strategies import OrnsteinUhlenbeck
from bbstrader.tseries import (
    load_and_prepare_data, get_prediction
)


class ArimaGarchStrategyBacktester(Strategy):
    """
    The `ArimaGarchStrategyBacktester` class extends the `Strategy` 
    class to implement a backtesting framework for trading strategies based on 
    ARIMA-GARCH models, incorporating a Hidden Markov Model (HMM) for risk management.

    ## Features

    - **ARIMA-GARCH Model**: Utilizes ARIMA for time series forecasting and GARCH 
    for volatility forecasting, aimed at predicting market movements.
    - **HMM Risk Management**: Employs a Hidden Markov Model to manage risks, 
    determining safe trading regimes.
    - **Event-Driven Backtesting**: Capable of simulating real-time trading conditions 
    by processing market data and signals sequentially.

    ### Key Methods

    - `get_data()`: Retrieves and prepares the data required for ARIMA-GARCH model predictions.
    - `create_signal()`: Generates trading signals based on model predictions and current market positions.
    - `calculate_signals(event)`: Listens for market events and triggers signal creation and event placement.
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
    
    Exemple:

        ```python
        import datetime
        from pathlib import Path
        import yfinance as yf
        from bbstrader.strategies import ArimaGarchStrategy
        from bbstrader.tseries import load_and_prepare_data
        from bbstrader.backtests import run_arch_backtest

        if __name__ == '__main__':
            # ARCH SPY Vectorize Backtest
            k = 252
            data = yf.download("SPY", start="2004-01-02", end="2015-12-31")
            arch = ArimaGarchStrategy("SPY", data, k=k)
            df = load_and_prepare_data(data)
            arch.show_arima_garch_results(df['diff_log_return'].values[-k:])
            arch.backtest_strategy()

            # ARCH SPY Event Driven backtest from 2004-01-02" to "2015-12-31"
            # with hidden Markov Model as a risk manager.
            data_dir = Path("/bbstrader/btengine/data/") # if you cloned this repo
            csv_dir = str(Path().cwd()) + str(data_dir) # Or the absolute path of your csv data directory
            hmm_csv = str(Path().cwd()) + str(data_dir/"spy_train.csv")

            symbol_list = ["SPY"]
            kwargs = {
                'tiker': 'SPY',
                'benchmark': 'SPY',
                'window_size': 252,
                'qty': 1000,
                'k': 50,
                'iterations': 1000,
                'strategy_name': 'ARIMA+GARCH & HMM'
            }
            start_date = datetime.datetime(2004, 1, 2)
            run_arch_backtest(
                symbol_list, csv_dir, hmm_csv, start_date, **kwargs
            )
        ```

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
    The `KLFStrategyBacktester` class implements a backtesting framework for a 
    [pairs trading](https://en.wikipedia.org/wiki/Pairs_trade) strategy using 
    Kalman Filter and Hidden Markov Models (HMM) for risk management. 
    This document outlines the structure and usage of the `KLFStrategyBacktester`, 
    including initialization parameters, main functions, and an example of how to run a backtest. 
    """

    def __init__(
        self, bars, events_queue, **kwargs
    ):
        """
        Args:
        - `bars`: Instance of `HistoricCSVDataHandler` for market data handling.
        - `events_queue`: A queue for managing events.
        - `**kwargs`: Additional keyword arguments including:
            - `tickers`: List of ticker symbols involved in the pairs trading strategy.
            - `quantity`: Quantity of assets to trade. Default is 100.
            - `delta`: Delta parameter for the Kalman Filter. Default is `1e-4`.
            - `vt`: Measurement noise covariance for the Kalman Filter. Default is `1e-3`.
            - `model`: Instance of `HMMRiskManager` for managing trading risks.
            - `window`: Window size for calculating returns for the HMM. Default is 50.
            - `hmm_tiker`: Ticker symbol used by the HMM for risk management.
        """
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
    
        Example Usage:
   
            ```python
            import datetime
            from pathlib import Path
            from bbstrader.backtests import run_kf_backtest

            if __name__ == '__main__':

                # KLF IEI TLT Event Driven backtest with Hidden Markov Model as a risk manager.
                symbol_list = ["IEI", "TLT"]
                kwargs = {
                    "tickers": symbol_list,
                    "quantity": 2000,
                    "time_frame": "D1",
                    "trading_hours": 6.5,
                    "benchmark": "TLT",
                    "window": 50,
                    "hmm_tiker": "TLT",
                    "iterations": 100,
                    'strategy_name': 'Kalman Filter & HMM'
                }
                start_date = datetime.datetime(2009, 8, 3, 10, 40, 0)
                data_dir = Path("/bbstrader/btengine/data/")
                csv_dir = str(Path().cwd()) + str(data_dir)
                hmm_csv = str(Path().cwd()) + str(data_dir/"tlt_train.csv")
                run_kf_backtest(symbol_list, csv_dir, hmm_csv, start_date, **kwargs)
            ```
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
        """
        Args:
            - `bars`: market data
            - `events`: event queue
            - `ticker`: Symbol of the financial instrument.
            - `p`: Lookback period for the OU process.
            - `n`: Minimum number of observations for signal generation.
            - `qty`: Quantity of assets to trade.
            - `csvfile`: Path to the CSV file containing historical market data.
            - `model`: HMM risk management model.
            - `window`: Lookback period for HMM.
        """
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
    Notes:
        It is imperative that the dataset provided through the `hmm_csv` parameter encompasses a time frame
        preceding that of the `backtest_csv` dataset. This requirement ensures the integrity and validity
        of the Hidden Markov Model's (HMM) risk management system by mandating that the data used for
        backtesting represents a forward-testing scenario. Consequently, the `backtest_csv` data must
        remain unseen by the HMM during its training phase to simulate real-world conditions accurately
        and to avoid look-ahead bias. This approach reinforces the robustness of the strategy's predictive
        capabilities and its applicability to future, unobserved market conditions.

        This strategy is optimized for assets that exhibit mean reversion characteristics. 
        Prior to executing this backtest, it is imperative to conduct a mean reversion test 
        on the intended asset to ensure its suitability for this approach.
    
    Example:

        ```python
        import datetime
        from pathlib import Path
        from bbstrader.backtests import run_ou_backtest

        if __name__ == '__main__':

            # OU Backtest
            symbol_list = ['GLD']
            kwargs = {
                "tiker": 'GLD',
                'n': 5,
                'p': 5,
                'qty': 2000,
                "window": 50,
                'strategy_name': 'Ornstein-Uhlenbeck & HMM'
            }
            data_dir = Path("/bbstrader/btengine/data/")
            csv_dir = str(Path().cwd()) + str(data_dir)
            ou_csv = str(Path().cwd()) + str(data_dir/"ou_gld_train.csv")
            hmm_csv = str(Path().cwd()) + str(data_dir/"hmm_gld_train.csv")
            # Backtest period
            start_date = datetime.datetime(2015, 1, 2)

            # Execute backtest
            run_ou_backtest(symbol_list, csv_dir, ou_csv, hmm_csv, start_date, **kwargs)
        ```
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
        """
        Args:
        - `bars`: A data handler object that provides market data.
        - `events`: An event queue object where generated signals are placed.
        - `short_window` (optional): The period for the short moving average. Default is 50.
        - `long_window` (optional): The period for the long moving average. Default is 200.
        - `model` (optional): The HMM risk management model to be used.
        - `quantity` (optional): The default quantity of assets to trade. Default is 1000.
        """
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

    Exemple:

        ```python
        import datetime
        from pathlib import Path
        from bbstrader.backtests import run_sma_backtest

        if __name__ == '__main__':

            # SMA Backtest
            symbol_list = ['SPY']
            kwargs = {
                "short_window": 50,
                "long_window": 200,
                'strategy_name': 'SMA & HMM',
                "quantity": 1000
            }
            data_dir = Path("/bbstrader/btengine/data/")
            csv_dir = str(Path().cwd()) + str(data_dir)
            hmm_csv = str(Path().cwd()) + str(data_dir/"spy_train.csv")

            # Backtest start date
            start_date = datetime.datetime(2004, 1, 2, 10, 40, 0)

            # Execute the backtest
            run_sma_backtest(
                symbol_list, csv_dir, hmm_csv, start_date, **kwargs
            )
         ```
    """
    hmm = HMMRiskManager(csv_filepath=hmm_csv, verbose=True, **kwargs)
    kwargs["model"] = hmm
    backtest = Backtest(
        backtest_csv, symbol_list, initial_capital, heartbeat,
        start_date, HistoricCSVDataHandler, SimulatedExecutionHandler,
        Portfolio, SMAStrategyBacktester, **kwargs
    )
    backtest.simulate_trading()
