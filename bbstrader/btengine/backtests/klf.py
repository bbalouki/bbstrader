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
        kf.P = self.P # Initial covariance
        kf.F = np.eye(2)  # State transition matrix
        kf.Q = self.wt  # Process noise covariance
        kf.R = 1. # Scalar measurement noise covariance

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

    Parameters
    ==========
    :param symbol_list (list): A list of ticker symbols to be included in the backtest.
    :param backtest_csv (str): The file path to the CSV containing the historical data for backtesting.
    :param hmm_csv (str): The file path to the CSV containing the data for initializing the HMM risk manager.
    :param start_date (datetime.datetime): The start date of the backtest.
    :param initial_capital (float, optional): The initial capital for the backtest. Default is 100000.0.
    :param heartbeat (float, optional): The heartbeat of the backtest, representing the frequency of data updates. Default is 0.0.
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
    kwargs['model'] = hmm
    backtest = Backtest(
        backtest_csv, symbol_list, initial_capital, heartbeat, start_date,
        HistoricCSVDataHandler, SimulatedExecutionHandler,
        Portfolio, KLFStrategyBacktester, **kwargs
    )
    backtest.simulate_trading()
