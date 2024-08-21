import pprint
import queue
import time
import numpy as np
import pandas as pd
import yfinance as yf
from queue import Queue
from datetime import datetime
from seaborn import saturate
from bbstrader.btengine.data import *
from bbstrader.btengine.execution import *
from bbstrader.btengine.portfolio import Portfolio
from bbstrader.btengine.strategy import Strategy
from bbstrader.btengine.event import SignalEvent
from bbstrader.models import HMMRiskManager
from filterpy.kalman import KalmanFilter
from bbstrader.strategies import OrnsteinUhlenbeck
from bbstrader.tseries import load_and_prepare_data
from bbstrader.tseries import get_prediction
from typing import Literal, Optional, List

__all__ = [
    "Backtest",
    "SMAStrategyBacktester",
    "KLFStrategyBacktester",
    "OUStrategyBacktester",
    "ArimaGarchStrategyBacktester",
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

    def _generate_trading_instances(self):
        """
        Generates the trading instance objects from
        their class types.
        """
        print(
            f"\nStarting Backtest on {self.symbol_list} "
            f"with ${self.initial_capital} Initial Capital\n"
        )
        self.data_handler: DataHandler = self.dh_cls(
            self.events, self.symbol_list, **self.kwargs
        )
        self.strategy: Strategy = self.strategy_cls(
            self.data_handler, self.events, **self.kwargs
        )
        self.portfolio = Portfolio(
            self.data_handler,
            self.events,
            self.start_date,
            self.initial_capital, **self.kwargs
        )
        self.execution_handler: ExecutionHandler = self.eh_cls(self.events)

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


class SMAStrategyBacktester(Strategy):
    """
    Carries out a basic Moving Average Crossover strategy bactesting with a
    short/long simple weighted moving average. Default short/long
    windows are 50/200 periods respectively and uses Hiden Markov Model 
    as risk Managment system for filteering signals.

    The trading strategy for this class is exceedingly simple and is used to bettter
    understood. The important issue is the risk management aspect (the Hmm model)

    The Long-term trend following strategy is of the classic moving average crossover type. 
    The rules are simple:
    -   At every bar calculate the 50-day and 200-day simple moving averages (SMA)
    -   If the 50-day SMA exceeds the 200-day SMA and the strategy is not invested, then go long
    -   If the 200-day SMA exceeds the 50-day SMA and the strategy is invested, then close the position
    """

    def __init__(
        self, bars: DataHandler, events: Queue, /, **kwargs
    ):
        """
        Args:
            bars (DataHandler): A data handler object that provides market data.
            events (Queue): An event queue object where generated signals are placed.
            short_window (int, optional): The period for the short moving average.
            long_window (int, optional): The period for the long moving average.
            hmm_model (optional): The risk management model to be used.
            quantity (int,  optional): The default quantity of assets to trade.
        """
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events

        self.short_window = kwargs.get("short_window", 50)
        self.long_window = kwargs.get("long_window", 200)
        self.hmm_model = kwargs.get("hmm_model")
        self.qty = kwargs.get("quantity", 100)

        self.bought = self._calculate_initial_bought()

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


class KLFStrategyBacktester(Strategy):
    """
    The `KLFStrategyBacktester` class implements a backtesting framework for a 
    [pairs trading](https://en.wikipedia.org/wiki/Pairs_trade) strategy using 
    Kalman Filter for signals and Hidden Markov Models (HMM) for risk management. 
    This document outlines the structure and usage of the `KLFStrategyBacktester`, 
    including initialization parameters, main functions, and an example of how to run a backtest. 
    """

    def __init__(
        self,
        bars: DataHandler, events_queue: Queue, **kwargs
    ):
        """
        Args:
            `bars`: `DataHandler` for market data handling.
            `events_queue`: A queue for managing events.
            kwargs : Additional keyword arguments including
                - `tickers`: List of ticker symbols involved in the pairs trading strategy.
                - `quantity`: Quantity of assets to trade. Default is 100.
                - `delta`: Delta parameter for the Kalman Filter. Default is `1e-4`.
                - `vt`: Measurement noise covariance for the Kalman Filter. Default is `1e-3`.
                - `hmm_model`: Instance of `HMMRiskManager` for managing trading risks.
                - `hmm_window`: Window size for calculating returns for the HMM. Default is 50.
                - `hmm_tiker`: Ticker symbol used by the HMM for risk management.
        """
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events_queue = events_queue

        self.tickers = kwargs.get("tickers")
        self.hmm_tiker = kwargs.get("hmm_tiker")
        self.hmm_model = kwargs.get("hmm_model")
        self._assert_tikers()
        self.hmm_window = kwargs.get("hmm_window", 50)
        self.qty = kwargs.get("quantity", 100)

        self.latest_prices = np.array([-1.0, -1.0])
        self.delta = kwargs.get("delta", 1e-4)
        self.wt = self.delta/(1-self.delta) * np.eye(2)
        self.vt = kwargs.get("vt", 1e-3)
        self.theta = np.zeros(2)
        self.P = np.zeros((2, 2))
        self.R = None
        self.kf = self._init_kalman()

        self.long_market = False
        self.short_market = False

    def _assert_tikers(self):
        if self.tickers is None:
            raise ValueError(
                "A list of 2 Tickers must be provide for this strategy")
        if self.hmm_tiker is None:
            raise ValueError(
                "You need to provide a ticker used by the HMM for risk management")

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
            self.hmm_tiker, "Returns", N=self.hmm_window
        )
        if len(returns) >= self.hmm_window:
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


class OUStrategyBacktester(Strategy):
    """
    The `OUBacktester` class is a specialized trading strategy that implements 
    the Ornstein-Uhlenbeck (OU) process for mean-reverting financial time series. 
    This class extends the generic `Strategy` class provided by a backtesting framework
    allowing it to integrate seamlessly with the ecosystem of data handling, signal generation
    event management, and execution handling components of the framework. 
    The strategy is designed to operate on historical market data, specifically 
    targeting a single financial instrument (or a list of instruments) 
    for which the trading signals are to be generated.

    Note:
    This strategy is based on a stochastic process, so it is normal that every time 
    you run the backtest you get different results.
    """

    def __init__(self, bars: DataHandler, events: Queue, **kwargs):
        """
        Args:
            `bars`: DataHandler
            `events`: event queue
            `ticker`: Symbol of the financial instrument.
            `p`: Lookback period for the OU process.
            `n`: Minimum number of observations for signal generation.
            `quantity`: Quantity of assets to trade.
            `ou_data`: DataFrame used to estimate Ornstein-Uhlenbeck process params
                (drift `(θ)`, volatility `(σ)`, and long-term mean `(μ)`).
            `hmm_model`: HMM risk management model.
            `hmm_window`: Lookback period for HMM.
        """
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events

        self.ticker = kwargs.get('tiker')
        self.p = kwargs.get('p', 20)
        self.n = kwargs.get('n', 10)
        self.qty = kwargs.get('quantity', 1000)

        self.data = kwargs.get("ou_data")
        self.ou_data = self._get_data(self.data)
        self.ou = OrnsteinUhlenbeck(self.ou_data["Close"].values)

        self.hmm = kwargs.get('hmm_model')
        self.window = kwargs.get('hmm_window', 50)

        self.LONG = False
        self.SHORT = False

    def _get_data(self, data):
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, str):
            return self._read_csv(data)

    def _read_csv(self, csv_file):
        df = pd.read_csv(csv_file, header=0,
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


class ArimaGarchStrategyBacktester(Strategy):
    """
    The `ArimaGarchStrategyBacktester` class extends the `Strategy` 
    class to implement a backtesting framework for trading strategies based on 
    ARIMA-GARCH models, incorporating a Hidden Markov Model (HMM) for risk management.

    Features
    ========
    - **ARIMA-GARCH Model**: Utilizes ARIMA for time series forecasting and GARCH for volatility forecasting, aimed at predicting market movements.
    
    - **HMM Risk Management**: Employs a Hidden Markov Model to manage risks, determining safe trading regimes.
    
    - **Event-Driven Backtesting**: Capable of simulating real-time trading conditions by processing market data and signals sequentially.

    Key Methods
    ===========
    - `get_data()`: Retrieves and prepares the data required for ARIMA-GARCH model predictions.
    - `create_signal()`: Generates trading signals based on model predictions and current market positions.
    - `calculate_signals(event)`: Listens for market events and triggers signal creation and event placement.
      
    """

    def __init__(self, bars: DataHandler, events: Queue, **kwargs):
        """
        Args:
            `bars`: DataHandler
            `events`: event queue
            `ticker`: Symbol of the financial instrument.
            `arima_window`: The window size for rolling prediction in backtesting.
            `quantity`: Quantity of assets to trade.
            `hmm_window`: Lookback period for HMM.
            `hmm_model`: HMM risk management model.

        """
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events

        self.tiker = kwargs.get('tiker')
        self.arima_window = kwargs.get('arima_window', 252)

        self.qty = kwargs.get('qauntity', 100)
        self.hmm_window = kwargs.get("hmm_window", 50)
        self.hmm_model = kwargs.get("hmm_model")

        self.long_market = False
        self.short_market = False

    def get_data(self):
        symbol = self.tiker
        M = self.arima_window
        N = self.hmm_window
        dt = self.bars.get_latest_bar_datetime(self.tiker)
        bars = self.bars.get_latest_bars_values(
            symbol, "Close", N=self.arima_window
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
            regime = self.hmm_model.which_trade_allowed(returns)

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


def _run_backtest(
        strategy_name: str,
        capital: float, symbol_list: list, kwargs: dict):
    """
    Executes a backtest of the specified strategy 
    integrating a Hidden Markov Model (HMM) for risk management.
    """
    hmm_data = yf.download(
        kwargs.get("hmm_tiker", symbol_list[0]),
        start=kwargs.get("hmm_start"), end=kwargs.get("hmm_end")
    )
    kwargs["hmm_model"] = HMMRiskManager(data=hmm_data, verbose=True)
    kwargs["strategy_name"] = strategy_name

    engine = Backtest(
        symbol_list, capital, 0.0, datetime.strptime(
            kwargs['yf_start'], "%Y-%m-%d"),
        kwargs.get("data_handler", YFHistoricDataHandler), 
        SimulatedExecutionHandler, kwargs.pop('backtester_class'), **kwargs
    )
    engine.simulate_trading()


def _run_arch_backtest(
        capital: float = 100000.0,
        quantity: int = 1000
):
    kwargs = {
        'tiker': 'SPY',
        'quantity': quantity,
        'yf_start': "2004-01-02",
        'backtester_class': ArimaGarchStrategyBacktester
    }
    _run_backtest("ARIMA+GARCH & HMM", capital, ["SPY"], kwargs)


def _run_ou_backtest(
        capital: float = 100000.0,
        quantity: int = 2000
):
    kwargs = {
        "tiker": 'GLD',
        'quantity': quantity,
        "n": 5,
        "p": 5,
        "hmm_window": 50,
        "yf_start": "2015-01-02",
        'backtester_class': OUStrategyBacktester,
        'ou_data': yf.download("GLD", start="2010-01-04", end="2014-12-31"),
        'hmm_end': "2014-12-31"
    }
    _run_backtest("Ornstein-Uhlenbeck & HMM", capital, ['GLD'], kwargs)


def _run_kf_backtest(
    capital: float = 100000.0,
    quantity: int = 2000
):
    symbol_list = ["IEI", "TLT"]
    kwargs = {
        "tickers": symbol_list,
        "quantity": quantity,
        "benchmark": "TLT",
        "yf_start": "2009-08-03",
        "hmm_tiker": "TLT",
        'backtester_class': KLFStrategyBacktester,
        'hmm_end': "2009-07-28"
    }
    _run_backtest("Kalman Filter & HMM", capital, symbol_list, kwargs)


def _run_sma_backtest(
    capital: float = 100000.0,
    quantity: int = 100
):
    kwargs = {
        "quantity": quantity,
        "hmm_end":  "2009-12-31",
        "hmm_tiker": "^GSPC",
        "yf_start": "2010-01-01",
        "hmm_start": "1990-01-01",
        "start_pos": "2023-01-01",
        "session_duration": 23.0,
        "backtester_class": SMAStrategyBacktester,
        "data_handler": MT5HistoricDataHandler
    }
    _run_backtest("SMA & HMM", capital, ["[SP500]"], kwargs)


_BACKTESTS = {
    'ou': _run_ou_backtest,
    'sma': _run_sma_backtest,
    'klf': _run_kf_backtest,
    'arch': _run_arch_backtest
}


def run_backtest(
    symbol_list:   List[str] = ...,
    start_date:    datetime = ...,
    data_handler:  DataHandler = ...,
    strategy:      Strategy = ...,
    exc_handler:   Optional[ExecutionHandler] = None,
    initial_capital: Optional[float] = 100000.0,
    heartbeat:     Optional[float] = 0.0,
    test_mode:     Optional[bool] = True,
    test_strategy: Literal['ou', 'sma', 'klf', 'arch'] = 'sma',
    test_quantity: Optional[int] = 1000,
    **kwargs
):
    """
    Runs a backtest simulation based on a `DataHandler`, `Strategy` and `ExecutionHandler`.

    Args:
        symbol_list (List[str]): List of symbol strings for the assets to be backtested. 
            This is required when `test_mode` is set to False.

        start_date (datetime): Start date of the backtest. This is required when `test_mode` is False.

        data_handler (DataHandler): An instance of the `DataHandler` class, responsible for managing 
            and processing market data. Required when `test_mode` is False.
            There are three DataHandler classes implemented in btengine module
           `HistoricCSVDataHandler`, `MT5HistoricDataHandler` and `YFHistoricDataHandler`
            See each of this class documentation for more details.
            You can create your `CustumDataHandler` but it must be a subclass of `DataHandler`.

        strategy (Strategy): The trading strategy to be employed during the backtest. Required when `test_mode` is False.
            The strategy must be an instance of `Strategy` and it must have `DataHandler` and `event queue`
            as required positional arguments to be used int the `Backtest` class. All other argument needed to be pass
            to the strategy class must be in `**kwargs`. The strategy class must have `calculate_signals`
            methods to generate `SignalEvent` in the backtest class.

        exc_handler (ExecutionHandler): The execution handler for managing order executions. If not provided,
            a `SimulatedExecutionHandler` will be used by default. Required when `test_mode` is False.
            The `exc_handler` must be an instance of `ExecutionHandler` and must have `execute_order`method
            used to handle `OrderEvent` in events queue in the `Backtest` class. 

        initial_capital (float, optional): The initial capital for the portfolio in the backtest. Default is 100,000.

        heartbeat (float, optional): Time delay (in seconds) between iterations of the event-driven backtest loop. 
            Default is 0.0, allowing the backtest to run as fast as possible. 
            It could be also used as time frame in live trading engine (e.g 1m, 5m, 15m etc.) when listening
            to a live market `DataHandler`.

        test_mode (bool, optional): If set to True, the function runs a predefined backtest using a selected strategy 
            (`ou`, `sma`, `klf`, `arch`). Default is True.

        test_strategy (Literal['ou', 'sma', 'klf', 'arch'], optional): The strategy to use in test mode. Default is `sma`.
            - `ou` Execute `OUStrategyBacktester`, for more detail see this class documentation.
            - `sma` Execute `SMAStrategyBacktester`, for more detail see this class documentation.
            - `klf` Execute `KLFStrategyBacktester`, for more detail see this class documentation.
            - `arch` Execute `ArimaGarchStrategyBacktester`, for more detail see this class documentation.

        test_quantity (int, optional): The quantity of assets to be used in the test backtest. Default is 1000.

        **kwargs: Additional parameters passed to the `Backtest` instance, which may include strategy-specific,
            data handler, protfolio or execution handler options.

    Usage:
        - To run a predefined test backtest, set `test_mode=True` and select a strategy using `test_strategy`.
        - To customize the backtest, set `test_mode=False` and provide the required parameters (`symbol_list`, 
          `start_date`, `data_handler`, `strategy`, `exc_handler`).

    Examples:
        >>> from bbstrader.btengine import run_backtest
        >>> run_backtest(test_mode=True, test_strategy='ou', test_quantity=2000)

        >>> run_backtest(
        ...     symbol_list=['AAPL', 'GOOG'],
        ...     start_date=datetime(2020, 1, 1),
        ...     data_handler=CustomDataHandler(),
        ...     strategy=MovingAverageStrategy(),
        ...     exc_handler=CustomExecutionHandler(),
        ...     initial_capital=500000.0,
        ...     heartbeat=1.0
        ... )
    """
    if test_mode:
        _BACKTESTS[test_strategy](
            capital=initial_capital,
            quantity=test_quantity
        )   
    else:
        execution_handler = kwargs.get("exc_handler", SimulatedExecutionHandler)
        engine = Backtest(
            symbol_list, initial_capital, heartbeat, start_date,
            data_handler, execution_handler, strategy, **kwargs
        )
        engine.simulate_trading()
