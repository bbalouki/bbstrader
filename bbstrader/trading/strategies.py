"""
Strategies module for trading strategies backtesting and execution.

# NOTE
These strategies inherit from the Strategy class, not from MT5Strategy, because we chose to demonstrate the modular approach to building and backtesting strategies using the bbstrader framework.

If these strategies need to be sent to the Mt5ExecutionEngine,
they must return signals as a list of bbstrader.metatrader.trade.TradingSignal objects.

Later, we will implement the Execution Engine for the Interactive Brokers TWS platform.

DISCLAIMER: 
This module is for educational purposes only and should not be
considered as financial advice. Always consult with a qualified financial advisor before making any investment decisions. 
The authors and contributors of this module are not responsible for any financial losses or damages incurred as a result of using 
this module or the information contained herein.
"""

from datetime import datetime
from queue import Queue
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf

from bbstrader.btengine.backtest import BacktestEngine
from bbstrader.btengine.data import DataHandler, MT5DataHandler, YFDataHandler
from bbstrader.btengine.event import Events, SignalEvent
from bbstrader.btengine.execution import MT5ExecutionHandler, SimExecutionHandler
from bbstrader.btengine.strategy import Strategy
from bbstrader.metatrader.account import Account
from bbstrader.metatrader.rates import Rates
from bbstrader.metatrader.trade import TradingMode
from bbstrader.models.risk import build_hmm_models
from bbstrader.tseries import ArimaGarchModel, KalmanFilterModel

__all__ = [
    "SMAStrategy",
    "ArimaGarchStrategy",
    "KalmanFilterStrategy",
    "StockIndexSTBOTrading",
    "test_strategy",
    "get_quantities",
]


def get_quantities(quantities, symbol_list):
    if isinstance(quantities, dict):
        return quantities
    elif isinstance(quantities, int):
        return {symbol: quantities for symbol in symbol_list}


class SMAStrategy(Strategy):
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
        self,
        bars: DataHandler = None,
        events: Queue = None,
        symbol_list: List[str] = None,
        mode: TradingMode = TradingMode.BACKTEST,
        **kwargs,
    ):
        """
        Args:
            bars (DataHandler): A data handler object that provides market data.
            events (Queue): An event queue object where generated signals are placed.
            symbol_list (List[str]): A list of symbols to consider for trading.
            mode TradingMode: The mode of operation for the strategy.
            short_window (int, optional): The period for the short moving average.
            long_window (int, optional): The period for the long moving average.
            time_frame (str, optional): The time frame for the data.
            session_duration (float, optional): The duration of the trading session.
            risk_window (int, optional): The window size for the risk model.
            quantities (int, dict | optional): The default quantities of each asset to trade.
        """
        self.bars = bars
        self.events = events
        self.symbol_list = symbol_list or self.bars.symbol_list
        self.mode = mode

        self.kwargs = kwargs
        self.short_window = kwargs.get("short_window", 50)
        self.long_window = kwargs.get("long_window", 200)
        self.tf = kwargs.get("time_frame", "D1")
        self.qty = get_quantities(kwargs.get("quantities", 100), self.symbol_list)
        self.sd = kwargs.get("session_duration", 23.0)
        self.risk_models = build_hmm_models(self.symbol_list, **self.kwargs)
        self.risk_window = kwargs.get("risk_window", self.long_window)
        self.bought = self._calculate_initial_bought()

    def _calculate_initial_bought(self):
        bought = {}
        for s in self.symbol_list:
            bought[s] = "OUT"
        return bought

    def get_backtest_data(self):
        symbol_data = {symbol: None for symbol in self.symbol_list}
        for s in self.symbol_list:
            bar_date = self.bars.get_latest_bar_datetime(s)
            bars = self.bars.get_latest_bars_values(s, "adj_close", N=self.long_window)
            returns_val = self.bars.get_latest_bars_values(
                s, "returns", N=self.risk_window
            )
            if len(bars) >= self.long_window and len(returns_val) >= self.risk_window:
                regime = self.risk_models[s].which_trade_allowed(returns_val)

                short_sma = np.mean(bars[-self.short_window :])
                long_sma = np.mean(bars[-self.long_window :])

                symbol_data[s] = (short_sma, long_sma, regime, bar_date)
        return symbol_data

    def create_backtest_signals(self):
        signals = {symbol: None for symbol in self.symbol_list}
        symbol_data = self.get_backtest_data()
        for s, data in symbol_data.items():
            signal = None
            if data is not None:
                price = self.bars.get_latest_bar_value(s, "adj_close")
                short_sma, long_sma, regime, bar_date = data
                dt = bar_date
                if regime == "LONG":
                    # Bulliqh regime
                    if short_sma < long_sma and self.bought[s] == "LONG":
                        print(f"EXIT: {bar_date}")
                        signal = SignalEvent(1, s, dt, "EXIT", price=price)
                        self.bought[s] = "OUT"

                    elif short_sma > long_sma and self.bought[s] == "OUT":
                        print(f"LONG: {bar_date}")
                        signal = SignalEvent(
                            1, s, dt, "LONG", quantity=self.qty[s], price=price
                        )
                        self.bought[s] = "LONG"

                elif regime == "SHORT":
                    # Bearish regime
                    if short_sma > long_sma and self.bought[s] == "SHORT":
                        print(f"EXIT: {bar_date}")
                        signal = SignalEvent(1, s, dt, "EXIT", price=price)
                        self.bought[s] = "OUT"

                    elif short_sma < long_sma and self.bought[s] == "OUT":
                        print(f"SHORT: {bar_date}")
                        signal = SignalEvent(
                            1, s, dt, "SHORT", quantity=self.qty[s], price=price
                        )
                        self.bought[s] = "SHORT"
                signals[s] = signal
        return signals

    def get_live_data(self):
        symbol_data = {symbol: None for symbol in self.symbol_list}
        for symbol in self.symbol_list:
            sig_rate = Rates(symbol, self.tf, 0, self.risk_window + 2, **self.kwargs)
            hmm_data = sig_rate.returns.values
            prices = sig_rate.close.values
            current_regime = self.risk_models[symbol].which_trade_allowed(hmm_data)
            assert len(prices) >= self.long_window and len(hmm_data) >= self.risk_window
            short_sma = np.mean(prices[-self.short_window :])
            long_sma = np.mean(prices[-self.long_window :])
            short_sma, long_sma, current_regime
            symbol_data[symbol] = (short_sma, long_sma, current_regime)
        return symbol_data

    def create_live_signals(self):
        signals = {symbol: None for symbol in self.symbol_list}
        symbol_data = self.get_live_data()
        for symbol, data in symbol_data.items():
            signal = None
            short_sma, long_sma, regime = data
            if regime == "LONG":
                if short_sma > long_sma:
                    signal = "LONG"
            elif regime == "SHORT":
                if short_sma < long_sma:
                    signal = "SHORT"
            signals[symbol] = signal
        return signals

    def calculate_signals(self, event=None):
        if self.mode == TradingMode.BACKTEST and event is not None:
            if event.type == Events.MARKET:
                signals = self.create_backtest_signals()
                for signal in signals.values():
                    if signal is not None:
                        self.events.put(signal)
        elif self.mode == TradingMode.LIVE:
            signals = self.create_live_signals()
            return signals


class ArimaGarchStrategy(Strategy):
    """
    The `ArimaGarchStrategy` class extends the `Strategy`
    class to implement a backtesting framework for trading strategies based on
    ARIMA-GARCH models, incorporating a Hidden Markov Model (HMM) for risk management.

    Features
    ========
    - **ARIMA-GARCH Model**: Utilizes ARIMA for time series forecasting and GARCH for volatility forecasting, aimed at predicting market movements.

    - **HMM Risk Management**: Employs a Hidden Markov Model to manage risks, determining safe trading regimes.

    - **Event-Driven Backtesting**: Capable of simulating real-time trading conditions by processing market data and signals sequentially.

    - **Live Trading**: Supports real-time trading by generating signals based on live ARIMA-GARCH predictions and HMM risk management.

    Key Methods
    ===========
    - `get_backtest_data()`: Retrieves historical data for backtesting.
    - `create_backtest_signal()`: Generates trading signals based on ARIMA-GARCH predictions and HMM risk management.
    - `get_live_data()`: Retrieves live data for real-time trading.
    - `create_live_signals()`: Generates trading signals based on live ARIMA-GARCH predictions and HMM risk management.
    - `calculate_signals()`: Determines the trading signals based on the mode of operation (backtest or live).

    """

    def __init__(
        self,
        bars: DataHandler = None,
        events: Queue = None,
        symbol_list: List[str] = None,
        mode: TradingMode = TradingMode.BACKTEST,
        **kwargs,
    ):
        """
        Args:
            `bars`: A data handler object that provides market data.
            `events`: An event queue object where generated signals are placed.
            `symbol_list`: A list of symbols to consider for trading.
            `mode`: The mode of operation for the strategy.
            `arima_window`: The window size for rolling prediction in backtesting.
            `time_frame`: The time frame for the data.
            `quantities`: Quantity of each assets to trade.
            `hmm_window`: Lookback period for HMM.
        """
        self.bars = bars
        self.events = events
        self.symbol_list = symbol_list or self.bars.symbol_list
        self.mode = mode

        self.qty = get_quantities(kwargs.get("quantities", 100), self.symbol_list)
        self.arima_window = kwargs.get("arima_window", 252)
        self.tf = kwargs.get("time_frame", "D1")
        self.sd = kwargs.get("session_duration", 23.0)
        self.risk_window = kwargs.get("hmm_window", 50)
        self.risk_models = build_hmm_models(self.symbol_list, **kwargs)
        self.arima_models = self._build_arch_models(**kwargs)

        self.long_market = {s: False for s in self.symbol_list}
        self.short_market = {s: False for s in self.symbol_list}

    def _build_arch_models(self, **kwargs) -> Dict[str, ArimaGarchModel]:
        arch_models = {symbol: None for symbol in self.symbol_list}
        for symbol in self.symbol_list:
            try:
                rates = Rates(symbol, self.tf, 0)
                data = rates.get_rates_from_pos()
                assert data is not None, f"No data for {symbol}"
            except AssertionError:
                data = yf.download(symbol, start=kwargs.get("yf_start"))
            arch = ArimaGarchModel(symbol, data, k=self.arima_window)
            arch_models[symbol] = arch
        return arch_models

    def get_backtest_data(self):
        symbol_data = {symbol: None for symbol in self.symbol_list}
        for symbol in self.symbol_list:
            M = self.arima_window
            N = self.risk_window
            dt = self.bars.get_latest_bar_datetime(symbol)
            bars = self.bars.get_latest_bars_values(
                symbol, "close", N=self.arima_window
            )
            returns = self.bars.get_latest_bars_values(
                symbol, "returns", N=self.risk_window
            )
            df = pd.DataFrame()
            df["Close"] = bars[-M:]
            df = df.dropna()
            arch_returns = self.arima_models[symbol].load_and_prepare_data(df)
            data = arch_returns["diff_log_return"].iloc[-self.arima_window :]
            if len(data) >= M and len(returns) >= N:
                symbol_data[symbol] = (data, returns[-N:], dt)
        return symbol_data

    def create_backtest_signal(self):
        signals = {symbol: None for symbol in self.symbol_list}
        for symbol in self.symbol_list:
            symbol_data = self.get_backtest_data()[symbol]
            if symbol_data is not None:
                data, returns, dt = symbol_data
                signal = None
                prediction = self.arima_models[symbol].get_prediction(data)
                regime = self.risk_models[symbol].which_trade_allowed(returns)
                price = self.bars.get_latest_bar_value(symbol, "adj_close")

                # If we are short the market, check for an exit
                if prediction > 0 and self.short_market[symbol]:
                    signal = SignalEvent(1, symbol, dt, "EXIT", price=price)
                    print(f"{dt}: EXIT SHORT")
                    self.short_market[symbol] = False

                # If we are long the market, check for an exit
                elif prediction < 0 and self.long_market[symbol]:
                    signal = SignalEvent(1, symbol, dt, "EXIT", price=price)
                    print(f"{dt}: EXIT LONG")
                    self.long_market[symbol] = False

                if regime == "LONG":
                    # If we are not in the market, go long
                    if prediction > 0 and not self.long_market[symbol]:
                        signal = SignalEvent(
                            1,
                            symbol,
                            dt,
                            "LONG",
                            quantity=self.qty[symbol],
                            price=price,
                        )
                        print(f"{dt}: LONG")
                        self.long_market[symbol] = True

                elif regime == "SHORT":
                    # If we are not in the market, go short
                    if prediction < 0 and not self.short_market[symbol]:
                        signal = SignalEvent(
                            1,
                            symbol,
                            dt,
                            "SHORT",
                            quantity=self.qty[symbol],
                            price=price,
                        )
                        print(f"{dt}: SHORT")
                        self.short_market[symbol] = True
                signals[symbol] = signal
        return signals

    def get_live_data(self):
        symbol_data = {symbol: None for symbol in self.symbol_list}
        for symbol in self.symbol_list:
            arch_data = Rates(symbol, self.tf, 0, self.arima_window)
            rates = arch_data.get_rates_from_pos()
            arch_returns = self.arima_models[symbol].load_and_prepare_data(rates)
            window_data = arch_returns["diff_log_return"].iloc[-self.arima_window :]
            hmm_returns = arch_data.returns.values[-self.risk_window :]
            symbol_data[symbol] = (window_data, hmm_returns)
        return symbol_data

    def create_live_signals(self):
        signals = {symbol: None for symbol in self.symbol_list}
        data = self.get_live_data()
        for symbol in self.symbol_list:
            symbol_data = data[symbol]
            if symbol_data is not None:
                window_data, hmm_returns = symbol_data
                prediction = self.arima_models[symbol].get_prediction(window_data)
                regime = self.risk_models[symbol].which_trade_allowed(hmm_returns)
                if regime == "LONG":
                    if prediction > 0:
                        signals[symbol] = "LONG"
                elif regime == "SHORT":
                    if prediction < 0:
                        signals[symbol] = "SHORT"
        return signals

    def calculate_signals(self, event=None):
        if self.mode == TradingMode.BACKTEST and event is not None:
            if event.type == Events.MARKET:
                signals = self.create_backtest_signal()
                for signal in signals.values():
                    if signal is not None:
                        self.events.put(signal)
        elif self.mode == TradingMode.LIVE:
            return self.create_live_signals()


class KalmanFilterStrategy(Strategy):
    """
    The `KalmanFilterStrategy` class implements a backtesting framework for a
    [pairs trading](https://en.wikipedia.org/wiki/Pairs_trade) strategy using
    Kalman Filter for signals and Hidden Markov Models (HMM) for risk management.
    This document outlines the structure and usage of the `KalmanFilterStrategy`,
    including initialization parameters, main functions, and an example of how to run a backtest.
    """

    def __init__(
        self,
        bars: DataHandler = None,
        events: Queue = None,
        symbol_list: List[str] = None,
        mode: TradingMode = TradingMode.BACKTEST,
        **kwargs,
    ):
        """
        Args:
            `bars`: `DataHandler` for market data handling.
            `events`: A queue for managing events.
            `symbol_list`: List of ticker symbols for the pairs trading strategy.
            `mode`: Mode of operation for the strategy.
            kwargs : Additional keyword arguments including
                - `quantity`: Quantity of assets to trade. Default is 100.
                - `hmm_window`: Window size for calculating returns for the HMM. Default is 50.
                - `hmm_tiker`: Ticker symbol used by the HMM for risk management.
                - `time_frame`: Time frame for the data. Default is 'D1'.
                - `session_duration`: Duration of the trading session. Default is 6.5.
        """
        self.bars = bars
        self.events_queue = events
        self.symbol_list = symbol_list or self.bars.symbol_list
        self.mode = mode

        self.hmm_tiker = kwargs.get("hmm_tiker")
        self._assert_tikers()
        self.account = Account(**kwargs)
        self.hmm_window = kwargs.get("hmm_window", 50)
        self.qty = kwargs.get("quantity", 100)
        self.tf = kwargs.get("time_frame", "D1")
        self.sd = kwargs.get("session_duration", 6.5)

        self.risk_model = build_hmm_models(self.symbol_list, **kwargs)
        self.kl_model = KalmanFilterModel(self.tickers, **kwargs)

        self.long_market = False
        self.short_market = False

    def _assert_tikers(self):
        if self.symbol_list is None or len(self.symbol_list) != 2:
            raise ValueError("A list of 2 Tickers must be provide for this strategy")
        self.tickers = self.symbol_list
        if self.hmm_tiker is None:
            raise ValueError(
                "You need to provide a ticker used by the HMM for risk management"
            )

    def calculate_btxy(self, etqt, regime, dt):
        # Make sure there is no position open
        if etqt is None:
            return
        et, sqrt_Qt = etqt
        theta = self.kl_model.theta
        p1 = self.bars.get_latest_bar_value(self.tickers[1], "adj_close")
        p0 = self.bars.get_latest_bar_value(self.tickers[0], "adj_close")
        if et >= -sqrt_Qt and self.long_market:
            print("CLOSING LONG: %s" % dt)
            y_signal = SignalEvent(1, self.tickers[1], dt, "EXIT", price=p1)
            x_signal = SignalEvent(1, self.tickers[0], dt, "EXIT", price=p0)
            self.events_queue.put(y_signal)
            self.events_queue.put(x_signal)
            self.long_market = False

        elif et <= sqrt_Qt and self.short_market:
            print("CLOSING SHORT: %s" % dt)
            y_signal = SignalEvent(1, self.tickers[1], dt, "EXIT", price=p1)
            x_signal = SignalEvent(1, self.tickers[0], dt, "EXIT", price=p0)
            self.events_queue.put(y_signal)
            self.events_queue.put(x_signal)
            self.short_market = False

        # Long Entry
        if regime == "LONG":
            if et <= -sqrt_Qt and not self.long_market:
                print("LONG: %s" % dt)
                y_signal = SignalEvent(
                    1, self.tickers[1], dt, "LONG", self.qty, 1.0, price=p1
                )
                x_signal = SignalEvent(
                    1, self.tickers[0], dt, "SHORT", self.qty, theta[0], price=p0
                )
                self.events_queue.put(y_signal)
                self.events_queue.put(x_signal)
                self.long_market = True

        # Short Entry
        elif regime == "SHORT":
            if et >= sqrt_Qt and not self.short_market:
                print("SHORT: %s" % dt)
                y_signal = SignalEvent(
                    1, self.tickers[1], dt, "SHORT", self.qty, 1.0, price=p1
                )
                x_signal = SignalEvent(
                    1, self.tickers[0], "LONG", self.qty, theta[0], price=p0
                )
                self.events_queue.put(y_signal)
                self.events_queue.put(x_signal)
                self.short_market = True

    def calculate_livexy(self):
        signals = {symbol: None for symbol in self.symbol_list}
        p0_price = self.account.get_tick_info(self.tickers[0]).ask
        p1_price = self.account.get_tick_info(self.tickers[1]).ask
        prices = np.array([p0_price, p1_price])
        et_std = self.kl_model.calculate_etqt(prices)
        if et_std is not None:
            et, std = et_std
            y_signal = None
            x_signal = None

            if et >= -std or et <= std:
                y_signal = "EXIT"
                x_signal = "EXIT"

            if et <= -std:
                y_signal = "LONG"
                x_signal = "SHORT"

            if et >= std:
                y_signal = "SHORT"
                x_signal = "LONG"

            signals[self.tickers[0]] = x_signal
            signals[self.tickers[1]] = y_signal
        return signals

    def calculate_backtest_signals(self):
        p0, p1 = self.tickers[0], self.tickers[1]
        dt = self.bars.get_latest_bar_datetime(p0)
        x = self.bars.get_latest_bar_value(p0, "close")
        y = self.bars.get_latest_bar_value(p1, "close")
        returns = self.bars.get_latest_bars_values(
            self.hmm_tiker, "returns", N=self.hmm_window
        )
        latest_prices = np.array([-1.0, -1.0])
        if len(returns) >= self.hmm_window:
            latest_prices[0] = x
            latest_prices[1] = y
            et_qt = self.kl_model.calculate_etqt(latest_prices)
            regime = self.risk_model[self.hmm_tiker].which_trade_allowed(returns)
            self.calculate_btxy(et_qt, regime, dt)

    def calculate_live_signals(self):
        # Data Retrieval
        signals = {symbol: None for symbol in self.symbol_list}
        initial_signals = self.calculate_livexy()
        hmm_data = Rates(self.hmm_ticker, self.tf, 0, self.hmm_window)
        returns = hmm_data.returns.values
        current_regime = self.risk_model[self.hmm_tiker].which_trade_allowed(returns)
        for symbol in self.symbol_list:
            if symbol in initial_signals:
                signal = initial_signals[symbol]
                if signal == "LONG" and current_regime == "LONG":
                    signals[symbol] = "LONG"
                elif signal == "SHORT" and current_regime == "SHORT":
                    signals[symbol] = "SHORT"
        return signals

    def calculate_signals(self, event=None):
        """
        Calculate the Kalman Filter strategy.
        """
        if self.mode == TradingMode.BACKTEST and event is not None:
            if event.type == Events.MARKET:
                self.calculate_backtest_signals()
        elif self.mode == TradingMode.LIVE:
            return self.calculate_live_signals()


class StockIndexSTBOTrading(Strategy):
    """
    The StockIndexSTBOTrading class implements a stock index Contract for Difference (CFD)
    Buy-Only trading strategy. This strategy is based on the assumption that stock markets
    typically follow a long-term uptrend. The strategy is designed to capitalize on market
    corrections and price dips, where stocks or indices temporarily drop but are expected
    to recover. It operates in two modes: backtest and live, and it is particularly
    tailored to index trading.
    """

    def __init__(
        self,
        bars: DataHandler = None,
        events: Queue = None,
        symbol_list: List[str] = None,
        mode: TradingMode = TradingMode.BACKTEST,
        **kwargs,
    ):
        """
        Args:
            `bars`: `DataHandler` for market data handling.
            `events`: A queue for managing events.
            `symbol_list`: List of ticker symbols for the pairs trading strategy.
            `mode`: Mode of operation for the strategy.
            kwargs : Additional keyword arguments including
                - rr (float, default: 3.0): The risk-reward ratio used to determine exit points.
                - epsilon (float, default: 0.1): The percentage threshold for price changes when considering new highs or lows.
                - expected_returns (dict): Expected return percentages for each symbol in the symbol list.
                - quantities (int, default: 100): The number of units to trade.
                - max_trades (dict): The maximum number of trades allowed per symbol.
                - logger: A logger object for tracking operations.
                - expert_id (int, default: 5134): Unique identifier for trade positions created by this strategy.
        """
        self.bars = bars
        self.events = events
        self.symbol_list = symbol_list or self.bars.symbol_list
        self.mode = mode

        self.account = Account()

        self.rr = kwargs.get("rr", 3.0)
        self.epsilon = kwargs.get("epsilon", 0.1)
        self._initialize(**kwargs)
        self.logger = kwargs.get("logger")
        self.ID = kwargs.get("expert_id", 5134)

    def _initialize(self, **kwargs):
        symbols = self.symbol_list.copy()
        returns = kwargs.get("expected_returns")
        quantities = kwargs.get("quantities", 100)
        max_trades = kwargs.get("max_trades")

        self.expeted_return = {index: returns[index] for index in symbols}
        self.max_trades = {index: max_trades[index] for index in symbols}
        self.last_price = {index: None for index in symbols}
        self.heightest_price = {index: None for index in symbols}
        self.lowerst_price = {index: None for index in symbols}

        if self.mode == TradingMode.BACKTEST:
            self.qty = get_quantities(quantities, symbols)
            self.num_buys = {index: 0 for index in symbols}
            self.buy_prices = {index: [] for index in symbols}

    def _calculate_pct_change(self, current_price, lh_price):
        return ((current_price - lh_price) / lh_price) * 100

    def calculate_live_signals(self):
        signals = {index: None for index in self.symbol_list}
        for index in self.symbol_list:
            current_price = self.account.get_tick_info(index).ask
            if self.last_price[index] is None:
                self.last_price[index] = current_price
                self.heightest_price[index] = current_price
                self.lowerst_price[index] = current_price
                continue
            else:
                if (
                    self._calculate_pct_change(
                        current_price, self.heightest_price[index]
                    )
                    >= self.epsilon
                ):
                    self.heightest_price[index] = current_price
                elif (
                    self._calculate_pct_change(current_price, self.lowerst_price[index])
                    <= -self.epsilon
                ):
                    self.lowerst_price[index] = current_price

                down_change = self._calculate_pct_change(
                    current_price, self.heightest_price[index]
                )

                if down_change <= -(self.expeted_return[index] / self.rr):
                    signals[index] = "LONG"

                positions = self.account.get_positions(symbol=index)
                if positions is not None:
                    buy_prices = [
                        position.price_open
                        for position in positions
                        if position.type == 0 and position.magic == self.ID
                    ]
                    if len(buy_prices) == 0:
                        continue
                    avg_price = sum(buy_prices) / len(buy_prices)
                    if (
                        self._calculate_pct_change(current_price, avg_price)
                        >= (self.expeted_return[index])
                    ):
                        signals[index] = "EXIT"
                self.logger.info(
                    f"SYMBOL={index} - Hp={self.heightest_price[index]} - "
                    f"Lp={self.lowerst_price[index]} - Cp={current_price} - %chg={round(down_change, 2)}"
                )
        return signals

    def calculate_backtest_signals(self):
        for index in self.symbol_list.copy():
            dt = self.bars.get_latest_bar_datetime(index)
            last_price = self.bars.get_latest_bars_values(index, "close", N=1)

            current_price = last_price[-1]
            if self.last_price[index] is None:
                self.last_price[index] = current_price
                self.heightest_price[index] = current_price
                self.lowerst_price[index] = current_price
                continue
            else:
                if (
                    self._calculate_pct_change(
                        current_price, self.heightest_price[index]
                    )
                    >= self.epsilon
                ):
                    self.heightest_price[index] = current_price
                elif (
                    self._calculate_pct_change(current_price, self.lowerst_price[index])
                    <= -self.epsilon
                ):
                    self.lowerst_price[index] = current_price

            down_change = self._calculate_pct_change(
                current_price, self.heightest_price[index]
            )

            if (
                down_change <= -(self.expeted_return[index] / self.rr)
                and self.num_buys[index] <= self.max_trades[index]
            ):
                signal = SignalEvent(
                    100,
                    index,
                    dt,
                    "LONG",
                    quantity=self.qty[index],
                    price=current_price,
                )
                self.events.put(signal)
                self.num_buys[index] += 1
                self.buy_prices[index].append(current_price)

            elif self.num_buys[index] > 0:
                av_price = sum(self.buy_prices[index]) / len(self.buy_prices[index])
                qty = self.qty[index] * self.num_buys[index]
                if (
                    self._calculate_pct_change(current_price, av_price)
                    >= (self.expeted_return[index])
                ):
                    signal = SignalEvent(
                        100, index, dt, "EXIT", quantity=qty, price=current_price
                    )
                    self.events.put(signal)
                    self.num_buys[index] = 0
                    self.buy_prices[index] = []

    def calculate_signals(self, event=None) -> Dict[str, Union[str, None]]:
        if self.mode == TradingMode.BACKTEST and event is not None:
            if event.type == Events.MARKET:
                self.calculate_backtest_signals()
        elif self.mode == TradingMode.LIVE:
            return self.calculate_live_signals()


def _run_backtest(strategy_name: str, capital: float, symbol_list: list, kwargs: dict):
    """
    Executes a backtest of the specified strategy
    integrating a Hidden Markov Model (HMM) for risk management.
    """
    kwargs["strategy_name"] = strategy_name
    engine = BacktestEngine(
        symbol_list,
        capital,
        0.0,
        datetime.strptime(kwargs["yf_start"], "%Y-%m-%d"),
        kwargs.get("data_handler", YFDataHandler),
        kwargs.get("exc_handler", SimExecutionHandler),
        kwargs.pop("backtester_class"),
        **kwargs,
    )
    engine.simulate_trading()


def _run_arch_backtest(capital: float = 100000.0, quantity: int = 1000):
    hmm_data = yf.download("^GSPC", start="1990-01-01", end="2009-12-31")
    kwargs = {
        "quantity": quantity,
        "yf_start": "2010-01-04",
        "hmm_data": hmm_data,
        "backtester_class": ArimaGarchStrategy,
        "data_handler": YFDataHandler,
    }
    _run_backtest("ARIMA+GARCH & HMM", capital, ["^GSPC"], kwargs)


def _run_kf_backtest(capital: float = 100000.0, quantity: int = 2000):
    symbol_list = ["IEI", "TLT"]
    tlt = yf.download("TLT", end="2008-07-09")
    iei = yf.download("IEI", end="2008-07-09")
    kwargs = {
        "quantity": quantity,
        "yf_start": "2009-08-03",
        "hmm_data": {"IEI": iei, "TLT": tlt},
        "hmm_tiker": "TLT",
        "session_duration": 6.5,
        "backtester_class": KalmanFilterStrategy,
        "data_handler": YFDataHandler,
    }
    _run_backtest("Kalman Filter & HMM", capital, symbol_list, kwargs)


def _run_sma_backtest(capital: float = 100000.0, quantity: int = 1):
    spx_data = yf.download("^GSPC", start="1990-01-01", end="2009-12-31")
    kwargs = {
        "quantities": quantity,
        "hmm_end": "2009-12-31",
        "yf_start": "2010-01-04",
        "hmm_data": spx_data,
        "mt5_start": datetime(2010, 1, 1),
        "mt5_end": datetime(2023, 1, 1),
        "backtester_class": SMAStrategy,
        "data_handler": MT5DataHandler,
        "exc_handler": MT5ExecutionHandler,
    }
    _run_backtest("SMA & HMM", capital, ["[SP500]"], kwargs)


def _run_sistbo_backtest(capital: float = 100000.0, quantity: int = None):
    ndx = "[NQ100]"
    spx = "[SP500]"
    dji = "[DJI30]"
    dax = "GERMANY40"

    symbol_list = [spx, dax, dji, ndx]
    start = datetime(2010, 6, 1, 2, 0, 0)
    quantity = {ndx: 15, spx: 30, dji: 5, dax: 10}
    kwargs = {
        "expected_returns": {ndx: 1.5, spx: 1.5, dji: 1.0, dax: 1.0},
        "quantities": quantity,
        "max_trades": {ndx: 3, spx: 3, dji: 3, dax: 3},
        "mt5_start": start,
        "yf_start": start.strftime("%Y-%m-%d"),
        "time_frame": "15m",
        "backtester_class": StockIndexSTBOTrading,
        "data_handler": MT5DataHandler,
        "exc_handler": MT5ExecutionHandler,
    }
    _run_backtest("Stock Index Short Term Buy Only ", capital, symbol_list, kwargs)


_BACKTESTS = {
    "sma": _run_sma_backtest,
    "klf": _run_kf_backtest,
    "arch": _run_arch_backtest,
    "sistbo": _run_sistbo_backtest,
}


def test_strategy(
    strategy: Literal["sma", "klf", "arch", "sistbo"] = "sma",
    quantity: Optional[int] = 100,
):
    """
    Executes a backtest of the specified strategy

    Args:
        strategy : The strategy to use in test mode. Default is `sma`.
            - `sma` Execute `SMAStrategy`, for more detail see this class documentation.
            - `klf` Execute `KalmanFilterStrategy`, for more detail see this class documentation.
            - `arch` Execute `ArimaGarchStrategy`, for more detail see this class documentation.
            - `sistbo` Execute `StockIndexSTBOTrading`, for more detail see this class documentation.
        quantity : The quantity of assets to be used in the test backtest. Default is 1000.

    """
    if strategy in _BACKTESTS:
        _BACKTESTS[strategy](quantity=quantity)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
