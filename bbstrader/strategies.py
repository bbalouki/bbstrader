"""
Trading Strategies module
"""


import numpy as np
import seaborn as sns
import pandas as pd
import time
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from filterpy.kalman import KalmanFilter
from scipy.optimize import minimize
from bbstrader.tseries import (
    load_and_prepare_data, fit_best_arima,
    fit_garch, predict_next_return, get_prediction
)
sns.set_theme()

__all__ = [
    "ArimaGarchStrategy",
    "KLFStrategy",
    "SMAStrategy",
    "OrnsteinUhlenbeck"
]

class ArimaGarchStrategy():
    """
    This class implements a trading strategy 
    that combines `ARIMA (AutoRegressive Integrated Moving Average)` 
    and `GARCH (Generalized Autoregressive Conditional Heteroskedasticity)` models
    to predict future returns based on historical price data. 
    It inherits from a abstract `Strategy` class and implement `calculate_signals()`.

    The strategy is implemented in the following steps:
    1. Data Preparation: Load and prepare the historical price data.
    2. Modeling: Fit the ARIMA model to the data and then fit the GARCH model to the residuals.
    3. Prediction: Predict the next return using the ARIMA model and the next volatility using the GARCH model.
    4. Trading Strategy: Execute the trading strategy based on the predictions.
    5. Vectorized Backtesting: Backtest the trading strategy using the historical data.

    Exemple:
    >>> import yfinance as yf
    >>> from bbstrader.strategies import ArimaGarchStrategy
    >>> from bbstrader.tseries import load_and_prepare_data

    >>> if __name__ == '__main__':
    >>>     # ARCH SPY Vectorize Backtest
    >>>     k = 252
    >>>     data = yf.download("SPY", start="2004-01-02", end="2015-12-31")
    >>>     arch = ArimaGarchStrategy("SPY", data, k=k)
    >>>     df = load_and_prepare_data(data)
    >>>     arch.show_arima_garch_results(df['diff_log_return'].values[-k:])
    >>>     arch.backtest_strategy()
    """

    def __init__(self, symbol, data, k: int = 252):
        """
        Initializes the ArimaGarchStrategy class.

        Args:
            symbol (str): The ticker symbol for the financial instrument.
            data (pd.DataFrame): `The raw dataset containing at least the 'Close' prices`.
            k (int): The window size for rolling prediction in backtesting.
        """
        self.symbol = symbol
        self.data = self.load_and_prepare_data(data)
        self.k = k

    # Step 1: Data Preparation
    def load_and_prepare_data(self, df):
        """
        Prepares the dataset by calculating logarithmic returns 
            and differencing if necessary.

        Args:
            df (pd.DataFrame): `The raw dataset containing at least the 'Close' prices`.

        Returns:
            pd.DataFrame: The dataset with additional columns 
                for log returns and differenced log returns.
        """
        return load_and_prepare_data(df)

    # Step 2: Modeling (ARIMA + GARCH)
    def fit_best_arima(self, window_data):
        """
        Fits the ARIMA model to the provided window of data, 
            selecting the best model based on AIC.

        Args:
            window_data (np.array): The dataset for a specific window period.

        Returns:
            ARIMA model: The best fitted ARIMA model based on AIC.
        """
        return fit_best_arima(window_data)

    def fit_garch(self, window_data):
        """
        Fits the GARCH model to the residuals of the best ARIMA model.

        Args:
            window_data (np.array): The dataset for a specific window period.

        Returns:
            tuple: Contains the ARIMA result and GARCH result.
        """
        return fit_garch(window_data)

    def show_arima_garch_results(self, window_data, acf=True, test_resid=True):
        """
        Displays the ARIMA and GARCH model results, including plotting 
        ACF of residuals and conducting , Box-Pierce and Ljung-Box tests.

        Args:
            window_data (np.array): The dataset for a specific window period.
            acf (bool, optional): If True, plot the ACF of residuals. Defaults to True.

            test_resid (bool, optional): 
                If True, conduct Box-Pierce and Ljung-Box tests on residuals. Defaults to True.
        """
        arima_result = self.fit_best_arima(window_data)
        resid = np.asarray(arima_result.resid)
        resid = resid[~(np.isnan(resid) | np.isinf(resid))]
        garch_model = arch_model(resid, p=1, q=1, rescale=False)
        garch_result = garch_model.fit(disp='off')
        residuals = garch_result.resid

        # TODO : Plot the ACF of the residuals
        if acf:
            fig = plt.figure(figsize=(12, 8))
            # Plot the ACF of ARIMA residuals
            ax1 = fig.add_subplot(211, ylabel='ACF')
            plot_acf(resid, alpha=0.05, ax=ax1, title='ACF of ARIMA Residuals')
            ax1.set_xlabel('Lags')
            ax1.grid(True)

            # Plot the ACF of GARCH residuals on the same axes
            ax2 = fig.add_subplot(212, ylabel='ACF')
            plot_acf(residuals, alpha=0.05, ax=ax2,
                     title='ACF of GARCH  Residuals')
            ax2.set_xlabel('Lags')
            ax2.grid(True)

            # Plot the figure
            plt.tight_layout()
            plt.show()

        # TODO : Conduct Box-Pierce and Ljung-Box Tests of the residuals
        if test_resid:
            print(arima_result.summary())
            print(garch_result.summary())
            bp_test = acorr_ljungbox(resid, return_df=True)
            print("Box-Pierce and Ljung-Box Tests Results  for ARIMA:\n", bp_test)

    # Step 3: Prediction
    def predict_next_return(self, arima_result, garch_result):
        """
        Predicts the next return using the ARIMA model 
            and the next volatility using the GARCH model.

        Args:
            arima_result (ARIMA model): The ARIMA model result.
            garch_result (GARCH model): The GARCH model result.

        Returns:
            float: The predicted next return.
        """
        return predict_next_return(arima_result, garch_result)

    def get_prediction(self, window_data):
        """
        Generates a prediction for the next return based on a window of data.

        Args:
            window_data (np.array): The dataset for a specific window period.

        Returns:
            float: The predicted next return.
        """
        return get_prediction(window_data)

    def calculate_signals(self, window_data):
        """
        Calculates the trading signal based on the prediction.

        Args:
            window_data (np.array): The dataset for a specific window period.

        Returns:
            str: The trading signal ('LONG', 'SHORT', or None).
        """
        prediction = self.get_prediction(window_data)
        if prediction > 0:
            signal = "LONG"
        elif prediction < 0:
            signal = "SHORT"
        else:
            signal = None
        return signal

    # Step 4: Trading Strategy

    def execute_trading_strategy(self, predictions):
        """
        Executes the trading strategy based on a list 
        of predictions, determining positions to take.

        Args:
            predictions (list): A list of predicted returns.

        Returns:
            list: A list of positions (1 for 'LONG', -1 for 'SHORT', 0 for 'HOLD').
        """
        positions = []  # Long if 1, Short if -1
        previous_position = 0  # Initial position
        for prediction in predictions:
            if prediction > 0:
                current_position = 1  # Long
            elif prediction < 0:
                current_position = -1  # Short
            else:
                current_position = previous_position  # Hold previous position
            positions.append(current_position)
            previous_position = current_position

        return positions

    # Step 5: Vectorized Backtesting
    def generate_predictions(self):
        """
        Generator that yields predictions one by one.
        """
        data = self.data
        window_size = self.k
        for i in range(window_size, len(data)):
            print(
                f"Processing window {i - window_size + 1}/{len(data) - window_size}...")
            window_data = data['diff_log_return'].iloc[i-window_size:i]
            next_return = self.get_prediction(window_data)
            yield next_return

    def backtest_strategy(self):
        """
        Performs a backtest of the strategy over 
        the entire dataset, plotting cumulative returns.
        """
        data = self.data
        window_size = self.k
        print(
            f"Starting backtesting for {self.symbol}\n"
            f"Window size {window_size}.\n"
            f"Total iterations: {len(data) - window_size}.\n")
        predictions_generator = self.generate_predictions()

        positions = self.execute_trading_strategy(predictions_generator)

        strategy_returns = np.array(
            positions[:-1]) * data['log_return'].iloc[window_size+1:].values
        buy_and_hold = data['log_return'].iloc[window_size+1:].values
        buy_and_hold_returns = np.cumsum(buy_and_hold)
        cumulative_returns = np.cumsum(strategy_returns)
        dates = data.index[window_size+1:]
        self.plot_cumulative_returns(
            cumulative_returns, buy_and_hold_returns, dates)

        print("\nBacktesting completed !!")

    # Function to plot the cumulative returns
    def plot_cumulative_returns(self, strategy_returns, buy_and_hold_returns, dates):
        """
        Plots the cumulative returns of the ARIMA+GARCH strategy against 
            a buy-and-hold strategy.

        Args:
            strategy_returns (np.array): Cumulative returns from the strategy.
            buy_and_hold_returns (np.array): Cumulative returns from a buy-and-hold strategy.
            dates (pd.Index): The dates corresponding to the returns.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(dates, strategy_returns, label='ARIMA+GARCH ', color='blue')
        plt.plot(dates, buy_and_hold_returns, label='Buy & Hold', color='red')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')
        plt.title(f'ARIMA+GARCH Strategy vs. Buy & Hold on ({self.symbol})')
        plt.legend()
        plt.grid(True)
        plt.show()


class KLFStrategy():
    """
    Implements a trading strategy based on the Kalman Filter, 
    a recursive algorithm used for estimating the state of a linear dynamic system 
    from a series of noisy measurements. It's designed to process market data, 
    estimate dynamic parameters such as the slope and intercept of price relationships,
    and generate trading signals based on those estimates.

    You can learn more here https://en.wikipedia.org/wiki/Kalman_filter
    """

    def __init__(self, tickers: list | tuple, **kwargs):
        """
        Initializes the Kalman Filter strategy.

        Args:
            tickers : 
            A list or tuple of ticker symbols representing financial instruments.

            kwargs : Keyword arguments for additional parameters,
             specifically `delta` and `vt`
        """
        self.tickers = tickers
        assert self.tickers is not None
        self.latest_prices = np.array([-1.0, -1.0])

        self.delta = kwargs.get("delta", 1e-4)
        self.wt = self.delta/(1-self.delta) * np.eye(2)
        self.vt = kwargs.get("vt", 1e-3)
        self.theta = np.zeros(2)
        self.P = np.zeros((2, 2))
        self.R = None
        self.kf = self._init_kalman()

    def _init_kalman(self):
        """
        Initializes and returns a Kalman Filter configured 
        for the trading strategy. The filter is set up with initial 
        state and covariance, state transition matrix, process noise
        and measurement noise covariances.
        """
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.zeros((2, 1))  # Initial state
        kf.P = self.P  # Initial covariance
        kf.F = np.eye(2)  # State transition matrix
        kf.Q = self.wt  # Process noise covariance
        kf.R = 1.  # Scalar measurement noise covariance

        return kf

    def calc_slope_intercep(self, prices: np.ndarray):
        """
        Calculates and returns the slope and intercept 
        of the relationship between the provided prices using the Kalman Filter. 
        This method updates the filter with the latest price and returns 
        the estimated slope and intercept.

        Args:
            prices : A numpy array of prices for two financial instruments.

        Returns:
            A tuple containing the slope and intercept of the relationship
        """
        kf = self.kf
        kf.H = np.array([[prices[1], 1.0]])
        kf.predict()
        kf.update(prices[0])
        slope = kf.x.copy().flatten()[0]
        intercept = kf.x.copy().flatten()[1]

        return slope, intercept

    def calculate_xy_signals(self, et, std):
        """
        Generates trading signals based on the forecast error 
        and standard deviation of the predictions. It returns signals for exiting, 
        going long, or shorting positions based on the comparison of 
        the forecast error with the standard deviation.

        Args:
            et : The forecast error.
            std : The standard deviation of the predictions.

        Returns:
            A tuple containing the trading signals for the two financial instruments.
        """
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

        return y_signal, x_signal

    def calculate_signals(self, prices: np.ndarray):
        """
        Calculates trading signals based on the latest prices 
        and the Kalman Filter's estimates. It updates the filter's state 
        with the latest prices, computes the slope and intercept
        and generates trading signals based on the forecast error 
        and prediction standard deviation.

        Args:
            prices : A numpy array of prices for two financial instruments.

        Returns:
            A dictionary containing trading signals for the two financial instruments.
        """

        self.latest_prices[0] = prices[0]
        self.latest_prices[1] = prices[1]

        if all(self.latest_prices > -1.0):
            slope, intercept = self.calc_slope_intercep(self.latest_prices)

            self.theta[0] = slope
            self.theta[1] = intercept

            # Create the observation matrix of the latest prices
            # of Y and the intercept value (1.0) as well as the
            # scalar value of the latest price from X
            F = np.asarray([self.latest_prices[0], 1.0]).reshape((1, 2))
            y = self.latest_prices[1]

            # The prior value of the states {\theta_t} is
            # distributed as a multivariate Gaussian with
            # mean a_t and variance-covariance {R_t}
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

            # {Q_t} is the variance of the prediction of
            # observations and hence sqrt_Qt is the
            # standard deviation of the predictions
            Qt = F.dot(self.R).dot(F.T) + self.vt
            sqrt_Qt = np.sqrt(Qt)

            # The posterior value of the states {\theta_t} is
            # distributed as a multivariate Gaussian with mean
            # {m_t} and variance-covariance {C_t}
            At = self.R.dot(F.T) / Qt
            self.theta = self.theta + At.flatten() * et
            self.C = self.R - At * F.dot(self.R)

            y_signal, x_signal = self.calculate_xy_signals(et, sqrt_Qt)

            return {
                self.tickers[1]: y_signal,
                self.tickers[0]: x_signal
            }


class SMAStrategy():

    """
    Carries out a basic Moving Average Crossover strategy with a
    short/long simple weighted moving average. Default short/long
    windows are 50/200 periods respectively and uses Hiden Markov Model 
    as risk Managment system for filteering signals.
    """

    def __init__(
        self, **kwargs
    ):
        self.short_window = kwargs.get("short_window", 50)
        self.long_window = kwargs.get("long_window", 200)

    def get_data(self, prices):
        assert len(prices) >= self.long_window
        short_sma = np.mean(prices[-self.short_window:])
        long_sma = np.mean(prices[-self.long_window:])
        return short_sma, long_sma

    def create_signal(self, prices):
        signal = None
        data = self.get_data(prices)
        short_sma, long_sma = data
        if short_sma > long_sma:
            signal = 'LONG'
        elif short_sma < long_sma:
            signal = 'SHORT'
        return signal

    def calculate_signals(self, prices):
        return self.create_signal(prices)


class OrnsteinUhlenbeck():
    """
    The Ornstein-Uhlenbeck process is a mathematical model 
    used to describe the behavior of a mean-reverting stochastic process. 
    We use it  to model the price dynamics of an asset that tends 
    to revert to a long-term mean.

    We Estimate the drift (θ), volatility (σ), and long-term mean (μ) 
    based on historical price data; then we Simulate the OU process 
    using the estimated parameters.

    https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """

    def __init__(
        self, prices: np.ndarray,
        returns: bool = True, timeframe: str = "D1"
    ):
        """
        Initializes the OrnsteinUhlenbeck instance.

        Args:
            prices (np.ndarray) : Historical close prices.

            retrurns (bool) : Use it to indicate weither 
                you want  to simulate the returns or your raw data

            timeframe (str) : The time  frame for the Historical prices 
                (1m, 5m, 15m, 30m, 1h, 4h, D1)
        """
        self.prices = prices
        if returns:
            series = pd.Series(self.prices)
            self.returns = series.pct_change().dropna().values
        else:
            self.returns = self.prices

        time_frame_mapping = {
            '1m':  1 / (24 * 60),    # 1 minute intervals
            '5m':  5 / (24 * 60),    # 5 minute intervals
            '15m': 15 / (24 * 60),   # 15 minute intervals
            '30m': 30 / (24 * 60),   # 30 minute intervals
            '1h':  1 / 24,           # 1 hour intervals
            '4h':  4 / 24,           # 4 hour intervals
            'D1':  1,                # Daily intervals
        }
        if timeframe not in time_frame_mapping:
            raise ValueError("Unsupported time frame")
        self.tf = time_frame_mapping[timeframe]

        params = self.estimate_parameters()
        self.mu_hat = params[0]  # Mean (μ)
        self.theta_hat = params[1]  # Drift (θ)
        self.sigma_hat = params[2]  # Volatility (σ)
        print(f'Estimated μ: {self.mu_hat}')
        print(f'Estimated θ: {self.theta_hat}')
        print(f'Estimated σ: {self.sigma_hat}')
        time.sleep(1)

    def ornstein_uhlenbeck(self, mu, theta, sigma, dt, X0, n):
        """
        Simulates the Ornstein-Uhlenbeck process.

        Args:
            mu (float): Estimated long-term mean.
            theta (float): Estimated drift.
            sigma (float): Estimated volatility.
            dt (float): Time step.
            X0 (float): Initial value.
            n (int): Number of time steps.

        Returns:
            np.ndarray : Simulated process.
        """
        x = np.zeros(n)
        x[0] = X0
        for t in range(1, n):
            dW = np.random.normal(loc=0, scale=np.sqrt(dt))
            # O-U process differential equation
            x[t] = x[t-1] + (theta * (mu - x[t-1]) * dt) + (sigma * dW)
            # dW is a Wiener process
            # (theta * (mu - x[t-1]) * dt) represents the mean-reverting tendency
            # (sigma * dW) represents the random volatility
        return x

    def estimate_parameters(self):
        """
        Estimates the mean-reverting parameters (μ, θ, σ) 
        using the negative log-likelihood.

        Returns:
            Tuple: Estimated μ, θ, and σ.
        """
        initial_guess = [0, 0.1, np.std(self.returns)]
        result = minimize(
            self._neg_log_likelihood, initial_guess, args=(self.returns,)
        )
        mu, theta, sigma = result.x
        return mu, theta, sigma

    def _neg_log_likelihood(self, params, returns):
        """
        Calculates the negative 
            log-likelihood for parameter estimation.

        Args:
            params (list): List of parameters [mu, theta, sigma].
            returns (np.ndarray): Historical returns.

        Returns:
            float: Negative log-likelihood.
        """
        mu, theta, sigma = params
        dt = self.tf
        n = len(returns)
        ou_simulated = self.ornstein_uhlenbeck(
            mu, theta, sigma, dt, 0, n + 1
        )
        residuals = ou_simulated[1:n + 1] - returns
        neg_ll = 0.5 * np.sum(
            residuals**2
        ) / sigma**2 + 0.5 * n * np.log(2 * np.pi * sigma**2)
        return neg_ll

    def simulate_process(self, rts=None, n=100, p=None):
        """
        Simulates the OU process multiple times .

        Args:
            rts (np.ndarray): Historical returns.
            n (int): Number of simulations to perform.
            p (int): Number of time steps.

        Returns:
            np.ndarray: 2D array representing simulated processes.
        """
        if rts is not None:
            returns = rts
        else:
            returns = self.returns
        if p is not None:
            T = p
        else:
            T = len(returns)
        dt = self.tf

        dW_matrix = np.random.normal(
            loc=0, scale=np.sqrt(dt), size=(n, T)
        )
        simulations_matrix = np.zeros((n, T))
        simulations_matrix[:, 0] = returns[-1]

        for t in range(1, T):
            simulations_matrix[:, t] = (
                simulations_matrix[:, t-1] +
                self.theta_hat * (
                    self.mu_hat - simulations_matrix[:, t-1]) * dt +
                self.sigma_hat * dW_matrix[:, t]
            )
        return simulations_matrix

    def calculate_signals(self, rts, p, n=10, th=1):
        """
        Calculate the SignalEvents based on the simulated processes.

        Args:
            rts (np.ndarray): Historical returns.
            p (int): Number of time steps.
            n (int): Number of simulations to perform.

        Returns
            np.ndarray: 2D array representing simulated processes.
        """
        simulations_matrix = self.simulate_process(rts=rts, n=n, p=p)
        last_values = simulations_matrix[:, -1]
        mean = last_values.mean()
        deviation = mean - self.mu_hat
        if deviation < -self.sigma_hat * th:
            signal = "LONG"
        elif deviation > self.sigma_hat * th:
            signal = "SHORT"
        else:
            signal = None
        return signal
