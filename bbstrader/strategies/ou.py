import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize

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

        Parameters
        ==========
        :param prices (np.ndarray): Historical close prices.
        :param retrurns (bool) : Use it to indicate weither 
            you want  to simulate the returns or your raw data
        :param timeframe (str) : 
            The time  frame for the Historical prices
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
        self.mu_hat    = params[0] # Mean (μ) 
        self.theta_hat = params[1] # Drift (θ)
        self.sigma_hat = params[2] # Volatility (σ)
        print(f'Estimated μ: {self.mu_hat}')
        print(f'Estimated θ: {self.theta_hat}')
        print(f'Estimated σ: {self.sigma_hat}')
        time.sleep(1)

    def ornstein_uhlenbeck(self, mu, theta, sigma, dt, X0, n):
        """
        Simulates the Ornstein-Uhlenbeck process.

        Parameters
        ==========
        :param mu (float): Estimated long-term mean.
        :param theta (float): Estimated drift.
        :param sigma (float): Estimated volatility.
        :param dt (float): Time step.
        :param X0 (float): Initial value.
        :param n (int): Number of time steps.

        Returns
        :return np.ndarray: Simulated process.
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

        Returns
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

        Parameters
        ==========
        :param params (list): List of parameters [mu, theta, sigma].
        :param returns (np.ndarray): Historical returns.

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

        Parameters
        ==========
        :param rts (np.ndarray): Historical returns.
        :param n(int): Number of simulations to perform.
        :param p(int): Number of time steps.

        Returns
        :return np.ndarray: 2D array 
            representing simulated processes.
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

        Parameters
        ==========
        :param rts (np.ndarray): Historical returns.
        :param p(int): Number of time steps.
        :param n(int): Number of simulations to perform.

        Returns
        :return np.ndarray: 2D array 
            representing simulated processes.
        """
        simulations_matrix = self.simulate_process(rts=rts,n=n,p=p)
        last_values = simulations_matrix[:, -1]
        mean = last_values.mean()
        deviation =  mean - self.mu_hat
        if deviation < -self.sigma_hat * th:
            signal = "LONG"
        elif deviation > self.sigma_hat * th:
            signal = "SHORT"
        else:
            signal =  None
        return signal

