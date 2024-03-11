import numpy as np
from filterpy.kalman import KalmanFilter


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

        Parameters
        ==========
        :param tickers: A list or tuple of ticker symbols 
            representing financial instruments.
        **kwargs: Keyword arguments for additional parameters,
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

        Parameters
        ==========
        :param prices: A numpy array of prices for two financial instruments.

        Returns
        =======
        :returns: A tuple containing the slope and intercept of the relationship
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

        Parameters
        ==========
        :param et: The forecast error.
        :param std: The standard deviation of the predictions.

        Returns
        =======
        :returns: A tuple containing the trading signals for 
            the two financial instruments.
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

        Parameters
        ==========
        :param prices: A numpy array of prices for two financial instruments.

        Returns
        =======
        :returns: A dictionary containing trading signals 
            for the two financial instruments.
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
