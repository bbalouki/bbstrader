"""
The `tseries` module is a designed for conducting
some simple time series analysis in financial markets.
"""

import pprint
import warnings
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import yfinance as yf
from filterpy.kalman import KalmanFilter
from pykalman import KalmanFilter as PyKalmanFilter
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from tqdm import tqdm


__all__ = [
    "run_kalman_filter",
    "KalmanFilterModel",
    "remove_correlated_assets",
    "check_stationarity",
    "remove_stationary_assets",
    "select_assets",
    "compute_pair_metrics",
    "find_cointegrated_pairs",
    "analyze_cointegrated_pairs",
    "select_candidate_pairs",
    "KFSmoother",
    "KFHedgeRatio",
]

# *******************************************
#          ARIMA AND GARCH MODELS          *
# *******************************************


def load_and_prepare_data(df):
    warnings.warn("`load_and_prepare_data` is removed.", DeprecationWarning)


def fit_best_arima(window_data):
    warnings.warn(
        "`fit_best_arima` is deprecated, use `pmdarima.auto_arima` instead.",
        DeprecationWarning,
    )


def fit_garch(window_data):
    warnings.warn(
        "`fit_garch` is deprecated, use `arch.arch_model` instead.",
        DeprecationWarning,
    )


def predict_next_return(arima_result, garch_result):
    warnings.warn(
        "`predict_next_return` is deprecated.",
        DeprecationWarning,
    )


def get_prediction(window_data):
    warnings.warn(
        "`get_prediction` is deprecated, ",
        DeprecationWarning,
    )


class ArimaGarchModel:
    def __init__(self, symbol, data, k: int = 252):
        warnings.warn(
            "`ArimaGarchModel` is deprecated, use `pmdarima.auto_arima` and `arch.arch_model` instead.",
            DeprecationWarning,
        )


# *********************************************
# STATS TEST (Cointegration , Mean Reverting)*
# *********************************************
def get_corr(tickers: Union[List[str], Tuple[str, ...]], start: str, end: str) -> None:
    warnings.warn(
        "`get_corr` is deprecated, use pandas DataFrame's `corr` method instead.",
        DeprecationWarning,
    )


def plot_price_series(df: pd.DataFrame, ts1: str, ts2: str):
    """
    Plot both time series on the same line graph for
    the specified date range.

    Args:
        df (pd.DataFrame):
            The DataFrame containing prices for each series
        ts1 (str): The first time series column name
        ts2 (str): The second time series column name
    """
    fig, ax = plt.subplots()
    ax.plot(df.index, df[ts1], label=ts1)
    ax.plot(df.index, df[ts2], label=ts2)

    fig.autofmt_xdate()
    plt.xlabel("Month/Year")
    plt.ylabel("Price ($)")
    plt.title(f"{ts1} and {ts2} Daily Prices ")
    plt.legend()
    plt.show()


def plot_scatter_series(df: pd.DataFrame, ts1: str, ts2: str):
    """
    Plot a scatter plot of both time series for
    via the provided DataFrame.

    Args:
        df (pd.DataFrame):
            The DataFrame containing prices for each series
        ts1 (str): The first time series column name
        ts2 (str): The second time series column name
    """
    plt.xlabel(f"{ts1} Price ($)")
    plt.ylabel(f"{ts2} Price ($)")
    plt.title(f"{ts1} and {ts2} Price Scatterplot")
    plt.scatter(df[ts1], df[ts2])

    # Plot the regression line
    plt.plot(
        df[ts1],
        results.fittedvalues,
        linestyle="--",
        color="red",
        linewidth=2,
        label="Regression Line",
    )
    plt.legend()
    plt.show()


def plot_residuals(df: pd.DataFrame):
    """
    Plot the residuals of OLS procedure for both
    time series.

    Args:
        df (pd.DataFrame):
            The DataFrame containing prices for each series
    """
    fig, ax = plt.subplots()
    ax.plot(df.index, df["res"], label="Residuals")

    fig.autofmt_xdate()
    plt.xlabel("Month/Year")
    plt.ylabel("Price ($)")
    plt.title("Residual Plot")
    plt.legend()
    plt.show()


def run_cadf_test(
    pair: Union[List[str], Tuple[str, ...]],
    start: str,
    end: str,
) -> None:
    """
    Performs the Cointegration Augmented Dickey-Fuller (CADF) test on a pair of stock tickers
    over a specified date range to check for cointegration.

    The function downloads historical adjusted closing prices for the specified pair of stock tickers,
    calculates the optimal hedge ratio (beta) using Ordinary Least Squares (OLS) regression, plots the
    time series and their residuals, and finally performs the CADF test on the residuals.

    Args:
        pair (List[str] or Tuple[str, ...]):
            A list or tuple containing two valid stock tickers (e.g., ['AAPL', 'MSFT']).
        start (str): The start date for the historical data in 'YYYY-MM-DD' format.
        end (str): The end date for the historical data in 'YYYY-MM-DD' format.

    Example:
        >>> from bbstrader.tseries import run_cadf_test
        >>> run_cadf_test(['AAPL', 'MSFT'], '2023-01-01', '2023-12-31')
        >>> Regression Metrics:
        >>> Optimal Hedge Ratio (Beta): 2.2485845594120333
        >>> Result Parmas:

        >>> const   -74.418034
        >>> AAPL      2.248585
        >>> dtype: float64

        >>> Regression Summary:
        >>>                              OLS Regression Results
        >>> ==============================================================================
        >>> Dep. Variable:                   MSFT   R-squared:                       0.900
        >>> Model:                            OLS   Adj. R-squared:                  0.900
        >>> Method:                 Least Squares   F-statistic:                     2244.
        >>> Date:                Sat, 20 Jul 2024   Prob (F-statistic):          2.95e-126
        >>> Time:                        13:36:58   Log-Likelihood:                -996.45
        >>> No. Observations:                 250   AIC:                             1997.
        >>> Df Residuals:                     248   BIC:                             2004.
        >>> Df Model:                           1
        >>> Covariance Type:            nonrobust
        >>> ==============================================================================
        >>>                  coef    std err          t      P>|t|      [0.025      0.975]
        >>> ------------------------------------------------------------------------------
        >>> const        -74.4180      8.191     -9.085      0.000     -90.551     -58.286
        >>> AAPL           2.2486      0.047     47.369      0.000       2.155       2.342
        >>> ==============================================================================
        >>> Omnibus:                        4.923   Durbin-Watson:                   0.121
        >>> Prob(Omnibus):                  0.085   Jarque-Bera (JB):                4.862
        >>> Skew:                           0.342   Prob(JB):                       0.0879
        >>> Kurtosis:                       2.993   Cond. No.                     1.71e+03
        >>> ==============================================================================

        >>> Notes:
        >>> [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
        >>> [2] The condition number is large, 1.71e+03. This might indicate that there are
        >>> strong multicollinearity or other numerical problems.

        >>> Cointegration TEST Results:
        >>> (np.float64(-3.204126144947765),
        >>> np.float64(0.019747080611767602),
        >>> 0,
        >>> 249,
        >>> {'1%': np.float64(-3.4568881317725864),
        >>> '10%': np.float64(-2.5729936189738876),
        >>> '5%': np.float64(-2.8732185133016057)},
        >>> np.float64(1364.3866758546171))
    """
    # Download historical data for required stocks
    p0, p1 = pair[0], pair[1]
    _p0 = yf.download(
        p0,
        start=start,
        end=end,
        progress=False,
        multi_level_index=False,
        auto_adjust=True,
    )
    _p1 = yf.download(
        p1,
        start=start,
        end=end,
        progress=False,
        multi_level_index=False,
        auto_adjust=True,
    )
    df = pd.DataFrame(index=_p0.index)
    df[p0] = _p0["Close"]
    df[p1] = _p1["Close"]
    df = df.dropna()

    # Calculate optimal hedge ratio "beta"
    # using statsmodels OLS
    X = sm.add_constant(df[p0])
    y = df[p1]
    model = sm.OLS(y, X)
    global results
    results = model.fit()
    beta_hr = results.params[p0]

    # Plot the two time series with regression line
    plot_price_series(df, p0, p1)

    # Display a scatter plot of the two time series
    # with regression line
    plot_scatter_series(df, p0, p1)

    # Calculate the residuals of the linear combination
    df["res"] = results.resid
    plot_residuals(df)

    # Display regression metrics
    print("\nRegression Metrics:")
    print(f"Optimal Hedge Ratio (Beta): {beta_hr}")
    print("Result Parmas: \n")
    print(results.params)
    print("\nRegression Summary:")
    print(results.summary())

    # Calculate and output the CADF test on the residuals
    print("\nCointegration TEST Results:")
    cadf = ts.adfuller(df["res"], autolag="AIC")
    pprint.pprint(cadf)


def run_hurst_test(symbol: str, start: str, end: str):
    warnings.warn(
        "`run_hurst_test` is deprecated, use `hurst.compute_Hc` instead.",
        DeprecationWarning,
    )


def test_cointegration(ticker1, ticker2, start, end):
    warnings.warn(
        "`test_cointegration` is deprecated, see statsmodels.tsa.stattools.coint instead.",
        DeprecationWarning,
    )


def run_coint_test(tickers: List[str], start: str, end: str) -> None:
    test_cointegration()


# *********************************
#          KALMAN FILTER         *
# *********************************
def draw_date_coloured_scatterplot(etfs, prices):
    """
    Create a scatterplot of the two ETF prices, which is
    coloured by the date of the price to indicate the
    changing relationship between the sets of prices
    """
    plen = len(prices)
    colour_map = plt.cm.get_cmap("YlOrRd")
    colours = np.linspace(0.1, 1, plen)

    scatterplot = plt.scatter(
        prices[etfs[0]],
        prices[etfs[1]],
        s=30,
        c=colours,
        cmap=colour_map,
        edgecolor="k",
        alpha=0.8,
    )

    colourbar = plt.colorbar(scatterplot)
    colourbar.ax.set_yticklabels([str(p.date()) for p in prices[:: plen // 9].index])

    plt.xlabel(prices.columns[0])
    plt.ylabel(prices.columns[1])
    plt.show()


def calc_slope_intercept_kalman(etfs, prices):
    """
    Utilize the Kalman Filter from the filterpy library
    to calculate the slope and intercept of the regressed
    ETF prices.
    """
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.zeros((2, 1))  # Initial state
    kf.P = np.ones((2, 2)) * 1000.0  # Initial covariance,
    # large to represent high uncertainty
    kf.F = np.eye(2)  # State transition matrix
    kf.Q = trans_cov  # Process noise covariance
    kf.R = 1.0  # Scalar measurement noise covariance

    state_means, state_covs = [], []
    for time, z in enumerate(prices[etfs[1]].values):
        # Dynamically update the observation matrix H
        # to include the current independent variable
        kf.H = np.array([[prices[etfs[0]][time], 1.0]])
        kf.predict()
        kf.update(z)
        state_means.append(kf.x.copy())
        state_covs.append(kf.P.copy())

    return np.array(state_means), np.array(state_covs)


def draw_slope_intercept_changes(prices, state_means):
    """
    Plot the slope and intercept of the regressed ETF prices
    between the two ETFs, with the changing values of the
    Kalman Filter over time.
    """
    print(f"First Slops : {state_means[0, 0]}")
    print(f"First intercept : {state_means[0, 1]}")
    pd.DataFrame(
        {
            "slope": state_means[:, 0].flatten(),
            "intercept": state_means[:, 1].flatten(),
        },
        index=prices.index,
    ).plot(subplots=True)
    plt.show()


def run_kalman_filter(
    etfs: Union[List[str], Tuple[str, ...]], start: str, end: str
) -> None:
    """
    Applies a Kalman filter to a pair of assets adjusted closing prices within a specified date range
    to estimate the slope and intercept over time.

    The function downloads historical adjusted closing prices for the specified pair of assets,
    visualizes their price relationship, calculates the Kalman filter estimates for the slope and
    intercept, and visualizes the changes in these estimates over time.

    Args:
        etfs (Union[List[str] , Tuple[str, ...]]):
        A list or tuple containing two valid assets tickers (e.g., ['SPY', 'QQQ']).
        start (str): The start date for the historical data in 'YYYY-MM-DD' format.
        end (str): The end date for the historical data in 'YYYY-MM-DD' format.

    Example:
    >>> from bbstrader.tseries import run_kalman_filter

    >>> run_kalman_filter(['SPY', 'QQQ'], '2023-01-01', '2023-12-31')
    """
    etf_df1 = yf.download(
        etfs[0], start, end, progress=False, multi_level_index=False, auto_adjust=True
    )
    etf_df2 = yf.download(
        etfs[1], start, end, progress=False, multi_level_index=False, auto_adjust=True
    )

    prices = pd.DataFrame(index=etf_df1.index)
    prices[etfs[0]] = etf_df1["Close"]
    prices[etfs[1]] = etf_df2["Close"]

    draw_date_coloured_scatterplot(etfs, prices)
    state_means, state_covs = calc_slope_intercept_kalman(etfs, prices)
    draw_slope_intercept_changes(prices, state_means)


class KalmanFilterModel:
    """
    Implements a Kalman Filter model a recursive algorithm used for estimating
    the state of a linear dynamic system from a series of noisy measurements.
    It's designed to process market data, estimate dynamic parameters such as
    the slope and intercept of price relationships,
    forecast error and standard deviation of the predictions

    You can learn more here https://en.wikipedia.org/wiki/Kalman_filter
    """

    def __init__(self, tickers: List | Tuple, **kwargs):
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

        self.R = None
        self.theta = np.zeros(2)
        self.P = np.zeros((2, 2))
        self.delta = kwargs.get("delta", 1e-4)
        self.vt = kwargs.get("vt", 1e-3)
        self.wt = self.delta / (1 - self.delta) * np.eye(2)
        self.latest_prices = np.array([-1.0, -1.0])
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
        kf.R = 1.0  # Scalar measurement noise covariance

        return kf

    Array = np.ndarray

    def calc_slope_intercep(self, prices: Array) -> Tuple:
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
        self.kf.H = np.array([[prices[1], 1.0]])
        self.kf.predict()
        self.kf.update(prices[1])
        slope = self.kf.x.copy().flatten()[0]
        intercept = self.kf.x.copy().flatten()[1]

        return slope, intercept

    def calculate_etqt(self, prices: Array) -> Tuple:
        """
        Calculates the ``forecast error`` and ``standard deviation`` of the predictions
        using the Kalman Filter.

        Args:
            prices : A numpy array of prices for two financial instruments.

        Returns:
            A tuple containing the ``forecast error`` and ``standard deviation`` of the predictions.
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
            return (et[0], sqrt_Qt.flatten()[0])
        else:
            return None


class OrnsteinUhlenbeck:
    def __init__(self, prices: np.ndarray, returns: bool = True, timeframe: str = "D1"):
        warnings.warn(
            "`OrnsteinUhlenbeck` is deprecated, use `statsmodels.tsa` instead.",
            DeprecationWarning,
        )


def remove_correlated_assets(df: pd.DataFrame, cutoff=0.99):
    """
    Removes highly correlated assets from a DataFrame based on a specified correlation cutoff threshold.
    This is useful in financial data analysis to reduce redundancy and multicollinearity in portfolios or datasets.

    Args:
        df (pd.DataFrame): A DataFrame where each column represents an asset
            and rows represent observations (e.g., time-series data).
        cutoff (float, optional, default=0.99): The correlation threshold.
            Columns with absolute correlation greater than this value will be considered for removal.

    Returns:
        pd.DataFrame: A DataFrame with less correlated assets.
            The columns that are highly correlated (above the cutoff) are removed.

    References
    ----------
    Stefan Jansen (2020). Machine Learning for Algorithmic Trading - Second Edition.
    chapter 9, Time-Series Models for Volatility Forecasts and Statistical Arbitrage.

    Example:
    >>> df = pd.DataFrame({
    ...     'AAPL': [100, 101, 102, 103, 104],
    ...     'MSFT': [200, 201, 202, 203, 204],
    ...     'GOOG': [300, 301, 302, 303, 304]
    ... })
    >>> df =  remove_correlated_assets(df)
    """
    corr = df.corr().stack()
    corr = corr[corr < 1]
    to_check = corr[corr.abs() > cutoff].index
    keep, drop = set(), set()
    for s1, s2 in to_check:
        if s1 not in keep:
            if s2 not in keep:
                keep.add(s1)
                drop.add(s2)
            else:
                drop.add(s1)
        else:
            keep.discard(s2)
            drop.add(s2)
    return df.drop(drop, axis=1)


def check_stationarity(df: pd.DataFrame):
    """
    Tests the stationarity of time-series data for each asset in the DataFrame
    using the Augmented Dickey-Fuller (ADF) test. Stationarity is a key property
    in time-series analysis, and non-stationary data can affect model performance.

    Args:
        df (pd.DataFrame): A DataFrame where each column represents a time series of an asset.

    Returns:
        pd.DataFrame: A DataFrame containing the ADF p-values for each asset,
        - ticker Asset name (column name from df).
        - adf p-value from the ADF test, indicating the probability of the null hypothesis (data is non-stationary).

    References
    ----------
    Stefan Jansen (2020). Machine Learning for Algorithmic Trading - Second Edition.
    chapter 9, Time-Series Models for Volatility Forecasts and Statistical Arbitrage.

    Example:
    >>> df = pd.DataFrame({
    ...     'AAPL': [100, 101, 102, 103, 104],
    ...     'MSFT': [200, 201, 202, 203, 204],
    ...     'GOOG': [300, 301, 302, 303, 304]
    ... })
    >>> df = check_stationarity(df)
    """
    results = []
    for ticker, prices in df.items():
        results.append([ticker, adfuller(prices, regression="ct")[1]])
    return pd.DataFrame(results, columns=["ticker", "adf"]).sort_values("adf")


def remove_stationary_assets(df: pd.DataFrame, pval=0.05):
    """
    Filters out stationary assets from the DataFrame based on the p-value obtained
    from the Augmented Dickey-Fuller test.
    Useful for focusing only on non-stationary time-series data.

    Args:
        df (pd.DataFrame): A DataFrame where each column represents a time series of an asset.
        pval (float, optional, default=0.05): The significance level to determine stationarity.
            Columns with an ADF test p-value below this threshold are considered stationary and removed.

    Returns:
        pd.DataFrame: A DataFrame containing only the non-stationary assets.

    References
    ----------
    Stefan Jansen (2020). Machine Learning for Algorithmic Trading - Second Edition.
    chapter 9, Time-Series Models for Volatility Forecasts and Statistical Arbitrage.

    Example:
    >>> df = pd.DataFrame({
    ...     'AAPL': [100, 101, 102, 103, 104],
    ...     'MSFT': [200, 201, 202, 203, 204],
    ...     'GOOG': [300, 301, 302, 303, 304]
    ... })
    >>> df = remove_stationary_assets(df)
    """
    test_result = check_stationarity(df)
    stationary = test_result.loc[test_result.adf <= pval, "ticker"].tolist()
    return df.drop(stationary, axis=1).sort_index()


def select_assets(df: pd.DataFrame, n=100, start=None, end=None, rolling_window=None):
    """
    Selects the top N assets based on the average trading volume from the input DataFrame.
    These assets are used as universe in which we can search cointegrated pairs for pairs trading strategies.

    Args:
        df (pd.DataFrame): A multi-index DataFrame with levels ['ticker', 'date'] containing market data.
            Must include columns 'close' (price) and 'volume'.
        n (int, optional): Number of assets to select based on highest average trading volume. Default is 100.
        start (str, optional): Start date for filtering the data. Default is the earliest date in the DataFrame.
        end (str, optional): End date for filtering the data. Default is the latest date in the DataFrame.
        rolling_window (int, optional): Rolling window for calculating the average trading volume. Default is None.

    Returns:
        pd.DataFrame: A DataFrame of selected assets with filtered, cleaned data, indexed by date.

    References
    ----------
    Stefan Jansen (2020). Machine Learning for Algorithmic Trading - Second Edition.
    chapter 9, Time-Series Models for Volatility Forecasts and Statistical Arbitrage.
    """
    required_columns = {"close", "volume"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Input DataFrame must contain {required_columns}, but got {df.columns.tolist()}."
        )

    if (
        not isinstance(df.index, pd.MultiIndex)
        or "ticker" not in df.index.names
        or "date" not in df.index.names
    ):
        raise ValueError("Index must be a MultiIndex with levels ['ticker', 'date'].")

    df = df.copy()
    idx = pd.IndexSlice
    start = start or df.index.get_level_values("date").min()
    end = end or df.index.get_level_values("date").max()
    df = (
        df.loc[lambda df: ~df.index.duplicated()]
        .sort_index()
        .loc[idx[:, f"{start}" : f"{end}"], :]
        .assign(dv=lambda df: df.close.mul(df.volume))
    )

    if rolling_window is None:
        most_traded = df.groupby(level="ticker").dv.mean().nlargest(n=n).index
    else:
        # Calculate the rolling average of dollar volume
        df["dv_rolling_avg"] = (
            df.groupby(level=0)
            .dv.rolling(window=rolling_window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        most_traded = df.groupby(level=0)["dv_rolling_avg"].mean().nlargest(n=n).index
    df = (
        df.loc[idx[most_traded, :], "close"]
        .unstack("ticker")
        .ffill(limit=5)
        .dropna(axis=1)
    )
    df = remove_correlated_assets(df)
    df = remove_stationary_assets(df)
    return df.sort_index()


def compute_pair_metrics(security: pd.Series, candidates: pd.DataFrame):
    """
    Calculates statistical and econometric metrics for a target security and a set of candidate securities.
    These metrics are useful in financial modeling and pairs trading strategies,
    providing information about drift, volatility, correlation, and cointegration.

    Args:
        security (pd.Series): A time-series of the target security's prices.
            The name of the Series should correspond to the security's identifier (e.g., ticker symbol).
        candidates (pd.DataFrame): A DataFrame where each column represents a time-series of prices
            for candidate securities to be evaluated against the target security.

    Returns:
        pd.DataFrame: A DataFrame combining:
            Drift: Estimated drift of spreads between the target security and each candidate.
            Volatility: Standard deviation of spreads.
            Correlation:
                ``corr``: Correlation of normalized prices between the target and each candidate.
                ``corr_ret``: Correlation of returns (percentage change) between the target and each candidate.
            Cointegration metrics:
                Engle-Granger test statistics (``t1``, ``t2``) and p-values (``p1``, ``p2``).
                Johansen test trace statistics (``trace0``, ``trace1``) and selected lag order (``k_ar_diff``).

    References
    ----------
    Stefan Jansen (2020). Machine Learning for Algorithmic Trading - Second Edition.
    chapter 9, Time-Series Models for Volatility Forecasts and Statistical Arbitrage.
    """
    security = security.div(security.iloc[0])
    ticker = security.name
    candidates = candidates.div(candidates.iloc[0])
    spreads = candidates.sub(security, axis=0)
    n, m = spreads.shape
    X = np.ones(shape=(n, 2))
    X[:, 1] = np.arange(1, n + 1)

    # compute drift
    drift = (np.linalg.inv(X.T @ X) @ X.T @ spreads).iloc[1].to_frame("drift")

    # compute volatility
    vol = spreads.std().to_frame("vol")

    # returns correlation
    corr_ret = (
        candidates.pct_change().corrwith(security.pct_change()).to_frame("corr_ret")
    )

    # normalized price series correlation
    corr = candidates.corrwith(security).to_frame("corr")
    metrics = drift.join(vol).join(corr).join(corr_ret).assign(n=n)

    tests = []
    # run cointegration tests
    for candidate, prices in tqdm(candidates.items()):
        df = pd.DataFrame({"s1": security, "s2": prices})
        var = VAR(df.values)
        lags = var.select_order()  # select VAR order
        k_ar_diff = lags.selected_orders["aic"]
        # Johansen Test with constant Term and estd. lag order
        cj0 = coint_johansen(df, det_order=0, k_ar_diff=k_ar_diff)
        # Engle-Granger Tests
        t1, p1 = coint(security, prices, trend="c")[:2]
        t2, p2 = coint(prices, security, trend="c")[:2]
        tests.append([ticker, candidate, t1, p1, t2, p2, k_ar_diff, *cj0.lr1])
    columns = ["s1", "s2", "t1", "p1", "t2", "p2", "k_ar_diff", "trace0", "trace1"]
    tests = pd.DataFrame(tests, columns=columns).set_index("s2")
    return metrics.join(tests)


__CRITICAL_VALUES = {
    0: {0.9: 13.4294, 0.95: 15.4943, 0.99: 19.9349},
    1: {0.9: 2.7055, 0.95: 3.8415, 0.99: 6.6349},
}


def find_cointegrated_pairs(
    securities: pd.DataFrame,
    candidates: pd.DataFrame,
    n=None,
    start=None,
    stop=None,
    coint=False,
):
    """
    Identifies cointegrated pairs between a target set of securities and candidate securities
    based on econometric tests. The function evaluates statistical relationships,
    such as cointegration and Engle-Granger significance, to determine pairs suitable
    for financial strategies like pairs trading.

    Args:
        securities (`pd.DataFrame`): A DataFrame where each column represents the time-series
            prices of target securities to evaluate.
        candidates (`pd.DataFrame`): A DataFrame where each column represents the time-series
            prices of candidate securities to compare against the target securities.
        n (`int`, optional): The number of top pairs to return. If `None`, returns all pairs.
        start (`str`, optional): Start date for slicing the data (e.g., 'YYYY-MM-DD').
        stop (`str`, optional): End date for slicing the data (e.g., 'YYYY-MM-DD').
        coint (`bool`, optional, default=False):
            - If `True`, filters for pairs identified as cointegrated.
            - If `False`, returns all evaluated pairs.

    Returns:
        - ``pd.DataFrame``: A DataFrame containing:
        - Johansen and Engle-Granger cointegration metrics:
            - `t1`, `t2`: Engle-Granger test statistics for two directions.
            - `p1`, `p2`: Engle-Granger p-values for two directions.
            - `trace0`, `trace1`: Johansen test trace statistics for 0 and 1 cointegration relationships.
        - Indicators and filters:
            - `joh_sig`: Indicates Johansen cointegration significance.
            - `eg_sig`: Indicates Engle-Granger significance (p-value < 0.05).
            - `s1_dep`: Indicates whether the first series depends on the second (based on p-values).
            - `coint`: Combined cointegration indicator (Johansen & Engle-Granger).
        - Spread and ranking:
            - `t`: Minimum of `t1` and `t2`.
            - `p`: Minimum of `p1` and `p2`.
    References
    ----------
    Stefan Jansen (2020). Machine Learning for Algorithmic Trading - Second Edition.
    chapter 9, Time-Series Models for Volatility Forecasts and Statistical Arbitrage.

    Example:
    >>>    import pandas as pd

    >>>    # Sample Data
    >>>    data_securities = {
    ...        'Security1': [100, 102, 101, 103, 105],
    ...        'Security2': [50, 52, 53, 51, 54]
    ...    }
    >>>    data_candidates = {
    ...        'Candidate1': [100, 101, 99, 102, 104],
    ...        'Candidate2': [200, 202, 201, 203, 205]
    ...    }

    >>>    securities = pd.DataFrame(data_securities, index=pd.date_range('2023-01-01', periods=5))
    >>>    candidates = pd.DataFrame(data_candidates, index=pd.date_range('2023-01-01', periods=5))

    >>>    # Find cointegrated pairs
    >>>    top_pairs = find_cointegrated_pairs(securities, candidates, n=2, coint=True)
    >>>    print(top_pairs)

    >>>    | s1       | s2        | t    | p     | joh_sig | eg_sig | coint |
    >>>    |----------|-----------|------|-------|---------|--------|-------|
    >>>    | Security1| Candidate1| -3.5 | 0.01  | 1       | 1      | 1     |
    >>>    | Security2| Candidate2| -2.9 | 0.04  | 1       | 1      | 1     |
    """
    trace0_cv = __CRITICAL_VALUES[0][
        0.95
    ]  # critical value for 0 cointegration relationships
    # critical value for 1 cointegration relationship
    trace1_cv = __CRITICAL_VALUES[1][0.95]
    spreads = []
    if start is not None and stop is not None:
        securities = securities.loc[str(start) : str(stop), :]
        candidates = candidates.loc[str(start) : str(stop), :]
    for i, (ticker, prices) in enumerate(securities.items(), 1):
        try:
            df = compute_pair_metrics(prices, candidates)
            spreads.append(df.set_index("s1", append=True))
        except np.linalg.LinAlgError:
            continue
    spreads = pd.concat(spreads)
    spreads.index.names = ["s2", "s1"]
    spreads = spreads.swaplevel()
    spreads["t"] = spreads[["t1", "t2"]].min(axis=1)
    spreads["p"] = spreads[["p1", "p2"]].min(axis=1)
    spreads["joh_sig"] = (
        (spreads.trace0 > trace0_cv) & (spreads.trace1 > trace1_cv)
    ).astype(int)
    spreads["eg_sig"] = (spreads.p < 0.05).astype(int)
    spreads["s1_dep"] = spreads.p1 < spreads.p2
    spreads["coint"] = (spreads.joh_sig & spreads.eg_sig).astype(int)
    # select top n pairs
    if coint:
        if n is not None:
            top_pairs = (
                spreads.query("coint == 1").sort_values("t", ascending=False).head(n)
            )
        else:
            top_pairs = spreads.query("coint == 1").sort_values("t", ascending=False)
    else:
        if n is not None:
            top_pairs = spreads.sort_values("t", ascending=False).head(n)
        else:
            top_pairs = spreads.sort_values("t", ascending=False)
    return top_pairs


def analyze_cointegrated_pairs(
    spreads: pd.DataFrame,
    plot_coint=True,
    crosstab=False,
    heuristics=False,
    log_reg=False,
    decis_tree=False,
):
    """
    Analyzes cointegrated pairs by visualizing, summarizing, and applying predictive models.

    Args:
        spreads (pd.DataFrame):
            A DataFrame containing cointegration metrics and characteristics.
            Required columns: 'coint', 't', 'trace0', 'trace1', 'drift', 'vol', 'corr', 'corr_ret', 'eg_sig', 'joh_sig'.
        plot_coint (bool, optional):
            If True, generates scatterplots and boxplots to visualize cointegration characteristics.
        cosstab (bool, optional):
            If True, displays crosstabulations of Engle-Granger and Johansen test significance.
        heuristics (bool, optional):
            If True, prints descriptive statistics for drift, volatility, and correlation grouped by cointegration status.
        log_reg (bool, optional):
            If True, fits a logistic regression model to predict cointegration and evaluates its performance.
        decis_tree (bool, optional):
            If True, fits a decision tree model to predict cointegration and evaluates its performance.

    References
    ----------
    Stefan Jansen (2020). Machine Learning for Algorithmic Trading - Second Edition.
    chapter 9, Time-Series Models for Volatility Forecasts and Statistical Arbitrage.

    Example:
    >>>   import pandas as pd
    >>>   from bbstrader.tseries import find_cointegrated_pairs, analyze_cointegrated_pairs

    >>>    # Sample Data
    >>>    securities = pd.DataFrame({
    ...        'SPY': [100, 102, 101, 103, 105],
    ...        'QQQ': [50, 52, 53, 51, 54]
    ...    })
    >>>    candidates = pd.DataFrame({
    ...        'AAPL': [100, 101, 99, 102, 104],
    ...        'MSFT': [200, 202, 201, 203, 205]
    ...    })

    >>>    pairs = find_cointegrated_pairs(securities, candidates, n=2, coint=True)
    >>>    analyze_cointegrated_pairs(pairs, plot_coint=True, cosstab=True, heuristics=True, log_reg=True, decis_tree=True
    """
    if plot_coint:
        trace0_cv = __CRITICAL_VALUES[0][0.95]
        spreads = spreads.reset_index()
        sns.scatterplot(
            x=np.log1p(spreads.t.abs()),
            y=np.log1p(spreads.trace1),
            hue="coint",
            data=spreads[spreads.trace0 > trace0_cv],
        )
        fig, axes = plt.subplots(ncols=4, figsize=(20, 5))
        for i, heuristic in enumerate(["drift", "vol", "corr", "corr_ret"]):
            sns.boxplot(x="coint", y=heuristic, data=spreads, ax=axes[i])
        fig.tight_layout()

    if heuristics:
        spreads = spreads.reset_index()
        h = (
            spreads.groupby(spreads.coint)[["drift", "vol", "corr"]]
            .describe()
            .stack(level=0)
            .swaplevel()
            .sort_index()
        )
        print(h)

    if log_reg:
        y = spreads.coint
        X = spreads[["drift", "vol", "corr", "corr_ret"]]
        log_reg = LogisticRegressionCV(
            Cs=np.logspace(-10, 10, 21), class_weight="balanced", scoring="roc_auc"
        )
        log_reg.fit(X=X, y=y)
        Cs = log_reg.Cs_
        scores = pd.DataFrame(log_reg.scores_[True], columns=Cs).mean()
        scores.plot(logx=True)
        res = f"C:{np.log10(scores.idxmax()):.2f}, AUC: {scores.max():.2%}"
        print(res)
        print(log_reg.coef_)

    if decis_tree:
        model = DecisionTreeClassifier(class_weight="balanced")
        decision_tree = GridSearchCV(
            model, param_grid={"max_depth": list(range(1, 10))}, cv=5, scoring="roc_auc"
        )
        y = spreads.coint
        X = spreads[["drift", "vol", "corr", "corr_ret"]]
        decision_tree.fit(X, y)
        res = f"{decision_tree.best_score_:.2%}, Depth: {decision_tree.best_params_['max_depth']}"
        print(res)

    if crosstab:
        pd.set_option("display.float_format", lambda x: f"{x:.2%}")
        print(pd.crosstab(spreads.eg_sig, spreads.joh_sig))
        print(pd.crosstab(spreads.eg_sig, spreads.joh_sig, normalize=True))


def select_candidate_pairs(pairs: pd.DataFrame, period=False):
    """
    Select candidate pairs from a DataFrame based on cointegration status.

    This function filters the input DataFrame to select pairs where the 'coint' column equals 1,
    indicating cointegration. It then determines the dependent and independent series for each pair
    and returns the selected pairs in a dictionary format.

    Args:
        pairs (pd.DataFrame): A DataFrame containing pairs of time series with columns 'coint', 's1', 's2', and 's1_dep'.
        period (bool, optional): If True, includes the 'period' column in the output. Defaults to False.

    Returns:
        list[dict]: A list of dictionaries, each containing the keys 'x' and 'y' (and optionally 'period') representing the selected pairs.

    References
    ----------
    Stefan Jansen (2020). Machine Learning for Algorithmic Trading - Second Edition.
    chapter 9, Time-Series Models for Volatility Forecasts and Statistical Arbitrage.
    """
    candidates = pairs.query("coint == 1").copy()
    candidates = candidates.reset_index()
    candidates["y"] = candidates.apply(
        lambda x: x["s1"] if x.s1_dep else x["s2"], axis=1
    )
    candidates["x"] = candidates.apply(
        lambda x: x["s2"] if x.s1_dep else x["s1"], axis=1
    )
    if period:
        return candidates[["x", "y", "period"]].to_dict(orient="records")
    return candidates[["x", "y"]].to_dict(orient="records")


def KFSmoother(prices: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """
    Estimate rolling mean using Kalman Smoothing.

    Args:
        prices : pd.Series or np.ndarray
            The input time series data to be smoothed. It must be either a pandas Series or a numpy array.

    Returns:
        pd.Series or np.ndarray
            The smoothed time series data. If the input is a pandas Series, the output will also be a pandas Series with the same index.
            If the input is a numpy array, the output will be a numpy array.

    References
    ----------
    Stefan Jansen (2020). Machine Learning for Algorithmic Trading - Second Edition.
    chapter 9, Time-Series Models for Volatility Forecasts and Statistical Arbitrage.

    Examples
    --------
    >>> import yfinance as yf
    >>> prices = yf.download('AAPL', start='2020-01-01', end='2021-01-01', multi_level_index=False)['Adj Close']
    >>> prices = KFSmoother(prices)
    >>> print(prices[:5])
    Date
    2020-01-02 00:00:00+00:00   36.39801407
    2020-01-03 00:00:00+00:00   49.06231000
    2020-01-06 00:00:00+00:00   55.86334436
    2020-01-07 00:00:00+00:00   60.02240894
    2020-01-08 00:00:00+00:00   63.15057948
    dtype: float64

    """
    if not isinstance(prices, (np.ndarray, pd.Series)):
        raise ValueError("Input must be either a numpy array or a pandas Series.")
    kf = PyKalmanFilter(
        transition_matrices=np.eye(1),
        observation_matrices=np.eye(1),
        initial_state_mean=0,
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.05,
    )
    if isinstance(prices, pd.Series):
        state_means, _ = kf.filter(prices.values)
        return pd.Series(state_means.flatten(), index=prices.index)
    elif isinstance(prices, np.ndarray):
        state_means, _ = kf.filter(prices)
        return state_means.flatten()


def KFHedgeRatio(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> np.ndarray:
    """
    Estimate Hedge Ratio using Kalman Filter.
    Args:
        x : pd.Series or np.ndarray
            The independent variable, which can be either a pandas Series or a numpy array.
        y : pd.Series or np.ndarray
            The dependent variable, which can be either a pandas Series or a numpy array.

    Returns:
        np.ndarray
            The estimated hedge ratio as a numpy array.

    The function returns the negative of the first state variable of each Kalman Filter estimate,
    which represents the estimated hedge ratio.

    References
    ----------
    Stefan Jansen (2020). Machine Learning for Algorithmic Trading - Second Edition.
    chapter 9, Time-Series Models for Volatility Forecasts and Statistical Arbitrage.
    """
    if not isinstance(x, (np.ndarray, pd.Series)) or not isinstance(
        y, (np.ndarray, pd.Series)
    ):
        raise ValueError(
            "Both x and y must be either a numpy array or a pandas Series."
        )

    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)

    kf = PyKalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=[0, 0],
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=2,
        transition_covariance=trans_cov,
    )
    y = y.values if isinstance(y, pd.Series) else y
    state_means, _ = kf.filter(y)
    # Indexing with [:, 0] in state_means[:, 0] extracts only the first state variable of
    # each Kalman Filter estimate, which is the estimated hedge ratio.
    return -state_means[:, 0]
