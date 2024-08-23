"""
The `tseries` module is a designed for conducting 
advanced time series analysis in financial markets. 
It leverages statistical models and algorithms to perform 
tasks such as cointegration testing, volatility modeling, 
and filter-based estimation to assist in trading strategy development, 
market analysis, and financial data exploration.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
from hurst import compute_Hc
from filterpy.kalman import KalmanFilter
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from itertools import combinations
from typing import Union, List, Tuple
import pprint
import warnings
warnings.filterwarnings("ignore")

# *******************************************
#          ARIMA AND GARCH MODELS          *
# *******************************************

__all__ = [
    "load_and_prepare_data",
    "fit_best_arima",
    "fit_garch",
    "predict_next_return",
    "get_prediction",
    "get_corr",
    "run_cadf_test",
    "run_hurst_test",
    "run_coint_test",
    "run_kalman_filter"
]

def load_and_prepare_data(df: pd.DataFrame):
    """
    Prepares financial time series data for analysis.

    This function takes a pandas DataFrame containing financial data,
    calculates logarithmic returns, and the first difference 
    of these logarithmic returns. It handles missing values 
    by filling them with zeros.

    Args:
        df (pd.DataFrame): DataFrame containing at least 
            a `Close` column with closing prices of a financial asset.

    Returns:
        pd.DataFrame: DataFrame with additional 
            columns for logarithmic returns (`log_return`) 
            and the first difference of logarithmic returns (`diff_log_return`), 
            with `NaN` values filled with `0`.
    """
    # Load data
    data = df.copy()
    # Calculate logarithmic returns
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    # Differencing if necessary
    data['diff_log_return'] = data['log_return'].diff()
    # Drop NaN values
    data.fillna(0, inplace=True)
    return data


def fit_best_arima(window_data: Union[pd.Series , np.ndarray]):
    """
    Identifies and fits the best `ARIMA` model 
    based on the Akaike Information Criterion `(AIC)`.

    Iterates through different combinations of `p` and `q` 
    parameters (within specified ranges) for the ARIMA model,
    fits them to the provided data, and selects the combination 
    with the lowest `AIC` value.

    Args:
        window_data (pd.Series or np.ndarray):
            Time series data to fit the `ARIMA` model on.

    Returns:
        ARIMA result object: The fitted `ARIMA` model with the lowest `AIC`.
    """
    model = pm.auto_arima(
        window_data,
        start_p=1,
        start_q=1,
        max_p=6,
        max_q=6,
        seasonal=False,
        stepwise=True
    )
    final_order = model.order
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    best_arima_model = ARIMA(
        window_data, order=final_order, missing='drop').fit()
    return best_arima_model


def fit_garch(window_data:  Union[pd.Series , np.ndarray]):
    """
    Fits an `ARIMA` model to the data to get residuals, 
    then fits a `GARCH(1,1)` model on these residuals.

    Utilizes the residuals from the best `ARIMA` model fit to 
    then model volatility using a `GARCH(1,1)` model.

    Args:
        window_data (pd.Series or np.ndarray): 
            Time series data for which to fit the `ARIMA` and `GARCH` models.

    Returns:
        tuple: A tuple containing the `ARIMA` result 
            object and the `GARCH` result object.
    """
    arima_result = fit_best_arima(window_data)
    resid = np.asarray(arima_result.resid)
    resid = resid[~(np.isnan(resid) | np.isinf(resid))]
    garch_model = arch_model(resid, p=1, q=1, rescale=False)
    garch_result = garch_model.fit(disp='off')
    return arima_result, garch_result


def predict_next_return(arima_result, garch_result):
    """
    Predicts the next return value using fitted `ARIMA` and `GARCH` models.

    Combines the next period forecast from the `ARIMA` model 
    with the next period volatility forecast from the `GARCH` model
    to predict the next return value.

    Args:
        arima_result (ARIMA result object): The fitted `ARIMA` model result.
        garch_result (ARCH result object): The fitted `GARCH` model result.

    Returns:
        float: The predicted next return, adjusted for predicted volatility.
    """
    # Predict next value with ARIMA
    arima_pred = arima_result.forecast(steps=1)
    # Predict next volatility with GARCH
    garch_pred = garch_result.forecast(horizon=1)
    next_volatility = garch_pred.variance.iloc[-1, 0]

    # Combine predictions (return + volatility)
    next_return = arima_pred.values[0] + next_volatility
    return next_return


def get_prediction(window_data:  Union[pd.Series , np.ndarray]):
    """
    Orchestrator function to get the next period's return prediction.

    This function ties together the process of fitting 
    both `ARIMA` and `GARCH` models on the provided data
    and then predicting the next period's return using these models.

    Args:
        window_data (Union[pd.Series , np.ndarray]): 
            Time series data to fit the models and predict the next return.

    Returns
        float: Predicted next return value.
    """
    arima_result, garch_result = fit_garch(window_data)
    prediction = predict_next_return(arima_result, garch_result)
    return prediction


# *********************************************
# STATS TEST (Cointegration , Mean Reverting)*
# *********************************************
def get_corr(tickers: Union[List[str] , Tuple[str, ...]], start: str, end: str) -> None:
    """
    Calculates and prints the correlation matrix of the adjusted closing prices 
    for a given list of stock tickers within a specified date range.

    Args:
        tickers (Union[List[str] , Tuple[str, ...]]): 
        A list or tuple of valid stock tickers (e.g., ['AAPL', 'MSFT', 'GOOG']).
        start (str): The start date for the historical data in 'YYYY-MM-DD' format.
        end (str): The end date for the historical data in 'YYYY-MM-DD' format.

    Example:
    >>> from bbstrader.tseries import get_corr
    >>> get_corr(['AAPL', 'MSFT', 'GOOG'], '2023-01-01', '2023-12-31')
    """
    # Download historical data
    data = yf.download(tickers, start=start, end=end)['Adj Close']

    # Calculate correlation matrix
    correlation_matrix = data.corr()

    # Display the matrix
    print(correlation_matrix)


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
    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title(f'{ts1} and {ts2} Daily Prices ')
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
    plt.xlabel(f'{ts1} Price ($)')
    plt.ylabel(f'{ts2} Price ($)')
    plt.title(f'{ts1} and {ts2} Price Scatterplot')
    plt.scatter(df[ts1], df[ts2])

    # Plot the regression line
    plt.plot(df[ts1], results.fittedvalues,
             linestyle='--', color='red', linewidth=2,
             label='Regression Line'
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
    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('Residual Plot')
    plt.legend()
    plt.show()


def run_cadf_test(pair: Union[List[str] , Tuple[str, ...]], start: str, end: str) -> None:
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
    _p0 = yf.download(p0, start=start, end=end)
    _p1 = yf.download(p1, start=start, end=end)
    df = pd.DataFrame(index=_p0.index)
    df[p0] = _p0["Adj Close"]
    df[p1] = _p1["Adj Close"]
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
    print(f'Result Parmas: \n')
    print(results.params)
    print("\nRegression Summary:")
    print(results.summary())

    # Calculate and output the CADF test on the residuals
    print("\nCointegration TEST Results:")
    cadf = ts.adfuller(df["res"], autolag='AIC')
    pprint.pprint(cadf)


def _hurst(ts):
    """
    Returns the Hurst Exponent of the time series vector ts,
    """
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0

# Function to calculate Hurst Exponent


def hurst(time_series):
    H, c, data_range = compute_Hc(time_series, kind='price', simplified=True)
    return H


def run_hurst_test(symbol: str, start: str, end: str):
    """
    Calculates and prints the Hurst Exponent for a given stock's adjusted closing prices
    within a specified date range, and for three generated series (Geometric Brownian Motion, 
    Mean-Reverting, and Trending).

    The Hurst Exponent is used to determine the long-term memory of a time series.

    Args:
        symbol (str): A valid stock ticker symbol (e.g., 'AAPL').
        start (str): The start date for the historical data in 'YYYY-MM-DD' format.
        end (str): The end date for the historical data in 'YYYY-MM-DD' format.

    Example:
    >>> from bbstrader.tseries import run_hurst_test

    >>> run_hurst_test('AAPL', '2023-01-01', '2023-12-31')
    """
    data = yf.download(symbol, start=start, end=end)

    # Create a Geometric Brownian Motion, Mean-Reverting, and Trending Series
    gbm = log(cumsum(randn(100000))+1000)
    mr = log(randn(100000)+1000)
    tr = log(cumsum(randn(100000)+1)+1000)

    # Output the Hurst Exponent for each of the series
    print(f"\nHurst(GBM):  {_hurst(gbm)}")
    print(f"Hurst(MR):   {_hurst(mr)}")
    print(f"Hurst(TR):   {_hurst(tr)}")
    print(f"Hurst({symbol}): {hurst(data['Adj Close'])}\n")


def test_cointegration(ticker1, ticker2, start, end):
    # Download historical data
    stock_data_pair = yf.download(
        [ticker1, ticker2], start=start, end=end
    )['Adj Close'].dropna()

    # Perform Johansen cointegration test
    result = coint_johansen(stock_data_pair, det_order=0, k_ar_diff=1)

    # Get the cointegration rank
    traces_stats = result.lr1
    print(f"\nTraces Stats: \n{traces_stats}")

    # Get the critical values for 95% confidence level
    critical_values = result.cvt
    print(f"\nCritical Values: \n{critical_values}")

    # Compare the cointegration rank with critical values
    if traces_stats[0] > critical_values[:, 1].all():
        print(f"\n{ticker1} and {ticker2} are cointegrated.\n")
    else:
        print(f"\nNo cointegration found for {ticker1} and {ticker2}.\n")


def run_coint_test(tickers: List[str], start: str, end: str) -> None:
    """
    Performs pairwise cointegration tests on a list of stock tickers over a specified date range.

    For each unique pair of tickers, the function downloads historical adjusted closing prices and
    tests for cointegration.

    Args:
        tickers (List[str]): A list of valid stock ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOG']).
        start (str): The start date for the historical data in 'YYYY-MM-DD' format.
        end (str): The end date for the historical data in 'YYYY-MM-DD' format.

    Example:
    >>> from bbstrader.tseries import run_coint_test

    >>> run_coint_test(['AAPL', 'MSFT', 'GOOG'], '2023-01-01', '2023-12-31')
    """
    # Loop through ticker combinations
    for ticker1, ticker2 in combinations(tickers, 2):
        test_cointegration(ticker1, ticker2, start, end)


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
    colour_map = plt.cm.get_cmap('YlOrRd')
    colours = np.linspace(0.1, 1, plen)

    scatterplot = plt.scatter(
        prices[etfs[0]], prices[etfs[1]],
        s=30, c=colours, cmap=colour_map,
        edgecolor='k', alpha=0.8
    )

    colourbar = plt.colorbar(scatterplot)
    colourbar.ax.set_yticklabels(
        [str(p.date()) for p in prices[::plen//9].index]
    )

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
    kf.P = np.ones((2, 2)) * 1000.  # Initial covariance,
    # large to represent high uncertainty
    kf.F = np.eye(2)  # State transition matrix
    kf.Q = trans_cov  # Process noise covariance
    kf.R = 1.  # Scalar measurement noise covariance

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
    pd.DataFrame({
        'slope': state_means[:, 0].flatten(),
        'intercept': state_means[:, 1].flatten()
    }, index=prices.index
    ).plot(subplots=True)
    plt.show()


def run_kalman_filter(etfs: Union[List[str] , Tuple[str, ...]], start: str, end: str) -> None:
    """
    Applies a Kalman filter to a pair of ETF adjusted closing prices within a specified date range
    to estimate the slope and intercept over time.

    The function downloads historical adjusted closing prices for the specified pair of ETFs,
    visualizes their price relationship, calculates the Kalman filter estimates for the slope and 
    intercept, and visualizes the changes in these estimates over time.

    Args:
        etfs (Union[List[str] , Tuple[str, ...]]):
        A list or tuple containing two valid ETF tickers (e.g., ['SPY', 'QQQ']).
        start (str): The start date for the historical data in 'YYYY-MM-DD' format.
        end (str): The end date for the historical data in 'YYYY-MM-DD' format.

    Example:
    >>> from bbstrader.tseries import run_kalman_filter

    >>> run_kalman_filter(['SPY', 'QQQ'], '2023-01-01', '2023-12-31')
    """
    etf_df1 = yf.download(etfs[0], start, end)
    etf_df2 = yf.download(etfs[1], start, end)

    prices = pd.DataFrame(index=etf_df1.index)
    prices[etfs[0]] = etf_df1["Adj Close"]
    prices[etfs[1]] = etf_df2["Adj Close"]

    draw_date_coloured_scatterplot(etfs, prices)
    state_means, state_covs = calc_slope_intercept_kalman(etfs, prices)
    draw_slope_intercept_changes(prices, state_means)
