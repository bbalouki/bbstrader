"""
The `tseries` module is a designed for conducting
advanced time series analysis in financial markets.
It leverages statistical models and algorithms to perform
tasks such as cointegration testing, volatility modeling,
and filter-based estimation to assist in trading strategy development,
market analysis, and financial data exploration.
"""

import pprint
import warnings
from itertools import combinations
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import yfinance as yf
from arch import arch_model
from filterpy.kalman import KalmanFilter
from hurst import compute_Hc
from pykalman import KalmanFilter as PyKalmanFilter
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from tqdm import tqdm

warnings.filterwarnings("ignore")


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
    "run_kalman_filter",
    "ArimaGarchModel",
    "KalmanFilterModel",
    "OrnsteinUhlenbeck",
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
    data["log_return"] = np.log(data["Close"] / data["Close"].shift(1))
    # Differencing if necessary
    data["diff_log_return"] = data["log_return"].diff()
    # Drop NaN values
    data.fillna(0, inplace=True)
    return data


def fit_best_arima(window_data: Union[pd.Series, np.ndarray]):
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
    if isinstance(window_data, pd.Series):
        window_data = window_data.values

    window_data = window_data[~(np.isnan(window_data) | np.isinf(window_data))]
    # Fit ARIMA model with best parameters
    model = pm.auto_arima(
        window_data,
        start_p=1,
        start_q=1,
        max_p=6,
        max_q=6,
        seasonal=False,
        stepwise=True,
    )
    final_order = model.order
    from arch.utility.exceptions import ConvergenceWarning as ArchWarning
    from statsmodels.tools.sm_exceptions import ConvergenceWarning as StatsWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=StatsWarning, module="statsmodels")
        warnings.filterwarnings("ignore", category=ArchWarning, module="arch")
        try:
            best_arima_model = ARIMA(
                window_data + 1e-5, order=final_order, missing="drop"
            ).fit()
            return best_arima_model
        except np.linalg.LinAlgError:
            # Catch specific linear algebra errors
            print("LinAlgError occurred, skipping this data point.")
            return None
        except Exception as e:
            # Catch any other unexpected errors and log them
            print(f"An error occurred: {e}")
            return None


def fit_garch(window_data: Union[pd.Series, np.ndarray]):
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
    if arima_result is None:
        return None, None
    resid = np.asarray(arima_result.resid)
    resid = resid[~(np.isnan(resid) | np.isinf(resid))]
    garch_model = arch_model(resid, p=1, q=1, rescale=False)
    garch_result = garch_model.fit(disp="off")
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
    if arima_result is None or garch_result is None:
        return 0
    # Predict next value with ARIMA
    arima_pred = arima_result.forecast(steps=1)
    # Predict next volatility with GARCH
    garch_pred = garch_result.forecast(horizon=1)
    next_volatility = garch_pred.variance.iloc[-1, 0]

    # Combine predictions (return + volatility)
    if not isinstance(arima_pred, np.ndarray):
        pred = arima_pred.values[0]
    else:
        pred = arima_pred[0]
    return pred + next_volatility


def get_prediction(window_data: Union[pd.Series, np.ndarray]):
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


class ArimaGarchModel:
    """
    This class implements a time serie model
    that combines `ARIMA (AutoRegressive Integrated Moving Average)`
    and `GARCH (Generalized Autoregressive Conditional Heteroskedasticity)` models
    to predict future returns based on historical price data.

    The model is implemented in the following steps:
    1. Data Preparation: Load and prepare the historical price data.
    2. Modeling: Fit the ARIMA model to the data and then fit the GARCH model to the residuals.
    3. Prediction: Predict the next return using the ARIMA model and the next volatility using the GARCH model.
    4. Trading Strategy: Execute the trading strategy based on the predictions.
    5. Vectorized Backtesting: Backtest the trading strategy using the historical data.

    Exemple:
        >>> import yfinance as yf
        >>> from bbstrader.tseries import ArimaGarchModel
        >>> from bbstrader.tseries import load_and_prepare_data

        >>> if __name__ == '__main__':
        >>>     # ARCH SPY Vectorize Backtest
        >>>     k = 252
        >>>     data = yf.download("SPY", start="2010-01-02", end="2015-12-31")
        >>>     arch = ArimaGarchModel("SPY", data, k=k)
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
        garch_result = garch_model.fit(disp="off")
        residuals = garch_result.resid

        # TODO : Plot the ACF of the residuals
        if acf:
            fig = plt.figure(figsize=(12, 8))
            # Plot the ACF of ARIMA residuals
            ax1 = fig.add_subplot(211, ylabel="ACF")
            plot_acf(resid, alpha=0.05, ax=ax1, title="ACF of ARIMA Residuals")
            ax1.set_xlabel("Lags")
            ax1.grid(True)

            # Plot the ACF of GARCH residuals on the same axes
            ax2 = fig.add_subplot(212, ylabel="ACF")
            plot_acf(residuals, alpha=0.05, ax=ax2, title="ACF of GARCH  Residuals")
            ax2.set_xlabel("Lags")
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
                f"Processing window {i - window_size + 1}/{len(data) - window_size}..."
            )
            window_data = data["diff_log_return"].iloc[i - window_size : i]
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
            f"Total iterations: {len(data) - window_size}.\n"
        )
        predictions_generator = self.generate_predictions()

        positions = self.execute_trading_strategy(predictions_generator)

        strategy_returns = (
            np.array(positions[:-1]) * data["log_return"].iloc[window_size + 1 :].values
        )
        buy_and_hold = data["log_return"].iloc[window_size + 1 :].values
        buy_and_hold_returns = np.cumsum(buy_and_hold)
        cumulative_returns = np.cumsum(strategy_returns)
        dates = data.index[window_size + 1 :]
        self.plot_cumulative_returns(cumulative_returns, buy_and_hold_returns, dates)

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
        plt.plot(dates, strategy_returns, label="ARIMA+GARCH ", color="blue")
        plt.plot(dates, buy_and_hold_returns, label="Buy & Hold", color="red")
        plt.xlabel("Time")
        plt.ylabel("Cumulative Returns")
        plt.title(f"ARIMA+GARCH Strategy vs. Buy & Hold on ({self.symbol})")
        plt.legend()
        plt.grid(True)
        plt.show()


# *********************************************
# STATS TEST (Cointegration , Mean Reverting)*
# *********************************************
def get_corr(tickers: Union[List[str], Tuple[str, ...]], start: str, end: str) -> None:
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
    data = yf.download(tickers, start=start, end=end, multi_level_index=False)[
        "Adj Close"
    ]

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
    print("Result Parmas: \n")
    print(results.params)
    print("\nRegression Summary:")
    print(results.summary())

    # Calculate and output the CADF test on the residuals
    print("\nCointegration TEST Results:")
    cadf = ts.adfuller(df["res"], autolag="AIC")
    pprint.pprint(cadf)


def _hurst(ts):
    """
    Returns the Hurst Exponent of the time series vector ts,
    """
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0


# Function to calculate Hurst Exponent


def hurst(time_series):
    H, c, data_range = compute_Hc(time_series, kind="price", simplified=True)
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
    data = yf.download(
        symbol,
        start=start,
        end=end,
        progress=False,
        multi_level_index=False,
        auto_adjust=True,
    )

    # Create a Geometric Brownian Motion, Mean-Reverting, and Trending Series
    gbm = np.log(np.cumsum(np.random.randn(100000)) + 1000)
    mr = np.log(np.random.randn(100000) + 1000)
    tr = np.log(np.cumsum(np.random.randn(100000) + 1) + 1000)

    # Output the Hurst Exponent for each of the series
    print(f"\nHurst(GBM):  {_hurst(gbm)}")
    print(f"Hurst(MR):   {_hurst(mr)}")
    print(f"Hurst(TR):   {_hurst(tr)}")
    print(f"Hurst({symbol}): {hurst(data['Adj Close'])}\n")


def test_cointegration(ticker1, ticker2, start, end):
    # Download historical data
    stock_data_pair = yf.download(
        [ticker1, ticker2],
        start=start,
        end=end,
        progress=False,
        multi_level_index=False,
        auto_adjust=True,
    )["Adj Close"].dropna()

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
    prices[etfs[0]] = etf_df1["Adj Close"]
    prices[etfs[1]] = etf_df2["Adj Close"]

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


# ******************************************
#         ORNSTEIN UHLENBECK PROCESS       *
# ******************************************


class OrnsteinUhlenbeck:
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

    def __init__(self, prices: np.ndarray, returns: bool = True, timeframe: str = "D1"):
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
            "1m": 1 / (24 * 60),  # 1 minute intervals
            "5m": 5 / (24 * 60),  # 5 minute intervals
            "15m": 15 / (24 * 60),  # 15 minute intervals
            "30m": 30 / (24 * 60),  # 30 minute intervals
            "1h": 1 / 24,  # 1 hour intervals
            "4h": 4 / 24,  # 4 hour intervals
            "D1": 1,  # Daily intervals
        }
        if timeframe not in time_frame_mapping:
            raise ValueError("Unsupported time frame")
        self.tf = time_frame_mapping[timeframe]

        params = self.estimate_parameters()
        self.mu_hat = params[0]  # Mean (μ)
        self.theta_hat = params[1]  # Drift (θ)
        self.sigma_hat = params[2]  # Volatility (σ)
        print(f"Estimated μ: {self.mu_hat}")
        print(f"Estimated θ: {self.theta_hat}")
        print(f"Estimated σ: {self.sigma_hat}")

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
            x[t] = x[t - 1] + (theta * (mu - x[t - 1]) * dt) + (sigma * dW)
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
        result = minimize(self._neg_log_likelihood, initial_guess, args=(self.returns,))
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
        ou_simulated = self.ornstein_uhlenbeck(mu, theta, sigma, dt, 0, n + 1)
        residuals = ou_simulated[1 : n + 1] - returns
        neg_ll = 0.5 * np.sum(residuals**2) / sigma**2 + 0.5 * n * np.log(
            2 * np.pi * sigma**2
        )
        return neg_ll

    def simulate_process(self, returns=None, n=100, p=None):
        """
        Simulates the OU process multiple times .

        Args:
            returns (np.ndarray): Historical returns.
            n (int): Number of simulations to perform.
            p (int): Number of time steps.

        Returns:
            np.ndarray: 2D array representing simulated processes.
        """
        if returns is None:
            returns = self.returns
        if p is not None:
            T = p
        else:
            T = len(returns)
        dt = self.tf

        dW_matrix = np.random.normal(loc=0, scale=np.sqrt(dt), size=(n, T))
        simulations_matrix = np.zeros((n, T))
        simulations_matrix[:, 0] = returns[-1]

        for t in range(1, T):
            simulations_matrix[:, t] = (
                simulations_matrix[:, t - 1]
                + self.theta_hat * (self.mu_hat - simulations_matrix[:, t - 1]) * dt
                + self.sigma_hat * dW_matrix[:, t]
            )
        return simulations_matrix


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
