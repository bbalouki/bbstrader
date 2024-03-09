"""
Cointegration Augmented Dickey-Fuller test
"""
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import pprint
import warnings
warnings.filterwarnings("ignore")


def get_corr(tickers: list[str] |tuple[str], start: str, end: str):
    # Download historical data
    data = yf.download(
        tickers, start=start, end=end)['Adj Close']

    # Calculate correlation matrix
    correlation_matrix = data.corr()

    # Display the matrix
    print(correlation_matrix)

def plot_price_series(df, ts1, ts2):
    """
    Plot both time series on the same line graph for
    the specified date range.

    :param df : (pd.DataFrame) 
        The DataFrame containing prices for each series 
    :param ts1 : (str) The first time series column name
    :param ts2 : (str) The second time series column name
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

def plot_scatter_series(df, ts1, ts2):
    """
    Plot a scatter plot of both time series for
    via the provided DataFrame.

    :param df : (pd.DataFrame) 
        The DataFrame containing prices for each series 
    :param ts1 : (str) The first time series column name
    :param ts2 : (str) The second time series column name
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

def plot_residuals(df):
    """
    Plot the residuals of OLS procedure for both
    time series.

    :param df : (pd.DataFrame) 
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

def run_test(pair: list[str] |tuple[str], start: str, end: str):
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