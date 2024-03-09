import pandas as pd
import yfinance as yf
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from itertools import combinations


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


def run_test(tickers: list, start: str, end: str):
    # Loop through ticker combinations
    for ticker1, ticker2 in combinations(tickers, 2):
        test_cointegration(ticker1, ticker2, start, end)
