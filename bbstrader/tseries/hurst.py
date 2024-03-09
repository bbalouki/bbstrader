from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
import yfinance as yf
from hurst import compute_Hc

def Hurst(ts):
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

def run_test(symbol: str, start: str, end: str):
    data = yf.download(symbol, start=start, end=end)

    # Create a Geometric Brownian Motion, Mean-Reverting, and Trending Series
    gbm = log(cumsum(randn(100000))+1000)
    mr = log(randn(100000)+1000)
    tr = log(cumsum(randn(100000)+1)+1000)

    # Output the Hurst Exponent for each of the series
    print(f"\nHurst(GBM):  {Hurst(gbm)}")
    print(f"Hurst(MR):   {Hurst(mr)}")
    print(f"Hurst(TR):   {Hurst(tr)}")
    print(f"\nHurst1({symbol}): {Hurst(data['Adj Close'].values)}")
    print(f"Hurst({symbol}): {hurst(data['Adj Close'])}\n")