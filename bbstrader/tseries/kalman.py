import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from filterpy.kalman import KalmanFilter
import warnings

warnings.filterwarnings('ignore')


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


def run_filter(etfs: list[str] |tuple[str], start: str, end: str):
    etf_df1 = yf.download(etfs[0], start, end)
    etf_df2 = yf.download(etfs[1], start, end)

    prices = pd.DataFrame(index=etf_df1.index)
    prices[etfs[0]] = etf_df1["Adj Close"]
    prices[etfs[1]] = etf_df2["Adj Close"]

    draw_date_coloured_scatterplot(etfs, prices)
    state_means, state_covs = calc_slope_intercept_kalman(etfs, prices)
    draw_slope_intercept_changes(prices, state_means)
