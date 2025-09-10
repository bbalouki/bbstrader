import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import seaborn as sns

from bbstrader.metatrader.account import check_mt5_connection, shutdown_mt5
from bbstrader.metatrader.utils import TIMEFRAMES

sns.set_theme()


def _get_data(symbol, timeframe, bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    df = pd.DataFrame(rates)
    return df


def volume_profile(df, bins):
    prices = (df["high"] + df["low"]) / 2
    volumes = df["tick_volume"]
    hist, bin_edges = np.histogram(prices, bins=bins, weights=volumes)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return hist, bin_edges, bin_centers


def value_area(hist, bin_centers, percentage):
    total_volume = np.sum(hist)
    poc_index = np.argmax(hist)
    poc = bin_centers[poc_index]

    sorted_indices = np.argsort(hist)[::-1]
    volume_accum = 0
    value_area_indices = []

    for idx in sorted_indices:
        volume_accum += hist[idx]
        value_area_indices.append(idx)
        if volume_accum >= percentage * total_volume:
            break

    vah = max(bin_centers[i] for i in value_area_indices)
    val = min(bin_centers[i] for i in value_area_indices)
    return poc, vah, val


def display_volume_profile(
    symbol,
    path,
    timeframe: str = "1m",
    bars: int = 1440,
    bins: int = 100,
    va_percentage: float = 0.7,
):
    """
    Display a volume profile chart for a given market symbol using historical data.

    This function retrieves historical price and volume data for a given symbol and
    plots a vertical volume profile chart showing the volume distribution across 
    price levels. It highlights key levels such as:
        - Point of Control (POC): Price level with the highest traded volume.
        - Value Area High (VAH): Upper bound of the value area.
        - Value Area Low (VAL): Lower bound of the value area.
        - Current Price: Latest bid price from MetaTrader 5.

    Args:
        symbol (str): Market symbol (e.g., "AAPL", "EURUSD").
        path (str): Path to the historical data see ``bbstrader.metatrader.account.check_mt5_connection()``.
        timeframe (str, optional): Timeframe for each candle (default is "1m").
        bars (int, optional): Number of historical bars to fetch (default is 1440).
        bins (int, optional): Number of price bins for volume profile calculation (default is 100).
        va_percentage (float, optional): Percentage of total volume to define the value area (default is 0.7).

    Returns:
        None: Displays a matplotlib chart of the volume profile.
    """
    check_mt5_connection(path=path)
    df = _get_data(symbol, TIMEFRAMES[timeframe], bars)
    if df.empty:
        raise ValueError(f"No data found for {symbol} in {path}")
    hist, bin_edges, bin_centers = volume_profile(df, bins)
    poc, vah, val = value_area(hist, bin_centers, va_percentage)
    current_price = mt5.symbol_info_tick(symbol).bid
    shutdown_mt5()

    plt.figure(figsize=(6, 10))
    plt.barh(bin_centers, hist, height=bin_centers[1] - bin_centers[0], color="skyblue")
    plt.axhline(poc, color="red", linestyle="--", label=f"POC: {poc:.5f}")
    plt.axhline(vah, color="green", linestyle="--", label=f"VAH: {vah:.5f}")
    plt.axhline(val, color="orange", linestyle="--", label=f"VAL: {val:.5f}")
    plt.axhline(
        current_price, color="black", linestyle=":", label=f"Price: {current_price:.5f}"
    )
    plt.legend()
    plt.title("Volume Profile")
    plt.xlabel("Volume")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
