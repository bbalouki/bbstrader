import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from scipy.stats import mstats
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import quantstats as qs
import warnings
warnings.filterwarnings("ignore")

sns.set_theme()

__all__ = [
    "create_drawdowns",
    "plot_performance",
    "create_sharpe_ratio",
    "create_sortino_ratio",
    "plot_returns_and_dd",
    "plot_monthly_yearly_returns",
    "show_qs_stats"
]


def create_sharpe_ratio(returns, periods=252) -> float:
    """
    Create the Sharpe ratio for the strategy, based on a
    benchmark of zero (i.e. no risk-free rate information).

    Args:
        returns : A pandas Series representing period percentage returns.
        periods (int): Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.

    Returns:
        S (float): Sharpe ratio
    """
    return qs.stats.sharpe(returns, periods=periods)

# Define a function to calculate the Sortino Ratio
def create_sortino_ratio(returns, periods=252) -> float:
    """
    Create the Sortino ratio for the strategy, based on a
    benchmark of zero (i.e. no risk-free rate information).

    Args:
        returns : A pandas Series representing period percentage returns.
        periods (int): Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.

    Returns:
        S (float): Sortino ratio
    """
    return qs.stats.sortino(returns, periods=periods)


def create_drawdowns(pnl):
    """
    Calculate the largest peak-to-trough drawdown of the PnL curve
    as well as the duration of the drawdown. Requires that the
    pnl_returns is a pandas Series.

    Args:
        pnl : A pandas Series representing period percentage returns.

    Returns:
        (tuple): drawdown, duration - high-water mark, duration.
    """
    # Calculate the cumulative returns curve
    # and set up the High Water Mark
    hwm = pd.Series(index=pnl.index)
    hwm.iloc[0] = 0

    # Create the drawdown and duration series
    idx = pnl.index
    drawdown = pd.Series(index=idx)
    duration = pd.Series(index=idx)

    # Loop over the index range
    for t in range(1, len(idx)):
        hwm.iloc[t] = max(hwm.iloc[t-1], pnl.iloc[t])
        drawdown.iloc[t] = (hwm.iloc[t] - pnl.iloc[t])
        duration.iloc[t] = (0 if drawdown.iloc[t] ==
                            0 else duration.iloc[t-1]+1)

    return drawdown, drawdown.max(), duration.max()


def plot_performance(df, title):
    """
    Plot the performance of the strategy:
        - (Portfolio value,  %)
        - (Period returns, %)
        - (Drawdowns, %)

    Args:
        df (pd.DataFrame):
        The DataFrame containing the strategy returns and drawdowns.
        title (str): The title of the plot.

    Note:
    The DataFrame should contain the following columns
    - Datetime: The timestamp of the data
    - Equity Curve: The portfolio value
    - Returns: The period returns
    - Drawdown: The drawdowns
    - Total : The total returns
    """
    data = df.copy()
    data = data.sort_values(by='Datetime')
    # Plot three charts: Equity curve,
    # period returns, drawdowns
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f'{title} Strategy Performance', fontsize=16)

    # Set the outer colour to white
    sns.set_theme()

    # Plot the equity curve
    ax1 = fig.add_subplot(311, ylabel='Portfolio value, %')
    data['Equity Curve'].plot(ax=ax1, color="blue", lw=2.)
    ax1.set_xlabel('')
    plt.grid(True)

    # Plot the returns
    ax2 = fig.add_subplot(312, ylabel='Period returns, %')
    data['Returns'].plot(ax=ax2, color="black", lw=2.)
    ax2.set_xlabel('')
    plt.grid(True)

    # Plot Drawdown
    ax3 = fig.add_subplot(313, ylabel='Drawdowns, %')
    data['Drawdown'].plot(ax=ax3, color="red", lw=2.)
    ax3.set_xlabel('')
    plt.grid(True)

    # Plot the figure
    plt.tight_layout()
    plt.show()


def plot_returns_and_dd(df: pd.DataFrame, benchmark: str, title):
    """
    Plot the returns and drawdowns of the strategy
    compared to a benchmark.

    Args:
        df (pd.DataFrame): 
            The DataFrame containing the strategy returns and drawdowns.
        benchmark (str): 
            The ticker symbol of the benchmark to compare the strategy to.
        title (str): The title of the plot.

    Note:
    The DataFrame should contain the following columns:
    - Datetime : The timestamp of the data
    - Equity Curve : The portfolio value
    - Returns : The period returns
    - Drawdown : The drawdowns
    - Total : The total returns
    """
    # Ensure data is sorted by Datetime
    data = df.copy()
    data.reset_index(inplace=True)
    data = data.sort_values(by='Datetime')
    data.sort_values(by='Datetime', inplace=True)

    # Get the first and last Datetime values
    first_date = data['Datetime'].iloc[0]
    last_date = data['Datetime'].iloc[-1]

    # Download benchmark data from Yahoo Finance
    # To avoid errors, we use the try-except block
    # in case the benchmark is not available
    try:
        bm = yf.download(benchmark, start=first_date, end=last_date)
        bm['log_return'] = np.log(bm['Close'] / bm['Close'].shift(1))
        # Use exponential to get cumulative returns
        bm_returns = np.exp(np.cumsum(bm['log_return'].fillna(0)))

        # Normalize bm series to start at 1.0
        bm_returns_normalized = bm_returns / bm_returns.iloc[0]
    except Exception:
        bm = None

    # Create figure and plot space
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(
        14, 8), gridspec_kw={'height_ratios': [3, 1]})

   # Plot the Equity Curve for the strategy
    ax1.plot(data['Datetime'], data['Equity Curve'],
             label='Backtest', color='green', lw=2.5)
    # Check benchmarck an Plot the Returns for the benchmark
    if bm is not None:
        ax1.plot(bm.index, bm_returns_normalized,
                 label='benchmark', color='gray', lw=2.5)
        ax1.set_title(f'{title} Strategy vs. Benchmark ({benchmark})')
    else:
        ax1.set_title(f'{title} Strategy Returns')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Returns')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    # Plot the Drawdown
    ax2.fill_between(data['Datetime'], data['Drawdown'],
                     0, color='red', step="pre", alpha=0.5)
    ax2.plot(data['Datetime'], data['Drawdown'], color='red',
             alpha=0.6, lw=2.5)  # Overlay the line
    ax2.set_title('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown')
    ax2.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_monthly_yearly_returns(df:pd.DataFrame, title):
    """
    Plot the monthly and yearly returns of the strategy.

    Args:
        df (pd.DataFrame): 
        The DataFrame containing the strategy returns and drawdowns.
        title (str): The title of the plot.

    Note:
    The DataFrame should contain the following columns:
    - Datetime : The timestamp of the data
    - Equity Curve : The portfolio value
    - Returns : The period returns
    - Drawdown : The drawdowns
    - Total : The total returns
    """
    equity_df = df.copy()
    equity_df.reset_index(inplace=True)
    equity_df['Datetime'] = pd.to_datetime(equity_df['Datetime'])
    equity_df.set_index('Datetime', inplace=True)

    # Calculate daily returns
    equity_df['Daily Returns'] = equity_df['Total'].pct_change()

    # Group by year and month to get monthly returns
    monthly_returns = equity_df['Daily Returns'].groupby(
        [equity_df.index.year, equity_df.index.month]
    ).apply(lambda x: (1 + x).prod() - 1)

    # Prepare monthly returns DataFrame
    monthly_returns_df = monthly_returns.unstack(level=-1) * 100
    monthly_returns_df.columns = monthly_returns_df.columns.map(
        lambda x: pd.to_datetime(x, format='%m').strftime('%b'))

    # Calculate and prepare yearly returns DataFrame
    yearly_returns_df = equity_df['Total'].resample(
        'A').last().pct_change().to_frame(name='Yearly Returns') * 100

    # Set the aesthetics for the plots
    sns.set_theme(style="darkgrid")

    # Initialize the matplotlib figure,
    # adjust the height_ratios to give more space to the yearly returns
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(
        12, 8), gridspec_kw={'height_ratios': [2, 1]})
    f.suptitle(f'{title} Strategy Monthly and Yearly Returns')
    # Find the min and max values in the data to set the color scale range.
    vmin = monthly_returns_df.min().min()
    vmax = monthly_returns_df.max().max()
    # Define the color palette for the heatmap
    cmap = sns.diverging_palette(10, 133, sep=3, n=256, center="light")

    # Create the heatmap with the larger legend
    sns.heatmap(monthly_returns_df, annot=True, fmt=".1f",
                linewidths=.5, ax=ax1, cbar_kws={"shrink": .8},
                cmap=cmap, center=0, vmin=vmin, vmax=vmax)

    # Rotate the year labels on the y-axis to vertical
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    ax1.set_ylabel('')
    ax1.set_xlabel('')

    # Create the bar plot
    yearly_returns_df.plot(kind='bar', ax=ax2, legend=None, color='skyblue')

    # Set plot titles and labels
    ax1.set_title('Monthly Returns (%)')
    ax2.set_title('Yearly Returns (%)')

    # Rotate the x labels for the yearly returns bar plot
    ax2.set_xticklabels(yearly_returns_df.index.strftime('%Y'), rotation=45)
    ax2.set_xlabel('')

    # Adjust layout spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

def show_qs_stats(returns, benchmark, strategy_name, save_dir=None):
    """
    Generate the full quantstats report for the strategy.

    Args:
        returns (pd.Serie): 
            The DataFrame containing the strategy returns and drawdowns.
        benchmark (str): 
            The ticker symbol of the benchmark to compare the strategy to.
        strategy_name (str): The name of the strategy.
    """
    # Load the returns data
    returns = returns.copy()

    # Drop duplicate index entries
    returns = returns[~returns.index.duplicated(keep='first')]

    # Extend pandas functionality with quantstats
    qs.extend_pandas()

    # Generate the full report with a benchmark
    qs.reports.full(returns, mode='full', benchmark=benchmark)
    qs.reports.html(returns, benchmark=benchmark, output=save_dir, title=strategy_name)