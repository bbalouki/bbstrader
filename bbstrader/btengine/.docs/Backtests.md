# ArimaGarchStrategy Backtester with HMM Risk Management

The `ArimaGarchStrategyBacktester` class extends the `Strategy` class to implement a backtesting framework for trading strategies based on ARIMA-GARCH models, incorporating a Hidden Markov Model (HMM) for risk management.

## Features

- **ARIMA-GARCH Model**: Utilizes ARIMA for time series forecasting and GARCH for volatility forecasting, aimed at predicting market movements.
- **HMM Risk Management**: Employs a Hidden Markov Model to manage risks, determining safe trading regimes.
- **Event-Driven Backtesting**: Capable of simulating real-time trading conditions by processing market data and signals sequentially.

## Usage

To use the `ArimaGarchStrategyBacktester`, instantiate it with necessary parameters such as market data, event queue, and strategy-specific settings like window sizes for ARIMA and HMM models, trading quantity, and the symbol of interest.

### Key Methods

- `get_data()`: Retrieves and prepares the data required for ARIMA-GARCH model predictions.
- `create_signal()`: Generates trading signals based on model predictions and current market positions.
- `calculate_signals(event)`: Listens for market events and triggers signal creation and event placement.

## `run_arch_backtest` Function

This function initiates and runs the backtest for the ARIMA-GARCH strategy with HMM risk management, setting up the environment including data, execution handlers, portfolio, and strategy.

### Parameters

- `symbol_list`: List of ticker symbols for the backtest.
- `backtest_csv`: Path to CSV file containing historical data for backtesting.
- `hmm_csv`: Path to CSV file for the HMM risk manager.
- `start_date`: Start date of the backtesting period.
- `initial_capital`: Initial capital for the backtest (default: 100000.0 USD).
- `heartbeat`: Simulation heartbeat in seconds (default: 0.0).

### Example Usage

```python
import datetime
from pathlib import Path
import yfinance as yf
from bbstrader.strategies import ArimaGarchStrategy
from bbstrader.tseries import load_and_prepare_data
from bbstrader.backtests import run_arch_backtest

if __name__ == '__main__':
    # ARCH SPY Vectorize Backtest
    k = 252
    data = yf.download("SPY", start="2004-01-02", end="2015-12-31")
    arch = ArimaGarchStrategy("SPY", data, k=k)
    df = load_and_prepare_data(data)
    arch.show_arima_garch_results(df['diff_log_return'].values[-k:])
    arch.backtest_strategy()

    # ARCH SPY Event Driven backtest from 2004-01-02" to "2015-12-31"
    # with hidden Markov Model as a risk manager.
    data_dir = Path("/bbstrader/btengine/data/") # if you cloned this repo
    csv_dir = str(Path().cwd()) + str(data_dir) # Or the absolute path of your csv data directory
    hmm_csv = str(Path().cwd()) + str(data_dir/"spy_train.csv")

    symbol_list = ["SPY"]
    kwargs = {
        'tiker': 'SPY',
        'benchmark': 'SPY',
        'window_size': 252,
        'qty': 1000,
        'k': 50,
        'iterations': 1000,
        'strategy_name': 'ARIMA+GARCH & HMM'
    }
    start_date = datetime.datetime(2004, 1, 2)
    run_arch_backtest(
        symbol_list, csv_dir, hmm_csv, start_date, **kwargs
    )
```

### Note

- The `csv_dir` and `hmm_csv` are located in the `/bbstrader/btengine/data/` directory for practical experimentation.
- The vectorized backtest tends to overestimate performance due to the absence of commissions and not employing HMM for risk management. It primarily aims to assess the ARIMA + GARCH model's efficacy before applying it within the event-driven backtesting engine.
- By default all bcktests uses Hidden Markov model as risk manager , if you want to not use it , you can do so by adjusting the `ArimaGarchStrategyBacktester` and `run_arch_backtest` . In the future we will give option to use different risk manager.

This documentation aims to provide a comprehensive understanding of the code's functionality, enabling users, to effectively utilize and adapt the backtesting strategy for their financial analysis and trading strategy development endeavors.

# KLFStrategyBacktester with Hidden Markov Model Risk Management

## Overview

The `KLFStrategyBacktester` class implements a backtesting framework for a [pairs trading](https://en.wikipedia.org/wiki/Pairs_trade) strategy using Kalman Filter and Hidden Markov Models (HMM) for risk management. This document outlines the structure and usage of the `KLFStrategyBacktester`, including initialization parameters, main functions, and an example of how to run a backtest.

## Classes and Functions

### KLFStrategyBacktester

#### Description

Implements a backtesting strategy for `KLFStrategy` that integrates a Hidden Markov Model for risk management, designed to simulate and evaluate the performance of a pairs trading strategy under historical market conditions.

#### Initialization Parameters

- `bars`: Instance of `HistoricCSVDataHandler` for market data handling.
- `events_queue`: A queue for managing events.
- `**kwargs`: Additional keyword arguments including:
  - `tickers`: List of ticker symbols involved in the pairs trading strategy.
  - `quantity`: Quantity of assets to trade. Default is 100.
  - `delta`: Delta parameter for the Kalman Filter. Default is `1e-4`.
  - `vt`: Measurement noise covariance for the Kalman Filter. Default is `1e-3`.
  - `model`: Instance of `HMMRiskManager` for managing trading risks.
  - `window`: Window size for calculating returns for the HMM. Default is 50.
  - `hmm_tiker`: Ticker symbol used by the HMM for risk management.

#### Key Methods

- `_init_kalman()`: Initializes the Kalman Filter.
- `calc_slope_intercep(prices)`: Calculates the slope and intercept using the Kalman Filter.
- `calculate_xy_signals(et, sqrt_Qt, regime)`: Determines trading signals based on the state and measurements.
- `calculate_signals_for_pairs()`: Calculates signals for pairs trading.
- `calculate_signals(event)`: Entry point for calculating trading signals on market events.

### `run_kf_backtest`

#### Description

Initializes and runs a backtest for the Kalman Filter strategy with a Hidden Markov Model risk manager.

#### Parameters

- `symbol_list`: List of ticker symbols for the backtest.
- `backtest_csv`: Path to the CSV file containing historical data for backtesting.
- `hmm_csv`: Path to the CSV file containing data for initializing the HMM risk manager.
- `start_date`: Start date of the backtest.
- `initial_capital`: Initial capital for the backtest. Default is `100000.0`.
- `heartbeat`: Frequency of data updates. Default is `0.0`.
- `**kwargs`: Additional parameters for risk management and portfolio configuration.

#### Notes

The dataset provided through `hmm_csv` should precede the timeframe of the `backtest_csv` to avoid look-ahead bias and ensure the validity of the HMM's risk management system.

## Example Usage

```python
import datetime
from pathlib import Path
from btengine.backtests import run_kf_backtest

if __name__ == '__main__':

    # KLF IEI TLT Event Driven backtest with Hidden Markov Model as a risk manager.
    symbol_list = ["IEI", "TLT"]
    kwargs = {
        "tickers": symbol_list,
        "quantity": 2000,
        "time_frame": "D1",
        "trading_hours": 6.5,
        "benchmark": "TLT",
        "window": 50,
        "hmm_tiker": "TLT",
        "iterations": 100,
        'strategy_name': 'Kalman Filter & HMM'
    }
    start_date = datetime.datetime(2009, 8, 3, 10, 40, 0)
    data_dir = Path("/bbstrader/btengine/data/")
    csv_dir = str(Path().cwd()) + str(data_dir)
    hmm_csv = str(Path().cwd()) + str(data_dir/"tlt_train.csv")
    run_kf_backtest(symbol_list, csv_dir, hmm_csv, start_date, **kwargs)
```

## Notes

- For experimentation, the `csv_dir` and `hmm_csv` paths are set to `/bbstrader/btengine/data/`, indicating that the required CSV files are located in this directory.

- By default all bcktests uses Hidden Markov model as risk manager , if you want to not use it , you can do so by adjusting the `KLFStrategyBacktester` and `run_kf_backtest`.
  To adjust the strategy backtester , you need to know which paramter to adjust, why and how , learn more on [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter). In the future we will give option to use different risk manager.

# OUBacktester Strategy with Hidden Markov Model Risk Management

The `OUBacktester` is a robust framework designed to execute backtesting on financial markets using the Ornstein-Uhlenbeck (OU) mean-reverting model combined with risk management through a Hidden Markov Model (HMM). This document provides detailed insights into the implementation, usage, and integration of the `OUBacktester` within a backtesting ecosystem.

## Overview

The `OUBacktester` extends a generic `Strategy` class, leveraging the OU process to identify and exploit mean-reverting characteristics in financial time series. This strategy is particularly tailored to historical market data, aiming to generate signals for a single or multiple financial instruments.

Key components of the backtesting framework include:

- **Data Handling**: Integration with `HistoricCSVDataHandler` for market data ingestion.
- **Signal Generation**: Utilizes the `OrnsteinUhlenbeck` and `HMMRiskManager` for signal and risk management.
- **Event Management**: Incorporates `SignalEvent` for signal broadcasting.
- **Execution Handling**: Employs `SimulatedExecutionHandler` for order execution simulation.

## Strategy Configuration

The `OUBacktester` class requires initialization with market data (`bars`), an event queue (`events`), and optional parameters that define strategy behavior:

- `ticker`: Symbol of the financial instrument.
- `p`: Lookback period for the OU process.
- `n`: Minimum number of observations for signal generation.
- `qty`: Quantity of assets to trade.
- `csvfile`: Path to the CSV file containing historical market data.
- `model`: HMM risk management model.
- `window`: Lookback period for HMM.

### Data Reading Function

- `_read_csv`: Reads market data from CSV into a DataFrame, indexing by date and parsing columns specific to market prices and volume.

### Signal Creation

- `create_signal`: Generates trading signals based on the OU process outcomes and HMM risk management decisions. It manages transitions between LONG and SHORT states and exits from positions as dictated by strategy logic and market conditions.

## Execution Function

`run_ou_backtest` facilitates the initialization and execution of the backtest using provided parameters:

- `symbol_list`: Symbols to trade.
- `backtest_csv`: Path to the backtesting CSV.
- `ou_csv`: CSV for the OU model.
- `hmm_csv`: CSV for the HMM model.
- `start_date`: Beginning of the backtesting period.
- `initial_capital`: Starting capital.
- `heartbeat`: Frequency of market data processing.

This function orchestrates the backtesting process by setting up the data handler, execution handler, portfolio, and integrating the `OUBacktester` strategy with HMM risk management.

## Example Usage

To execute a backtest for the `GLD` symbol using specific data sources and strategy parameters:

```python
import datetime
from pathlib import Path
from bbstrader.backtests import run_ou_backtest

if __name__ == '__main__':

    # OU Backtest
    symbol_list = ['GLD']
    kwargs = {
        "tiker": 'GLD',
        'n': 5,
        'p': 5,
        'qty': 2000,
        "window": 50,
        'strategy_name': 'Ornstein-Uhlenbeck & HMM'
    }
    data_dir = Path("/bbstrader/btengine/data/")
    csv_dir = str(Path().cwd()) + str(data_dir)
    ou_csv = str(Path().cwd()) + str(data_dir/"ou_gld_train.csv")
    hmm_csv = str(Path().cwd()) + str(data_dir/"hmm_gld_train.csv")
    # Backtest period
    start_date = datetime.datetime(2015, 1, 2)

    # Execute backtest
    run_ou_backtest(symbol_list, csv_dir, ou_csv, hmm_csv, start_date, **kwargs)
```

## Notes

- For effectiveness, it's crucial that the `hmm_csv` data covers a period preceding the `backtest_csv` dataset to prevent look-ahead bias and ensure the integrity of the HMM's risk management.

- By default all bcktests uses Hidden Markov model as risk manager , if you want to not use it , you can do so by adjusting the `OUBacktester` and `run_ou_backtest`.
  In the future we will give option to use different risk manager.

- This strategy is optimized for assets that exhibit mean reversion characteristics. Prior to executing this backtest, it is imperative to conduct a mean reversion test on the intended asset to ensure its suitability for this approach.
  It's important to understand that mean reversion is a theory suggesting that asset prices and returns eventually move back towards the mean or average. This concept can be applied to various financial instruments, including stocks, bonds, and currencies. Here's a more detailed explanation on how to test for mean reversion:

1. Statistical Methods
   a. Augmented Dickey-Fuller (ADF) Test:
   The ADF test is a common statistical test used to determine if a time series is stationary, which is a prerequisite for mean reversion. It tests the null hypothesis that a unit root is present in a time series sample. If the test statistic is less than the critical value, we can reject the null hypothesis and conclude that the series is stationary and possibly mean-reverting.

b. Hurst Exponent:
The Hurst exponent measures the autocorrelation of a time series and its tendency to revert to the mean. A Hurst exponent close to 0.5 suggests a random walk (non-mean reverting), less than 0.5 indicates mean reversion, and greater than 0.5 suggests a trending market.

2. Visual Inspection
   a. Moving Averages:
   Plotting short-term and long-term moving averages can provide a visual indication of mean reversion. If the asset price frequently crosses over its moving averages, it may suggest mean-reverting behavior.

b. Mean and Standard Deviation:
Plotting the mean and standard deviation bands around the price can also provide visual cues. If the price regularly oscillates around the mean and returns within the standard deviation bands, it may indicate mean reversion.

3. Quantitative Models
   a. Cointegration Tests:
   For pairs trading or multiple assets, cointegration tests can identify whether a linear combination of the assets is mean-reverting, even if the individual assets themselves are not.

b. Mean-Reverting Time Series Models:
Implementing models like Ornstein-Uhlenbeck or AR(1) (Autoregressive Model) can help in identifying mean reversion. These models quantify the speed of mean reversion and the volatility of the asset around its mean.

Practical Steps:
Collect Data: Gather historical price data for the asset in question.
Preprocess Data: Ensure the data is clean and adjusted for dividends and splits.
Statistical Testing: Apply the ADF test and calculate the Hurst exponent for the data series.
Model Fitting: For a more sophisticated analysis, fit a mean-reverting model to the data and estimate its parameters.
Backtesting: After identifying mean reversion, backtest the strategy on historical data to check its viability before live implementation.

## Conclusion

The `OUBacktester` framework provides a comprehensive approach for backtesting trading strategies based on mean-reversion and sophisticated risk management through HMM. By adhering to this documentation, users can effectively configure and execute backtests to evaluate the performance of the Ornstein-Uhlenbeck process in conjunction with HMM risk management across different financial instruments and market conditions.

# Moving Average Cross Strategy with Hidden Markov Model Risk Management

## Overview

The `SMAStrategyBacktester` class implements a basic Moving Average Crossover strategy for backtesting, incorporating a simple weighted moving average for short and long periods, typically 50 and 200 days respectively. This strategy utilizes a Hidden Markov Model (HMM) for risk management, filtering trading signals based on the current market regime. It is designed to demonstrate a straightforward long-term trend-following strategy with an emphasis on risk management.

## Key Features

- **Simple Moving Average Crossover:** Trades are based on the crossover of short-term and long-term moving averages, a fundamental technique for identifying potential market entry and exit points.
- **Risk Management with HMM:** Incorporates a Hidden Markov Model to identify the current market regime and filter signals accordingly, enhancing decision-making by acknowledging underlying market conditions.
- **Configurable Windows:** Short and long window periods are customizable, allowing for flexibility and experimentation with different averaging periods.

## Class: `SMAStrategyBacktester`

### Initialization

- **Parameters:**
  - `bars`: A data handler object that provides market data.
  - `events`: An event queue object where generated signals are placed.
  - `short_window` (optional): The period for the short moving average. Default is 50.
  - `long_window` (optional): The period for the long moving average. Default is 200.
  - `model` (optional): The HMM risk management model to be used.
  - `quantity` (optional): The default quantity of assets to trade. Default is 1000.

### Key Methods

- `_calculate_initial_bought`: Initializes the investment status of each symbol in the portfolio as 'OUT'.
- `get_data`: Retrieves and processes the latest market data for each symbol, calculating short and long simple moving averages (SMAs) and determining the current market regime using the HMM model.
- `create_signal`: Generates trading signals based on the strategy logic and the current market regime identified by the HMM model.
- `calculate_signals`: Responds to market data updates and triggers signal creation.

## Function: `run_sma_backtest`

Executes the backtest of the Moving Average Cross Strategy incorporating the HMM for risk management.

### Parameters

- `symbol_list`: List of symbol strings for the backtest.
- `backtest_csv`: Path to the CSV file with backtest data.
- `hmm_csv`: Path to the CSV file with HMM model training data.
- `start_date`: The start date of the backtest as a `datetime` object.
- `initial_capital`: Initial capital for the backtest. Default is 100,000.0.
- `heartbeat`: Heartbeat of the backtest in seconds. Default is 0.0.

### Additional Notes

The integration of the HMM model requires the `hmm_csv` dataset to precede the `backtest_csv` dataset temporally. This ensures that the HMM's risk management is based on a forward-testing scenario, where the backtest data remains unseen by the model during training, thus avoiding look-ahead bias and ensuring the strategy's applicability to future market conditions.

## Dependencies

- `numpy` for numerical operations.
- `datetime` for handling date and time.
- `filterpy` for the Kalman filter, used within the HMM model.
- Custom modules for backtesting framework: `backtester.strategy`, `backtester.event`, `backtester.backtest`, `backtester.data`, `backtester.portfolio`, `backtester.execution`.
- `risk_models.hmm` for the Hidden Markov Model risk management.

## Example of Usage

Below is an example demonstrating how to set up and execute a backtest using the `MovingAverageCrossStrategy` class with HMM risk management. This example assumes you have prepared your environment with the necessary data and modules.

### Setting Up Your Backtest

First, define the directory paths for your CSV data, the specific CSV file for training the HMM model, the list of symbols to be included in the backtest, and other strategy parameters. Here, we're focusing on 'SPY' as our trading symbol with a specific short and long window setup for the moving averages.

```python
import datetime
from pathlib import Path
from bbstrader.backtests import run_sma_backtest

if __name__ == '__main__':

    # SMA Backtest
    symbol_list = ['SPY']
    kwargs = {
        "short_window": 50,
        "long_window": 200,
        'strategy_name': 'SMA & HMM',
        "quantity": 1000
    }
    data_dir = Path("/bbstrader/btengine/data/")
    csv_dir = str(Path().cwd()) + str(data_dir)
    hmm_csv = str(Path().cwd()) + str(data_dir/"spy_train.csv")

    # Backtest start date
    start_date = datetime.datetime(2004, 1, 2, 10, 40, 0)

    # Execute the backtest
    run_sma_backtest(
        symbol_list, csv_dir, hmm_csv, start_date, **kwargs
    )
```

### Accessing the Data

For those interested in experimenting with this strategy, the CSV data used in this example (`csv_dir` and `hmm_csv`) are located in the `/bbstrader/btengine/data/` folder. This allows you to directly copy the setup and play with the data and parameters to understand the impact of different configurations on the strategy's performance.

### Note:

Make sure to adjust the paths for `csv_dir` and `hmm_csv` as per your file system structure or where you have saved the CSV data files. The paths provided here are just examples and need to be tailored to your environment.

## Conclusion

The Moving Average Cross Strategy with HMM risk management offers a foundational approach to systematic trading, emphasizing the importance of risk management through market regime identification. This strategy serves as a starting point for developing more sophisticated trading models and strategies.
