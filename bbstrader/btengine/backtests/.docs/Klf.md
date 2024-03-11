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
from btengine.backtests.klf import run_kf_backtest
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
csv_dir = '/backtests/results/klf_hmm/'
hmm_csv = '/backtests/results/klf_hmm/tlt_train.csv'
run_kf_backtest(symbol_list, csv_dir, hmm_csv, start_date, **kwargs)
```

## Notes
- For experimentation, the `csv_dir` and `hmm_csv` paths are set to `/backtests/results/klf_hmm/`, indicating that the required CSV files are located in this directory.

- By default all bcktests uses Hidden Markov model as risk manager  , if you want to not use it , you can do so by adjusting the `KLFStrategyBacktester` and `run_kf_backtest`.
To adjust the strategy backtester , you need to know which paramter to adjust, why and how , learn more on [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter). In the future we will give option to use different risk manager.