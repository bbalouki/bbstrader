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
from bbstrader.backtests.sma import run_sma_backtest

# Directory paths for CSV data
csv_dir = '/backtests/results/sma_hmm/'
hmm_csv = '/backtests/results/sma_hmm/spy_train.csv'

# Symbol list and strategy parameters
symbol_list = ['SPY']
kwargs = {
    "short_window": 50,
    "long_window": 200,
    'strategy_name': 'SMA & HMM',
    "quantity": 1000
}

# Backtest start date
start_date = datetime.datetime(2004, 1, 2, 10, 40, 0)

# Execute the backtest
run_sma_backtest(
    symbol_list, csv_dir, hmm_csv, start_date, **kwargs
)
```

### Accessing the Data

For those interested in experimenting with this strategy, the CSV data used in this example (`csv_dir` and `hmm_csv`) are located in the `/backtests/results/sma_hmm` folder. This allows you to directly copy the setup and play with the data and parameters to understand the impact of different configurations on the strategy's performance.

### Note:

Make sure to adjust the paths for `csv_dir` and `hmm_csv` as per your file system structure or where you have saved the CSV data files. The paths provided here are just examples and need to be tailored to your environment.

## Conclusion

The Moving Average Cross Strategy with HMM risk management offers a foundational approach to systematic trading, emphasizing the importance of risk management through market regime identification. This strategy serves as a starting point for developing more sophisticated trading models and strategies.
