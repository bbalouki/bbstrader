# Trading Strategies Module

## Overview

The Trading Strategies Module is a Python-based framework designed to facilitate the research, development, and evaluation of various algorithmic trading strategies on financial markets. This module integrates several advanced strategies including ARIMA+GARCH, Kalman Filter, Ornstein-Uhlenbeck processes, and Simple Moving Averages (SMA) with Hidden Markov Model (HMM) for risk management. The module aims to provide a robust toolset for traders and researchers to simulate, backtest, and optimize their trading algorithms.

## Key Features

- **ARIMA+GARCH Strategy**: Utilizes ARIMA for predicting future price movements and GARCH for modeling volatility, suitable for markets with time-varying volatility.
- **Kalman Filter Strategy (KLFStrategy)**: Employs the Kalman Filter to estimate dynamic states such as the slope and intercept in price relationships, allowing for adaptive trading signals.
- **Ornstein-Uhlenbeck Process**: Implements a mean-reverting model for asset prices, which is ideal for strategies based on statistical arbitrage.
- **Simple Moving Average (SMA) Strategy**: A traditional technical analysis strategy that signals trades based on short-term and long-term moving average crossovers, integrated with HMM for filtering signals under uncertain market conditions.

## Installation

1. Ensure you have Python 3.8 or later installed on your machine.
2. Clone or download this module to your preferred directory.
3. Install the required Python packages. While the exact dependencies may vary based on the strategy, the common ones include:

```bash
pip install bbstrader
```
## [Documentation](./Ts.md)

## Basic Usage

Each strategy within the module is encapsulated in its class, designed to be flexible and extendable. Here’s a basic example to use the ARIMA+GARCH strategy:

```python
from bbstrader.strategies import ArimaGarchStrategy
import pandas as pd

# Load your historical price data into a DataFrame
data = pd.read_csv('path_to_your_data.csv')

# Initialize the strategy with your data
strategy = ArimaGarchStrategy(symbol='AAPL', data=data)

# Run backtest
strategy.backtest_strategy()
```

Similarly, you can instantiate and utilize other strategies (`KLFStrategy`, `OrnsteinUhlenbeck`, `SMAStrategy`) by following the pattern above, adjusting parameters as needed, see `/strategies/.docs/` for more details.

## Customizing Strategies

You are encouraged to modify and extend the strategies to suit your trading style and the characteristics of the financial instruments you're trading. For example, you can adjust the window sizes in the SMA strategy or the parameters of the ARIMA and GARCH models to fit different assets or market conditions.

## Contributing

Contributions to the Trading Strategies Module are welcome. If you have suggestions for new strategies or improvements, please fork the repository and submit a pull request with your changes.

## License

This project is open source and available under the MIT License.