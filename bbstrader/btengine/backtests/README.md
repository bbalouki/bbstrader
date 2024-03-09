# Trading Strategies Backtesting Module

## Overview

This backtesting module is designed to test various trading strategies using an event-driven system. It supports strategies based on statistical and econometric models, including ARIMA-GARCH, Kalman Filter, Ornstein-Uhlenbeck processes, and Simple Moving Average (SMA) crossovers. Integrating risk management through Hidden Markov Models (HMM), it provides a robust framework for evaluating the effectiveness of trading strategies under different market conditions.

## Module Components

- **ArimaGarchStrategyBacktester**: Tests trading strategies based on ARIMA-GARCH models, suitable for volatile markets.
- **KLFStrategyBacktester**: Utilizes the Kalman Filter for estimating hidden variables and making trading decisions.
- **OUBacktester**: Implements the Ornstein-Uhlenbeck process to exploit mean-reverting behaviors in financial time series.
- **SMAStrategyBacktester**: A classic strategy based on Simple Moving Average crossovers, integrated with HMM for risk management.

## Features

- **Event-Driven Architecture**: Mimics real-world trading environments by processing market data and generating signals in response to market events.
- **Risk Management**: Integrates Hidden Markov Models for assessing and managing trading risks effectively.
- **Statistical and Econometric Models**: Supports a range of models for developing sophisticated trading strategies.
- **Performance Visualization**: Includes tools for plotting strategy performance metrics such as equity curves, returns, and drawdowns.

## Getting Started

### Installation

1. Ensure Python 3.8+ is installed on your system.
2. Install required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn filterpy yfinance
```

3. Clone or download this module to your local machine.

### Configuration

Before running a backtest, configure the strategy parameters and data sources in the respective strategy file. Each strategy file (`arch.py`, `klf.py`, `ou.py`, `sma.py`) contains a `run_backtest` function that you need to customize according to your data and preferences.

### Running a Backtest

Navigate to the directory containing the backtest files and run the desired backtest script from the command ; see `/backtests/.docs` for more details on how run each backtest.

## Customizing Strategies

To develop custom trading strategies, extend the `Strategy` class and implement the `calculate_signals` method according to your strategy logic. Integrate your strategy with the backtesting engine by updating the respective `run_backtest` function.

## Contributing

Contributions to enhance the module with new trading strategies, features, or improvements are welcome. Please fork the repository and submit a pull request with your changes.

## License

This backtesting module is open source and available under the MIT License.
