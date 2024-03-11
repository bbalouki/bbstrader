
# MT5 Trading Strategies Execution Module

## Overview

The MT5 Trading Strategies Execution Module is a versatile and powerful Python-based framework designed for automating the execution of a variety of algorithmic trading strategies directly on the MetaTrader 5 (MT5) platform. It integrates with MT5 to offer real-time trading capabilities across multiple strategies including ARIMA+GARCH, Machine Learning models, Ornstein-Uhlenbeck processes, Pair Trading, and Simple Moving Averages. With a focus on flexibility, efficiency, and effectiveness, this module caters to both seasoned traders and those new to algorithmic trading, allowing for the exploration and implementation of complex trading strategies in a structured and risk-managed environment.

## Features

- **Multiple Trading Strategies**: Support for diverse trading strategies including statistical models, machine learning predictions, mean-reversion strategies, pair trading, and technical indicators.
- **Risk Management**: Integrated risk management through Hidden Markov Models (HMM), enabling dynamic adjustment to trading behavior based on market regime changes.
- **Real-Time Trading**: Automated trading on the MT5 platform, with real-time order execution based on strategy signals.
- **Flexible Timeframes**: Operates across various timeframes, from minutes to daily, accommodating a wide range of trading styles.
- **Dynamic Position Management**: Capabilities to manage positions based on strategy signals, risk levels, and predefined trading rules.

## Installation

1. Ensure MetaTrader 5 is installed and configured on your system.
2. Install the required Python packages:

```bash
pip install MetaTrader5 pandas numpy scipy matplotlib datetime
```

3. Clone or download this module to your local machine.

## Configuration

Before using the module, configure the MetaTrader 5 terminal to allow automated trading and ensure your account details are correctly set up. Edit the trading strategy files as needed to match your trading preferences and risk management profile.

## Usage

This module consists of several scripts, each corresponding to a specific trading strategy. To run a strategy, execute the relevant script after configuring your MT5 details and strategy parameters. For example, to run the ARIMA+GARCH trading strategy:

- First you need to create a .py file , let's say `trade.py`.
- In and in this file , importe the strategy you want to run 

```python
from trading.mt5.run import run_arch_trading

if __name__ == '__main__':
    run_arch_trading()
```
- And on the terminal tape:
```bash
$ python trade.py --symbol "QQQ" --period "week" --std_stop True  
```
You can refer to `Trade` class and `run_arch_trading` for specfic paramter.

## Customization

You can customize each trading strategy by adjusting its parameters or extending the strategy logic. For example, modify the window sizes for moving averages or adjust the risk management settings.

## Contributing

We welcome contributions to improve existing strategies or add new ones. If you have a trading strategy or an enhancement you'd like to share, please fork the repository and submit a pull request.

## Disclaimer

Trading involves significant risk of loss and is not suitable for all investors. The developers of this module are not responsible for any financial losses incurred from its use. Always trade responsibly and at your own risk.

## License

This project is open source and available under the MIT License.
