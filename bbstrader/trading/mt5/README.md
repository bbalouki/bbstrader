
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

## New Utility Functions

### Time Frame Mapping

- `tf_mapping()`: Returns a dictionary mapping string representations of timeframes to their numeric value in minutes.

### Initializing Trades

- `init_trade(args, symbol=None)`: Initializes and returns a `Trade` object with parameters specified by `args` and optionally overrides the symbol.

### Argument Parsing for Trades

- `add_trade_arguments(parser, pair=False, pchange_sl=None, strategy=None)`: Adds command-line arguments for configuring trade parameters, customizable based on the strategy.

### Common Trading Arguments

- `add_common_trading_arguments(parser, strategy=None)`: Adds common trading-related arguments to the parser, allowing for customization based on the selected strategy.

### Strategy-Specific Argument Functions

- `add_sma_trading_arguments(parser)`: Adds arguments specific to the SMA trading strategy.
- `add_pair_trading_arguments(parser, pair=True, pchange_sl=2.5)`: Adds arguments for configuring pair trading strategies.
- `add_ou_trading_arguments(parser)`: Sets up arguments for Ornstein-Uhlenbeck strategy configuration.
- `add_arch_trading_arguments(parser)`: Configures arguments for ARIMA+GARCH strategy execution.

These utility functions enhance the module's flexibility, allowing users to easily customize and execute different trading strategies based on their requirements.

## Usage
### Command Line Arguments

Each trading strategy can be executed with specific command line arguments. Here are examples for each supported strategy:

#### ARIMA+GARCH Strategy
- **Command Line Argument Example:**

```bash
python trade.py --expert "YourExpertName" --id 1 --version 1.0 --symbol "QQQ" --mr 5.0 --t 2.0 --dr 0.25 --maxt 20 --acl True --tfm "D1" --start "13:35" --fint "19:50" --endt "19:55" --std False --rr 3.0 --psl 2.5
```

#### Simple Moving Averages (SMA) Strategy

- **Command Line Argument Example:**

```bash
python trade.py --expert "YourExpertName" --symbol "QQQ" --tfm "1h" --start "13:35" --sma 35 --lma 80 --rm "hmm"
```

#### Pair Trading Strategy

- **Command Line Argument Example:**

```bash
python trade.py --expert "YourExpertName" --tfm "D1" --start "13:35" --pair "GOOG" "MSFT" --psl 2.5
```

#### Ornstein-Uhlenbeck Strategy

- **Command Line Argument Example:**

```bash
python trade.py --expert "YourExpertName" --symbol "AAPL" --tfm "1h" --start "13:35" --p 20 --n 20 --ouw 2000
```

## Customization

You can customize each trading strategy by adjusting its parameters or extending the strategy logic. The utility functions provided in `utils.py` make it easier to add or modify command-line arguments for different trading setups.

## Contributing

We welcome contributions to improve existing strategies or add new ones. If you have a trading strategy or an enhancement you'd like to share, please fork the repository and submit a pull request.

## Disclaimer

Trading involves a significant risk of loss and is not suitable for all investors. The developers of this module are not responsible for any financial losses incurred from its use. Always trade responsibly and at your own risk.

## License

This project is open source and available under the MIT License.
