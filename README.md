# Simplified Investment & Trading Toolkit
[![Documentation Status](https://readthedocs.org/projects/bbstrader/badge/?version=latest)](https://bbstrader.readthedocs.io/en/latest/?badge=latest)
[![PYPI Version](https://img.shields.io/pypi/v/bbstrader)](https://pypi.org/project/bbstrader/)
[![PyPi status](https://img.shields.io/pypi/status/bbstrader.svg?maxAge=60)](https://pypi.python.org/pypi/bbstrader)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/bbstrader)](https://pypi.org/project/bbstrader/)
[![PyPI Downloads](https://static.pepy.tech/badge/bbstrader)](https://pepy.tech/projects/bbstrader)
[![CodeFactor](https://www.codefactor.io/repository/github/bbalouki/bbstrader/badge)](https://www.codefactor.io/repository/github/bbalouki/bbstrader)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-grey?logo=Linkedin&logoColor=white)](https://www.linkedin.com/in/bertin-balouki-simyeli-15b17a1a6/)
[![PayPal Me](https://img.shields.io/badge/PayPal%20Me-blue?logo=paypal)](https://paypal.me/bertinbalouki?country.x=SN&locale.x=en_US)

[Dcoumentation](https://bbstrader.readthedocs.io/en/latest/index.html)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Core Components](#core-components)
- [Getting Started](#getting-started)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Customization and Contribution](#customization-and-contribution)
- [Contributing to `bbstrader`](#contributing-to-bbstrader)
- [Disclaimer](#disclaimer)
- [License](#license)

`bbstrader` is a trading system suite developed for MetaTrader 5 (MT5) and IBKR platforms (coming soon), designed to offer a comprehensive set of tools for developing, backtesting, executing, and managing a wide array of trading strategies. It targets algorithmic traders, quantitative analysts, and developers looking to build, test, and deploy trading strategies. With an emphasis on algorithmic and quantitative trading, `bbstrader` aims to provide users with a robust platform for exploring and deploying sophisticated trading strategies.

## Overview

`bbstrader` aims to empower traders by providing a comprehensive and flexible suite of tools that simplify the development-to-deployment pipeline for algorithmic trading strategies. Our philosophy centers on offering powerful, accessible technology to navigate the complexities of financial markets, enabling users to efficiently design, test, and execute their trading ideas. By focusing on robust analytics and seamless platform integration, `bbstrader` strives to be an indispensable partner for traders seeking to enhance their market analysis and execution capabilities.

## Features

- **Comprehensive Backtesting**: Rigorously test strategies with historical data to optimize performance before live deployment.
- **Integrated Risk Management**: Utilize sophisticated techniques to manage risk and adapt to fluctuating market conditions.
- **Automated Trading Execution**: Seamlessly execute trades on MT5, with real-time management of orders and positions. (IBKR support coming soon).
- **Trade Copier**: Effortlessly replicate trades across multiple accounts.
- **Flexible Strategy Framework**: Customize existing strategies or develop new ones with our adaptable, modular architecture.
- **Advanced Time Series Analysis**: Uncover market patterns and insights with powerful tools for in-depth financial data analysis.
- **Multi-Platform Support**: Designed for MetaTrader 5 with Interactive Brokers (IBKR) integration under active development.

You can read the full documentation [here](https://bbstrader.readthedocs.io/en/latest/index.html)

## Core Components

`bbstrader` is organized into several key modules, each designed to address specific aspects of the trading workflow:

### Backtesting Engine (`btengine`)
The **`btengine`** module enables traders to rigorously test their trading strategies using historical market data. It features an event-driven architecture, provides comprehensive performance metrics, and supports parameter optimization to evaluate and refine strategies before live deployment.

### MetaTrader5 Module (`metatrader`)
This **`metatrader`** module facilitates direct interaction with the MetaTrader 5 platform. It allows for seamless execution of trading strategies, including managing accounts, sending orders, and monitoring positions and balances in real-time.

### Trading Strategies (`trading.strategies`)
The **`trading.strategies`** sub-module offers a collection of pre-built trading strategies, such as those based on ARIMA+GARCH models, Kalman Filters, and Simple Moving Averages. These strategies often come equipped with risk management features, like Hidden Markov Models, and serve as practical examples or starting points for custom development.

### Models Module (`models`)
The **`models`** module provides a versatile framework for implementing and utilizing various types of financial models. This includes statistical models for market analysis, machine learning models for predictive insights, NLP models for sentiment analysis, optimization algorithms for portfolio balancing, and risk management models to safeguard investments.

### Time Series Module (`tseries`)
Specialized for advanced analysis of financial time series, the **`tseries`** module offers tools for cointegration testing, volatility modeling (e.g., GARCH), and various filtering techniques. These capabilities help in identifying market regimes, understanding asset correlations, and forecasting.

### Live Trading Engine (`trading`)
The **`trading`** module serves as a higher-level interface for implementing and managing live trading logic. It coordinates between strategy signals, risk management, and execution modules like `metatrader` and `ibkr` to manage the full lifecycle of trades.

### Interactive Brokers Module (`ibkr`)
Currently under development, the **`ibkr`** module aims to provide integration with the Interactive Brokers platform. It is expected to offer functionalities similar to the `metatrader` module, including account interaction, order execution, and position management for IBKR users.

### Core Utilities (`core`)
The **`core`** module is the backbone of `bbstrader`, providing fundamental data structures, utility functions, configuration management, and shared functionalities. These components are used across the entire `bbstrader` ecosystem to ensure consistency and efficiency.

### Configuration (`config`)
This **`config`** component handles the management of all system settings, including API keys, broker executable paths, database connections, and logging configurations, making it easier to customize `bbstrader` to specific user environments.

### Compatibility Layer (`compat`)
The **`compat`** module is designed to enhance cross-platform development and testing. It achieves this by mocking the MetaTrader 5 environment, allowing developers on non-Windows systems to work with `bbstrader`'s core functionalities without needing a live MT5 instance.

## Getting Started

To begin using `bbstrader`, please ensure your system meets the following prerequisites and follow the installation steps.

### Prerequisites

*   **Python**: Python 3.8+ is required.
*   **MetaTrader 5 (MT5)**:
    *   The MetaTrader 5 platform must be installed on your system (primarily for Windows users needing live trading or direct MT5 interaction).
    *   An active trading account with a MetaTrader 5 broker. `bbstrader` currently supports:
        *   [Admirals Group AS](https://one.justmarkets.link/a/tufvj0xugm/registration/trader) (for Stocks, ETFs, Indices, Commodities, Futures, Forex)
        *   [Just Global Markets Ltd.](https://one.justmarkets.link/a/tufvj0xugm/registration/trader) (for Stocks, Crypto, Indices, Commodities, Forex)
        *   [FTMO](https://trader.ftmo.com/?affiliates=JGmeuQqepAZLMcdOEQRp) (Proprietary Firm)


### Installation

It is highly recommended to install `bbstrader` in a virtual environment to manage dependencies effectively.

1.  **Create and activate a virtual environment:**

    *   On macOS and Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```

2.  **Install `bbstrader`:**

    *   **For Windows users (with MetaTrader 5):**
        To include the official MetaTrader 5 package (which is Windows-only), install using:
        ```bash
        pip install bbstrader[MT5]
        ```
    *   **For macOS, Linux users, or Windows users not needing direct MT5 interaction:**
        Install the base package. The `MetaTrader5` package will be mocked by our `compat` module, allowing development and use of non-MT5 specific features.
        ```bash
        pip install bbstrader
        ```

With these steps completed, you are ready to explore the features and modules of `bbstrader`!

## Usage Examples

This section provides examples of how to use `bbstrader` for various tasks. Remember to replace placeholder values (like account numbers, server names, file paths, and strategy parameters) with your actual details.

### Connecting to MetaTrader 5 (Conceptual)

`bbstrader` scripts and modules that interact with MetaTrader 5 handle the connection process internally, typically based on your configuration (`~/.bbstrader/config/config.ini` or environment variables).

If you were to connect to MetaTrader 5 manually using the `MetaTrader5` library, it would look something like this:

```python
import MetaTrader5 as mt5

# Ensure the MetaTrader 5 terminal is running
# For Windows, specify the path to terminal64.exe
# For Linux/MacOS with Wine, specify the path and use mt5.wine_mode()

# Example for Windows:
# path_to_mt5 = r"C:\Program Files\MetaTrader 5\terminal64.exe"
# if not mt5.initialize(path=path_to_mt5, login=123456, server="YourServer", password="YourPassword"):
# For default path lookup (often sufficient if MT5 is installed and logged in):
if not mt5.initialize(login=123456, server="YourServer", password="YourPassword"):
    print("initialize() failed, error code =", mt5.last_error())
    quit()

# Display account information
account_info = mt5.account_info()
if account_info is not None:
    print(account_info)
else:
    print("Failed to get account info, error code =", mt5.last_error())

# ... your trading logic would go here ...

# Shutdown connection
mt5.shutdown()
```
**Note:** `bbstrader`'s `metatrader` module and execution scripts abstract this process, using configured credentials and settings.

### Programmatic Backtesting Example

You can run backtests on strategies programmatically. The following example demonstrates how to test the `sistbo` (Stock Index Short Term Buy Only) strategy:

```python
from bbstrader.trading.strategies import test_strategy

if __name__ == '__main__':
    # Run backtesting for Stock Index Short Term Buy Only Strategy
    # This function call will use default parameters for the 'sistbo' strategy
    # and save results to the default 'data/results' directory.
    test_strategy(strategy='sistbo')
```
This will typically output performance metrics and charts to a results directory.
### Backtesting Results
![Backtesting Results 1](https://github.com/bbalouki/bbstrader/blob/main/assets/bbs_.png?raw=true)
![Backtesting Results 2](https://github.com/bbalouki/bbstrader/blob/main/assets/qs_metrics_1.png?raw=true)
![Backtesting Results 3](https://github.com/bbalouki/bbstrader/blob/main/assets/qs_metrics_2.png?raw=true)
![Backtesting Results 4](https://github.com/bbalouki/bbstrader/blob/main/assets/qs_plots_1_.png?raw=true)
![Backtesting Results 5](https://github.com/bbalouki/bbstrader/blob/main/assets/qs_plots_2_.png?raw=true)

### Command-Line Interface (CLI) Examples

`bbstrader` provides a CLI for various operations, including running live strategies, backtests, and utilities like the trade copier.

#### CLI - Running a Live Strategy

To run a live strategy, you first need to define its parameters in an `execution.json` file. By default, `bbstrader` looks for this file at `~/.bbstrader/execution/execution.json`.

1.  **Create `execution.json`**:
    Create the directory `~/.bbstrader/execution/` if it doesn't exist. Inside, create `execution.json` with content like this for an `SMAStrategy`:

    ```json
    {
      "SMAStrategy": {
        "MY_MT5_ACCOUNT_1": {
          "symbol_list": ["EURUSD", "GBPUSD"],
          "trades_kwargs": {"magic": 12345, "comment": "SMAStrategy_Live"},
          "short_window": 20,
          "long_window": 50,
          "time_frame": "H1",
          "quantities": 0.1
        }
      }
    }
    ```
    Replace `MY_MT5_ACCOUNT_1` with your account identifier used in `bbstrader`'s configuration. Adjust strategy parameters as needed.

2.  **Run the strategy via CLI**:
    Open your terminal and run:
    ```bash
    python -m bbstrader --run execution -s SMAStrategy -a MY_MT5_ACCOUNT_1
    ```
    *   `-s SMAStrategy`: Specifies the strategy class name to run.
    *   `-a MY_MT5_ACCOUNT_1`: Specifies the account name (must match a key in `execution.json` under the strategy).

    The `SMAStrategy` (and other built-in strategies) should be discoverable by Python as they are part of the `bbstrader` package. For custom strategies, ensure they are in your `PYTHONPATH` or use the `-p` option to specify the directory.

#### CLI - Running a Backtest

You can also initiate backtests via the CLI. This is useful for quick tests or integrating into automated scripts.

To see all available options for backtesting:
```bash
python -m bbstrader --run backtest --help
```

Example command to backtest an `SMAStrategy`:
```bash
python -m bbstrader --run backtest --strategy SMAStrategy
```

#### CLI - Trade Copier

`bbstrader` includes a trade copier utility to replicate trades between different MetaTrader 5 accounts.

To see the available options for the trade copier:
```bash
python -m bbstrader --run copier --help
```
This will display detailed instructions on how to specify source and target accounts, along with other relevant parameters for copying trades.

## Configuration

`bbstrader` uses a combination of user-defined JSON files and internal Python scripts for configuration. Understanding these will help you customize the system to your needs.

### User Configuration Directory: `~/.bbstrader/`

`bbstrader` uses a hidden directory in your user's home folder, `~/.bbstrader/`, to store user-specific files. This typically includes:
*   `execution/execution.json`: For live strategy execution parameters.
*   `logs/`: Default directory for log files.
*   Potentially other configuration files for different modules in the future.

You may need to create the `~/.bbstrader/` directory and its subdirectories (like `execution/` or `logs/`) manually if they don't exist upon first use.

### Strategy Execution (`execution.json`)

*   **Purpose**: Defines parameters for live strategy execution when using the `python -m bbstrader --run execution` command.
*   **Default Location**: `~/.bbstrader/execution/execution.json`. You'll likely need to create this file and its parent directory.
*   **Structure**:
    *   The file is a JSON object where top-level keys are strategy class names (e.g., `"SMAStrategy"`).
    *   Each strategy key contains another JSON object where keys are your account identifiers (e.g., `"MY_MT5_ACCOUNT_1"`). These account identifiers should match those you've configured for MT5 connections (often set via environment variables or a central configuration not detailed here but handled by the `metatrader` module).
    *   Under each account, you specify:
        *   `"symbol_list"`: A list of symbols the strategy will trade (e.g., `["EURUSD", "GBPUSD"]`).
        *   `"trades_kwargs"`: A dictionary for MetaTrader 5 specific order parameters, commonly including:
            *   `"magic"`: The magic number for orders placed by this strategy instance.
            *   `"comment"`: A comment for orders.
        *   Custom strategy parameters: Any other parameters required by your strategy's `__init__` method (e.g., `"short_window"`, `"long_window"`, `"time_frame"`, `"quantities"`).
*   **Example**: Refer to the example in the "Usage Examples" -> "CLI - Running a Live Strategy" section.

### MetaTrader 5 Broker Paths (`bbstrader/config.py`)

*   The file `bbstrader/config.py` within the installed package contains a dictionary named `BROKERS_PATHS`. This dictionary maps broker shortnames (e.g., "AMG", "FTMO") to the default installation paths of their MetaTrader 5 `terminal64.exe`.
*   **Customization**:
    *   If your MT5 terminal is installed in a non-standard location, or you use a broker not listed, `bbstrader` might not find the terminal.
    *   Ideally, future versions might support environment variables or a user-specific configuration file to override these paths.
    *   Currently, the most direct way to change these is by modifying `bbstrader/config.py` in your Python environment's `site-packages` directory. This should be done with caution as changes might be overwritten during package updates.
    *   Alternatively, when initializing `MetaTrader5` in your custom scripts, you can often pass the `path` argument directly to `mt5.initialize(path="C:\\path\\to\\your\\terminal64.exe", ...)`. `bbstrader`'s internal scripts might not use this method by default.

### Logging Configuration (`bbstrader/config.py`)

*   **Setup**: The `config_logger` function in `bbstrader/config.py` sets up application-wide logging.
*   **Log Files**:
    *   By default, logs are typically saved to a file within the `~/.bbstrader/logs/` directory (e.g., `bbstrader.log`). The exact path might depend on how `config_logger` is invoked by the application.
    *   The default file logging level is `INFO`.
*   **Console Logging**:
    *   If enabled (usually by default for CLI operations), console logging is set to `DEBUG` level, providing more verbose output.
*   **Customization**:
    *   You can modify logging behavior (e.g., log levels, output formats, log file location) by editing the `config_logger` function in `bbstrader/config.py` within your `site-packages`. This is subject to the same caveats as modifying broker paths (potential overwrites on update).
    *   For programmatic use, you can re-configure logging after importing `bbstrader` modules if needed, though this might affect internal `bbstrader` logging.

### General Advice

*   For detailed configuration options specific to certain modules or advanced use cases, always refer to the official `bbstrader` documentation (if available) or consult the source code of the respective modules.
*   Keep an eye on the `~/.bbstrader/` directory for any new configuration files or logs that might appear as you use different features.

## Documentation

For comprehensive information, including detailed API references, tutorials, and advanced guides for each module, please refer to our full documentation hosted on ReadTheDocs:

[**View the Full Documentation**](https://bbstrader.readthedocs.io/en/latest/)

Additionally, the codebase is commented and includes docstrings, which can be a valuable resource for understanding the implementation details of specific functions and classes.

## Customization and Contribution

`bbstrader`'s modular design allows for easy customization and extension. Traders and developers are encouraged to modify existing strategies, add new ones, or enhance the system's capabilities. Contributions to the `bbstrader` project are welcome.

## Contributing to `bbstrader`

We warmly welcome contributions from the trading and development community! Whether you're interested in fixing bugs, adding new features, or improving documentation, your help is invaluable to making `bbstrader` more robust and versatile. Here's how you can contribute:

### Ways to Contribute

- **Develop New Strategies**: Implement and share your unique trading strategies or models.
- **Enhance Existing Modules**: Optimize the performance, extend the functionality, or improve the usability of existing modules.
- **Report Bugs**: Identify and report bugs to help us improve the system's stability and performance. (See "Reporting Issues" below).
- **Improve Documentation**: Contribute to the project's documentation for clearer guidance and better usability.
- **Share Insights and Best Practices**: Provide examples, tutorials, or best practices on utilizing `bbstrader` effectively.
- **Request Features**: Suggest new functionalities or improvements. (See "Requesting Features" below).

### Reporting Issues

If you encounter a bug or unexpected behavior, please help us by reporting it on GitHub Issues. A well-detailed bug report makes it easier and faster to identify and fix the problem.

[**Report an Issue on GitHub**](https://github.com/bbalouki/bbstrader/issues)

Please include the following in your bug report:

*   **Clear Title**: A concise and descriptive title for the issue.
*   **Steps to Reproduce**: Detailed steps that consistently reproduce the bug.
*   **Expected Behavior**: What you expected to happen.
*   **Actual Behavior**: What actually happened.
*   **Environment Details**:
    *   Python version (e.g., `python --version`).
    *   `bbstrader` version (e.g., `pip show bbstrader`).
    *   Operating System (e.g., Windows 10, Ubuntu 22.04, macOS Sonoma).
*   **Logs and Error Messages**: Any relevant console output, error messages, or snippets from log files (`~/.bbstrader/logs/`). Please use code blocks for formatting.

### Requesting Features

We are always open to suggestions for new features and improvements! If you have an idea that could make `bbstrader` better, please submit it via GitHub Issues.

[**Request a Feature on GitHub**](https://github.com/bbalouki/bbstrader/issues)

When submitting a feature request, please:

*   **Clear Title**: A concise and descriptive title for the feature.
*   **Describe the Feature**: Clearly explain the proposed functionality and its potential benefits to users.
*   **Use Case / Problem Solved**: Describe the specific scenario or problem this feature would address.
*   **Suggested Implementation (Optional)**: If you have ideas on how the feature could be implemented, feel free to share them.

### How to Get Started

1. **Fork the Repository**: Start by forking the `bbstrader` repository to your GitHub account.
2. **Clone Your Fork**: Clone your forked repository to your local machine to start making changes.
3. **Set Up Your Development Environment**: Ensure you have the necessary development environment set up, including Python, MetaTrader 5, and any dependencies.
4. **Create a New Branch**: Make your changes in a new git branch, branching off from the main branch.
5. **Implement Your Changes**: Work on bug fixes, features, or documentation improvements.
6. **Test Your Changes**: Ensure your changes do not introduce new issues and that they work as intended.
7. **Submit a Pull Request**: Once you're ready, submit a pull request (PR) against the main `bbstrader` repository. Include a clear description of the changes and any other relevant information.

### Contribution Guidelines

Please adhere to the following guidelines to ensure a smooth contribution process:

- **Follow the Coding Standards**: Write clean, readable code and follow the coding conventions used throughout the project.
- **Document Your Changes**: Add comments and update the README.md files as necessary to explain your changes or additions.
- **Respect the License**: All contributions are subject to the MIT License under which `bbstrader` is distributed.

We're excited to see your contributions and to welcome you to the `bbstrader` community. Together, we can build a powerful tool that serves the needs of traders around the world.


## Disclaimer

Trading financial instruments involves a high level of risk and may not be suitable for all investors. The developers of `bbstrader` are not responsible for any financial losses incurred from the use of this software. Trade responsibly and at your own risk.

## License
`bbstrader` is open source and available under the MIT License.
