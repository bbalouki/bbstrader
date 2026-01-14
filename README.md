# Simplified Investment & Trading Toolkitlkit

[![Documentation Status](https://readthedocs.org/projects/bbstrader/badge/?version=latest)](https://bbstrader.readthedocs.io/en/latest/?badge=latest)
[![PYPI Version](https://img.shields.io/pypi/v/bbstrader)](https://pypi.org/project/bbstrader/)
[![PyPi status](https://img.shields.io/pypi/status/bbstrader.svg?maxAge=60)](https://pypi.python.org/pypi/bbstrader)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/bbstrader)](https://pypi.org/project/bbstrader/)
[![PyPI Downloads](https://static.pepy.tech/badge/bbstrader)](https://pepy.tech/projects/bbstrader)
[![CodeFactor](https://www.codefactor.io/repository/github/bbalouki/bbstrader/badge)](https://www.codefactor.io/repository/github/bbalouki/bbstrader)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-grey?logo=Linkedin&logoColor=white)](https://www.linkedin.com/in/bertin-balouki-s-15b17a1a6)

### **[Get Premium Data Packs]([bertin@bbs-trading.com])** | **[Need Custom Strategies](mailto:[bertin@bbs-trading.com])** | ☕ **[Support the Dev](https://paypal.me/bertinbalouki?country.x=SN&locale.x=en_US)**

## Why bbstrader?

`bbstrader` is not just another Python wrapper. It is an **institutional-grade** trading system designed for **speed and accuracy**.

- **Financial Rigor:** Built by a developer with a background in **Finance & Accounting**, ensuring that PnL calculations, fees, and margin logic are mathematically precise.
- **Proven Scale:** Trusted by **51,000+** users for backtesting and live execution.

## Professional Services & Consulting

Do you need to move faster? I offer specialized services for traders and funds:

### 1. Custom Strategy Implementation

Stop struggling with code. I will translate your manual strategy into a high-speed automated bot.

- **Deliverable:** A fully backtested, live-ready Python bot.
- **Cost:** Starting at **$200**.
- **[Click Here to Request a Quote](mailto:[bertin@bbs-trading.com])**

### 2. High-Performance Optimization

Is your current backtest taking hours?

- I can optimize your Python code using C++ bindings to reduce backtest time from **hours to minutes**.
- Perfect for HFT (High-Frequency Trading) or heavy Multi-Asset strategies.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Core Components](#core-components)
- [Getting Started](#getting-started)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Disclaimer](#disclaimer)
- [License](#license)

## Overview

`bbstrader` aims to empower traders by providing a comprehensive and flexible suite of tools that simplify the development-to-deployment pipeline for algorithmic trading strategies. Our philosophy centers on offering powerful, accessible technology to navigate the complexities of financial markets.

## Features

- **High-Speed Backtesting**: Rigorously test strategies with historical data using event-driven architecture.
- **Integrated Risk Management**: Utilize sophisticated techniques (Kalman Filters, HMM) to manage risk.
- **Automated Execution**: Seamlessly execute trades on **MetaTrader 5**, with real-time management. (IBKR support in dev).
- **Trade Copier**: Effortlessly replicate trades across multiple accounts.
- **Advanced Time Series**: Cointegration testing, GARCH volatility modeling, and regime detection.

You can read the full documentation [here](https://bbstrader.readthedocs.io/en/latest/index.html)

## Core Components

`bbstrader` is organized into several key modules:

### Backtesting Engine (`btengine`)

Features an event-driven architecture, comprehensive performance metrics (Sharpe, Sortino, MaxDD), and parameter optimization.

### MetaTrader5 Module (`metatrader`)

Facilitates direct interaction with the MT5 platform. Manage accounts, send orders, and monitor positions in real-time.

### Models Module (`models`)

A framework for financial modeling, including:

- **Statistical Models:** ARIMA+GARCH, Cointegration.
- **Machine Learning:** LSTM/Transformers for prediction (via plugins).
- **Risk:** Value at Risk (VaR), CVaR.

### Time Series Module (`tseries`)

Specialized for advanced analysis: Cointegration testing, volatility modeling, and filtering techniques.

## Getting Started

### Prerequisites

- **Python**: Python 3.12+ is required.
- **MetaTrader 5 (MT5)**: Required for live execution (Windows).
- **Brokers Supported**: [Admirals](https://one.justmarkets.link/a/tufvj0xugm/registration/trader), [JustMarkets](https://one.justmarkets.link/a/tufvj0xugm/registration/trader), [FTMO](https://trader.ftmo.com/?affiliates=JGmeuQqepAZLMcdOEQRp).

### Installation

It is highly recommended to install `bbstrader` in a virtual environment.

```bash
# Create virtual env
python -m venv venv
# Activate
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install
pip install bbstrader[MT5]  # Windows with MT5
pip install bbstrader       # Linux/Mac (Backtesting only)
```

## Usage Examples

### Programmatic Backtesting Example

```python
from bbstrader.trading.strategies import test_strategy

if __name__ == '__main__':
    # Run backtesting for Stock Index Short Term Buy Only Strategy
    test_strategy(strategy='sistbo')
```

### Backtesting Results

![Backtesting Results](https://github.com/bbalouki/bbstrader/blob/main/assets/qs_metrics_1.png?raw=true)

### Command-Line Interface (CLI)

**Run a Live Strategy:**

```bash
python -m bbstrader --run execution -s SMAStrategy -a MY_MT5_ACCOUNT_1
```

**Run a Backtest:**

```bash
python -m bbstrader --run backtest --strategy SMAStrategy
```

## Configuration

`bbstrader` uses a `~/.bbstrader/` directory in your user home folder for configuration and logs.

**Strategy Execution (`execution.json`):**
Define your strategy parameters in `~/.bbstrader/execution/execution.json`:

```json
{
  "SMAStrategy": {
    "MY_MT5_ACCOUNT_1": {
      "symbol_list": ["EURUSD", "GBPUSD"],
      "trades_kwargs": { "magic": 12345, "comment": "SMA_Live" },
      "short_window": 20,
      "long_window": 50
    }
  }
}
```

## Documentation

For comprehensive information, including detailed API references and math explanations:
[**View the Full Documentation**](https://bbstrader.readthedocs.io/en/latest/)

## Contributing

We welcome contributions! Whether you are a developer or a trader:

1.  Fork the repository.
2.  Create a feature branch.
3.  Submit a Pull Request.

_Note: If you find a security vulnerability, please email the maintainer directly._

## Disclaimer

Trading financial instruments involves a high level of risk. The developers of `bbstrader` are not responsible for any financial losses incurred from the use of this software.

## License

`bbstrader` is open source and available under the MIT License.
