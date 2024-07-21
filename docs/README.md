# Simplified Investment & Trading Toolkit
![bbstrader](../assets/BBSTrader.png)

## Overview

BBSTrader is a trading system suite developed for MetaTrader 5 (MT5) and IBKR platforms (comming soon), designed to offer a comprehensive set of tools for developping, backtesting, executing, and managing a wide array of trading strategies. With an emphasis on algorithmic and quantitative trading, BBSTrader integrates advanced statistical models, machine learning algorithms, to provide traders with a robust platform for exploring and deploying sophisticated trading strategies.

## Modules Overview

`bbstrader` is comprised of several key modules, each focus on specific aspects of trading strategy development and execution:

- [Backtesting Module](btengine/README.md): Enables traders to rigorously test their trading strategies using historical data to evaluate performance before live deployment.
- [Trading Strategies Module](strategies/README.md): A collection of predefined trading strategies, including ARIMA+GARCH models, Kalman Filters, Ornstein-Uhlenbeck processes, and Simple Moving Averages, equipped with risk management through Hidden Markov Models.
- **MT5 Execution Module**: Facilitates the direct execution of trading strategies on the MetaTrader 5 platform, supporting real-time trading across multiple financial instruments.
- [Modles Module](models/README.md): Serves as a framework for implementing various types of financial models (risk managment models, Machine learing models etc).

## Features

- **Comprehensive Backtesting**: Assess the performance of trading strategies with historical market data to optimize parameters and strategies for live trading environments.
- **Diverse Trading Strategies**: From statistical and econometric models to machine learning predictions and traditional technical analysis, explore a wide range of trading strategies.
- **Integrated Risk Management**: Leverage advanced risk management techniques to adapt to changing market conditions and maintain control over risk exposure.
- **Automated Trading**: Execute trades automatically on the MT5 platform, with support for managing orders, positions, and risk in real-time.
- **Flexible Framework**: Customize existing strategies or develop new ones with the flexible, modular architecture designed to accommodate traders' evolving needs.

## Getting Started

1. Ensure MetaTrader 5 is installed and configured on your system for automated trading.
2. Install Python 3.8+ and the required dependencies listed in each module's documentation.
3. Clone the `bbstrader` repository to your local machine.
4. Configure each module according to your trading strategy and risk management preferences.

## Usage

Refer to the README.md file of each module within `bbstrader` for detailed instructions on configuring and running the individual components. Customize strategy parameters, backtest with historical data, and execute trades on MT5 as per your requirements.

## Examples
### Backtesting Module
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
### Backtesting Results
#### Hidden States
![Hidden States](../bbstrader/btengine/assets/klf_hidden_states.png)

#### Performance
![Performance](../bbstrader/btengine/assets/klf_performance.png)

#### Risk Return
![Risk Return](../bbstrader/btengine/assets/klf_risk_return.png)

#### Returns Distribution
![Returns Distribution](../bbstrader/btengine/assets/klf_returns_distribution.png)

#### Portfolio
![Portfolio](../bbstrader/btengine/assets/klf_summary_stats.png)

## Customization and Contribution

`bbstrader`'s modular design allows for easy customization and extension. Traders and developers are encouraged to modify existing strategies, add new ones, or enhance the system's capabilities. Contributions to the `bbstrader` project are welcome.

## Contributing to BBSTrader

We warmly welcome contributions from the trading and development community! Whether you're interested in fixing bugs, adding new features, or improving documentation, your help is invaluable to making `bbstrader` more robust and versatile. Here's how you can contribute:

### Ways to Contribute

- **Develop New Strategies**: Implement and share your unique trading strategies or models.
- **Enhance Existing Modules**: Optimize the performance, extend the functionality, or improve the usability of existing modules.
- **Report Bugs**: Identify and report bugs to help us improve the system's stability and performance.
- **Improve Documentation**: Contribute to the project's documentation for clearer guidance and better usability.
- **Share Insights and Best Practices**: Provide examples, tutorials, or best practices on utilizing `bbstrader` effectively.

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