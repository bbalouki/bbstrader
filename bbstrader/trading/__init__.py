"""
Overview
========

The Trading Module is responsible for the execution of trading strategies. It provides a
structured framework for implementing and managing trading strategies, from signal generation
to order execution. This module is designed to be flexible and extensible, allowing for the
customization of trading logic and integration with various execution handlers.

Features
========

- **Strategy Execution Framework**: Defines a clear structure for creating and executing trading strategies.
- **Signal Generation**: Supports the generation of trading signals based on market data and strategy logic.
- **Order Management**: Manages the creation and execution of orders based on generated signals.
- **Extensibility**: Allows for the implementation of custom strategies and execution handlers.

Components
==========

- **Execution**: Handles the execution of trades, with a base class for creating custom execution handlers.
- **Strategy**: Defines the core logic of the trading strategy, including signal generation and order creation.
- **Utils**: Provides utility functions to support the trading process.

Examples
========

>>> from bbstrader.trading import MovingAverageCrossStrategy, SimulatedExecutionHandler
>>> from bbstrader.btengine.data import CsvDataHandler
>>> from bbstrader.btengine.portfolio import Portfolio
>>> from datetime import datetime
>>> data_handler = CsvDataHandler(csv_dir='path/to/csv', symbol_list=['AAPL'])
>>> portfolio = Portfolio(data_handler, 100000.0)
>>> execution_handler = SimulatedExecutionHandler()
>>> strategy = MovingAverageCrossStrategy(data_handler, portfolio, execution_handler)
>>> # Run the strategy within a backtesting or live trading loop

Notes
=====

This module can be used in both backtesting and live trading environments by swapping out the
execution handler.
"""
from bbstrader.trading.execution import *  # noqa: F403
from bbstrader.trading.strategy import *  # noqa: F403
