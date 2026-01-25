"""
Overview
========

This Backtesting Module provides a comprehensive suite of tools to test trading strategies in an event-driven system. 
It simulates the execution of trades in historical market conditions to evaluate the performance of trading strategies 
before applying them in live trading environments. Designed with modularity and extensibility in mind, it caters to 
both novices and experts in algorithmic trading.

Features
========

- **Event-Driven Architecture**: Processes market data, generates signals, executes orders, and manages portfolio updates in response to events, closely mimicking live trading environments.
- **Historical Market Data Support**: Utilizes historical OHLCV data from CSV files, Yahoo finance and MT5 terminal allowing for the testing of strategies over various market conditions and time frames.
- **Performance Metrics Calculation**: Includes tools for calculating key performance indicators, such as `Sharpe Ratio`, `Sortino Ratio`, and `drawdowns`, to evaluate the effectiveness of trading strategies.
- **Visualization**: Generates plots of the `equity curve`, `returns`, `drawdowns`, and other metrics for comprehensive strategy `performance analysis`.

Components
==========

- **BacktestEgine**: Orchestrates the backtesting process, managing events and invoking components.
- **Event**: Abstract class for events, with implementations for market data, signals, fill and order events.
- **DataHandler**: Abstract class for market data handling, with an implementation for `CSVDataHandler`, `MT5DataHandler`, `YFDataHandler`. We will add another data handling in the future such as MacroEconomic Data, Fundamental Data, TICK Data and Real-time Data.
- **Portfolio**: Manages positions and calculates performance metrics, responding to market data and signals.
- **ExecutionHandler**: Abstract class for order execution, with a simulated execution handler provided with an implementation for `SimExecutionHandler`.
- **Performance**: Utility functions for calculating performance metrics and visualizing strategy performance.

Examples
========

>>> from bbstrader.btengine import run_backtest
>>> from datetime import datetime
>>> run_backtest(
...     symbol_list=['AAPL', 'GOOG'],
...     start_date=datetime(2020, 1, 1),
...     data_handler=DataHandler,
...     strategy=Strategy,
...     exc_handler=ExecutionHandler,
...     initial_capital=500000.0,
...     heartbeat=1.0
... )

Notes
=====

See `bbstrader.btengine.backtest.run_backtest` for more details on the backtesting process and its parameters.
"""
from bbstrader.btengine.backtest import *  # noqa: F403
from bbstrader.btengine.data import *  # noqa: F403
from bbstrader.btengine.event import *  # noqa: F403
from bbstrader.btengine.execution import *  # noqa: F403
from bbstrader.btengine.performance import *  # noqa: F403
from bbstrader.btengine.portfolio import *  # noqa: F403
from bbstrader.btengine.strategy import *  # noqa: F403
