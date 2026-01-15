"""
Overview
========

The Core Module provides the fundamental building blocks and abstract base classes for the
trading system. It defines the essential components that are extended by other modules to
create a complete trading application, ensuring a consistent and modular architecture.

Features
========

- **Abstract Base Classes**: Defines the interfaces for key components like data handlers and strategies, promoting a standardized approach to development.
- **Modularity**: Enforces a modular design by providing a clear separation of concerns between data handling, strategy logic, and execution.
- **Extensibility**: Designed to be easily extended with concrete implementations, allowing for the creation of custom data sources and trading strategies.

Components
==========

- **Data**: Contains the abstract base class `DataHandler`, which defines the interface for managing market data from various sources.
- **Strategy**: Contains the abstract base class `Strategy`, which provides the framework for developing trading strategies.

Examples
========

>>> from bbstrader.core import DataHandler, Strategy
>>> class CustomDataHandler(DataHandler):
...     def get_latest_bars(self, symbol, N=1):
...         pass
...     def update_bars(self):
...         pass
>>> class CustomStrategy(Strategy):
...     def calculate_signals(self, event):
...         pass

Notes
=====

This module contains the abstract classes that form the foundation of the trading system.
Implementations of these classes can be found in other modules like `btengine` and `trading`.
"""
from bbstrader.core.data import *  # noqa: F403
from bbstrader.core.strategy import * # noqa: F403
