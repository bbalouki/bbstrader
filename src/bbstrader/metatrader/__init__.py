"""
Overview
========

This MetaTrader Module provides a direct interface to the MetaTrader 5 trading platform,
enabling seamless integration of Python-based trading strategies with a live trading environment.
It offers a comprehensive set of tools for account management, trade execution, market data retrieval,
and risk management, all tailored for the MetaTrader 5 platform.

Features
========

- **Direct MetaTrader 5 Integration**: Connects to the MetaTrader 5 terminal to access its full range of trading functionalities.
- **Account and Trade Management**: Provides tools for querying account information, managing open positions, and executing trades.
- **Market Data Retrieval**: Fetches historical and real-time market data, including rates and ticks, directly from MetaTrader 5.
- **Risk Management**: Includes utilities for managing risk, such as setting stop-loss and take-profit levels.
- **Trade Copying**: Functionality to copy trades between different MetaTrader 5 accounts.

Components
==========

- **Account**: Manages account information, including balance, equity, and margin.
- **Broker**: Handles the connection to the MetaTrader 5 terminal.
- **Copier**: Copies trades between accounts.
- **Rates**: Retrieves historical and current market rates.
- **Risk**: Provides risk management functionalities.
- **Trade**: Manages trade execution and position management.
- **Utils**: Contains utility functions for the MetaTrader module.

Examples
========

>>> from bbstrader.metatrader import Account
>>> account = Account()
>>> print(account.get_account_info())

Notes
=====

This module requires the MetaTrader 5 terminal to be installed and running.
"""

from bbstrader.metatrader.account import *  # noqa: F403
from bbstrader.metatrader.rates import *  # noqa: F403
from bbstrader.metatrader.risk import *  # noqa: F403
from bbstrader.metatrader.trade import *  # noqa: F403
from bbstrader.metatrader.utils import *  # noqa: F403
from bbstrader.metatrader.copier import *  # noqa: F403
