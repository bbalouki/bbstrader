"""
Overview
========

The API Module provides a high-level client for interacting with the MetaTrader 5 platform.
It serves as the primary interface for connecting to, retrieving data from, and sending commands
to a MetaTrader 5 terminal from a Python application. The module is designed to simplify
interactions and provide convenient data handling.

Features
========

- **High-Level MT5 Client**: A simplified client (`Mt5client`) for easy access to MetaTrader 5 functionalities.
- **Dynamic Object Representation**: Automatically patches MetaTrader 5 data objects for better string representation, making debugging easier.
- **DataFrame Conversion**: Includes a utility function (`trade_object_to_df`) to quickly convert lists of MT5 trade objects (like deals, orders, positions) into pandas DataFrames for analysis.
- **Handler-Based Architecture**: Uses a handler class (`Mt5Handlers`) to manage the underlying MetaTrader 5 API calls, promoting modularity.

Components
==========

- **Mt5client**: The main client instance used to interact with the MetaTrader 5 terminal.
- **Mt5Handlers**: A class that encapsulates the direct calls to the MetaTrader 5 API.
- **Helper Functions**:
  - `trade_object_to_df`: Converts lists of trade-related objects to pandas DataFrames.
  - Dynamic patching of `__str__` and `__repr__` for improved object inspection.

Notes
=====

This module requires a running MetaTrader 5 terminal and the `MetaTrader5` Python package to be installed.
The connection is managed by the `Mt5client`.
"""

from operator import attrgetter

import pandas as pd

from bbstrader.api.handlers import Mt5Handlers
from bbstrader.api.client import *  # type: ignore # noqa: F403

# ruff: noqa: F405
classes_to_patch = [
    AccountInfo,
    BookInfo,
    OrderCheckResult,
    OrderSentResult,
    RateInfo,
    SymbolInfo,
    TerminalInfo,
    TickInfo,
    TradeDeal,
    TradeOrder,
    TradePosition,
    TradeRequest,
]


def dynamic_str(self):
    fields = set()
    for name in dir(self):
        if name.startswith("_"):
            continue
        try:
            value = getattr(self, name)
            if not callable(value):
                fields.add(f"{name}={value!r}")
        except Exception:
            pass
    return f"{type(self).__name__}({', '.join(fields)})"


for cls in classes_to_patch:
    cls.__str__ = dynamic_str
    cls.__repr__ = dynamic_str


def trade_object_to_df(obj_list):
    """
    Fast conversion of a list of C++ bound objects to a pandas DataFrame.
    """
    if not obj_list:
        return pd.DataFrame()

    first_obj = obj_list[0]
    columns = [
        name
        for name in dir(first_obj)
        if not name.startswith("_") and not callable(getattr(first_obj, name))
    ]
    fetcher = attrgetter(*columns)
    data = [fetcher(obj) for obj in obj_list]
    df = pd.DataFrame(data, columns=columns)
    return df


Mt5client = MetaTraderClient(Mt5Handlers)
