MetaTrader5 (MT5) Trading
=========================

Overview
--------

The MT5 Trading Module is a Python package designed to revolutionize algorithmic trading on the MetaTrader5 (MT5) platform. This module is born out of the realization that while traditional assets have been the mainstay of trading for many, Contracts for Difference (CFDs) present an untapped avenue with immense potential. Leveraging the MetaTrader5 platform's capabilities, this module demystifies the use of leverage in CFD trading, presenting it as a powerful tool when wielded with knowledge and precision.

The integration of MT5 with Python opens up a world of possibilities for traders, allowing for the development of sophisticated trading strategies that were previously inaccessible to the individual trader. This module capitalizes on Python's analytical prowess to provide a robust framework for executing and managing trades with greater efficiency and precision.

Understanding the high entry barriers in traditional asset trading, this module also highlights the lower capital requirements of CFDs, making advanced trading accessible to a wider audience. With CFDs, traders can gain exposure to the price movements of major assets without the need for substantial upfront capital, leveling the playing field for individual traders and small institutions alike.

Features
--------

- **Leverage Strategy Optimization**: Navigate the high-reward potential of leverage in CFD trading with strategies designed to maximize gains while mitigating risks.
- **Sophisticated Strategy Development**: Utilize Python's extensive libraries and tools to develop, test, and implement complex trading strategies.
- **Lower Capital Requirement**: Engage in trading with significantly lower capital compared to traditional assets, with access to the same market opportunities.
- **Comprehensive Trading Toolkit**: From trade execution to risk management, this module offers a complete suite of tools to empower traders at all levels.
- **Free and Integrated Platform**: Benefit from the no-cost MT5 platform, seamlessly integrated with Python for an enriched trading experience.
- **Trade Execution**: Simplify the process of opening and closing buy/sell positions with advanced order types, including support for stop loss, take profit, and deviation parameters.
- **Risk Management**: Implement robust risk management strategies using the integrated `risk.py` module, which calculates optimal lot sizes, stop loss, and take profit levels based on predefined risk parameters.
- **Market Data Access**: Retrieve real-time rates and historical data for analysis and strategy backtesting.
- **Account Management**: Easily access and display account information, including balance, equity, margin, and profit, to monitor trading performance and make informed decisions.
- **Symbol Information**: Query detailed symbol information, such as trading conditions, costs, and constraints, essential for strategy development and optimization.

Installation
------------

Before you can use the MT5 Trading Module, you need to have MetaTrader 5 (MT5) installed on your computer and an active MT5 trading account. This module currently supports two brokers:

- For trading `Stocks`, `ETFs`, `Indices`, `Commodities`, `Futures`, and `Forex`, see `Admirals Group AS`_.
- For trading `Stocks`, `Crypto`, `Indices`, `Commodities`, and `Forex`, see `Just Global Markets Ltd.`_.
- If you are looking for a prop firm, see `FTMO`_.

Then, you can install `bbstrader` using pip:

.. code-block:: bash
   
   pip install bbstrader # Mac or Linux
   pip install bbstrader[MT5] # Windows

Usage
-----

The module consists of several components, each responsible for different aspects of trading on the MT5 platform:

- **Account Management** (`Account()`): Access and manage your MT5 account information, symbols and symbol information, trades and orders information, trades and orders history.
- **Market Data** (`Rates()`): Fetch real-time and historical market data.
- **Risk Management** (`RiskManagement()`): Apply risk management strategies to your trading.
- **Trade Execution** (`Trade()`): Execute trades based on your strategies.
- **Trade Copier** (`TradeCopier()`): Copy trades from one account to another.

Customization
-------------

You can customize and extend the module to fit your trading strategies and requirements. Implement custom trading strategies by subclassing and overriding methods in the `Trade` class. Adjust risk parameters in the `risk.py` module to align with your risk tolerance and trading goals.


.. _Admirals Group AS: https://cabinet.a-partnership.com/visit/?bta=35537&brand=admiralmarkets
.. _Just Global Markets Ltd.: https://one.justmarkets.link/a/tufvj0xugm/registration/trader
.. _FTMO: https://trader.ftmo.com/?affiliates=JGmeuQqepAZLMcdOEQRp


.. automodule:: metatrader
   :members:
   :undoc-members:
   :show-inheritance:


Aaccount
-------

.. automodule:: metatrader.account
   :members:
   :undoc-members:
   :show-inheritance:

TradeCopier
------------

.. automodule:: metatrader.copier
   :members:
   :show-inheritance:
   :undoc-members:

Rates
-----

.. automodule:: metatrader.rates
   :members:
   :undoc-members:
   :show-inheritance:

RiskManagement
--------------

.. automodule:: metatrader.risk
   :members:
   :undoc-members:
   :show-inheritance:

scripts
-------

.. automodule:: metatrader.scripts
   :members:
   :show-inheritance:
   :undoc-members:

Trade
-----

.. automodule:: metatrader.trade
   :members:
   :undoc-members:
   :show-inheritance:

utils
------

.. automodule:: metatrader.utils
   :members:
   :undoc-members:
   :show-inheritance:
