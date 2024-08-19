"""
## MetaTrader5 (MT5) Trading Module

The MT5 Trading Module is a Python package designed to revolutionize 
algorithmic trading on the MetaTrader5 (MT5) platform.

The integration of MT5 with Python allows traders for the development of sophisticated 
trading strategies that were previously inaccessible to them. 

This module capitalizes on Python's analytical prowess to provide a robust framework 
for executing and managing trades with greater efficiency and precision.
"""

from bbstrader.metatrader.account import Account
from bbstrader.metatrader.rates import Rates
from bbstrader.metatrader.risk import RiskManagement
from bbstrader.metatrader.trade import Trade, create_trade_instance
from bbstrader.metatrader.utils import *