import pytest
from mtrader5.account import Account
import pandas as pd

# Make sure an instance of MT5 is running with a demo account
def test_get_account_info():
    account = Account()
    info = account.get_account_info()
    assert info is not None
    assert hasattr(info, "balance")

def test_get_terminal_info():
    account = Account()
    info = account.get_terminal_info()
    assert isinstance(info, pd.DataFrame)
    assert 'property' in info.columns
    assert 'value' in info.columns

def test_get_symbols():
    account = Account()
    symbols = account.get_symbols()
    assert isinstance(symbols, list)
    assert len(symbols) > 0

def test_get_symbol_info():
    account = Account()
    symbols = account.get_symbols()
    assert isinstance(symbols[0], str)
    symbol_info = account.get_symbol_info(symbols[0])
    assert symbol_info is not None
    assert hasattr(symbol_info, 'ask')
    assert hasattr(symbol_info, "bid")

def test_get_orders():
    account = Account()
    # This test assumes there are open orders. 
    # If not, it will fail.
    orders = account.get_orders()
    if orders is not None:
        assert isinstance(orders, pd.DataFrame)
        assert len(orders) > 0

def test_get_positions():
    account = Account()
    # This test assumes there are open positions. 
    # If not, it may fail or need adjustment.
    positions = account.get_positions()
    if positions is not None:
        assert isinstance(positions, pd.DataFrame)
        assert len(positions) > 0

def test_get_trade_history():
    account = Account()
    history = account.get_trade_history()
    if history is not None:
        assert isinstance(history, pd.DataFrame)
        assert len(history) > 0