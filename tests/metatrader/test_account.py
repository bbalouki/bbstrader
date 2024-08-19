import pytest
import unittest
import pandas as pd
import MetaTrader5 as mt5
from typing import Tuple
from bbstrader.metatrader.account import Account
from bbstrader.metatrader.utils import (
    raise_mt5_error, AccountInfo, TerminalInfo,
    SymbolInfo, TickInfo, TradeRequest, OrderCheckResult,
    OrderSentResult, TradePosition, TradeOrder, TradeDeal,
)


class TestAccount(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.account = Account()

    def test_get_account_info(self):
        info = self.account.get_account_info()
        self.assertIsNotNone(info)
        self.assertIsInstance(info, AccountInfo)
        self.assertTrue(hasattr(info, "balance"))
        self.assertTrue(hasattr(info, "assets"))

    def test_get_terminal_info(self):
        info = self.account.get_terminal_info()
        self.assertIsNotNone(info)
        self.assertIsInstance(info, TerminalInfo)
        self.assertTrue(hasattr(info, "trade_allowed"))
        self.assertTrue(hasattr(info, "tradeapi_disabled"))

    def test_get_symbols(self):
        symbols = self.account.get_symbols()
        self.assertIsInstance(symbols, list)
        self.assertGreater(len(symbols), 0)

    def test_get_symbol_info(self):
        symbols = self.account.get_symbols()
        self.assertIsInstance(symbols[0], str)
        symbol_info = self.account.get_symbol_info(symbols[0])
        self.assertIsNotNone(symbol_info)
        self.assertIsInstance(symbol_info, SymbolInfo)
        self.assertTrue(hasattr(symbol_info, 'ask'))
        self.assertTrue(hasattr(symbol_info, "bid"))

    def test_get_symbol_type(self):
        stocks = self.account.get_symbols(symbol_type='STK')
        self.assertGreater(len(stocks), 0)
        symbol_type = self.account.get_symbol_type(stocks[0])
        self.assertEqual(symbol_type, 'STK')
        
        forex = self.account.get_symbols(symbol_type='FX')
        self.assertGreater(len(forex), 0)
        symbol_type = self.account.get_symbol_type(forex[0])
        self.assertEqual(symbol_type, 'FX')
        
        indices = self.account.get_symbols(symbol_type='IDX')
        self.assertGreater(len(indices), 0)
        symbol_type = self.account.get_symbol_type(indices[0])
        self.assertEqual(symbol_type, 'IDX')
        
        commodities = self.account.get_symbols(symbol_type='COMD')
        self.assertGreater(len(commodities), 0)
        symbol_type = self.account.get_symbol_type(commodities[0])
        self.assertEqual(symbol_type, 'COMD')
        
        unknown = '_ABC_' 
        symbol_type = self.account.get_symbol_type(unknown)
        self.assertEqual(symbol_type, 'unknown')

    def test_get_tick_info(self):
        symbols = self.account.get_symbols(symbol_type='FX')
        tick_info = self.account.get_tick_info(symbols[0])
        self.assertIsNotNone(tick_info)
        self.assertIsInstance(tick_info, TickInfo)
        self.assertTrue(hasattr(tick_info, 'last'))

    def test_order_check(self):
        symbol = 'EURUSD'
        point = self.account.get_symbol_info(symbol).point
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 10.0,
            "type": mt5.ORDER_TYPE_BUY,
            "price": self.account.get_tick_info(symbol).ask,
            "sl": self.account.get_tick_info(symbol).ask-100*point,
            "tp": self.account.get_tick_info(symbol).ask+100*point,
            "deviation": 10,
            "magic": 234000,
            "comment": "test order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        result = self.account.check_order(request)
        self.assertIsInstance(result, OrderCheckResult)
        self.assertTrue(hasattr(result, 'retcode'))
        self.assertTrue(hasattr(result, 'margin_free'))
        self.assertTrue(hasattr(result, 'request'))
        self.assertIsInstance(result.request, TradeRequest)
        self.assertTrue(hasattr(result.request, 'action'))
        self.assertTrue(hasattr(result.request, 'magic'))
        self.assertTrue(hasattr(result.request, 'symbol'))
        self.assertEqual(result.request.symbol, symbol)

    def test_order_send(self):
        symbol = 'EURUSD'
        point = self.account.get_symbol_info(symbol).point
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 10.0,
            "type": mt5.ORDER_TYPE_BUY,
            "price": self.account.get_tick_info(symbol).ask,
            "sl": self.account.get_tick_info(symbol).ask-100*point,
            "tp": self.account.get_tick_info(symbol).ask+100*point,
            "deviation": 10,
            "magic": 234000,
            "comment": "test order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        result = self.account.send_order(request)
        self.assertIsInstance(result, OrderSentResult)
        self.assertTrue(hasattr(result, 'deal'))
        self.assertTrue(hasattr(result, 'order'))
        self.assertTrue(hasattr(result, 'request_id'))
        self.assertIsInstance(result.request, TradeRequest)
        self.assertTrue(hasattr(result.request, 'price'))
        self.assertTrue(hasattr(result.request, 'volume'))
        self.assertTrue(hasattr(result.request, 'order'))
        self.assertEqual(result.request.symbol, symbol)

    def test_get_orders(self):
        orders = self.account.get_orders()
        if orders is not None:
            self.assertIsInstance(orders, Tuple)
            self.assertIsInstance(orders[0], TradeOrder)
            self.assertTrue(hasattr(orders[0], 'time_setup'))
        orders = self.account.get_orders(to_df=True)
        if orders is not None:
            self.assertIsInstance(orders, pd.DataFrame)
            self.assertGreater(len(orders), 0)

    def test_get_positions(self):
        positions = self.account.get_positions()
        if positions is not None:
            self.assertIsInstance(positions, Tuple)
            self.assertIsInstance(positions[0], TradePosition)
            self.assertTrue(hasattr(positions[0], 'price_open'))
        positions = self.account.get_positions(to_df=True)
        if positions is not None:
            self.assertIsInstance(positions, pd.DataFrame)
            self.assertGreater(len(positions), 0)

    def test_get_trade_history(self):
        history = self.account.get_trades_history()
        if history is not None:
            self.assertIsInstance(history, pd.DataFrame)
            self.assertGreater(len(history), 0)
        history = self.account.get_trades_history(to_df=False)
        if history is not None:
            self.assertIsInstance(history, Tuple)
            self.assertIsInstance(history[0], TradeDeal)
            self.assertTrue(hasattr(history[0], 'profit'))

    def test_get_orders_history(self):
        history = self.account.get_orders_history()
        if history is not None:
            self.assertIsInstance(history, pd.DataFrame)
            self.assertGreater(len(history), 0)
        history = self.account.get_orders_history(to_df=False)
        if history is not None:
            self.assertIsInstance(history, Tuple)
            self.assertIsInstance(history[0], TradeOrder)
            self.assertTrue(hasattr(history[0], 'type_filling'))

if __name__ == '__main__':
    unittest.main()
