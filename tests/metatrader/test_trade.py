import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

from bbstrader.metatrader.account import __BROKERS__, Broker
from bbstrader.metatrader.trade import (
    Trade,
    TradeAction,
    TradeSignal,
    create_trade_instance,
)

# Mock MetaTrader5 module and its constants/functions
mock_mt5 = MagicMock()
mock_mt5.TRADE_ACTION_DEAL = 0
mock_mt5.ORDER_TYPE_BUY = 0
mock_mt5.ORDER_TYPE_SELL = 1
mock_mt5.ORDER_FILLING_FOK = 0
mock_mt5.ORDER_TIME_GTC = 0
mock_mt5.TRADE_RETCODE_DONE = 10009
mock_mt5.TRADE_ACTION_PENDING = 1
mock_mt5.ORDER_TYPE_BUY_LIMIT = 2
mock_mt5.ORDER_TYPE_SELL_LIMIT = 3
mock_mt5.ORDER_TYPE_BUY_STOP = 4
mock_mt5.ORDER_TYPE_SELL_STOP = 5
mock_mt5.ORDER_TYPE_BUY_STOP_LIMIT = 6
mock_mt5.ORDER_TYPE_SELL_STOP_LIMIT = 7
mock_mt5.TRADE_ACTION_SLTP = 6
mock_mt5.TRADE_ACTION_REMOVE = 2
mock_mt5.TRADE_ACTION_MODIFY = 1


# Mock AccountInfo object
MockAccountInfo = MagicMock()
MockAccountInfo.login = 12345
MockAccountInfo.name = "Test Account"
MockAccountInfo.server = "Test Server"
MockAccountInfo.currency = "USD"
MockAccountInfo.balance = 10000.0
MockAccountInfo.equity = 10000.0
MockAccountInfo.profit = 0.0
MockAccountInfo.margin = 0.0
MockAccountInfo.margin_free = 10000.0
MockAccountInfo.leverage = 100
MockAccountInfo.broker = Broker(__BROKERS__["AMG"])

# Mock SymbolInfo object
MockSymbolInfo = MagicMock()
MockSymbolInfo.name = "EURUSD"
MockSymbolInfo.visible = True
MockSymbolInfo.point = 0.00001
MockSymbolInfo.digits = 5
MockSymbolInfo.spread = 10
MockSymbolInfo.trade_tick_size = 0.00001
MockSymbolInfo.trade_tick_value = 1.0
MockSymbolInfo.trade_contract_size = 100000
MockSymbolInfo.volume_min = 0.01
MockSymbolInfo.volume_max = 100.0
MockSymbolInfo.volume_step = 0.01


# Mock TickInfo object
MockTickInfo = MagicMock()
MockTickInfo.ask = 1.10000
MockTickInfo.bid = 1.09990
MockTickInfo.last = 1.10000
MockTickInfo.volume = 100
MockTickInfo.time_msc = 0
MockTickInfo.flags = 0


# Mock TradePosition object
MockTradePosition = MagicMock()
MockTradePosition.ticket = 1
MockTradePosition.symbol = "EURUSD"
MockTradePosition.type = 0  # BUY
MockTradePosition.magic = 98181105
MockTradePosition.volume = 0.1
MockTradePosition.price_open = 1.10000
MockTradePosition.price_current = 1.10010
MockTradePosition.sl = 1.09900
MockTradePosition.tp = 1.10100
MockTradePosition.profit = 1.0
MockTradePosition.swap = 0.0
MockTradePosition.commission = 0.0


# Mock TradeDeal object
MockTradeDeal = MagicMock()
MockTradeDeal.ticket = 1
MockTradeDeal.order = 1
MockTradeDeal.symbol = "EURUSD"
MockTradeDeal.type = 0  # BUY
MockTradeDeal.entry = 0  # IN
MockTradeDeal.magic = 98181105
MockTradeDeal.volume = 0.1
MockTradeDeal.price = 1.10000
MockTradeDeal.profit = 1.0
MockTradeDeal.swap = 0.0
MockTradeDeal.commission = 0.0
MockTradeDeal.fee = 0.0
MockTradeDeal.time = 0
MockTradeDeal.position_id = 1


# Mock Logger
mock_logger = MagicMock()

# Mock for quantstats
mock_qs = MagicMock()
mock_qs.stats.sharpe.return_value = 1.5


# Patching MetaTrader5 and other external dependencies
# These are the existing patch objects
patch_mt5 = patch("bbstrader.metatrader.trade.Mt5")
patch_logger = patch("bbstrader.metatrader.trade.LOGGER")
patch_qs = patch("bbstrader.metatrader.trade.qs", mock_qs)
patch_check_mt5_connection = patch("bbstrader.metatrader.trade.check_mt5_connection")
patch_raise_mt5_error = patch("bbstrader.metatrader.trade.raise_mt5_error")


# Mock functions that interact with Mt5
mock_mt5.initialize.return_value = True
mock_mt5.symbol_select.return_value = True
mock_mt5.symbol_info.return_value = MockSymbolInfo
mock_mt5.symbol_info_tick.return_value = MockTickInfo
mock_mt5.account_info.return_value = MockAccountInfo
mock_mt5.positions_get.return_value = [MockTradePosition]
mock_mt5.orders_get.return_value = []
mock_mt5.history_deals_get.return_value = [MockTradeDeal]
mock_mt5.order_send.return_value = MagicMock(
    retcode=mock_mt5.TRADE_RETCODE_DONE, order=12345
)

patch_trade_retcode_message = patch(
    "bbstrader.metatrader.trade.trade_retcode_message",
    MagicMock(return_value="Mocked retcode message"),
)


class TestTrade(unittest.TestCase):
    @patch_raise_mt5_error
    @patch_check_mt5_connection
    @patch_logger
    @patch_mt5
    def setUp(
        self,
        mock_mt5_init_arg,
        mock_logger_init_arg,
        mock_check_mt5_connection_arg,
        mock_raise_mt5_error_arg,
    ):

        self.mock_mt5_trade_module = (
            mock_mt5_init_arg  # This is the mock of Mt5 in trade.py
        )
        self.mock_logger_trade_module = (
            mock_logger_init_arg  # This is the mock of LOGGER in trade.py
        )
        self.mock_check_mt5_connection = mock_check_mt5_connection_arg
        self.mock_raise_mt5_error = mock_raise_mt5_error_arg


        # Reset the state of these mocks (which are now the actual globally defined mocks)
        mock_mt5.reset_mock()  # Use the global mock_mt5 directly
        mock_logger.reset_mock()  # Use the global mock_logger directly
        self.mock_check_mt5_connection.reset_mock()  # This is now an instance attribute from arg
        self.mock_raise_mt5_error.reset_mock()  # This is now an instance attribute from arg

        # Update mock configurations as before, using the global mocks
        mock_mt5.account_info.return_value = MockAccountInfo
        mock_mt5.symbol_info.return_value = MockSymbolInfo
        mock_mt5.symbol_info_tick.return_value = MockTickInfo

        # Default parameters for Trade initialization
        self.default_params = {
            "symbol": "EURUSD",
            "expert_name": "TestExpert",
            "expert_id": 12345,
            "version": "1.0",
            "target": 5.0,
            "start_time": "09:00",
            "finishing_time": "17:00",
            "ending_time": "17:30",
            "verbose": False,  # Keep verbose False to prevent printing during tests
            "console_log": False,
            "logger": mock_logger,  # Use the global mock_logger
            # RiskManagement params
            "max_risk": 2.0,
            "daily_risk": 1.0,
            "max_trades": 5,
            "rr": 2.0,
            "account_leverage": True,
            "std_stop": True,
            "sl": 20,
            "tp": 40,  # rr = 2.0, so tp = sl * rr
            "be": 10,
        }
       

    def test_trade_signal_initialization(self):
        signal = TradeSignal(
            id=1,
            symbol="EURUSD",
            action=TradeAction.BUY,
            price=1.1000,
            comment="Test Signal",
        )
        self.assertEqual(signal.id, 1)
        self.assertEqual(signal.symbol, "EURUSD")
        self.assertEqual(signal.action, TradeAction.BUY)
        self.assertEqual(signal.price, 1.1000)
        self.assertIsNone(signal.stoplimit)
        self.assertEqual(signal.comment, "Test Signal")

    def test_trade_signal_initialization_with_stoplimit(self):
        signal = TradeSignal(
            id=2,
            symbol="EURUSD",
            action=TradeAction.SELL,
            price=1.1200,
            stoplimit=1.1100,
            comment="Test Signal SL",
        )
        self.assertEqual(signal.id, 2)
        self.assertEqual(signal.symbol, "EURUSD")
        self.assertEqual(signal.action, TradeAction.SELL)
        self.assertEqual(signal.price, 1.1200)
        self.assertEqual(signal.stoplimit, 1.1100)
        self.assertEqual(signal.comment, "Test Signal SL")

    def test_trade_signal_invalid_action_type(self):
        with self.assertRaisesRegex(
            TypeError, "action must be of type TradeAction, not <class 'str'>"
        ):
            TradeSignal(id=3, symbol="EURUSD", action="INVALID_ACTION", price=1.1000)

    def test_trade_signal_stoplimit_without_price(self):
        with self.assertRaisesRegex(
            ValueError, "stoplimit cannot be set without price"
        ):
            TradeSignal(id=4, symbol="EURUSD", action=TradeAction.BUY, stoplimit=1.1000)

    def test_trade_signal_repr(self):
        signal = TradeSignal(
            id=1,
            symbol="EURUSD",
            action=TradeAction.BUY,
            price=1.1000,
            comment="Test Signal",
        )
        expected_repr = "TradeSignal(id=1, symbol='EURUSD', action='LONG', price=1.1, stoplimit=None), comment='Test Signal'"
        self.assertEqual(repr(signal), expected_repr)

        signal_sl = TradeSignal(
            id=2,
            symbol="GBPUSD",
            action=TradeAction.SSTP,
            price=1.2500,
            stoplimit=1.2400,
            comment="Sell Stop",
        )
        expected_repr_sl = "TradeSignal(id=2, symbol='GBPUSD', action='SSTP', price=1.25, stoplimit=1.24), comment='Sell Stop'"
        self.assertEqual(repr(signal_sl), expected_repr_sl)

    def test_trade_initialization_defaults(self):
        # Test with minimal parameters, relying on defaults
        # Class-level patches for check_mt5_connection and raise_mt5_error are active.
        # If a specific test needs to override them, it can use its own @patch decorator.
        trade = Trade(symbol="EURUSD", logger=mock_logger, verbose=False)
        self.assertEqual(trade.symbol, "EURUSD")
        self.assertEqual(trade.expert_name, "bbstrader")  # Default
        self.assertEqual(trade.expert_id, 98181105)  # Default
        self.assertFalse(trade.verbose)
        self.assertEqual(trade.logger, mock_logger)

        # Check RiskManagement defaults (a few examples)
        self.assertEqual(
            trade.kwargs.get("max_risk"), None
        )  # Default from RiskManagement
        self.assertEqual(trade.kwargs.get("rr"), None)  # Default from RiskManagement

    def test_trade_initialization_custom_params(self):
        # Class-level patches active.
        params = self.default_params.copy()
        trade = Trade(**params)
        self.assertEqual(trade.symbol, "EURUSD")
        self.assertEqual(trade.expert_name, "TestExpert")
        self.assertEqual(trade.expert_id, 12345)
        self.assertEqual(trade.version, "1.0")
        self.assertEqual(trade.target, 5.0)
        self.assertEqual(trade.start, "09:00")
        self.assertEqual(trade.finishing, "17:00")
        self.assertEqual(trade.end, "17:30")
        self.assertFalse(trade.verbose)
        self.assertFalse(trade.console_log)
        self.assertEqual(trade.logger, mock_logger)
        self.assertEqual(
            trade.tf, "D1"
        )  # Default from RiskManagement as time_frame is not in params

        # Check RiskManagement params
        self.assertEqual(trade.kwargs.get("max_risk"), 2.0)
        self.assertEqual(trade.kwargs.get("daily_risk"), 1.0)
        self.assertEqual(trade.kwargs.get("max_trades"), 5)
        self.assertEqual(trade.kwargs.get("rr"), 2.0)
        self.assertTrue(trade.kwargs.get("account_leverage"))
        self.assertTrue(trade.kwargs.get("std_stop"))
        self.assertEqual(trade.kwargs.get("sl"), 20)
        self.assertEqual(trade.kwargs.get("tp"), 40)
        self.assertEqual(trade.kwargs.get("be"), 10)

    def test_trade_initialization_verbose_mode(self):
        # Test verbose mode calls summary and risk_managment
        # We need to mock summary and risk_managment to prevent actual printing
        # Class-level patches active for check_mt5_connection and raise_mt5_error.
        with patch.object(
            Trade, "summary", return_value=None
        ) as mock_summary, patch.object(
            Trade, "risk_managment", return_value=None
        ) as mock_risk_managment:
            params = self.default_params.copy()
            params["verbose"] = True
            trade = Trade(**params)

            self.assertTrue(trade.verbose)
            mock_summary.assert_called_once()
            mock_risk_managment.assert_called_once()

    def test_trade_initialization_logger_string(self):
        # Test when logger is a string
        # Class-level patches active.
        with patch("bbstrader.metatrader.trade.config_logger") as mock_config_logger:
            mock_config_logger.return_value = (
                mock_logger  # Ensure our global mock_logger is used
            )
            params = self.default_params.copy()
            params["logger"] = "test_trade.log"
            trade = Trade(**params)

            mock_config_logger.assert_called_once_with(
                unittest.mock.ANY, False
            )  # Path will be absolute
            self.assertEqual(trade.logger, mock_logger)

    def test_trade_initialization_mt5_connection_failure_initialize(self):
        mock_mt5_local_init_fail = MagicMock()  # Local mock for this test
        mock_mt5_local_init_fail.initialize.side_effect = Exception(
            "MT5 Connection Failed During Initialize"
        )

        # Override class-level patch for Mt5 and check_mt5_connection for this specific test
        with patch("bbstrader.metatrader.trade.Mt5", mock_mt5_local_init_fail), patch(
            "bbstrader.metatrader.trade.check_mt5_connection",
            side_effect=Exception("MT5 Connection Failed"),
        ):
            # We expect the logger to be called with an error
            Trade(**self.default_params)  # Initialize Trade
            # Check that the global mock_logger.error was called due to check_mt5_connection failure
            # The exception in initialize() within Trade class is caught and logged.
            self.assertTrue(mock_logger.error.called)
            # Get the call arguments for the logger.error
            call_args_list = mock_logger.error.call_args_list
            # Check if any of the calls contain the expected error message
            self.assertTrue(
                any(
                    "During initialization: MT5 Connection Failed" in str(call_args)
                    for call_args in call_args_list
                )
            )

    def test_trade_property_retcodes(self):
        trade = Trade(**self.default_params)
        self.assertEqual(trade.retcodes, [])  # Initially empty
        trade._retcodes.append(10004)  # Simulate a retcode being added
        self.assertEqual(trade.retcodes, [10004])

    def test_trade_property_logger(self):
        trade = Trade(**self.default_params)
        self.assertEqual(trade.logger, mock_logger)  # Global mock_logger

    def test_trade_property_orders_no_orders(self):
        mock_mt5.orders_get.return_value = []  # No current orders
        trade = Trade(**self.default_params)
        trade.opened_orders = []  # No manually tracked orders
        self.assertIsNone(trade.orders)

    def test_trade_property_orders_with_current_orders(self):
        mock_order_1 = MagicMock(ticket=101)
        mock_order_2 = MagicMock(ticket=102)
        mock_mt5.orders_get.return_value = [mock_order_1, mock_order_2]
        trade = Trade(**self.default_params)
        trade.opened_orders = []
        self.assertEqual(trade.orders, None)

    def test_trade_property_orders_with_opened_orders(self):
        mock_mt5.orders_get.return_value = []
        trade = Trade(**self.default_params)
        trade.opened_orders = [103, 104]  # Manually tracked
        self.assertEqual(set(trade.orders), {103, 104})

    def test_trade_property_orders_with_mixed_orders(self):
        mock_order_1 = MagicMock(ticket=101)
        mock_mt5.orders_get.return_value = [mock_order_1]
        trade = Trade(**self.default_params)
        trade.opened_orders = [101, 105]  # 101 is duplicate, should be handled by set
        self.assertEqual(set(trade.orders), {101, 105})

    def test_trade_property_positions_no_positions(self):
        mock_mt5.positions_get.return_value = []
        trade = Trade(**self.default_params)
        trade.opened_positions = []
        self.assertIsNone(trade.positions)

    def test_trade_property_positions_with_current_positions(self):
        mock_pos_1 = MagicMock(ticket=201)
        mock_pos_2 = MagicMock(ticket=202)
        mock_mt5.positions_get.return_value = [mock_pos_1, mock_pos_2]
        trade = Trade(**self.default_params)
        trade.opened_positions = []
        self.assertEqual(trade.positions, None)

    def test_trade_property_positions_with_opened_positions(self):
        mock_mt5.positions_get.return_value = []
        trade = Trade(**self.default_params)
        trade.opened_positions = [203, 204]
        self.assertEqual(trade.positions, [203, 204])

    def test_trade_property_positions_with_mixed_positions(self):
        mock_pos_1 = MagicMock(ticket=201)
        mock_mt5.positions_get.return_value = [mock_pos_1]
        trade = Trade(**self.default_params)
        trade.opened_positions = [201, 205]  # 201 is duplicate
        self.assertEqual(trade.positions, [201, 205])

    def test_trade_property_buypos_no_buy_positions(self):
        with patch.object(Trade, "get_current_buys", return_value=None):
            trade = Trade(**self.default_params)
            trade.buy_positions = []
            self.assertIsNone(trade.buypos)

    def test_trade_property_buypos_with_current_buy_positions(self):
        with patch.object(
            Trade, "get_current_buys", return_value=[301, 302]
        ) as mock_get_buys:
            trade = Trade(**self.default_params)
            trade.buy_positions = []
            self.assertEqual(trade.buypos, [301, 302])
            mock_get_buys.assert_called_once_with()

    def test_trade_property_buypos_with_opened_buy_positions(self):
        with patch.object(Trade, "get_current_buys", return_value=None):
            trade = Trade(**self.default_params)
            trade.buy_positions = [303, 304]
            self.assertEqual(set(trade.buypos), {303, 304})

    def test_trade_property_sellpos_no_sell_positions(self):
        with patch.object(Trade, "get_current_sells", return_value=None):
            trade = Trade(**self.default_params)
            trade.sell_positions = []
            self.assertIsNone(trade.sellpos)

    def test_trade_property_sellpos_with_current_sell_positions(self):
        with patch.object(
            Trade, "get_current_sells", return_value=[401, 402]
        ) as mock_get_sells:
            trade = Trade(**self.default_params)
            trade.sell_positions = []
            self.assertEqual(trade.sellpos, [401, 402])
            mock_get_sells.assert_called_once_with()

    def test_trade_property_sellpos_with_opened_sell_positions(self):
        with patch.object(Trade, "get_current_sells", return_value=None):
            trade = Trade(**self.default_params)
            trade.sell_positions = [403, 404]
            self.assertEqual(trade.sellpos, [403, 404])

    def test_trade_property_bepos_no_be_positions(self):
        trade = Trade(**self.default_params)
        trade.break_even_status = []
        self.assertIsNone(trade.bepos)

    def test_trade_property_bepos_with_be_positions(self):
        trade = Trade(**self.default_params)
        trade.break_even_status = [501, 502]
        self.assertEqual(trade.bepos, [501, 502])

    def test_get_logger_with_string_path(self):
        # This is also covered in test_trade_initialization_logger_string,
        # but a focused test is also good.
        with patch("bbstrader.metatrader.trade.config_logger") as mock_config_logger:
            mock_config_logger.return_value = mock_logger  # Global mock
            params_str_logger = self.default_params.copy()
            params_str_logger["logger"] = "some_log_file.log"
            trade = Trade(**params_str_logger)
            mock_config_logger.assert_called_with(unittest.mock.ANY, False)
            self.assertEqual(trade.logger, mock_logger)

    def test_order_type_method(self):
        trade = Trade(**self.default_params)
        order_types = trade._order_type()
        self.assertEqual(order_types["BMKT"], (mock_mt5.ORDER_TYPE_BUY, "BUY"))
        self.assertEqual(
            order_types["SMKT"], (mock_mt5.ORDER_TYPE_BUY, "SELL")
        )  # Note: This seems to be (Mt5.ORDER_TYPE_BUY, "SELL") in source
        self.assertEqual(
            order_types["BLMT"], (mock_mt5.ORDER_TYPE_BUY_LIMIT, "BUY_LIMIT")
        )
        self.assertEqual(
            order_types["SLMT"], (mock_mt5.ORDER_TYPE_SELL_LIMIT, "SELL_LIMIT")
        )
        self.assertEqual(
            order_types["BSTP"], (mock_mt5.ORDER_TYPE_BUY_STOP, "BUY_STOP")
        )
        self.assertEqual(
            order_types["SSTP"], (mock_mt5.ORDER_TYPE_SELL_STOP, "SELL_STOP")
        )
        self.assertEqual(
            order_types["BSTPLMT"],
            (mock_mt5.ORDER_TYPE_BUY_STOP_LIMIT, "BUY_STOP_LIMIT"),
        )
        self.assertEqual(
            order_types["SSTPLMT"],
            (mock_mt5.ORDER_TYPE_SELL_STOP_LIMIT, "SELL_STOP_LIMIT"),
        )

    def test_get_trail_after_points_integer(self):
        trade = Trade(**self.default_params)
        self.assertEqual(trade._get_trail_after_points(50), 50)

    def test_get_trail_after_points_string_sl(self):
        trade = Trade(**self.default_params)
        # Default sl is 20 from default_params
        self.assertEqual(trade._get_trail_after_points("SL"), 20)

    @patch("bbstrader.metatrader.trade.datetime")
    def test_current_datetime(self, mock_datetime):
        trade = Trade(**self.default_params)
        mock_datetime.now.return_value.strftime.return_value = "2023-10-27 10:30:00"
        self.assertEqual(trade.current_datetime(), "2023-10-27 10:30:00")
        mock_datetime.now().strftime.assert_called_once_with("%Y-%m-%d %H:%M:%S")

    @patch("bbstrader.metatrader.trade.datetime")
    def test_current_time_without_seconds(self, mock_datetime):
        trade = Trade(**self.default_params)
        mock_datetime.now.return_value.strftime.return_value = "10:30"
        self.assertEqual(trade.current_time(), "10:30")
        mock_datetime.now().strftime.assert_called_once_with("%H:%M")

    @patch("bbstrader.metatrader.trade.datetime")
    def test_current_time_with_seconds(self, mock_datetime):
        trade = Trade(**self.default_params)
        mock_datetime.now.return_value.strftime.return_value = "10:30:05"
        self.assertEqual(trade.current_time(seconds=True), "10:30:05")
        mock_datetime.now().strftime.assert_called_once_with("%H:%M:%S")

    @patch("builtins.print")  # Mock print to avoid console output during tests
    @patch("bbstrader.metatrader.trade.tabulate")  # Mock tabulate
    def test_summary_method(self, mock_tabulate, mock_print):
        trade = Trade(**self.default_params)
        # Reset call counts from __init__ if verbose was True (it's False by default)
        mock_print.reset_mock()
        mock_tabulate.reset_mock()

        trade.summary()

        # Check that tabulate was called with summary_data
        self.assertTrue(mock_tabulate.called)
        call_args_list = mock_tabulate.call_args_list
        # Check if any call to tabulate had "Expert Advisor Name" in its data
        self.assertTrue(
            any("Expert Advisor Name" in str(call[0][0]) for call in call_args_list)
        )

        # Check that print was called
        self.assertTrue(mock_print.called)

    # Helper method to reset mocks commonly used in position opening tests
    def _reset_position_opening_mocks(self):
        mock_mt5.order_send.reset_mock()
        mock_mt5.symbol_info_tick.reset_mock()  # For current price
        mock_logger.info.reset_mock()
        mock_logger.error.reset_mock()
        

    @patch.object(Trade, "check", return_value=True)  # Assume checks pass by default
    @patch.object(
        Trade, "request_result", return_value=True
    )  # Assume request_result is successful
    @patch.object(Trade, "get_lot", return_value=0.01)
    @patch.object(Trade, "get_stop_loss", return_value=20)  # in points
    @patch.object(Trade, "get_take_profit", return_value=40)  # in points
    @patch.object(Trade, "get_deviation", return_value=5)  # in points
    def test_open_buy_position_market_order_mm(
        self,
        mock_get_dev,
        mock_get_tp,
        mock_get_sl,
        mock_get_lot,
        mock_req_res,
        mock_check,
    ):
        self._reset_position_opening_mocks()
        trade = Trade(**self.default_params)
        mock_mt5.symbol_info_tick.return_value.ask = 1.10000  # Set current ask price

        result = trade.open_buy_position(
            action="BMKT", mm=True, comment="Test Buy Market"
        )

        self.assertTrue(result)
        mock_check.assert_called_once_with("Test Buy Market")

        mock_req_res.assert_called_once()

    @patch.object(Trade, "check", return_value=True)
    @patch.object(Trade, "request_result", return_value=True)
    @patch.object(Trade, "get_lot", return_value=0.01)
    @patch.object(Trade, "get_deviation", return_value=5)
    def test_open_buy_position_market_order_no_mm(
        self, mock_get_dev, mock_get_lot, mock_req_res, mock_check
    ):
        self._reset_position_opening_mocks()
        trade = Trade(**self.default_params)
        mock_mt5.symbol_info_tick.return_value.ask = 1.10000

        result = trade.open_buy_position(
            action="BMKT", mm=False, comment="Test Buy Market No MM"
        )

        self.assertTrue(result)
        mock_check.assert_called_once_with("Test Buy Market No MM")

        mock_req_res.assert_called_once()

    @patch.object(Trade, "check", return_value=True)
    @patch.object(Trade, "request_result", return_value=True)
    @patch.object(Trade, "get_lot", return_value=0.01)
    @patch.object(Trade, "get_stop_loss", return_value=20)
    @patch.object(Trade, "get_take_profit", return_value=40)
    def test_open_buy_position_limit_order(
        self, mock_get_tp, mock_get_sl, mock_get_lot, mock_req_res, mock_check
    ):
        self._reset_position_opening_mocks()
        trade = Trade(**self.default_params)

        order_price = 1.09000
        result = trade.open_buy_position(
            action="BLMT", price=order_price, mm=True, comment="Test Buy Limit"
        )

        self.assertTrue(result)
        mock_check.assert_called_once_with("Test Buy Limit")

        args, _ = mock_req_res.call_args
        self.assertEqual(args[0], order_price)
        self.assertEqual(args[2], "BLMT")

    def test_open_buy_position_pending_order_no_price(self):
        self._reset_position_opening_mocks()
        trade = Trade(**self.default_params)
        with self.assertRaisesRegex(
            ValueError, "You need to set a price for pending orders"
        ):
            trade.open_buy_position(action="BLMT", mm=True)

    @patch.object(Trade, "check", return_value=True)
    @patch.object(Trade, "request_result", return_value=True)
    def test_open_buy_position_buy_stop_limit(self, mock_req_res, mock_check):
        self._reset_position_opening_mocks()
        trade = Trade(**self.default_params)
        order_price = 1.10000
        stoplimit_price = 1.09900  # Must be < order_price for BSTPLMT
        trade.open_buy_position(
            action="BSTPLMT", price=order_price, stoplimit=stoplimit_price, mm=False
        )

        args, _ = mock_req_res.call_args
        sent_request = args[1]
        self.assertEqual(sent_request["type"], mock_mt5.ORDER_TYPE_BUY_STOP_LIMIT)
        self.assertEqual(sent_request["price"], order_price)
        self.assertEqual(sent_request["stoplimit"], stoplimit_price)

    def test_open_buy_position_buy_stop_limit_no_stoplimit_price(self):
        trade = Trade(**self.default_params)
        with self.assertRaisesRegex(
            ValueError, "You need to set a stoplimit price for BSTPLMT orders"
        ):
            trade.open_buy_position(action="BSTPLMT", price=1.10000, mm=False)

    def test_open_buy_position_buy_stop_limit_invalid_stoplimit_price(self):
        trade = Trade(**self.default_params)
        with self.assertRaisesRegex(
            ValueError, "Stoplimit price must be less than the price"
        ):
            trade.open_buy_position(
                action="BSTPLMT", price=1.10000, stoplimit=1.10100, mm=False
            )

    @patch.object(Trade, "check", return_value=True)
    @patch.object(Trade, "request_result", return_value=True)
    @patch.object(Trade, "get_lot", return_value=0.01)
    @patch.object(Trade, "get_stop_loss", return_value=20)
    @patch.object(Trade, "get_take_profit", return_value=40)
    @patch.object(Trade, "get_deviation", return_value=5)
    def test_open_sell_position_market_order_mm(
        self,
        mock_get_dev,
        mock_get_tp,
        mock_get_sl,
        mock_get_lot,
        mock_req_res,
        mock_check,
    ):
        self._reset_position_opening_mocks()
        trade = Trade(**self.default_params)
        mock_mt5.symbol_info_tick.return_value.bid = 1.09990  # Set current bid price

        result = trade.open_sell_position(
            action="SMKT", mm=True, comment="Test Sell Market"
        )

        self.assertTrue(result)
        mock_check.assert_called_once_with("Test Sell Market")

        mock_req_res.assert_called_once()

    # Similar tests for open_sell_position (limit, stop, stop_limit, no_mm, errors) would follow...
    # For brevity, only showing one sell market example.

    @patch.object(Trade, "open_buy_position")
    @patch.object(Trade, "open_sell_position")
    def test_open_position_buy_action(self, mock_open_sell, mock_open_buy):
        trade = Trade(**self.default_params)
        trade.open_position(action="BMKT", price=1.1, comment="TestGenericBuy")
        mock_open_buy.assert_called_once_with(
            action="BMKT",
            price=1.1,
            stoplimit=None,
            id=None,
            mm=True,
            trail=True,
            comment="TestGenericBuy",
            symbol=None,
            volume=None,
            sl=None,
            tp=None,
        )
        mock_open_sell.assert_not_called()

    @patch.object(Trade, "open_buy_position")
    @patch.object(Trade, "open_sell_position")
    def test_open_position_sell_action(self, mock_open_sell, mock_open_buy):
        trade = Trade(**self.default_params)
        trade.open_position(action="SLMT", price=1.1, comment="TestGenericSell")
        mock_open_sell.assert_called_once_with(
            action="SLMT",
            price=1.1,
            stoplimit=None,
            id=None,
            mm=True,
            trail=True,
            comment="TestGenericSell",
            symbol=None,
            volume=None,
            sl=None,
            tp=None,
        )
        mock_open_buy.assert_not_called()

    def test_open_position_invalid_action(self):
        trade = Trade(**self.default_params)
        with self.assertRaisesRegex(ValueError, "Invalid action type 'INVALID'"):
            trade.open_position(action="INVALID")

    @patch.object(Trade, "check", return_value=False)  # Simulate check failing
    def test_open_buy_position_check_fails(self, mock_check_fail):
        self._reset_position_opening_mocks()
        trade = Trade(**self.default_params)
        result = trade.open_buy_position(action="BMKT")
        self.assertFalse(result)  # Should return False if check fails
        mock_check_fail.assert_called_once()
        mock_mt5.order_send.assert_not_called()  # request_result should not be called

    # Tests for condition checking methods

    @patch.object(Trade, "days_end", return_value=False)
    @patch.object(Trade, "trading_time", return_value=True)
    @patch.object(Trade, "is_risk_ok", return_value=True)
    @patch.object(Trade, "is_max_trades_reached", return_value=False)
    @patch.object(Trade, "profit_target", return_value=False)
    @patch.object(Trade, "_check")  # Mock _check to see if it's called
    def test_check_all_conditions_pass(
        self,
        mock_internal_check,
        mock_profit_target,
        mock_max_trades,
        mock_risk_ok,
        mock_trading_time,
        mock_days_end,
    ):
        trade = Trade(**self.default_params)
        trade.copy_mode = False  # Ensure copy_mode is off for these checks

        self.assertTrue(trade.check("Test Comment"))

        mock_internal_check.assert_not_called()

    @patch.object(Trade, "days_end", return_value=True)  # Fails here
    @patch.object(Trade, "trading_time", return_value=True)
    @patch.object(Trade, "is_risk_ok", return_value=True)
    @patch.object(Trade, "is_max_trades_reached", return_value=False)
    @patch.object(Trade, "profit_target", return_value=False)
    def test_check_fails_days_end(
        self,
        mock_profit_target,
        mock_max_trades,
        mock_risk_ok,
        mock_trading_time,
        mock_days_end,
    ):
        trade = Trade(**self.default_params)
        trade.copy_mode = False
        self.assertFalse(trade.check("Test Comment"))


    @patch.object(Trade, "days_end", return_value=False)
    @patch.object(Trade, "trading_time", return_value=True)
    @patch.object(Trade, "is_risk_ok", return_value=True)
    @patch.object(Trade, "is_max_trades_reached", return_value=False)
    @patch.object(
        Trade, "profit_target", return_value=True
    )  # Fails here (but means success for profit)
    @patch.object(Trade, "_check")
    def test_check_profit_target_reached(
        self,
        mock_internal_check_pt,
        mock_profit_target_pt,
        mock_max_trades_pt,
        mock_risk_ok_pt,
        mock_trading_time_pt,
        mock_days_end_pt,
    ):
        trade = Trade(**self.default_params)
        trade.copy_mode = False
        # If profit target is reached, check() returns True, but calls _check
        self.assertTrue(trade.check("Test Comment"))
        mock_internal_check_pt.assert_called_once_with(
            "Profit target Reached !!! SYMBOL=EURUSD"
        )

    def test_check_copy_mode_true(self):
        trade = Trade(**self.default_params)
        trade.copy_mode = True
        # All checks should be bypassed if copy_mode is True
        self.assertTrue(trade.check("Test Comment"))

    @patch.object(Trade, "positive_profit", return_value=True)
    @patch.object(
        Trade, "get_current_positions", return_value=[MockTradePosition]
    )  # Has positions
    @patch.object(Trade, "close_positions")
    @patch.object(Trade, "statistics")
    def test_internal_check_positive_profit_and_positions(
        self, mock_stats, mock_close_pos, mock_get_curr_pos, mock_pos_profit
    ):
        trade = Trade(**self.default_params)
        mock_logger.reset_mock()  # Reset from setup

        trade._check("Risk Reached")

        mock_pos_profit.assert_called_once_with(id=trade.expert_id)
        mock_close_pos.assert_called_once_with(position_type="all")
        mock_logger.info.assert_any_call("Risk Reached")
        mock_stats.assert_called_once_with(save=True)

    @patch.object(Trade, "positive_profit", return_value=False)  # Not positive profit
    @patch.object(
        Trade, "get_current_positions", return_value=None
    )  # No current positions
    @patch.object(Trade, "close_positions")
    @patch.object(Trade, "statistics")
    def test_internal_check_no_positive_profit_no_positions(
        self, mock_stats, mock_close_pos, mock_get_curr_pos, mock_pos_profit
    ):
        trade = Trade(**self.default_params)
        mock_logger.reset_mock()

        trade._check("Risk Reached")

        # positive_profit is called, returns False.
        # get_current_positions is then called, returns None.
        # Because get_current_positions is None, the OR condition is met.
        mock_pos_profit.assert_called_once_with(id=trade.expert_id)
        mock_get_curr_pos.assert_called_once_with()
        mock_close_pos.assert_called_once_with(position_type="all")
        mock_logger.info.assert_any_call("Risk Reached")
        mock_stats.assert_called_once_with(save=True)

    @patch("bbstrader.metatrader.trade.datetime")
    def test_days_end(self, mock_datetime_days_end):
        trade = Trade(**self.default_params)  # end_time="17:30"

        # Case 1: Current time is before end time
        mock_datetime_days_end.now.return_value.time.return_value = datetime.strptime(
            "17:00", "%H:%M"
        ).time()
        mock_datetime_days_end.strptime = (
            datetime.strptime
        )  # Ensure original strptime is used by the method
        self.assertFalse(trade.days_end())

        # Case 2: Current time is at end time
        mock_datetime_days_end.now.return_value.time.return_value = datetime.strptime(
            "17:30", "%H:%M"
        ).time()
        self.assertTrue(trade.days_end())

        # Case 3: Current time is after end time
        mock_datetime_days_end.now.return_value.time.return_value = datetime.strptime(
            "18:00", "%H:%M"
        ).time()
        self.assertTrue(trade.days_end())

    @patch("bbstrader.metatrader.trade.datetime")
    def test_trading_time(self, mock_datetime_trading_time):
        trade = Trade(
            **self.default_params
        )  # start_time="09:00", finishing_time="17:00"
        mock_datetime_trading_time.strptime = (
            datetime.strptime
        )  # Ensure original strptime

        # Case 1: Before trading window
        mock_datetime_trading_time.now.return_value.time.return_value = (
            datetime.strptime("08:00", "%H:%M").time()
        )
        self.assertFalse(trade.trading_time())

        # Case 2: At start of trading window
        mock_datetime_trading_time.now.return_value.time.return_value = (
            datetime.strptime("09:00", "%H:%M").time()
        )
        self.assertTrue(trade.trading_time())

        # Case 3: Within trading window
        mock_datetime_trading_time.now.return_value.time.return_value = (
            datetime.strptime("12:00", "%H:%M").time()
        )
        self.assertTrue(trade.trading_time())

        # Case 4: At end of trading window (finishing_time)
        mock_datetime_trading_time.now.return_value.time.return_value = (
            datetime.strptime("17:00", "%H:%M").time()
        )
        self.assertTrue(trade.trading_time())  # Inclusive of end time

        # Case 5: After trading window
        mock_datetime_trading_time.now.return_value.time.return_value = (
            datetime.strptime("17:01", "%H:%M").time()
        )
        self.assertFalse(trade.trading_time())

    @patch.object(Trade, "get_today_deals")
    @patch.object(
        Trade, "max_trade", return_value=3
    )  # From RiskManagement, mock directly or ensure it's set
    def test_is_max_trades_reached(self, mock_max_trade_prop, mock_get_today_deals):
        trade = Trade(**self.default_params)

        # Scenario 1: Fewer negative deals than max_trades
        mock_get_today_deals.return_value = [
            MagicMock(profit=-10),
            MagicMock(profit=5),
            MagicMock(profit=-5),  # 2 negative deals
        ]
        self.assertFalse(trade.is_max_trades_reached())

        # Scenario 2: Equal negative deals to max_trades
        mock_get_today_deals.return_value = [
            MagicMock(profit=-10),
            MagicMock(profit=-5),
            MagicMock(profit=-2),  # 3 negative deals
        ]
        self.assertTrue(trade.is_max_trades_reached())

        # Scenario 3: More negative deals than max_trades
        mock_get_today_deals.return_value = [
            MagicMock(profit=-10),
            MagicMock(profit=-5),
            MagicMock(profit=-2),
            MagicMock(profit=-1),  # 4 negative
        ]
        self.assertTrue(trade.is_max_trades_reached())

        # Scenario 4: No deals
        mock_get_today_deals.return_value = []
        self.assertFalse(trade.is_max_trades_reached())

        # Scenario 5: Only positive deals
        mock_get_today_deals.return_value = [MagicMock(profit=10), MagicMock(profit=20)]
        self.assertFalse(trade.is_max_trades_reached())

    # Tests for order/position retrieval methods

    def _create_mock_order(self, ticket, magic, type, symbol="EURUSD"):
        order = MagicMock()
        order.ticket = ticket
        order.magic = magic
        order.type = type
        order.symbol = symbol
        # Add other attributes if get_filtered_tickets uses them for filtering
        return order

    def _create_mock_position(self, ticket, magic, type, profit, symbol="EURUSD"):
        position = MagicMock()
        position.ticket = ticket
        position.magic = magic
        position.type = type  # 0 for buy, 1 for sell
        position.profit = profit
        position.symbol = symbol
        # Mock other necessary attributes like price_open, volume, point for win_trade if used by profitables
        position.price_open = 1.0
        position.volume = 0.1
        position.sl = 0.9
        position.tp = 1.1
        position.price_current = 1.05 if type == 0 else 0.95  # for win_trade check
        return position

    def test_get_filtered_tickets_no_items(self):
        trade = Trade(**self.default_params)
        mock_mt5.positions_get.return_value = []
        mock_mt5.orders_get.return_value = []
        self.assertIsNone(trade.get_filtered_tickets(filter_type="positions"))
        self.assertIsNone(trade.get_filtered_tickets(filter_type="orders"))

    @patch("bbstrader.metatrader.trade.datetime")
    @patch.object(Trade, "get_trades_history")
    def test_get_today_deals(self, mock_get_history, mock_dt):
        trade = Trade(**self.default_params)  # expert_id = 12345

        # Setup mock datetime
        current_test_time = datetime(2023, 1, 15, 12, 0, 0)
        mock_dt.now.return_value = current_test_time
        # When get_trades_history calls datetime.fromtimestamp, it needs a real datetime
        mock_dt.fromtimestamp = datetime.fromtimestamp

        # Mock deals from history
        deal1_today_expert = MagicMock(
            position_id=1, magic=12345, time=int(current_test_time.timestamp())
        )
        deal2_today_other_expert = MagicMock(
            position_id=2, magic=67890, time=int(current_test_time.timestamp())
        )
        deal3_yesterday_expert = MagicMock(
            position_id=3,
            magic=12345,
            time=int((current_test_time - timedelta(days=1)).timestamp()),
        )

        # This is the initial return for history to get all position_ids
        mock_get_history.return_value = [
            deal1_today_expert,
            deal2_today_other_expert,
            deal3_yesterday_expert,
        ]

        # Side effect for subsequent calls to get_trades_history with position_id
        def history_side_effect(*args, **kwargs):
            pos_id = kwargs.get("position")
            if pos_id == 1:  # Today, our expert
                # Simulate it returns [open_deal, close_deal]
                return [MagicMock(magic=12345), deal1_today_expert]
            if (
                pos_id == 2
            ):  # Today, other expert (should be filtered out by magic check on position_ids)
                return [MagicMock(magic=67890), deal2_today_other_expert]
            if pos_id == 3:  # Yesterday, our expert
                return [MagicMock(magic=12345), deal3_yesterday_expert]
            # Default for the first call without position kwarg
            return [
                deal1_today_expert,
                deal2_today_other_expert,
                deal3_yesterday_expert,
            ]

        mock_get_history.side_effect = history_side_effect

        today_deals = trade.get_today_deals(group="EURUSD")

        self.assertEqual(len(today_deals), 1)
        self.assertIn(deal1_today_expert, today_deals)
        self.assertNotIn(deal2_today_other_expert, today_deals)  # Filtered by magic
        self.assertNotIn(deal3_yesterday_expert, today_deals)  # Filtered by date

        # Check calls to get_trades_history
        # First call: date_from, group
        mock_get_history.assert_any_call(
            date_from=unittest.mock.ANY, group="EURUSD", to_df=False
        )
        # Subsequent calls: date_from, position_id
        mock_get_history.assert_any_call(
            date_from=unittest.mock.ANY, position=1, to_df=False
        )

        mock_get_history.assert_any_call(
            date_from=unittest.mock.ANY, position=3, to_df=False
        )

    @patch("bbstrader.metatrader.trade.datetime")
    @patch.object(Trade, "get_trades_history")
    def test_get_today_deals_no_deals_in_history(self, mock_get_history, mock_dt):
        trade = Trade(**self.default_params)
        current_test_time = datetime(2023, 1, 15, 12, 0, 0)
        mock_dt.now.return_value = current_test_time
        mock_get_history.return_value = []  # No deals in history

        today_deals = trade.get_today_deals(group="EURUSD")
        self.assertEqual(today_deals, [])

    @patch("bbstrader.metatrader.trade.datetime")
    @patch.object(Trade, "get_trades_history")
    def test_get_today_deals_history_returns_none(self, mock_get_history, mock_dt):
        trade = Trade(**self.default_params)
        current_test_time = datetime(2023, 1, 15, 12, 0, 0)
        mock_dt.now.return_value = current_test_time
        mock_get_history.return_value = None  # get_trades_history returns None

        today_deals = trade.get_today_deals(group="EURUSD")
        self.assertEqual(today_deals, [])

    # Tests for break-even and trailing stop logic

    def _setup_symbol_info_for_be_tests(self, mock_symbol_info_method):
        # Helper to provide a consistent MockSymbolInfo for BE tests
        # Ensure this mock_symbol_info has all necessary attributes for win_trade, set_break_even etc.
        info = MagicMock()
        info.point = 0.00001
        info.digits = 5
        info.spread = 2  # in points
        info.trade_tick_size = 0.00001
        info.trade_tick_value = (
            1.0  # Assume 1 USD per point for simplicity in profit calc
        )
        mock_symbol_info_method.return_value = info
        return info

    @patch.object(Trade, "get_positions")
    @patch.object(Trade, "get_symbol_info")
    @patch.object(Trade, "get_break_even", return_value=10)  # BE points
    @patch.object(Trade, "set_break_even")  # Mock the method that sends the order
    def test_break_even_set_first_time(
        self, mock_set_be, mock_get_be_val, mock_get_sym_info, mock_get_positions
    ):
        trade = Trade(**self.default_params)  # expert_id = 12345
        mock_sym_info = self._setup_symbol_info_for_be_tests(mock_get_sym_info)

        # Simulate a position that has moved enough for break-even
        # profit_points = position.profit * (size / value / position.volume)
        # Let size/value = 1. profit_points = profit / volume
        # We need profit_points >= be_points (10)
        # If profit = 2, volume = 0.1 => 20 points. 20 >= 10.
        mock_pos = self._create_mock_position(1, 12345, 0, profit=2.0)  # BUY position
        mock_pos.volume = 0.1
        mock_get_positions.return_value = [mock_pos]

        with patch.object(mock_sym_info, "trade_tick_size", 1), patch.object(
            mock_sym_info, "trade_tick_value", 1
        ):
            trade.break_even(
                mm=True, id=12345, trail=False
            )  # trail=False for this test

        mock_set_be.assert_called_once_with(
            mock_pos, 10, price=None
        )  # price=None means auto-calc
        self.assertIn(mock_pos.ticket, trade.break_even_status)
        self.assertEqual(trade.break_even_points[mock_pos.ticket], 10)

    @patch.object(Trade, "get_positions")
    @patch.object(Trade, "get_symbol_info")
    @patch.object(Trade, "get_break_even", return_value=10)
    @patch.object(Trade, "set_break_even")
    def test_break_even_trailing_stop(
        self, mock_set_be, mock_get_be_val, mock_get_sym_info, mock_get_positions
    ):
        trade = Trade(**self.default_params)
        mock_sym_info = self._setup_symbol_info_for_be_tests(mock_get_sym_info)

        # Position initially set to BE
        mock_pos = self._create_mock_position(
            1, 12345, 0, profit=2.0
        )  # 20 points profit (size/value=1, vol=0.1)
        mock_pos.volume = 0.1
        mock_pos.price_open = 1.10000
        mock_pos.price_current = 1.10200  # Corresponds to 20 points profit if price_open = 1.10000 and tick_value=0.1 for 0.1 lot

        trade.break_even_status = [mock_pos.ticket]
        trade.break_even_points[mock_pos.ticket] = 10  # Initial BE was at 10 points

        mock_get_positions.return_value = [mock_pos]

        # Simulate favorable move for trail: new_be_points = 10 (current_be) + round(10*0.10) = 11
        # Current profit points = 20. 20 >= 11. So, trail.
        # stop_trail (trail_points) = round(10 * 0.50) = 5 points
        # new_price for SL = current_price - (trail_points * point)
        # = 1.10200 - (5 * 0.00001) = 1.10200 - 0.00005 = 1.10195
        # new_level for validation = price_open + (new_be_points * point)
        # = 1.10000 + (11 * 0.00001) = 1.10000 + 0.00011 = 1.10011

        with patch.object(mock_sym_info, "trade_tick_size", 1), patch.object(
            mock_sym_info, "trade_tick_value", 1
        ), patch.object(
            mock_sym_info, "point", 0.00001
        ):  # ensure point is correct for calc
            trade.break_even(
                mm=True, id=12345, trail=True, stop_trail=None, trail_after_points=None
            )

        expected_new_sl_price = round(1.10200 - (5 * 0.00001), 5)
        expected_new_level = round(1.10000 + (11 * 0.00001), 5)
        mock_set_be.assert_called_once_with(
            mock_pos, 10, price=expected_new_sl_price, level=expected_new_level
        )

    @patch.object(Trade, "get_symbol_info")
    @patch.object(Trade, "get_tick_info")
    @patch.object(Trade, "get_stats")
    @patch.object(Trade, "break_even_request")  # Mock the actual order sending part
    def test_set_break_even_buy_position(
        self, mock_be_request, mock_get_stats, mock_get_tick, mock_get_sym_info
    ):
        trade = Trade(**self.default_params)
        mock_sym_info = self._setup_symbol_info_for_be_tests(mock_get_sym_info)  # noqa: F841
        mock_get_stats.return_value = ({"average_fee": -0.2}, {})  # fee = -0.2
        # currency_risk is part of RiskManagement, assume it's set up to return trade_profit > 0
        with patch.object(
            trade, "currency_risk", return_value={"trade_profit": 0.1}
        ):  # profit of 0.1 per point
            # fees_points = round(-0.2 / 0.1) = -2 points.
            # mock_sym_info.spread = 2 points

            mock_tick = MagicMock(ask=1.10150)  # Current ask price > break_even_level
            mock_get_tick.return_value = mock_tick

            # Buy position, current price > open price
            mock_pos = self._create_mock_position(
                1, 12345, 0, profit=1.5
            )  # Type 0 = BUY
            mock_pos.price_open = 1.10000
            mock_pos.price_current = 1.10150  # Matches tick.ask
            mock_pos.tp = 1.10500

            be_points = 10  # from @patch on test_win_trade, but here it's direct arg

            trade.set_break_even(mock_pos, be_points, price=None, level=None)
            mock_be_request.assert_called_once()

    @patch.object(Trade, "check_order")  # from RiskManagement
    @patch.object(Trade, "send_order")  # from RiskManagement
    def test_break_even_request_success(self, mock_send_order, mock_check_order):
        trade = Trade(**self.default_params)
        mock_send_order.return_value = MagicMock(retcode=mock_mt5.TRADE_RETCODE_DONE)

        ticket = 123
        price = 1.10050
        request_dict = {
            "action": mock_mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": price,
            "tp": 1.10500,
        }

        trade.break_even_request(ticket, price, request_dict)

        mock_check_order.assert_called_once()
        mock_send_order.assert_called_once()
        self.assertIn(ticket, trade.break_even_status)

    # Tests for order/position closing and modification methods
    @patch.object(Trade, "check_order")
    @patch.object(Trade, "send_order")
    def test_close_request_position_success(self, mock_send_order, mock_check_order):
        trade = Trade(**self.default_params)
        mock_send_order.return_value = MagicMock(retcode=mock_mt5.TRADE_RETCODE_DONE)

        request_data = {
            "action": mock_mt5.TRADE_ACTION_DEAL,
            "symbol": "EURUSD",
            "volume": 0.1,
            "type": mock_mt5.ORDER_TYPE_SELL,
            "position": 123,
            "price": 1.1000,
            "deviation": 5,
            "magic": trade.expert_id,
            "comment": f"@{trade.expert_name}",
            "type_time": mock_mt5.ORDER_TIME_GTC,
            "type_filling": mock_mt5.ORDER_FILLING_FOK,
        }

        result = trade.close_request(request_data, type="position")
        self.assertTrue(result)
        mock_check_order.assert_called_once()
        mock_send_order.assert_called_once()

    @patch.object(Trade, "get_orders")
    @patch.object(Trade, "check_order")
    @patch.object(Trade, "send_order")
    def test_modify_order_success(
        self, mock_send_order, mock_check_order, mock_get_orders
    ):
        trade = Trade(**self.default_params)

        mock_existing_order = self._create_mock_order(
            789, trade.expert_id, mock_mt5.ORDER_TYPE_BUY_LIMIT
        )
        mock_existing_order.price_open = 1.0900
        mock_existing_order.sl = 1.0800
        mock_existing_order.tp = 1.1000
        mock_existing_order.price_stoplimit = 0  # No stoplimit initially
        mock_get_orders.return_value = [mock_existing_order]

        mock_send_order.return_value = MagicMock(retcode=mock_mt5.TRADE_RETCODE_DONE)

        new_price = 1.0950
        new_sl = 1.0850
        trade.modify_order(ticket=789, price=new_price, sl=new_sl)

        mock_get_orders.assert_called_once_with(ticket=789)
        mock_check_order.assert_called_once()
        mock_send_order.assert_called_once()
        mock_logger.info.assert_called_once()

    @patch.object(Trade, "get_orders", return_value=[])  # Order not found
    def test_modify_order_not_found(self, mock_get_orders):
        trade = Trade(**self.default_params)
        mock_logger.reset_mock()
        trade.modify_order(ticket=999, price=1.1)
        mock_logger.error.assert_called_with(
            f"Order #999 not found, SYMBOL={trade.symbol}, PRICE=1.1"
        )

    @patch.object(Trade, "close_request", return_value=True)
    def test_close_order(self, mock_cr):
        trade = Trade(**self.default_params)
        ticket_to_close = 111
        custom_comment = "Closing this order"

        trade.close_order(ticket_to_close, id=trade.expert_id, comment=custom_comment)

        mock_cr.assert_called_once()

    @patch.object(Trade, "get_positions")
    @patch.object(Trade, "get_tick_info")
    @patch.object(Trade, "close_request", return_value=True)
    @patch.object(Trade, "get_deviation", return_value=5)
    def test_close_position_full(
        self, mock_get_dev, mock_cr, mock_get_tick, mock_get_pos
    ):
        trade = Trade(**self.default_params)  # expert_id = 12345

        mock_pos_to_close = self._create_mock_position(
            222, 12345, 0, 10
        )  # Buy position
        mock_pos_to_close.volume = 0.5
        mock_get_pos.return_value = [mock_pos_to_close]

        mock_sell_price = 1.12340  # Price for closing a buy
        mock_get_tick.return_value = MagicMock(ask=1.12350, bid=mock_sell_price)

        custom_id = 12345
        trade.close_position(ticket=222, id=custom_id, pct=1.0, comment="Full Close")

        mock_get_pos.assert_called_once_with(ticket=222)
        mock_get_tick.assert_called_with(
            trade.symbol
        )  # Ensure get_tick_info is called for the trade's symbol

        mock_cr.assert_called_once()

    @patch.object(Trade, "get_positions", return_value=None)  # Position not found
    def test_close_position_not_found(self, mock_get_pos):
        trade = Trade(**self.default_params)
        # If position not found, close_request should not be called.
        # The method currently doesn't explicitly log or return False if pos not found, relies on close_request.
        # Let's assume close_request is not called.
        with patch.object(trade, "close_request") as mock_cr_local:
            trade.close_position(ticket=998, id=12345)
            mock_cr_local.assert_not_called()

    def test_bulk_close_orders(self):
        trade = Trade(**self.default_params)
        mock_close_order_func = MagicMock()

        tickets_to_close = [1, 2, 3]
        # Simulate some successes and some failures
        mock_close_order_func.side_effect = [True, False, True]

        mock_logger.reset_mock()
        trade.bulk_close(
            tickets_to_close,
            "orders",
            mock_close_order_func,
            "buy_limits",
            id=12345,
            comment="Bulk Close Test",
        )

        self.assertEqual(mock_close_order_func.call_count, 3)
        mock_close_order_func.assert_any_call(1, id=12345, comment="Bulk Close Test")
        mock_close_order_func.assert_any_call(2, id=12345, comment="Bulk Close Test")
        mock_close_order_func.assert_any_call(3, id=12345, comment="Bulk Close Test")

        # tickets_to_close list is modified in place
        self.assertEqual(
            tickets_to_close, [2]
        )  # Only the one that failed (returned False) remains

    @patch.object(Trade, "get_current_buy_stops", return_value=[10, 11])
    @patch.object(Trade, "bulk_close")
    def test_close_orders_by_type(self, mock_bulk_cl, mock_get_cbs):
        trade = Trade(**self.default_params)
        custom_id = 999
        trade.close_orders(
            order_type="buy_stops", id=custom_id, comment="Close All Buy Stops"
        )

        mock_get_cbs.assert_called_once_with(id=custom_id)
        mock_bulk_cl.assert_called_once_with(
            [10, 11],
            "orders",
            trade.close_order,
            "buy_stops",
            id=custom_id,
            comment="Close All Buy Stops",
        )

    @patch.object(Trade, "get_current_positions", return_value=[20, 21])
    @patch.object(Trade, "bulk_close")
    def test_close_positions_by_type(self, mock_bulk_cl_pos, mock_get_cp):
        trade = Trade(**self.default_params)
        trade.close_positions(position_type="all", comment="Close All Positions")

        mock_get_cp.assert_called_once_with(
            id=trade.expert_id
        )  # Uses default expert_id
        mock_bulk_cl_pos.assert_called_once_with(
            [20, 21],
            "positions",
            trade.close_position,
            "all",
            id=trade.expert_id,
            comment="Close All Positions",
        )

    # Tests for statistics and utility methods

    @patch.object(Trade, "get_today_deals")
    @patch.object(Trade, "get_trades_history")  # This is called twice in get_stats
    @patch.object(Trade, "get_account_info")
    def test_get_stats_no_deals(
        self, mock_get_acc_info, mock_get_hist, mock_get_today_deals
    ):
        trade = Trade(**self.default_params)
        mock_get_today_deals.return_value = []  # No deals for session stats
        mock_get_hist.return_value = None  # No deals for total stats
        mock_get_acc_info.return_value = MagicMock(balance=10000)

        stats1, stats2 = trade.get_stats()

        expected_stats1 = {
            "deals": 0,
            "profit": 0,
            "win_trades": 0,
            "loss_trades": 0,
            "total_fees": 0,
            "average_fee": 0,
            "win_rate": 0,
        }
        expected_stats2 = {"total_profit": 0, "profitability": "No"}

        self.assertEqual(stats1, expected_stats1)
        self.assertEqual(stats2, expected_stats2)

    @patch.object(Trade, "get_today_deals")
    @patch.object(Trade, "get_trades_history")
    @patch.object(Trade, "get_account_info")
    def test_get_stats_with_deals(
        self, mock_get_acc_info, mock_get_hist_main, mock_get_today_deals
    ):
        trade = Trade(**self.default_params)
        mock_get_acc_info.return_value = MagicMock(
            balance=10000, profit=150
        )  # profit for current open positions

        # --- Setup for session stats (stats1) ---
        # Mock deals for get_today_deals
        # These are "closing" deals of today's trades
        today_closing_deal1 = MagicMock(profit=100, position_id=1)  # win
        today_closing_deal2 = MagicMock(profit=-50, position_id=2)  # loss
        mock_get_today_deals.return_value = [today_closing_deal1, today_closing_deal2]

        # Mock get_trades_history for when it's called by get_stats for session deals
        # It's called per position_id from today_deals
        # Deal 1: win (profit 100, comm -2, swap -1, fee -1 = net 96)
        history_deal1_open = MagicMock(commission=-2, swap=-1, fee=-1)
        history_deal1_close = today_closing_deal1  # profit is on the closing deal
        # Deal 2: loss (profit -50, comm -2, swap 0, fee -1 = net -53)
        history_deal2_open = MagicMock(commission=-2, swap=0, fee=-1)
        history_deal2_close = today_closing_deal2

        def session_history_side_effect(*args, **kwargs):
            pos_id = kwargs.get("position")
            if pos_id == 1:
                return [history_deal1_open, history_deal1_close]
            if pos_id == 2:
                return [history_deal2_open, history_deal2_close]
            return None  # Should not happen if logic is correct

        mock_df_data = {
            "profit": [
                0,
                100,
                -50,
            ],  # First row is often a dummy/header like in example
            "commission": [0, -2, -2],
            "fee": [0, -1, -1],
            "swap": [0, -1, 0],
        }
        mock_historical_df = pd.DataFrame(mock_df_data)

        # Configure side effects for the main mock_get_hist_main
        # It will be called first for session stats (iteratively), then for total stats (DataFrame)
        call_count_get_hist = 0

        def main_history_side_effect(*args, **kwargs):
            nonlocal call_count_get_hist
            call_count_get_hist += 1
            if kwargs.get("position"):  # Called for session stats
                return session_history_side_effect(*args, **kwargs)
            else:  # Called for total stats (expecting DataFrame)
                return mock_historical_df

        mock_get_hist_main.side_effect = main_history_side_effect

        stats1, stats2 = trade.get_stats()

        # Assertions for stats1 (session)
        self.assertEqual(stats1["deals"], 2)
        self.assertEqual(
            stats1["profit"], 100 - 50
        )  # Sum of raw profits from closing deals
        self.assertEqual(stats1["win_trades"], 1)
        self.assertEqual(stats1["loss_trades"], 1)
        self.assertEqual(
            stats1["total_fees"], (-2 - 1 - 1) + (-2 - 1)
        )  # Sum of comm+swap+fee for both
        self.assertAlmostEqual(stats1["average_fee"], ((-2 - 1 - 1) + (-2 - 1)) / 2.0)
        self.assertEqual(stats1["win_rate"], 50.0)

        # Assertions for stats2 (total)
        # total_profit = sum of (profit + commission + fee + swap) from df2.iloc[1:]
        # Deal 1 net = 100 - 2 - 1 - 1 = 96
        # Deal 2 net = -50 - 2 - 1 - 0 = -53
        # Total = 96 - 53 = 43
        self.assertAlmostEqual(stats2["total_profit"], 43)
        self.assertEqual(
            stats2["profitability"], "Yes"
        )  # Since balance (10000) > initial_balance (10000 - 43)

    @patch.object(Trade, "get_trades_history", return_value=None)
    def test_sharpe_ratio_no_history(self, mock_get_hist_sharpe_none):
        trade = Trade(**self.default_params)
        self.assertEqual(trade.sharpe(), 0.0)
        mock_qs.stats.sharpe.assert_not_called()  # Should not be called if no history

    @patch("bbstrader.metatrader.trade.datetime")
    def test_sleep_time_weekday(self, mock_dt_sleep):
        # Params: start_time="09:00"
        trade = Trade(**self.default_params)

        # Simulate current time is 17:30 on a weekday (after self.start="09:00")
        # current_time() will be "17:30"
        mock_dt_sleep.now.return_value.strftime = MagicMock(
            side_effect=lambda fmt: "17:30" if fmt == "%H:%M" else "dummy"
        )
        mock_dt_sleep.strptime = datetime.strptime  # Use real strptime

        # Expected: (24*60) - minutes_from_start_to_current
        # start = 09:00, current = 17:30
        # duration = 17:30 - 09:00 = 8 hours 30 mins = 510 minutes
        # sleep_time = (24*60) - 510 = 1440 - 510 = 930 minutes
        self.assertEqual(trade.sleep_time(weekend=False), 930)

    @patch("bbstrader.metatrader.trade.datetime")
    def test_sleep_time_weekend(self, mock_dt_sleep_wk):
        # Params: start_time="09:00"
        trade = Trade(**self.default_params)

        # Simulate current time is Friday 18:00
        mock_dt_sleep_wk.now.return_value.strftime = MagicMock(
            side_effect=lambda fmt: "18:00" if fmt == "%H:%M" else "dummy"
        )
        mock_dt_sleep_wk.strptime = datetime.strptime  # Use real strptime

        # Expected: from Friday 18:00 to Monday 09:00
        # Friday 18:00 to Monday 18:00 = 3 days * 24 * 60 = 4320 mins
        # From Monday 18:00 back to Monday 09:00 = -9 hours = -540 mins
        # Total = 4320 - 540 = 3780 mins
        # Formula: (monday_start_time - friday_current_time) + 3_days_in_minutes
        # friday_time_obj = 18:00, monday_time_obj = 09:00
        # intra_day_diff_seconds = ( (9*3600) - (18*3600) ) = -32400 seconds = -540 minutes
        # inter_day_diff_minutes = 3 * 24 * 60 = 4320 minutes
        # total_minutes = 4320 + (-540) = 3780
        self.assertEqual(trade.sleep_time(weekend=True), 3780)

    # Tests for create_trade_instance function
    # We need to patch 'bbstrader.metatrader.trade.Trade' to check its instantiation
    @patch(
        "bbstrader.metatrader.trade.Trade"
    )  # Patch the Trade class in the trade module
    def test_create_trade_instance_single_symbol(self, MockTradeClass):
        mock_trade_instance = MagicMock()
        MockTradeClass.return_value = mock_trade_instance

        symbols = ["EURUSD"]
        base_params = {
            "expert_name": "CreatorTest",
            "target": 3.0,
            "logger": mock_logger,
        }

        instances = create_trade_instance(symbols, base_params.copy())  # Use .copy()

        self.assertEqual(len(instances), 1)
        self.assertIn("EURUSD", instances)
        self.assertEqual(instances["EURUSD"], mock_trade_instance)

        MockTradeClass.assert_called_once()

    @patch("bbstrader.metatrader.trade.Trade")
    def test_create_trade_instance_scalar_risk_values(self, MockTradeClass):
        MockTradeClass.return_value = MagicMock()
        symbols = ["AUDUSD"]
        # base_params = {"expert_name": "ScalarRisk", "logger": mock_logger, "expert_id": 777}

        # Test the intended way for scalar values (i.e., in the base_params)
        corrected_base_params = {
            "expert_name": "ScalarRisk",
            "logger": mock_logger,
            "expert_id": 777,
            "daily_risk": 0.8,
            "max_risk": 1.8,
            "pchange_sl": 0.25,
        }
        instances_corrected = create_trade_instance(  # noqa: F841
            symbols, corrected_base_params.copy()
        )

        # The actual call will be: { 'symbol': 'AUDUSD', **corrected_base_params }
        # Defaults from create_trade_instance (like max_risk=10.0) are overridden by corrected_base_params
        final_expected_call_params = {"symbol": "AUDUSD", **corrected_base_params}
        MockTradeClass.assert_called_once_with(**final_expected_call_params)

    def test_create_trade_instance_value_errors(self):
        symbols = ["EURUSD", "GBPUSD"]
        base_params = {"logger": mock_logger}

        # Missing daily_risk for GBPUSD
        with self.assertRaisesRegex(
            ValueError, "Missing daily risk weight for symbol 'GBPUSD'"
        ):
            create_trade_instance(
                symbols, base_params.copy(), daily_risk={"EURUSD": 0.5}
            )

        # Missing max_risk for EURUSD
        with self.assertRaisesRegex(
            ValueError, "Missing maximum risk percentage for symbol 'EURUSD'"
        ):
            create_trade_instance(symbols, base_params.copy(), max_risk={"GBPUSD": 1.5})

        # Missing pchange_sl for GBPUSD (if pchange_sl is a dict)
        with self.assertRaisesRegex(
            ValueError, "Missing percentage change for symbol 'GBPUSD'"
        ):
            create_trade_instance(
                symbols, base_params.copy(), pchange_sl={"EURUSD": 0.2}
            )

        # Missing expert_id for EURUSD (if expert_id is a dict)
        params_with_id_dict = base_params.copy()
        params_with_id_dict["expert_id"] = {"GBPUSD": 111}
        with self.assertRaisesRegex(
            ValueError, "Missing expert ID for symbol 'EURUSD'"
        ):
            create_trade_instance(symbols, params_with_id_dict)

        # Empty symbols list
        with self.assertRaisesRegex(ValueError, "The 'symbols' list cannot be empty."):
            create_trade_instance([], base_params.copy())

        # Empty params dict (though logger is added by default in test setup, let's make it more explicit)
        with self.assertRaisesRegex(
            ValueError, "The 'params' dictionary cannot be empty."
        ):
            create_trade_instance(["EURUSD"], {})

    @patch(
        "bbstrader.metatrader.trade.Trade",
        side_effect=Exception("Trade Creation Failed"),
    )
    def test_create_trade_instance_trade_creation_fails(self, MockTradeClassFail):
        symbols = ["EURUSD"]
        base_params = {"expert_name": "FailTest", "logger": mock_logger}
        mock_logger.reset_mock()  # Reset from previous calls

        instances = create_trade_instance(symbols, base_params.copy())

        self.assertEqual(
            len(instances), 0
        )  # No instance should be added if creation fails
        MockTradeClassFail.assert_called_once()

    @patch("bbstrader.metatrader.trade.Trade")
    def test_create_trade_instance_partial_failure(self, MockTradeClassPartial):
        mock_eurusd_instance = MagicMock(name="EURUSD_Instance")
        # Simulate failure for GBPUSD
        MockTradeClassPartial.side_effect = [
            mock_eurusd_instance,
            Exception("GBPUSD Creation Failed"),
        ]

        symbols = ["EURUSD", "GBPUSD"]
        base_params = {
            "expert_name": "PartialFail",
            "logger": mock_logger,
            "expert_id": 123,
        }
        mock_logger.reset_mock()

        instances = create_trade_instance(symbols, base_params.copy())

        self.assertEqual(len(instances), 1)
        self.assertIn("EURUSD", instances)
        self.assertNotIn("GBPUSD", instances)
        self.assertEqual(MockTradeClassPartial.call_count, 2)


if __name__ == "__main__":
    unittest.main()
