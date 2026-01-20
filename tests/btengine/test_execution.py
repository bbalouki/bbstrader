import unittest
from collections import namedtuple
from datetime import datetime
from queue import Queue
from unittest.mock import Mock, patch

from bbstrader.btengine.data import DataHandler
from bbstrader.btengine.event import FillEvent, OrderEvent
from bbstrader.btengine.execution import MT5ExecutionHandler, SimExecutionHandler
from bbstrader.metatrader.utils import SymbolType


class TestSimExecutionHandler(unittest.TestCase):
    """
    Tests for the SimExecutionHandler class.
    """

    def setUp(self):
        """Set up test fixtures before each test."""
        self.events_queue = Queue()
        self.mock_data_handler = Mock(spec=DataHandler)
        self.mock_logger = Mock()

        self.test_time = datetime(2023, 1, 1, 12, 0, 0)
        self.mock_data_handler.get_latest_bar_datetime.return_value = self.test_time

        self.handler = SimExecutionHandler(
            events=self.events_queue,
            data=self.mock_data_handler,
            logger=self.mock_logger,
            commission=5.0,
            exchange="SIMEX",
        )

    def test_execute_order_creates_and_queues_fill_event(self):
        """
        Tests that SimExecutionHandler correctly creates a FillEvent
        from an OrderEvent and puts it onto the events queue.
        """
        order_event = OrderEvent(
            symbol="AAPL",
            order_type="MKT",
            quantity=100,
            direction="BUY",
            price=150.0,
            signal="LONG",
        )

        self.handler.execute_order(order_event)

        self.mock_data_handler.get_latest_bar_datetime.assert_called_once_with("AAPL")

        self.assertEqual(self.events_queue.qsize(), 1)
        fill_event = self.events_queue.get()

        self.assertIsInstance(fill_event, FillEvent)
        self.assertEqual(fill_event.timeindex, self.test_time)
        self.assertEqual(fill_event.symbol, "AAPL")
        self.assertEqual(fill_event.exchange, "SIMEX")
        self.assertEqual(fill_event.quantity, 100)
        self.assertEqual(fill_event.direction, "BUY")
        self.assertIsNone(fill_event.fill_cost)
        self.assertEqual(fill_event.commission, 5.0)
        self.assertEqual(fill_event.order, "LONG")

        expected_log_msg = (
            "BUY ORDER FILLED: SYMBOL=AAPL, QUANTITY=100, PRICE @150.0 EXCHANGE=SIMEX"
        )
        self.mock_logger.info.assert_called_once_with(
            expected_log_msg, custom_time=self.test_time
        )


SymbolInfo = namedtuple(
    "SymbolInfo", ["trade_contract_size", "volume_min", "volume_max"]
)


class TestMT5ExecutionHandler(unittest.TestCase):
    """
    Tests for the MT5ExecutionHandler class.
    The 'Account' class dependency is mocked to avoid any real API calls.
    """

    def setUp(self):
        """Set up test fixtures before each test."""
        self.events_queue = Queue()
        self.mock_data_handler = Mock(spec=DataHandler)
        self.mock_logger = Mock()

        self.test_time = datetime(2023, 1, 1, 12, 0, 0)
        self.mock_data_handler.get_latest_bar_datetime.return_value = self.test_time

        self.patcher = patch("bbstrader.btengine.execution.Account")
        MockAccount = self.patcher.start()

        self.mock_account_instance = MockAccount.return_value

        self.handler = MT5ExecutionHandler(
            events=self.events_queue,
            data=self.mock_data_handler,
            logger=self.mock_logger,
        )

    def tearDown(self):
        """Stop the patcher after each test."""
        self.patcher.stop()

    def test_execute_order_us_stock(self):
        """Tests commission calculation for a US stock."""
        symbol, qty, price = "AAPL", 100, 150.0

        self.mock_account_instance.get_symbol_type.return_value = SymbolType.STOCKS
        self.mock_account_instance.get_symbol_info.return_value = SymbolInfo(
            1, 0.01, 1000
        )
        self.mock_account_instance.get_stocks_from_country.return_value = [symbol]

        order_event = OrderEvent(symbol, "MKT", qty, "BUY", price, "LONG")
        self.handler.execute_order(order_event)

        fill_event = self.events_queue.get_nowait()

        expected_commission = max(1.0, 100 * 0.02)  # 2.0
        self.assertEqual(fill_event.commission, expected_commission)
        self.assertEqual(fill_event.exchange, "MT5")

    def test_execute_order_forex(self):
        """Tests commission calculation for a Forex pair."""
        symbol, qty, price = "EURUSD", 10000, 1.05

        self.mock_account_instance.get_symbol_type.return_value = SymbolType.FOREX
        self.mock_account_instance.get_symbol_info.return_value = SymbolInfo(
            100000, 0.01, 1000
        )
        self.mock_account_instance.broker.validate_lot_size.return_value = 0.1

        order_event = OrderEvent(symbol, "MKT", qty, "BUY", price, "LONG")
        self.handler.execute_order(order_event)

        fill_event = self.events_queue.get_nowait()

        # lot = 10000 * 1.05 / 100000 = 0.105
        # round(0.105, 2) is 0.10 (round half to even)
        # commission = 3.0 * 0.10 = 0.3
        expected_commission = 0.3
        self.assertAlmostEqual(fill_event.commission, expected_commission)

    def test_execute_order_commodity(self):
        """Tests commission calculation for a Commodity."""
        symbol, qty, price = "XAUUSD", 10, 1800

        self.mock_account_instance.get_symbol_type.return_value = SymbolType.COMMODITIES
        self.mock_account_instance.get_symbol_info.return_value = SymbolInfo(
            100, 0.01, 1000
        )
        self.mock_account_instance.broker.validate_lot_size.return_value = 0.1

        order_event = OrderEvent(symbol, "MKT", qty, "BUY", price, "LONG")
        self.handler.execute_order(order_event)

        fill_event = self.events_queue.get_nowait()

        expected_commission = 3.0 * 0.1
        self.assertAlmostEqual(fill_event.commission, expected_commission)

    def test_execute_order_index(self):
        """Tests commission calculation for an Index."""
        symbol, qty, price = "GER30", 10, 15000

        self.mock_account_instance.get_symbol_type.return_value = SymbolType.INDICES
        self.mock_account_instance.get_symbol_info.return_value = SymbolInfo(1, 0.1, 50)
        self.mock_account_instance.broker.validate_lot_size.return_value = 10

        order_event = OrderEvent(symbol, "MKT", qty, "BUY", price, "LONG")
        self.handler.execute_order(order_event)

        fill_event = self.events_queue.get_nowait()

        expected_commission = 0.25 * 10
        self.assertAlmostEqual(fill_event.commission, expected_commission)

    def test_lot_capping_at_minimum(self):
        """Tests that lot size is floored at the symbol's volume_min."""
        symbol, qty, price = "EURUSD", 100, 1.05

        self.mock_account_instance.get_symbol_type.return_value = SymbolType.FOREX
        self.mock_account_instance.get_symbol_info.return_value = SymbolInfo(
            100000, 0.01, 1000
        )
        self.mock_account_instance.broker.validate_lot_size.return_value = 0.01

        order_event = OrderEvent(symbol, "MKT", qty, "BUY", price, "LONG")
        self.handler.execute_order(order_event)

        fill_event = self.events_queue.get_nowait()

        expected_commission = 3.0 * 0.01
        self.assertAlmostEqual(fill_event.commission, expected_commission)

    def test_custom_commission_overrides_calculation(self):
        """Tests that a provided commission value overrides the calculated one."""
        handler_with_commission = MT5ExecutionHandler(
            events=self.events_queue, data=self.mock_data_handler, commission=99.99
        )

        symbol, qty, price = "AAPL", 100, 150.0
        self.mock_account_instance.get_symbol_type.return_value = SymbolType.STOCKS
        self.mock_account_instance.get_symbol_info.return_value = SymbolInfo(
            1, 0.01, 1000
        )
        self.mock_account_instance.get_stocks_from_country.return_value = [symbol]

        order_event = OrderEvent(symbol, "MKT", qty, "BUY", price, "LONG")
        handler_with_commission.execute_order(order_event)

        fill_event = self.events_queue.get_nowait()

        self.assertEqual(fill_event.commission, 99.99)


if __name__ == "__main__":
    unittest.main()
