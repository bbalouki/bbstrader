import datetime
import time
import unittest


try:
    import MetaTrader5 as mt5
except ImportError:
    import bbstrader.compat  # noqa: F401

from bbstrader.api.handlers import Mt5Handlers
from bbstrader.api.client import MetaTraderClient  # type: ignore

# CONFIGURATION
LOGIN: int = 12345
PASSWORD: str = ""
SERVER: str = ""
PATH: str = ""
SYMBOL: str = ""


class Handlers:
    pass


class TestMetaTraderClient(unittest.TestCase):
    client: MetaTraderClient = None

    @classmethod
    def setUpClass(cls):
        """Initialize the client once before all tests."""
        cls.client = MetaTraderClient(Mt5Handlers)
        if not cls.client.initialize(
            path=PATH,
            login=LOGIN,
            password=PASSWORD,
            server=SERVER,
            timeout=5000,
            portable=False,
        ):
            raise Exception(f"Setup failed: {cls.client.last_error()}")

        # Enable symbol for data tests
        cls.client.symbol_select(SYMBOL, True)
        time.sleep(1)  # Allow time for data syncing

    @classmethod
    def tearDownClass(cls):
        """Clean up at the very end."""
        if cls.client:
            cls.client.shutdown()

    def setUp(self):
        """
        Run before EVERY test.
        Ensures that if a previous test (like shutdown) killed the connection,
        it is restored before the next test runs.
        """
        if not self.client.terminal_info():
            self.client.initialize(
                path=PATH,
                login=LOGIN,
                password=PASSWORD,
                server=SERVER,
                timeout=5000,
                portable=False,
            )
            self.client.symbol_select(SYMBOL, True)

    # =========================================================================
    # INIT & CONNECTION METHODS
    # =========================================================================
    def test_initialize_simple(self):
        """Test: initialize(self)"""
        result = self.client.initialize()
        self.assertTrue(result, "initialize() failed")

    def test_initialize_path(self):
        """Test: initialize(self, path: str)"""
        result = self.client.initialize(PATH)
        self.assertTrue(result, "initialize(path) failed")

    def test_initialize_full(self):
        """Test: initialize(self, path, login, password, server, timeout, portable)"""
        result = self.client.initialize(PATH, LOGIN, PASSWORD, SERVER, 5000, False)
        self.assertTrue(result, "initialize(full_args) failed")

    def test_login(self):
        """Test: login(self, login, password, server, timeout)"""
        result = self.client.login(LOGIN, PASSWORD, SERVER, 5000)
        self.assertTrue(result, "login() failed")

    def test_shutdown(self):
        """Test: shutdown(self)"""
        self.client.shutdown()

        info = self.client.terminal_info()
        self.assertIsNone(info, "Shutdown did not close the connection")

        is_connected = self.client.initialize(
            path=PATH,
            login=LOGIN,
            password=PASSWORD,
            server=SERVER,
            timeout=5000,
            portable=False,
        )
        self.assertTrue(
            is_connected, "Failed to recover connection after shutdown test"
        )

    # =========================================================================
    # INFO METHODS
    # =========================================================================

    def test_version(self):
        """Test: version(self)"""
        ver = self.client.version()
        self.assertIsInstance(ver, tuple)
        self.assertEqual(len(ver), 3)

    def test_last_error(self):
        """Test: last_error(self)"""
        err = self.client.last_error()
        self.assertIsInstance(err, tuple)
        self.assertEqual(len(err), 2)

    def test_account_info(self):
        """Test: account_info(self)"""
        info = self.client.account_info()
        self.assertIsNotNone(info)
        self.assertEqual(info.login, LOGIN)

    def test_terminal_info(self):
        """Test: terminal_info(self)"""
        info = self.client.terminal_info()
        self.assertIsNotNone(info)
        self.assertTrue(info.connected)

    # =========================================================================
    # SYMBOL METHODS
    # =========================================================================

    def test_symbol_select(self):
        """Test: symbol_select(self, symbol, enable)"""
        res = self.client.symbol_select(SYMBOL, True)
        self.assertTrue(res)

    def test_symbol_info(self):
        """Test: symbol_info(self, symbol)"""
        info = self.client.symbol_info(SYMBOL)
        self.assertIsNotNone(info)
        self.assertEqual(info.name, SYMBOL)

    def test_symbol_info_tick(self):
        """Test: symbol_info_tick(self, symbol)"""
        tick = self.client.symbol_info_tick(SYMBOL)
        self.assertIsNotNone(tick)
        self.assertGreater(tick.ask, 0.0)

    def test_symbols_total(self):
        """Test: symbols_total(self)"""
        total = self.client.symbols_total()
        self.assertIsInstance(total, int)
        self.assertGreater(total, 0)

    def test_symbols_get_all(self):
        """Test: symbols_get(self)"""
        syms = self.client.symbols_get()
        self.assertIsInstance(syms, list)
        self.assertGreater(len(syms), 0)

    def test_symbols_get_group(self):
        """Test: symbols_get(self, group)"""
        syms = self.client.symbols_get(group="USD*")
        if syms is not None:
            self.assertIsInstance(syms, list)

    # =========================================================================
    # MARKET BOOK METHODS
    # =========================================================================

    def test_market_book_add(self):
        """Test: market_book_add(self, symbol)"""
        res = self.client.market_book_add(SYMBOL)
        self.assertTrue(res)

    def test_market_book_get(self):
        """Test: market_book_get(self, symbol)"""
        self.client.market_book_add(SYMBOL)
        book = self.client.market_book_get(SYMBOL)
        if book is not None:
            self.assertIsInstance(book, list)

    def test_market_book_release(self):
        """Test: market_book_release(self, symbol)"""
        res = self.client.market_book_release(SYMBOL)
        self.assertTrue(res)

    # =========================================================================
    # COPY DATA METHODS
    # =========================================================================

    def test_copy_rates_from_int(self):
        """Test: copy_rates_from(self, symbol, timeframe, date_from, count)"""
        now = datetime.datetime.now()
        rates = self.client.copy_rates_from(
            SYMBOL, mt5.TIMEFRAME_H1, int(now.timestamp()), 5
        )
        self.assertIsNotNone(rates)
        self.assertEqual(len(rates), 5)

    def test_copy_rates_from_date(self):
        """Test: copy_rates_from(self, symbol, timeframe, date_from, count)"""
        now = datetime.datetime.now()
        rates = self.client.copy_rates_from(SYMBOL, mt5.TIMEFRAME_H1, now, 5)
        self.assertIsNotNone(rates)
        self.assertEqual(len(rates), 5)

    def test_copy_rates_from_pos(self):
        """Test: copy_rates_from_pos(self, symbol, timeframe, start_pos, count)"""
        rates = self.client.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 5)
        self.assertIsNotNone(rates)
        self.assertEqual(len(rates), 5)

    def test_copy_rates_range_int(self):
        """Test: copy_rates_range(self, symbol, timeframe, date_from, date_to)"""
        now = datetime.datetime.now()
        start = now - datetime.timedelta(days=1)
        rates = self.client.copy_rates_range(
            SYMBOL, mt5.TIMEFRAME_H1, int(start.timestamp()), int(now.timestamp())
        )
        self.assertIsNotNone(rates)

    def test_copy_rates_range_date(self):
        """Test: copy_rates_range(self, symbol, timeframe, date_from, date_to)"""
        now = datetime.datetime.now()
        start = now - datetime.timedelta(days=1)
        rates = self.client.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H1, start, now)
        self.assertIsNotNone(rates)

    def test_copy_ticks_from_int(self):
        """Test: copy_ticks_from(self, symbol, date_from, count, flags)"""
        now = datetime.datetime.now()
        ticks = self.client.copy_ticks_from(
            SYMBOL, int(now.timestamp()), 5, mt5.COPY_TICKS_ALL
        )
        if ticks is not None:
            self.assertEqual(len(ticks), 5)

    def test_copy_ticks_from_date(self):
        """Test: copy_ticks_from(self, symbol, date_from, count, flags)"""
        now = datetime.datetime.now()
        ticks = self.client.copy_ticks_from(SYMBOL, now, 5, mt5.COPY_TICKS_ALL)
        if ticks is not None:
            self.assertEqual(len(ticks), 5)

    def test_copy_ticks_range_int(self):
        """Test: copy_ticks_range(self, symbol, date_from, date_to, flags)"""
        now = datetime.datetime.now()
        start = now - datetime.timedelta(hours=1)
        ticks = self.client.copy_ticks_range(
            SYMBOL, int(start.timestamp()), int(now.timestamp()), mt5.COPY_TICKS_ALL
        )
        if ticks is not None:
            self.assertTrue(hasattr(ticks, "size"))

    def test_copy_ticks_range_date(self):
        """Test: copy_ticks_range(self, symbol, date_from, date_to, flags)"""
        now = datetime.datetime.now()
        start = now - datetime.timedelta(hours=1)
        ticks = self.client.copy_ticks_range(SYMBOL, start, now, mt5.COPY_TICKS_ALL)
        if ticks is not None:
            self.assertTrue(hasattr(ticks, "size"))

    # =========================================================================
    # ORDERS (LIVE) METHODS
    # =========================================================================

    def test_orders_total(self):
        """Test: orders_total(self)"""
        total = self.client.orders_total()
        self.assertIsInstance(total, int)

    def test_orders_get_all(self):
        """Test: orders_get(self)"""
        orders = self.client.orders_get()
        if orders is not None:
            self.assertIsInstance(orders, list)

    def test_orders_get_symbol(self):
        """Test: orders_get(self, symbol)"""
        orders = self.client.orders_get(SYMBOL)
        if orders is not None:
            self.assertIsInstance(orders, list)

    def test_orders_get_by_group(self):
        """Test: orders_get_by_group(self, group)"""
        orders = self.client.orders_get_by_group("*")
        if orders is not None:
            self.assertIsInstance(orders, list)

    def test_order_get_by_ticket(self):
        """Test: order_get_by_ticket(self, ticket)"""
        # We try to get a fake ticket 12345. It should return None, not crash.
        order = self.client.order_get_by_ticket(12345)
        self.assertIsNone(order)

    # =========================================================================
    # POSITIONS METHODS
    # =========================================================================

    def test_positions_total(self):
        """Test: positions_total(self)"""
        total = self.client.positions_total()
        self.assertIsInstance(total, int)

    def test_positions_get_all(self):
        """Test: positions_get(self)"""
        pos = self.client.positions_get()
        if pos is not None:
            self.assertIsInstance(pos, list)

    def test_positions_get_symbol(self):
        """Test: positions_get(self, symbol)"""
        pos = self.client.positions_get(SYMBOL)
        if pos is not None:
            self.assertIsInstance(pos, list)

    def test_positions_get_by_group(self):
        """Test: positions_get_by_group(self, group)"""
        pos = self.client.positions_get_by_group("*")
        if pos is not None:
            self.assertIsInstance(pos, list)

    def test_position_get_by_ticket(self):
        """Test: position_get_by_ticket(self, ticket)"""
        pos = self.client.position_get_by_ticket(12345)
        self.assertIsNone(pos)

    # =========================================================================
    # HISTORY METHODS
    # =========================================================================

    def test_history_orders_total_from(self):
        """Test: history_orders_total(self, date_from, date_to)"""
        now = datetime.datetime.now()
        start = now - datetime.timedelta(days=30)
        total = self.client.history_orders_total(
            int(start.timestamp()), int(now.timestamp())
        )
        self.assertIsInstance(total, int)

    def test_history_orders_total_from_date(self):
        """Test: history_orders_total(self, date_from, date_to)"""
        now = datetime.datetime.now()
        start = now - datetime.timedelta(days=30)
        total = self.client.history_orders_total(start, now)
        self.assertIsInstance(total, int)

    def test_history_orders_get_group(self):
        """Test: history_orders_get(self, date_from, date_to, group)"""
        now = datetime.datetime.now()
        start = now - datetime.timedelta(days=30)
        orders = self.client.history_orders_get(
            int(start.timestamp()), int(now.timestamp()), group="*"
        )
        if orders is not None:
            self.assertIsInstance(orders, list)

    def test_history_orders_get_ticket(self):
        """Test: history_orders_get(self, ticket)"""
        res = self.client.history_orders_get(12345)
        self.assertTrue(res is None or isinstance(res, list))

    def test_history_orders_get_by_pos(self):
        """Test: history_orders_get_by_pos(self, position_id)"""
        res = self.client.history_orders_get_by_pos(12345)
        if res is not None:
            self.assertIsInstance(res, list)

    def test_history_deals_total_int(self):
        """Test: history_deals_total(self, date_from, date_to)"""
        now = datetime.datetime.now()
        start = now - datetime.timedelta(days=30)
        total = self.client.history_deals_total(
            int(start.timestamp()), int(now.timestamp())
        )
        self.assertIsInstance(total, int)

    def test_history_deals_total_date(self):
        """Test: history_deals_total(self, date_from, date_to)"""
        now = datetime.datetime.now()
        start = now - datetime.timedelta(days=30)
        total = self.client.history_deals_total(start, now)
        self.assertIsInstance(total, int)

    def test_history_deals_get_group(self):
        """Test: history_deals_get(self, date_from, date_to, group)"""
        now = datetime.datetime.now()
        start = now - datetime.timedelta(days=30)
        deals = self.client.history_deals_get(
            int(start.timestamp()), int(now.timestamp()), group="*"
        )
        if deals is not None:
            self.assertIsInstance(deals, list)

    def test_history_deals_get_ticket(self):
        """Test: history_deals_get(self, ticket)"""
        res = self.client.history_deals_get(12345)
        self.assertTrue(res is None or isinstance(res, list))

    def test_history_deals_get_by_pos(self):
        """Test: history_deals_get_by_pos(self, position_id)"""
        res = self.client.history_deals_get_by_pos(12345)
        if res is not None:
            self.assertIsInstance(res, list)

    # =========================================================================
    # TRADING / CALCULATION METHODS
    # =========================================================================

    def test_order_calc_margin(self):
        """Test: order_calc_margin(self, action, symbol, volume, price)"""
        margin = self.client.order_calc_margin(mt5.ORDER_TYPE_BUY, SYMBOL, 0.01, 1.1000)
        self.assertIsNotNone(margin)

    def test_order_calc_profit(self):
        """Test: order_calc_profit(self, action, symbol, volume, price_open, price_close)"""
        profit = self.client.order_calc_profit(
            mt5.ORDER_TYPE_BUY, SYMBOL, 0.01, 1.1000, 1.1100
        )
        self.assertIsNotNone(profit)

    def test_order_check(self):
        """Test: order_check(self, request)"""
        tick = self.client.symbol_info_tick(SYMBOL)
        price = tick.ask if tick else 1.0

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": SYMBOL,
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_BUY_LIMIT,
            "price": price - 0.5,
            "magic": 999,
        }
        res = self.client.order_check(request)
        self.assertIsNotNone(res)
        self.assertTrue(hasattr(res, "retcode"))

    def test_order_send(self):
        """Test: order_send(self, request)"""
        tick = self.client.symbol_info_tick(SYMBOL)
        price = tick.ask if tick else 1.0

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": SYMBOL,
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_BUY_LIMIT,
            "price": price - 1.0,
            "magic": 999999,
            "comment": "Unittest",
            "type_time": mt5.ORDER_TIME_DAY,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }

        result = self.client.order_send(request)
        self.assertIsNotNone(result)

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            time.sleep(0.5)
            cancel_req = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": result.order,
            }
            self.client.order_send(cancel_req)


if __name__ == "__main__":
    print("Starting MetaTraderClient Test Suite...")
    unittest.main()
