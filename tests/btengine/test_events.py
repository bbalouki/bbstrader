import io
import unittest
from contextlib import redirect_stdout
from datetime import datetime

from bbstrader.btengine.event import (
    Event,
    Events,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)


class TestEventsEnum(unittest.TestCase):
    """Tests the Events Enum."""

    def test_enum_values(self):
        self.assertEqual(Events.MARKET.value, "MARKET")
        self.assertEqual(Events.SIGNAL.value, "SIGNAL")
        self.assertEqual(Events.ORDER.value, "ORDER")
        self.assertEqual(Events.FILL.value, "FILL")


class TestMarketEvent(unittest.TestCase):
    """Tests the MarketEvent class."""

    def test_market_event_creation(self):
        """Test MarketEvent initialization and type."""
        event = MarketEvent()
        self.assertIsInstance(event, Event)
        self.assertEqual(event.type, Events.MARKET)


class TestSignalEvent(unittest.TestCase):
    """Tests the SignalEvent class."""

    def setUp(self):
        """Set up common data for tests."""
        self.timestamp = datetime(2023, 10, 27, 10, 30, 0)
        self.strategy_id = 1
        self.symbol = "AAPL"
        self.signal_type_long = "LONG"
        self.signal_type_short = "SHORT"
        self.signal_type_exit = "EXIT"
        self.quantity = 150
        self.strength = 1.5
        self.price = 170.50
        self.stoplimit = 168.00

    def test_signal_event_creation_long(self):
        """Test SignalEvent initialization with all parameters for LONG."""
        event = SignalEvent(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            datetime=self.timestamp,
            signal_type=self.signal_type_long,
            quantity=self.quantity,
            strength=self.strength,
            price=self.price,
            stoplimit=self.stoplimit,
        )
        self.assertIsInstance(event, Event)
        self.assertEqual(event.type, Events.SIGNAL)
        self.assertEqual(event.strategy_id, self.strategy_id)
        self.assertEqual(event.symbol, self.symbol)
        self.assertEqual(event.datetime, self.timestamp)
        self.assertEqual(event.signal_type, self.signal_type_long)
        self.assertEqual(event.quantity, self.quantity)
        self.assertEqual(event.strength, self.strength)
        self.assertEqual(event.price, self.price)
        self.assertEqual(event.stoplimit, self.stoplimit)

    def test_signal_event_creation_short_defaults(self):
        """Test SignalEvent initialization with defaults for SHORT."""
        event = SignalEvent(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            datetime=self.timestamp,
            signal_type=self.signal_type_short,
        )
        self.assertIsInstance(event, Event)
        self.assertEqual(event.type, Events.SIGNAL)
        self.assertEqual(event.strategy_id, self.strategy_id)
        self.assertEqual(event.symbol, self.symbol)
        self.assertEqual(event.datetime, self.timestamp)
        self.assertEqual(event.signal_type, self.signal_type_short)
        # Check defaults
        self.assertEqual(event.quantity, 100)  # Default quantity
        self.assertEqual(event.strength, 1.0)  # Default strength
        self.assertIsNone(event.price)  # Default price
        self.assertIsNone(event.stoplimit)  # Default stoplimit

    def test_signal_event_creation_exit(self):
        """Test SignalEvent initialization for EXIT."""
        event = SignalEvent(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            datetime=self.timestamp,
            signal_type=self.signal_type_exit,
            quantity=50,  # Different quantity for exit
        )
        self.assertIsInstance(event, Event)
        self.assertEqual(event.type, Events.SIGNAL)
        self.assertEqual(event.signal_type, self.signal_type_exit)
        self.assertEqual(event.quantity, 50)


class TestOrderEvent(unittest.TestCase):
    """Tests the OrderEvent class."""

    def setUp(self):
        """Set up common data for tests."""
        self.symbol = "GOOG"
        self.quantity = 75
        self.price_lmt = 135.00
        self.signal_ref = "Signal_123"

    def test_order_event_creation_mkt_buy(self):
        """Test MKT BUY OrderEvent initialization."""
        event = OrderEvent(
            symbol=self.symbol,
            order_type="MKT",
            quantity=self.quantity,
            direction="BUY",
            signal=self.signal_ref,
        )
        self.assertIsInstance(event, Event)
        self.assertEqual(event.type, Events.ORDER)
        self.assertEqual(event.symbol, self.symbol)
        self.assertEqual(event.order_type, "MKT")
        self.assertEqual(event.quantity, self.quantity)
        self.assertEqual(event.direction, "BUY")
        self.assertIsNone(event.price)  # Price is None for MKT
        self.assertEqual(event.signal, self.signal_ref)

    def test_order_event_creation_lmt_sell(self):
        """Test LMT SELL OrderEvent initialization."""
        event = OrderEvent(
            symbol=self.symbol,
            order_type="LMT",
            quantity=self.quantity,
            direction="SELL",
            price=self.price_lmt,
            signal=self.signal_ref,
        )
        self.assertIsInstance(event, Event)
        self.assertEqual(event.type, Events.ORDER)
        self.assertEqual(event.symbol, self.symbol)
        self.assertEqual(event.order_type, "LMT")
        self.assertEqual(event.quantity, self.quantity)
        self.assertEqual(event.direction, "SELL")
        self.assertEqual(event.price, self.price_lmt)
        self.assertEqual(event.signal, self.signal_ref)

    def test_print_order(self):
        """Test the print_order method output."""
        event = OrderEvent(
            symbol=self.symbol,
            order_type="LMT",
            quantity=self.quantity,
            direction="SELL",
            price=self.price_lmt,
        )
        # Redirect stdout to capture print output
        captured_output = io.StringIO()
        try:
            with redirect_stdout(captured_output):
                event.print_order()  # Call the *instance* method
            output = captured_output.getvalue().strip()
            expected_output = f"Order: Symbol={self.symbol}, Type=LMT, Quantity={self.quantity}, Direction=SELL, Price={self.price_lmt}"
            self.assertEqual(output, expected_output)
        except AttributeError:
            self.fail(
                "print_order is likely defined inside __init__ and not as a class method."
            )


class TestFillEvent(unittest.TestCase):
    """Tests the FillEvent class."""

    def setUp(self):
        """Set up common data for tests."""
        self.timestamp = datetime(2023, 10, 27, 10, 35, 15)
        self.symbol = "MSFT"
        self.exchange = "NYSE"
        self.fill_cost = 330.25
        self.order_ref = "Order_456"

    def create_fill_event(self, quantity, commission=None, direction="BUY"):
        """Helper method to create a FillEvent."""
        return FillEvent(
            timeindex=self.timestamp,
            symbol=self.symbol,
            exchange=self.exchange,
            quantity=quantity,
            direction=direction,
            fill_cost=self.fill_cost
            * quantity,  # Example fill cost based on price * quantity
            commission=commission,
            order=self.order_ref,
        )

    def test_fill_event_creation_buy_calculated_commission_small(self):
        """Test FillEvent BUY with calculated commission (qty <= 500)."""
        quantity = 100
        event = self.create_fill_event(quantity=quantity)

        self.assertIsInstance(event, Event)
        self.assertEqual(event.type, Events.FILL)
        self.assertEqual(event.timeindex, self.timestamp)
        self.assertEqual(event.symbol, self.symbol)
        self.assertEqual(event.exchange, self.exchange)
        self.assertEqual(event.quantity, quantity)
        self.assertEqual(event.direction, "BUY")
        self.assertEqual(event.fill_cost, self.fill_cost * quantity)
        self.assertEqual(event.order, self.order_ref)

        # Test commission calculation (<= 500)
        expected_commission = max(1.3, 0.013 * quantity)
        self.assertAlmostEqual(
            event.commission, expected_commission
        )  # Use assertAlmostEqual for floats
        self.assertAlmostEqual(event.commission, 1.30)  # 0.013 * 100 = 1.3

    def test_fill_event_creation_sell_calculated_commission_large(self):
        """Test FillEvent SELL with calculated commission (qty > 500)."""
        quantity = 600
        event = self.create_fill_event(quantity=quantity, direction="SELL")

        self.assertIsInstance(event, Event)
        self.assertEqual(event.type, Events.FILL)
        self.assertEqual(event.direction, "SELL")
        self.assertEqual(event.quantity, quantity)

        # Test commission calculation (> 500)
        expected_commission = max(1.3, 0.008 * quantity)
        self.assertAlmostEqual(event.commission, expected_commission)
        self.assertAlmostEqual(event.commission, 4.8)  # 0.008 * 600 = 4.8

    def test_fill_event_creation_calculated_commission_edge_500(self):
        """Test FillEvent with calculated commission (qty == 500)."""
        quantity = 500
        event = self.create_fill_event(quantity=quantity)

        # Test commission calculation (<= 500 rule applies)
        expected_commission = max(1.3, 0.013 * quantity)
        self.assertAlmostEqual(event.commission, expected_commission)
        self.assertAlmostEqual(event.commission, 6.5)  # 0.013 * 500 = 6.5

    def test_fill_event_creation_calculated_commission_minimum(self):
        """Test FillEvent with calculated commission hitting minimum."""
        quantity = 50
        event = self.create_fill_event(quantity=quantity)

        # Test commission calculation (minimum rule applies)
        expected_commission = max(1.3, 0.013 * quantity)  # 0.013 * 50 = 0.65
        self.assertAlmostEqual(event.commission, expected_commission)
        self.assertAlmostEqual(event.commission, 1.3)  # max(1.3, 0.65) = 1.3

    def test_fill_event_creation_provided_commission(self):
        """Test FillEvent with explicitly provided commission."""
        quantity = 200
        provided_commission = 5.0
        event = self.create_fill_event(
            quantity=quantity, commission=provided_commission
        )

        self.assertIsInstance(event, Event)
        self.assertEqual(event.type, Events.FILL)
        self.assertEqual(event.quantity, quantity)

        # Test that provided commission overrides calculation
        self.assertEqual(event.commission, provided_commission)

    def test_calculate_ib_commission_method(self):
        """Directly test the commission calculation method."""
        # Create a dummy event just to call the method (attributes don't matter here)
        event_small = FillEvent(datetime.now(), "SYM", "EXCH", 100, "BUY", 1000)
        event_large = FillEvent(datetime.now(), "SYM", "EXCH", 600, "BUY", 6000)
        event_edge = FillEvent(datetime.now(), "SYM", "EXCH", 500, "BUY", 5000)
        event_min = FillEvent(datetime.now(), "SYM", "EXCH", 10, "BUY", 100)

        self.assertAlmostEqual(
            event_small.calculate_ib_commission(), 1.3
        )  # max(1.3, 0.013*100=1.3)
        self.assertAlmostEqual(
            event_large.calculate_ib_commission(), 4.8
        )  # max(1.3, 0.008*600=4.8)
        self.assertAlmostEqual(
            event_edge.calculate_ib_commission(), 6.5
        )  # max(1.3, 0.013*500=6.5)
        self.assertAlmostEqual(
            event_min.calculate_ib_commission(), 1.3
        )  # max(1.3, 0.013*10=0.13)


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
