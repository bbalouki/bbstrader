import unittest
from datetime import datetime
from unittest.mock import MagicMock
from btengine.event import (
    Event, MarketEvent, SignalEvent, OrderEvent, FillEvent
)
# Event Test Cases
class TestEventClasses(unittest.TestCase):
    def test_market_event(self):
        event = MarketEvent()
        self.assertEqual(event.type, 'MARKET')
    
    def test_signal_event(self):
        event = SignalEvent(1, 'AAPL', datetime.now(), 'LONG', 100, 1.0)
        self.assertEqual(event.type, 'SIGNAL')
        self.assertEqual(event.symbol, 'AAPL')
        self.assertEqual(event.signal_type, 'LONG')
        self.assertEqual(event.quantity, 100)
        self.assertEqual(event.strength, 1.0)
    
    def test_order_event(self):
        event = OrderEvent('GOOG', 'MKT', 50, 'BUY')
        self.assertEqual(event.type, 'ORDER')
        self.assertEqual(event.symbol, 'GOOG')
        self.assertEqual(event.order_type, 'MKT')
        self.assertEqual(event.quantity, 50)
        self.assertEqual(event.direction, 'BUY')
    
    def test_fill_event(self):
        event = FillEvent(datetime.now(), 'MSFT', 'NASDAQ', 100, 'SELL', 200, 1.3)
        self.assertEqual(event.type, 'FILL')
        self.assertEqual(event.symbol, 'MSFT')
        self.assertEqual(event.exchange, 'NASDAQ')
        self.assertEqual(event.quantity, 100)
        self.assertEqual(event.direction, 'SELL')
        self.assertEqual(event.fill_cost, 200)
        self.assertEqual(event.commission, 1.3)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
