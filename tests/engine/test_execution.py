import unittest
from datetime import datetime
from unittest.mock import MagicMock
from bbstrader.btengine.event import (
    Event, MarketEvent, SignalEvent, OrderEvent, FillEvent
)
class ExecutionHandler:
    def execute_order(self, event):
        raise NotImplementedError("Should implement execute_order()")

class SimulatedExecutionHandler(ExecutionHandler):
    def __init__(self, events):
        self.events = events

    def execute_order(self, event):
        if event.type == 'ORDER':
            fill_event = FillEvent(
                datetime.now(), event.symbol, 
                'SIMULATED_EXCHANGE', event.quantity, 
                event.direction, None
            )
            self.events.append(fill_event)

class TestExecutionHandler(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.execution_handler = SimulatedExecutionHandler(self.events)

    def test_order_execution(self):
        order_event = OrderEvent('AAPL', 'MKT', 100, 'BUY')
        self.execution_handler.execute_order(order_event)
        self.assertEqual(len(self.events), 1)
        fill_event = self.events[0]

        self.assertIsInstance(fill_event, FillEvent)
        self.assertEqual(fill_event.symbol, 'AAPL')
        self.assertEqual(fill_event.quantity, 100)
        self.assertEqual(fill_event.direction, 'BUY')
        self.assertEqual(fill_event.exchange, 'SIMULATED_EXCHANGE')

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
