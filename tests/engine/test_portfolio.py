import unittest
from unittest.mock import MagicMock
from bbstrader.btengine.portfolio import Portfolio
from bbstrader.btengine.event import FillEvent, SignalEvent


class MockPortfolio(Portfolio):
    def __init__(self, bars, events, start_date, initial_capital=100000.0, symbol_list=[]):
        super().__init__(bars, events, start_date, initial_capital, symbol_list=symbol_list)
        for symbol in symbol_list:
            if symbol not in self.current_positions:
                self.current_positions[symbol] = 0
            if symbol not in self.current_holdings:
                self.current_holdings[symbol] = 0.0
        self.current_holdings['Cash'] = initial_capital
        self.current_holdings['Commission'] = 0.0
        self.current_holdings['Total'] = initial_capital


class TestPortfolio(unittest.TestCase):

    def setUp(self):
        self.mock_bars = MagicMock()
        self.mock_events_queue = MagicMock()
        self.mock_bars.get_latest_bar_value.return_value = 100
        self.mock_bars.get_latest_bar_datetime.return_value = "2020-01-01"
        self.symbol_list = ['AAPL']
        self.start_date = "2020-01-01"
        self.initial_capital = 100000.0
        self.portfolio = MockPortfolio(
            self.mock_bars,
            self.mock_events_queue,
            self.start_date,
            self.initial_capital,
            symbol_list=self.symbol_list
        )

    def test_initial_setup(self):
        self.assertEqual(
            self.portfolio.current_holdings['Cash'], self.initial_capital)
        self.assertEqual(
            self.portfolio.current_holdings['Total'], self.initial_capital)

    def test_update_positions_from_fill(self):
        fill_event = FillEvent(timeindex="2020-01-01", symbol="AAPL", exchange="NASDAQ",
                               quantity=100, direction="BUY", fill_cost=None, commission=0.0)
        self.portfolio.update_positions_from_fill(fill_event)
        self.assertEqual(self.portfolio.current_positions["AAPL"], 100)

    def test_update_holdings_from_fill(self):
        fill_event = FillEvent(timeindex="2020-01-01", symbol="AAPL", exchange="NASDAQ",
                               quantity=100, direction="BUY", fill_cost=None, commission=0.0)
        self.portfolio.update_holdings_from_fill(fill_event)
        self.assertEqual(self.portfolio.current_holdings["AAPL"], 10000)
        self.assertEqual(
            self.portfolio.current_holdings['Cash'], self.initial_capital - 10000)

    def test_generate_naive_order(self):
        signal_event = SignalEvent(strategy_id=1, symbol="AAPL", datetime="2020-01-01",
                                   signal_type="LONG", quantity=100, strength=1.0)
        order_event = self.portfolio.generate_naive_order(signal_event)
        self.assertIsNotNone(order_event)
        self.assertEqual(order_event.quantity, 100)
        self.assertEqual(order_event.symbol, "AAPL")
        self.assertEqual(order_event.direction, "BUY")


if __name__ == '__main__':
    unittest.main()
