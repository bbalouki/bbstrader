import unittest
from unittest.mock import MagicMock
from btengine.data import HistoricCSVDataHandler
from btengine.event import MarketEvent

class MockHistoricCSVDataHandler(HistoricCSVDataHandler):
    def _open_convert_csv_files(self):
        self.symbol_data = {
            "AAPL": [("2020-01-01", 100), ("2020-01-02", 105)],
            "GOOG": [("2020-01-01", 1100), ("2020-01-02", 1150)]
        }
        for symbol in self.symbol_list:
            self.symbol_data[symbol] = iter(self.symbol_data[symbol])
            self.latest_symbol_data[symbol] = []

class TestDataHandler(unittest.TestCase):

    def setUp(self):
        self.events_queue = MagicMock()
        self.csv_dir = "mock/csv/dir"
        self.symbol_list = ["AAPL", "GOOG"]

        self.data_handler = MockHistoricCSVDataHandler(
            self.events_queue, self.csv_dir, self.symbol_list
        )

    def test_get_latest_bar(self):
        self.data_handler.update_bars()
        self.data_handler.update_bars()
        
        latest_bar = self.data_handler.get_latest_bar("AAPL")
        self.assertEqual(latest_bar[0], '2020-01-02') 

if __name__ == '__main__':
    unittest.main()