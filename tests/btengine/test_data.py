import unittest
from unittest.mock import MagicMock, patch, mock_open, call
import pandas as pd
import numpy as np
from queue import Queue
from bbstrader.btengine.data import *
from pathlib import Path

class TestDataHandler(unittest.TestCase):
    def test_abstract_methods(self):
        with self.assertRaises(TypeError):
            dh = DataHandler()


class TestBaseCSVDataHandler(unittest.TestCase):
    def setUp(self):
        self.events_queue = Queue()
        self.symbol_list = ['AAPL', 'GOOG']
        self.csv_dir = '/path/to/csvs'

        # Mock the _load_and_process_data to avoid actual file IO
        with patch.object(BaseCSVDataHandler, '_load_and_process_data'):
            self.handler = BaseCSVDataHandler(
                self.events_queue, self.symbol_list, self.csv_dir
            )

    def test_get_latest_bar(self):
        self.handler.latest_symbol_data = {
            'AAPL': [(pd.Timestamp('2022-01-01'), pd.Series({'Open': 100}))],
            'GOOG': [(pd.Timestamp('2022-01-02'), pd.Series({'Open': 150}))]
        }
        self.assertEqual(self.handler.get_latest_bar('AAPL')[0], pd.Timestamp('2022-01-01'))

    def test_get_latest_bars(self):
        self.handler.latest_symbol_data = {
            'AAPL': [
                (pd.Timestamp('2022-01-01'), pd.Series({'Open': 100})),
                (pd.Timestamp('2022-01-02'), pd.Series({'Open': 105}))
            ]
        }
        self.assertEqual(len(self.handler.get_latest_bars('AAPL', N=2)), 2)

    def test_get_latest_bar_datetime(self):
        self.handler.latest_symbol_data = {
            'AAPL': [(pd.Timestamp('2022-01-01'), pd.Series({'Open': 100}))]
        }
        self.assertEqual(self.handler.get_latest_bar_datetime('AAPL'), pd.Timestamp('2022-01-01'))

    def test_get_latest_bar_value(self):
        self.handler.latest_symbol_data = {
            'AAPL': [(pd.Timestamp('2022-01-01'), pd.Series({'Open': 100}))]
        }
        self.assertEqual(self.handler.get_latest_bar_value('AAPL', 'Open'), 100)

    def test_get_latest_bars_values(self):
        self.handler.latest_symbol_data = {
            'AAPL': [
                (pd.Timestamp('2022-01-01'), pd.Series({'Open': 100})),
                (pd.Timestamp('2022-01-02'), pd.Series({'Open': 105}))
            ]
        }
        values = self.handler.get_latest_bars_values('AAPL', 'Open', N=2)
        np.testing.assert_array_equal(values, np.array([100, 105]))

    def test_update_bars(self):
        # Initialize latest_symbol_data with empty lists
        self.handler.latest_symbol_data = {s: [] for s in self.symbol_list}
        
        # Mock _get_new_bar and MarketEvent
        mock_market_event = MagicMock()
        mock_new_bar = MagicMock(return_value=iter([(pd.Timestamp('2022-01-03'), pd.Series({'Open': 110}))]))

        self.handler._get_new_bar = mock_new_bar
        with patch('bbstrader.btengine.event.MarketEvent', mock_market_event):
            self.handler.update_bars()

        self.assertEqual(self.handler.latest_symbol_data['AAPL'][-1][1]['Open'], 110)


class TestHistoricCSVDataHandler(unittest.TestCase):
    def setUp(self):
        self.events_queue = Queue()
        self.symbol_list = ['AAPL', 'GOOG']
        self.csv_dir = '/path/to/csvs'

        with patch('bbstrader.btengine.data.BaseCSVDataHandler.__init__', return_value=None):
            self.handler = HistoricCSVDataHandler(self.events_queue, self.symbol_list, csv_dir=self.csv_dir)
            self.handler.csv_dir = self.csv_dir  


    def test_initialization(self):
        self.assertEqual(self.handler.csv_dir, self.csv_dir)


class TestMT5HistoricDataHandler(unittest.TestCase):
    @patch('bbstrader.metatrader.rates.Rates')
    @patch('bbstrader.btengine.data.pd.read_csv')
    @patch.object(MT5HistoricDataHandler, '_download_data', return_value='/mocked/path')
    def setUp(self, mock_download_data, mock_read_csv, mock_rates):
        self.events_queue = Queue()
        self.symbol_list = ['AAPL', 'GOOG']

        # Mock read_csv to return a dummy DataFrame
        mock_read_csv.return_value = pd.DataFrame({
            'Datetime': pd.to_datetime(['2022-01-01', '2022-01-02']),
            'Open': [100, 105],
            'High': [110, 115],
            'Low': [90, 95],
            'Close': [108, 110],
            'Adj Close': [107, 109],
            'Volume': [1000, 1200]
        }).set_index('Datetime')

        self.handler = MT5HistoricDataHandler(
            self.events_queue, self.symbol_list,
            time_frame='D1', max_bars=100, start_pos=0
        )
        self.handler.csv_dir = '/mocked/path' 


    def test_initialization(self):
        self.assertEqual(self.handler.tf, 'D1')
        self.assertEqual(self.handler.max_bars, 100)


class TestYFHistoricDataHandler(unittest.TestCase):
    def setUp(self):
        self.yf_start='2020-01-01' 
        self.yf_end_date='2020-12-31'
        self.cache_dir = 'yf_cache'
        self.handler  = YFHistoricDataHandler(
        Queue(), ["SPY", "QQQ"],
        yf_start=self.yf_start, yf_end=self.yf_end_date
        )
    def test_initialization(self):
        self.assertEqual(self.handler.start_date, self.yf_start)
        self.assertEqual(self.handler.end_date, self.yf_end_date)
        self.assertEqual(self.handler.cache_dir, self.cache_dir)

if __name__ == '__main__':
    unittest.main()
