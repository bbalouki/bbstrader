import unittest
import pandas as pd
from datetime import datetime
from unittest.mock import patch,  MagicMock
from bbstrader.metatrader.rates import Rates, TIMEFRAMES



class TestRates(unittest.TestCase):

    def setUp(self):
        """Set up any required parameters or mock objects."""
        self.symbol = "EURUSD"
        self.time_frame = "D1"
        self.start_pos = 0
        self.count = 100
        self.session_duration = 8
        self.rates = Rates(
            symbol=self.symbol, 
            time_frame=self.time_frame, 
            start_pos=self.start_pos, 
            count=self.count, 
            session_duration=self.session_duration
        )

    def test_initialization(self):
        """Test the initialization of the Rates class."""
        self.assertEqual(self.rates.symbol, self.symbol)
        self.assertEqual(self.rates.time_frame, TIMEFRAMES[self.time_frame])
        self.assertEqual(self.rates.count, self.count)
        self.assertEqual(self.rates.start_pos, self.start_pos)

    def test_invalid_time_frame(self):
        """Test that an invalid timeframe raises an error."""
        with self.assertRaises(ValueError):
            Rates(symbol="EURUSD", time_frame="invalid_timeframe")

    def test_get_start_pos_with_int(self):
        """Test that start_pos is correctly assigned when an integer is provided."""
        self.assertEqual(self.rates.start_pos, self.start_pos)

    def test_get_start_pos_with_str(self):
        """Test that start_pos is correctly calculated when a string date is provided."""
        start_date = "2024-01-01"
        rates = Rates(symbol="EURUSD", time_frame="1h", start_pos=start_date, session_duration=8)
        expected_start_pos = rates._get_pos_index(start_date, "1h", 8)
        self.assertEqual(rates.start_pos, expected_start_pos)

    def test_get_pos_index(self):
        """Test the calculation of position index from a start date."""
        start_date = "2024-01-01"
        time_frame = "1h"
        session_duration = 8
        rates = Rates(symbol="EURUSD", time_frame="1h", start_pos=start_date, session_duration=session_duration)
        index = rates._get_pos_index(start_date, time_frame, session_duration)
        self.assertGreaterEqual(index, 0)

    def test_format_dataframe(self):
        """Test the formatting of the DataFrame returned by MT5."""
        data = {
            'time': [1637854800],
            'open': [1.1234],
            'high': [1.1250],
            'low': [1.1200],
            'close': [1.1220],
            'tick_volume': [1000]
        }
        df = pd.DataFrame(data)
        formatted_df = self.rates._format_dataframe(df)
        
        self.assertIn('Open', formatted_df.columns)
        self.assertIn('High', formatted_df.columns)
        self.assertIn('Adj Close', formatted_df.columns)
        self.assertEqual(formatted_df.index.name, 'Date')

    def test_get_rates_from_pos_no_mt5(self):
        """Test getting rates from position without relying on MT5."""
        # Assuming _fetch_data is supposed to return a DataFrame
        mock_df = pd.DataFrame({
            'Date': [datetime(2024, 1, 1)],
            'Open': [1.1234],
            'High': [1.1250],
            'Low': [1.1200],
            'Close': [1.1220],
            'Adj Close': [1.1220],
            'Volume': [1000]
        })
        
        self.rates._fetch_data = MagicMock(return_value=mock_df)
        df = self.rates.get_rates_from_pos()
        
        self.assertIsNotNone(df)
        self.assertEqual(df.shape[0], 1)
        self.assertIn('Open', df.columns)

    def test_get_historical_data_no_mt5(self):
        """Test getting historical data within a date range without relying on MT5."""
        mock_df = pd.DataFrame({
            'Date': [datetime(2024, 1, 1)],
            'Open': [1.1234],
            'High': [1.1250],
            'Low': [1.1200],
            'Close': [1.1220],
            'Adj Close': [1.1220],
            'Volume': [1000]
        })

        self.rates._fetch_data = MagicMock(return_value=mock_df)
        date_from = datetime(2024, 1, 1)
        date_to = datetime(2024, 1, 10)
        df = self.rates.get_historical_data(date_from, date_to)
        
        self.assertIsNotNone(df)
        self.assertEqual(df.shape[0], 1)
        self.assertIn('Open', df.columns)

    def test_save_csv(self):
        """Test saving historical data to a CSV file."""
        mock_df = pd.DataFrame({
            'Date': [datetime(2024, 1, 1)],
            'Open': [1.1234],
            'High': [1.1250],
            'Low': [1.1200],
            'Close': [1.1220],
            'Adj Close': [1.1220],
            'Volume': [1000]
        })

        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            self.rates._fetch_data = MagicMock(return_value=mock_df)
            date_from = datetime(2024, 1, 1)
            date_to = datetime(2024, 1, 10)
            df = self.rates.get_historical_data(date_from, date_to, save_csv=True)

            mock_to_csv.assert_called_once_with(f"{self.symbol}.csv")
            self.assertIsNotNone(df)

if __name__ == '__main__':
    unittest.main()
