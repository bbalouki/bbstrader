import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from bbstrader.btengine.data import (
    CSVDataHandler,
    EODHDataHandler,
    FMPDataHandler,
    MT5DataHandler,
    YFDataHandler,
)
from bbstrader.btengine.event import MarketEvent


class TestCSVDataHandler(unittest.TestCase):
    @patch("pandas.read_csv")
    @patch("pandas.DataFrame.to_csv")
    def setUp(self, mock_to_csv, mock_read_csv):
        # Mock CSV content
        date_rng = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "datetime": date_rng,
                "open": np.random.rand(5),
                "high": np.random.rand(5),
                "low": np.random.rand(5),
                "close": np.random.rand(5),
                "adj_close": np.random.rand(5),
                "volume": np.random.randint(100, 1000, size=5),
            }
        ).set_index("datetime")
        mock_read_csv.return_value = df

        self.events = Queue()
        self.symbol_list = ["AAPL"]
        self.handler = CSVDataHandler(
            self.events, self.symbol_list, csv_dir="/fake/dir"
        )

        # Manually trigger symbol iterrows generator
        self.handler.symbol_data["AAPL"] = iter(df.iterrows())
        self.handler.latest_symbol_data["AAPL"] = list(df.iterrows())

    def test_get_latest_bar(self):
        latest = self.handler.get_latest_bar("AAPL")
        self.assertIsInstance(latest, tuple)

    def test_get_latest_bars(self):
        bars = self.handler.get_latest_bars("AAPL", N=2)
        self.assertEqual(len(bars), 2)

    def test_get_latest_bar_value(self):
        val = self.handler.get_latest_bar_value("AAPL", "close")
        self.assertIsInstance(val, float)

    def test_get_latest_bars_values(self):
        vals = self.handler.get_latest_bars_values("AAPL", "close", N=3)
        self.assertEqual(len(vals), 3)

    def test_update_bars(self):
        self.handler.update_bars()
        self.assertFalse(self.events.empty())


class TestMT5DataHandler(unittest.TestCase):
    """
    Tests for the MT5DataHandler class, mocked to run without an MT5 terminal.
    """

    def setUp(self):
        """Set up a temporary directory and a mock events queue for each test."""
        # Create a temporary directory that will be automatically cleaned up
        self.temp_dir_context = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir_context.name)

        # A mock queue to check if events are being put correctly
        self.events = Queue()
        self.symbol_list = ["EURUSD"]
        self.start_dt = datetime(2023, 1, 1)
        self.end_dt = datetime(2023, 1, 5)

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        self.temp_dir_context.cleanup()

    def _create_sample_mt5_df(self):
        """Helper function to create a realistic mock DataFrame."""
        date_rng = pd.date_range(start="2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "open": np.random.uniform(1.05, 1.06, 5),
                "high": np.random.uniform(1.06, 1.07, 5),
                "low": np.random.uniform(1.04, 1.05, 5),
                "close": np.random.uniform(1.05, 1.06, 5),
                "volume": np.random.randint(10000, 50000, size=5),
            },
            index=date_rng,
        )
        df.index.name = "time"  # MT5 data uses 'time' index
        # The handler normalizes column names to lowercase
        return df

    # We only need to patch the function that makes the external call.
    @patch("bbstrader.btengine.data.download_historical_data")
    def test_initialization_downloads_and_caches_data(self, mock_download):
        """
        Verify that the handler correctly calls the download function,
        caches the data to a CSV, and loads it.
        """
        sample_df = self._create_sample_mt5_df()
        mock_download.return_value = sample_df
        time_frame_arg = "D1"
        handler = MT5DataHandler(
            self.events,
            self.symbol_list,
            time_frame=time_frame_arg,
            mt5_start=self.start_dt,
            mt5_end=self.end_dt,
            data_dir=self.temp_dir_path,
        )

        # 1. Was the download function called with the correct parameters?
        mock_download.assert_called_once_with(
            # Explicitly set arguments
            symbol="EURUSD",
            timeframe=time_frame_arg,
            date_from=self.start_dt,
            date_to=self.end_dt,
            utc=False,
            filter=False,
            fill_na=False,
            lower_colnames=True,
            # Arguments passed via **self.kwargs
            time_frame=time_frame_arg,
            mt5_start=self.start_dt,
            mt5_end=self.end_dt,
            data_dir=self.temp_dir_path,
            backtest=True,
        )

        # 2. Was a CSV file created in our temporary directory?
        expected_filepath = Path(self.temp_dir_path) / "EURUSD.csv"
        self.assertTrue(expected_filepath.exists())

        # 3. Was the data loaded correctly into the handler?
        # The data is stored as a generator, so we check by consuming one item.
        self.assertIn("EURUSD", handler.data)
        first_bar_tuple = next(handler.symbol_data["EURUSD"])

        # The first element of the tuple is the timestamp
        self.assertEqual(first_bar_tuple[0], pd.Timestamp("2023-01-01"))
        # The second is the Series of bar data
        self.assertAlmostEqual(first_bar_tuple[1]["close"], sample_df["close"].iloc[0])

    @patch("bbstrader.btengine.data.download_historical_data")
    def test_update_bars_and_get_latest_bar(self, mock_download):
        """
        Verify the full data flow from initialization to updating and retrieving bars.
        """
        sample_df = self._create_sample_mt5_df()
        mock_download.return_value = sample_df

        handler = MT5DataHandler(
            self.events, self.symbol_list, data_dir=self.temp_dir_path
        )

        # Update bars for the first time
        handler.update_bars()

        # Assert 1
        self.assertEqual(self.events.qsize(), 1)  # A MarketEvent should be in the queue
        self.assertIsInstance(self.events.get(), MarketEvent)
        self.assertEqual(
            handler.get_latest_bar_datetime("EURUSD"), pd.Timestamp("2023-01-01")
        )
        self.assertAlmostEqual(
            handler.get_latest_bar_value("EURUSD", "close"), sample_df["close"].iloc[0]
        )

        #  Update bars for the second time
        handler.update_bars()

        # Assert 2
        self.assertEqual(
            handler.get_latest_bar_datetime("EURUSD"), pd.Timestamp("2023-01-02")
        )
        latest_bars_df = handler.get_latest_bars("EURUSD", N=2)
        self.assertEqual(len(latest_bars_df), 2)
        self.assertAlmostEqual(
            latest_bars_df.iloc[1]["close"], sample_df["close"].iloc[1]
        )

    @patch("bbstrader.btengine.data.download_historical_data")
    def test_download_failure_raises_valueerror(self, mock_download):
        """
        Verify that if the download function fails, a descriptive ValueError is raised.
        """
        # Arrange: Configure the mock to raise an error
        mock_download.side_effect = Exception("MT5 Connection Failed")

        # Act & Assert: Check that instantiating the handler raises the correct error
        with self.assertRaisesRegex(
            ValueError, "Error downloading EURUSD: .*MT5 Connection Failed.*"
        ):
            MT5DataHandler(self.events, self.symbol_list, data_dir=self.temp_dir_path)


class TestYFDataHandler(unittest.TestCase):
    """Tests for the YFDataHandler class."""

    def setUp(self):
        """Set up a temporary directory and mock queue for each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mock_events_queue = MagicMock(spec=Queue)
        self.symbol_list = ["AAPL"]
        self.start_date = "2023-01-01"
        self.end_date = "2023-01-05"

    def tearDown(self):
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()

    def _create_sample_yf_df(self):
        """Creates a sample DataFrame mimicking yfinance output."""
        dates = pd.to_datetime(["2023-01-03", "2023-01-04"])
        data = {
            "Open": [130.0, 127.0],
            "High": [131.0, 128.0],
            "Low": [129.0, 126.0],
            "Close": [130.5, 127.5],
            "Volume": [100000, 110000],
        }
        df = pd.DataFrame(data, index=dates)
        df.index.name = "Date"
        # YFDataHandler expects 'Adj Close', but it's added if missing.
        # We'll test the case where it's provided.
        df["Adj Close"] = df["Close"]
        return df

    @patch("bbstrader.btengine.data.yf.download")
    def test_successful_initialization_and_caching(self, mock_yf_download):
        """Verify successful initialization, download, and caching."""
        sample_df = self._create_sample_yf_df()
        mock_yf_download.return_value = sample_df

        handler = YFDataHandler(
            self.mock_events_queue,
            self.symbol_list,
            yf_start=self.start_date,
            yf_end=self.end_date,
            data_dir=self.temp_dir.name,
        )

        # 1. Verify yf.download was called correctly
        mock_yf_download.assert_called_once_with(
            "AAPL",
            start=self.start_date,
            end=self.end_date,
            multi_level_index=False,
            auto_adjust=True,
            progress=False,
        )

        # 2. Verify the CSV file was created
        expected_filepath = Path(self.temp_dir.name) / "AAPL.csv"
        self.assertTrue(expected_filepath.exists())

        # 3. Verify data is loaded into the handler
        self.assertIn("AAPL", handler.data)
        df_from_csv = pd.read_csv(expected_filepath)
        self.assertEqual(len(df_from_csv), 2)
        self.assertIn(
            "adj_close", df_from_csv.columns
        )  # Check for normalized column name

    @patch("bbstrader.btengine.data.yf.download")
    def test_download_failure_raises_value_error(self, mock_yf_download):
        """Test that a download exception is handled and raises a ValueError."""
        mock_yf_download.side_effect = Exception("Network Error")

        with self.assertRaisesRegex(
            ValueError, "Error downloading AAPL: .*Network Error.*"
        ):
            YFDataHandler(
                self.mock_events_queue,
                self.symbol_list,
                yf_start=self.start_date,
                data_dir=self.temp_dir.name,
            )

    @patch("bbstrader.btengine.data.yf.download")
    def test_empty_data_raises_value_error(self, mock_yf_download):
        """Test that empty data from API raises a ValueError."""
        mock_yf_download.return_value = pd.DataFrame()

        with self.assertRaisesRegex(ValueError, "Error downloading AAPL: 'Close'"):
            YFDataHandler(
                self.mock_events_queue,
                self.symbol_list,
                yf_start=self.start_date,
                data_dir=self.temp_dir.name,
            )


class TestEODHDataHandler(unittest.TestCase):
    """Tests for the EODHDataHandler class."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mock_events_queue = MagicMock(spec=Queue)
        self.symbol_list = ["MSFT.US"]
        self.start_date = "2023-01-01"
        self.end_date = "2023-01-05"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_sample_eodhd_df(self):
        """Creates a sample DataFrame mimicking EODHD daily output."""
        data = {
            "date": ["2023-01-03", "2023-01-04"],
            "open": [240.0, 235.0],
            "high": [241.0, 236.0],
            "low": [239.0, 234.0],
            "close": [240.5, 235.5],
            "adjusted_close": [240.5, 235.5],
            "volume": [200000, 210000],
            "symbol": ["MSFT.US", "MSFT.US"],
            "interval": ["d", "d"],
        }
        return pd.DataFrame(data).set_index("date")

    def test_missing_api_key_raises_error(self):
        """Verify that not providing an API key raises a ValueError."""
        with self.assertRaisesRegex(ValueError, "API key is required"):
            EODHDataHandler(
                self.mock_events_queue,
                self.symbol_list,
                eodhd_start=self.start_date,
                eodhd_api_key=None,  # Explicitly set to None
                data_dir=self.temp_dir.name,
            )

    @patch("bbstrader.btengine.data.APIClient")
    def test_successful_initialization_daily(self, mock_api_client):
        """Test successful initialization for daily data."""
        mock_instance = mock_api_client.return_value
        mock_instance.get_historical_data.return_value = self._create_sample_eodhd_df()

        handler = EODHDataHandler(
            self.mock_events_queue,
            self.symbol_list,
            eodhd_start=self.start_date,
            eodhd_end=self.end_date,
            eodhd_api_key="fake_key",
            data_dir=self.temp_dir.name,
        )

        mock_api_client.assert_called_once_with(api_key="fake_key")
        mock_instance.get_historical_data.assert_called_once_with(
            symbol="MSFT.US",
            interval="d",
            iso8601_start=self.start_date,
            iso8601_end=self.end_date,
        )

        expected_filepath = Path(self.temp_dir.name) / "MSFT.US.csv"
        self.assertTrue(expected_filepath.exists())
        self.assertIn("MSFT.US", handler.data)

    @patch("bbstrader.btengine.data.APIClient")
    def test_empty_data_raises_value_error(self, mock_api_client):
        """Test that empty data from the API raises a ValueError."""
        mock_instance = mock_api_client.return_value
        mock_instance.get_historical_data.return_value = pd.DataFrame()

        with self.assertRaisesRegex(ValueError, "No data found"):
            EODHDataHandler(
                self.mock_events_queue,
                self.symbol_list,
                eodhd_api_key="fake_key",
                data_dir=self.temp_dir.name,
            )


class TestFMPDataHandler(unittest.TestCase):
    """Tests for the FMPDataHandler class."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mock_events_queue = MagicMock(spec=Queue)
        self.symbol_list = ["GOOG"]
        self.start_date = "2023-01-01"
        self.end_date = "2023-01-05"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_sample_fmp_df(self):
        """Creates a sample DataFrame mimicking FMP output."""
        dates = pd.period_range("2023-01-03", periods=2, freq="D")
        data = {
            "Open": [90.0, 88.0],
            "High": [91.0, 89.0],
            "Low": [89.0, 87.0],
            "Close": [90.5, 88.5],
            "Volume": [300000, 310000],
            "Dividends": [0, 0],
            "Return": [0.01, -0.01],
            "Volatility": [0.1, 0.1],
            "Excess Return": [0.005, -0.015],
            "Excess Volatility": [0.1, 0.1],
            "Cumulative Return": [1.01, 0.99],
        }
        df = pd.DataFrame(data, index=pd.Index(dates, name="date"))
        return df

    def test_missing_api_key_raises_error(self):
        """Verify that not providing an API key raises a ValueError."""
        with self.assertRaisesRegex(ValueError, "API key is required"):
            FMPDataHandler(
                self.mock_events_queue,
                self.symbol_list,
                fmp_start=self.start_date,
                fmp_api_key=None,  # Explicitly set to None
                data_dir=self.temp_dir.name,
            )

    @patch("bbstrader.btengine.data.Toolkit")
    def test_successful_initialization_and_formatting(self, mock_toolkit):
        """Test successful initialization and data formatting."""
        mock_instance = mock_toolkit.return_value
        mock_instance.get_historical_data.return_value = self._create_sample_fmp_df()

        handler = FMPDataHandler(
            self.mock_events_queue,
            self.symbol_list,
            fmp_start=self.start_date,
            fmp_end=self.end_date,
            fmp_api_key="fake_key",
            data_dir=self.temp_dir.name,
        )

        mock_toolkit.assert_called_once_with(
            "GOOG",
            api_key="fake_key",
            start_date=self.start_date,
            end_date=self.end_date,
            benchmark_ticker=None,
            progress_bar=False,
        )
        mock_instance.get_historical_data.assert_called_once_with(
            period="daily", progress_bar=False
        )

        expected_filepath = Path(self.temp_dir.name) / "GOOG.csv"
        self.assertTrue(expected_filepath.exists())
        self.assertIn("GOOG", handler.data)

        # Check that formatting worked by reading the CSV
        df_from_csv = pd.read_csv(expected_filepath)
        self.assertNotIn("Dividends", df_from_csv.columns)
        self.assertNotIn("Return", df_from_csv.columns)
        self.assertIn("adj_close", df_from_csv.columns)

    @patch("bbstrader.btengine.data.Toolkit")
    def test_empty_data_raises_value_error(self, mock_toolkit):
        """Test that empty data from the API raises a ValueError."""
        mock_instance = mock_toolkit.return_value
        mock_instance.get_historical_data.return_value = pd.DataFrame()

        with self.assertRaisesRegex(ValueError, "No data found"):
            FMPDataHandler(
                self.mock_events_queue,
                self.symbol_list,
                fmp_api_key="fake_key",
                data_dir=self.temp_dir.name,
            )

    @patch("bbstrader.btengine.data.Toolkit")
    @patch("bbstrader.btengine.event.MarketEvent")
    def test_data_flow_and_base_handler_methods(self, mock_market_event, mock_toolkit):
        """Test update_bars and get_latest_* methods from the base class."""
        mock_instance = mock_toolkit.return_value
        sample_df = self._create_sample_fmp_df()
        mock_instance.get_historical_data.return_value = sample_df

        handler = FMPDataHandler(
            self.mock_events_queue,
            self.symbol_list,
            fmp_api_key="fake_key",
            data_dir=self.temp_dir.name,
        )

        # 1. First update
        handler.update_bars()

        # Check that a MarketEvent was put on the queue
        self.mock_events_queue.put.assert_called_once()
        event_put_on_queue = self.mock_events_queue.put.call_args[0][0]
        self.assertIsInstance(
            event_put_on_queue, MarketEvent
        )  # Check against the real class

        # 2. Check latest bar data
        latest_bar = handler.get_latest_bar("GOOG")[1]  # Bar data is the 2nd element
        self.assertAlmostEqual(latest_bar["close"], 90.5)

        latest_dt = handler.get_latest_bar_datetime("GOOG")
        self.assertEqual(latest_dt, pd.to_datetime("2023-01-03"))

        latest_val = handler.get_latest_bar_value("GOOG", "high")
        self.assertAlmostEqual(latest_val, 91.0)

        # 3. Second update
        handler.update_bars()

        # Check latest bars data (N=2)
        latest_2_bars = handler.get_latest_bars("GOOG", N=2)
        self.assertEqual(len(latest_2_bars), 2)
        self.assertAlmostEqual(latest_2_bars.iloc[1]["close"], 88.5)

        latest_2_vals = handler.get_latest_bars_values("GOOG", "low", N=2)
        self.assertEqual(len(latest_2_vals), 2)
        self.assertAlmostEqual(latest_2_vals[0], 89.0)
        self.assertAlmostEqual(latest_2_vals[1], 87.0)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
