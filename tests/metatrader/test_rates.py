from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from bbstrader.metatrader.rates import Rates


@pytest.fixture
def mock_mt5_rates():
    """Fixture to mock the MetaTrader5 functions used in Rates."""
    with patch("bbstrader.metatrader.rates.Mt5") as mock_mt5:
        yield mock_mt5


@pytest.fixture
def mock_account():
    """Fixture to mock the Account class."""
    with patch("bbstrader.metatrader.rates.Account") as mock_account_class:
        mock_account_instance = mock_account_class.return_value
        mock_account_instance.get_symbol_type.return_value = "FOREX"
        mock_account_instance.get_currency_rates.return_value = {
            "bc": "EUR",
            "mc": "EUR",
            "pc": "USD",
            "ac": "USD",
        }
        yield mock_account_instance


@patch("bbstrader.metatrader.rates.TIMEFRAMES", {"1h": 2})
def test_rates_initialization(mock_account):
    """Test the initialization of the Rates class."""
    rates = Rates("EURUSD", "1h")
    assert rates.symbol == "EURUSD"
    assert rates.time_frame == 2  # 1h
    assert rates.start_pos == 0
    assert rates.count == 10_000_000


def test_rates_invalid_timeframe(mock_account):
    """Test that an invalid timeframe raises a ValueError."""
    with pytest.raises(ValueError):
        Rates("EURUSD", "invalid_tf")


@patch("bbstrader.metatrader.rates.Rates._fetch_data")
def test_get_rates_from_pos(mock_fetch_data, mock_account):
    """Test get_rates_from_pos."""
    mock_df = pd.DataFrame({"Close": [1.1, 1.2]})
    mock_fetch_data.return_value = mock_df

    rates = Rates("EURUSD", "1h")
    df = rates.get_rates_from_pos()

    mock_fetch_data.assert_called_once_with(
        0, 10_000_000, lower_colnames=False, utc=False
    )
    pd.testing.assert_frame_equal(df, mock_df)


@patch("bbstrader.metatrader.rates.Rates._fetch_data")
def test_get_historical_data(mock_fetch_data, mock_account):
    """Test get_historical_data."""
    mock_df = pd.DataFrame({"Close": [1.1, 1.2]})
    mock_fetch_data.return_value = mock_df

    rates = Rates("EURUSD", "1h")
    date_from = datetime(2023, 1, 1)
    date_to = datetime(2023, 1, 2)
    df = rates.get_historical_data(date_from, date_to)

    mock_fetch_data.assert_called_once()
    pd.testing.assert_frame_equal(df, mock_df)


@patch("bbstrader.metatrader.rates.Rates")
def test_download_historical_data(mock_rates_class, mock_account):
    """Test the download_historical_data standalone function."""
    from bbstrader.metatrader.rates import download_historical_data

    mock_rates_instance = mock_rates_class.return_value
    mock_df = pd.DataFrame({"Close": [1.1, 1.2]})
    mock_rates_instance.get_historical_data.return_value = mock_df

    date_from = datetime(2023, 1, 1)
    date_to = datetime(2023, 1, 2)

    df = download_historical_data("EURUSD", "1h", date_from, date_to)

    mock_rates_class.assert_called_with("EURUSD", "1h")
    mock_rates_instance.get_historical_data.assert_called_with(
        date_from=date_from,
        date_to=date_to,
        save_csv=False,
        utc=False,
        filter=False,
        lower_colnames=True,
    )
    pd.testing.assert_frame_equal(df, mock_df)


@patch("bbstrader.metatrader.rates.Rates")
def test_get_data_from_pos(mock_rates_class, mock_account):
    """Test the get_data_from_pos standalone function."""
    from bbstrader.metatrader.rates import get_data_from_pos

    mock_rates_instance = mock_rates_class.return_value
    mock_df = pd.DataFrame({"Close": [1.1, 1.2]})
    mock_rates_instance.get_rates_from_pos.return_value = mock_df

    df = get_data_from_pos("EURUSD", "1h")

    mock_rates_class.assert_called()
    mock_rates_instance.get_rates_from_pos.assert_called()
    pd.testing.assert_frame_equal(df, mock_df)


@patch("bbstrader.metatrader.rates.Rates")
def test_get_data_from_date(mock_rates_class, mock_account):
    """Test the get_data_from_date standalone function."""
    from bbstrader.metatrader.rates import get_data_from_date

    mock_rates_instance = mock_rates_class.return_value
    mock_df = pd.DataFrame({"Close": [1.1, 1.2]})
    mock_rates_instance.get_rates_from.return_value = mock_df

    date_from = datetime(2023, 1, 1)

    df = get_data_from_date("EURUSD", "1h", date_from)

    mock_rates_class.assert_called_with("EURUSD", "1h")
    mock_rates_instance.get_rates_from.assert_called()
    pd.testing.assert_frame_equal(df, mock_df)
