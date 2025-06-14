from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

mock_mt5_module = MagicMock()
mock_mt5_module.TIMEFRAME_M1 = 1
mock_mt5_module.TIMEFRAME_D1 = 16408
mock_mt5_module.TIMEFRAME_W1 = 32769
mock_mt5_module.TIMEFRAME_H12 = 16396
mock_mt5_module.TIMEFRAME_MN1 = 49153


from bbstrader.metatrader.rates import (  # noqa: E402
    Rates,
    download_historical_data,
    get_data_from_date,  # noqa: F401
    get_data_from_pos,  # noqa: F401
)
from bbstrader.metatrader.utils import SymbolType  # noqa: E402


@pytest.fixture
def mock_mt5_api(mocker):
    """Fixture to configure the mocked MetaTrader5 API calls."""
    rates_data = np.array(
        [
            (1672617600, 1.06, 1.09, 1.05, 1.07, 1200),  # 2023-01-02 00:00:00 UTC
            (1672704000, 1.07, 1.10, 1.06, 1.08, 1100),  # 2023-01-03 00:00:00 UTC
            (1672790400, 1.08, 1.11, 1.07, 1.09, 1300),  # 2023-01-04 00:00:00 UTC
        ],
        dtype=[
            ("time", "<i8"),
            ("open", "<f8"),
            ("high", "<f8"),
            ("low", "<f8"),
            ("close", "<f8"),
            ("tick_volume", "<i8"),
        ],
    )

    # 1. Mock for bbstrader.metatrader.rates.Mt5 (data fetching)
    # This is the Mt5 alias used in rates.py
    mock_rates_mt5 = mocker.patch("bbstrader.metatrader.rates.Mt5")
    mock_rates_mt5.copy_rates_from_pos.return_value = rates_data
    mock_rates_mt5.copy_rates_range.return_value = rates_data
    mock_rates_mt5.copy_rates_from.return_value = rates_data

    # 2. Mock for bbstrader.metatrader.utils.MT5 (for last_error and TIMEFRAME constants)
    mock_utils_mt5 = mocker.patch("bbstrader.metatrader.utils.MT5")
    mock_utils_mt5.last_error.return_value = (0, "Success")  # Default success

    
    mock_utils_mt5.TIMEFRAME_M1 = 1
    mock_utils_mt5.TIMEFRAME_M2 = 2
    mock_utils_mt5.TIMEFRAME_M3 = 3
    mock_utils_mt5.TIMEFRAME_M4 = 4
    mock_utils_mt5.TIMEFRAME_M5 = 5
    mock_utils_mt5.TIMEFRAME_M6 = 6
    mock_utils_mt5.TIMEFRAME_M10 = 10
    mock_utils_mt5.TIMEFRAME_M12 = 12
    mock_utils_mt5.TIMEFRAME_M15 = 15
    mock_utils_mt5.TIMEFRAME_M20 = 20
    mock_utils_mt5.TIMEFRAME_M30 = 30
    mock_utils_mt5.TIMEFRAME_H1 = 16385
    mock_utils_mt5.TIMEFRAME_H2 = 16386
    mock_utils_mt5.TIMEFRAME_H3 = 16387
    mock_utils_mt5.TIMEFRAME_H4 = 16388
    mock_utils_mt5.TIMEFRAME_H6 = 16390
    mock_utils_mt5.TIMEFRAME_H8 = 16392
    mock_utils_mt5.TIMEFRAME_H12 = (
        mock_mt5_module.TIMEFRAME_H12
    )  # Use value from global mock
    mock_utils_mt5.TIMEFRAME_D1 = (
        mock_mt5_module.TIMEFRAME_D1
    )  # Use value from global mock
    mock_utils_mt5.TIMEFRAME_W1 = (
        mock_mt5_module.TIMEFRAME_W1
    )  # Use value from global mock
    mock_utils_mt5.TIMEFRAME_MN1 = (
        mock_mt5_module.TIMEFRAME_MN1
    )  # Use value from global mock

    # 3. Patch bbstrader.metatrader.utils.TIMEFRAMES directly to ensure it uses correct integer values
    patched_timeframes = {
        "1m": mock_utils_mt5.TIMEFRAME_M1,
        "2m": mock_utils_mt5.TIMEFRAME_M2,
        "3m": mock_utils_mt5.TIMEFRAME_M3,
        "4m": mock_utils_mt5.TIMEFRAME_M4,
        "5m": mock_utils_mt5.TIMEFRAME_M5,
        "6m": mock_utils_mt5.TIMEFRAME_M6,
        "10m": mock_utils_mt5.TIMEFRAME_M10,
        "12m": mock_utils_mt5.TIMEFRAME_M12,
        "15m": mock_utils_mt5.TIMEFRAME_M15,
        "20m": mock_utils_mt5.TIMEFRAME_M20,
        "30m": mock_utils_mt5.TIMEFRAME_M30,
        "1h": mock_utils_mt5.TIMEFRAME_H1,
        "2h": mock_utils_mt5.TIMEFRAME_H2,
        "3h": mock_utils_mt5.TIMEFRAME_H3,
        "4h": mock_utils_mt5.TIMEFRAME_H4,
        "6h": mock_utils_mt5.TIMEFRAME_H6,
        "8h": mock_utils_mt5.TIMEFRAME_H8,
        "12h": mock_utils_mt5.TIMEFRAME_H12,
        "D1": mock_utils_mt5.TIMEFRAME_D1,
        "W1": mock_utils_mt5.TIMEFRAME_W1,
        "MN1": mock_utils_mt5.TIMEFRAME_MN1,
    }
    mocker.patch.dict("bbstrader.metatrader.utils.TIMEFRAMES", patched_timeframes)

    # The fixture should return the mock that is directly called by the Rates class for data fetching.
    return mock_rates_mt5


@pytest.fixture
def mock_account(mocker):
    """
    Fixture to mock the Account class *specifically where it's used in the rates module*.
    This is the key change.
    """
    mock_account_instance = MagicMock()
    mock_account_instance.get_symbol_type.return_value = SymbolType.FOREX
    mock_account_instance.get_symbol_info.return_value = MagicMock(
        path="Group\\Forex\\EURUSD"
    )
    mock_account_instance.get_currency_rates.return_value = {"mc": "USD"}

    mocker.patch(
        "bbstrader.metatrader.rates.Account", return_value=mock_account_instance
    )

    return mock_account_instance


@pytest.fixture
def mock_check_connection(mocker):
    """Fixture to mock the check_mt5_connection function."""
    return mocker.patch("bbstrader.metatrader.rates.check_mt5_connection")


def test_rates_initialization(mock_mt5_api, mock_account, mock_check_connection):
    """
    Test the successful initialization of the Rates class with proper mocks.
    """
    rates = Rates(symbol="EURUSD", timeframe="D1", start_pos=0, count=100)

    # 1. Check if MT5 connection was checked
    mock_check_connection.assert_called_once()

    # 2. Check if Account was instantiated (the patch in mock_account handles this)
    # We can check its methods were NOT called since filter=False by default.
    mock_account.get_symbol_type.assert_not_called()

    # 3. Check if data was fetched on initialization
    # Access TIMEFRAME_D1 from mock_mt5_module as it's defined there for tests
    mock_mt5_api.copy_rates_from_pos.assert_called_once_with(
        "EURUSD", mock_mt5_module.TIMEFRAME_D1, 0, 100
    )

    # 4. Check if the internal data DataFrame is correctly populated
    assert isinstance(rates._Rates__data, pd.DataFrame)
    assert not rates._Rates__data.empty
    assert "Close" in rates._Rates__data.columns
    assert rates._Rates__data.index.name == "Date"


def test_rates_initialization_invalid_timeframe():
    """
    Test that initializing with an invalid timeframe raises a ValueError.
    """
    with pytest.raises(ValueError, match="Unsupported time frame 'INVALID_TF'"):
        # We don't need full mocks for this, as it fails before they are used.
        Rates(symbol="EURUSD", timeframe="INVALID_TF")


def test_get_start_pos_with_string_date(
    mocker, mock_check_connection, mock_account, mock_mt5_api
):
    """
    Test the _get_pos_index calculation for a string date start_pos.
    """
    mock_dt = mocker.patch("bbstrader.metatrader.rates.datetime")
    mock_dt.now.return_value = datetime(2023, 12, 31)

    rates = Rates(
        symbol="EURUSD", timeframe="D1", start_pos="2023-12-20", session_duration=24
    )
    assert rates.start_pos == 6


def test_get_historical_data(mock_mt5_api, mock_account, mock_check_connection):
    """
    Test the get_historical_data method.
    """
    rates = Rates("GBPUSD", "D1")
    date_from = datetime(2023, 1, 1)
    date_to = datetime(2023, 1, 31)

    df = rates.get_historical_data(
        date_from=date_from, date_to=date_to, lower_colnames=True
    )

    mock_mt5_api.copy_rates_range.assert_called_once_with(
        "GBPUSD", mock_mt5_module.TIMEFRAME_D1, date_from, date_to
    )
    assert isinstance(df, pd.DataFrame)
    assert "close" in df.columns
    assert df.index.name == "date"


def test_data_filtering_for_stock(mock_mt5_api, mock_account, mock_check_connection):
    """
    Test the filtering mechanism for a stock symbol.
    """
    mock_account.get_symbol_type.return_value = SymbolType.STOCKS
    mock_account.get_stocks_from_exchange.return_value = ["AAPL"]

    with patch("bbstrader.metatrader.rates.AMG_EXCHANGES", ["'XNYS'"]):
        rates = Rates("AAPL", "D1")
        date_from = pd.Timestamp("2023-01-01")
        date_to = pd.Timestamp("2023-01-04")

        df = rates.get_historical_data(
            date_from=date_from, date_to=date_to, filter=True
        )

    mock_account.get_symbol_type.assert_called()
    mock_account.get_stocks_from_exchange.assert_called()

    assert not df.isnull().values.any()


def test_data_filtering_with_fill_na(mock_mt5_api, mock_account, mock_check_connection):
    """
    Test filtering with fill_na=True for a D1 timeframe.
    """
    mock_account.get_symbol_type.return_value = SymbolType.FOREX  # Use a 24/5 calendar

    rates = Rates("EURUSD", "D1")
    date_from = pd.Timestamp("2023-01-01")
    date_to = pd.Timestamp("2023-01-04")

    df = rates.get_historical_data(
        date_from=date_from, date_to=date_to, filter=True, fill_na=True
    )

    # The 'us_futures' calendar (used for FOREX) is closed on Jan 2nd.
    # With fill_na=True, this day should be present and filled.
    assert not df.isnull().values.any()


def test_properties_access(mock_mt5_api, mock_account, mock_check_connection):
    """
    Test the data properties like .open, .close, .returns.
    """
    rates = Rates("EURUSD", "D1")

    pd.testing.assert_series_equal(
        rates.close, rates._Rates__data["Close"], check_names=False
    )

    returns = rates.returns
    assert isinstance(returns, pd.Series)
    assert not returns.isnull().any()
    expected_return = (1.08 - 1.07) / 1.07
    assert np.isclose(returns.iloc[0], expected_return)


def test_download_historical_data_wrapper(mocker):
    """
    Test the wrapper function to ensure it instantiates Rates and calls the correct method.
    """
    # Here, we mock the entire Rates class since we are testing the wrapper function, not the class itself.
    mock_rates_class = mocker.patch("bbstrader.metatrader.rates.Rates")
    mock_rates_instance = mock_rates_class.return_value

    date_from = datetime(2022, 1, 1)

    download_historical_data(
        symbol="USDCAD", timeframe="H12", date_from=date_from, filter=True
    )

    # Check that Rates was initialized correctly
    # Note: the timeframe "H12" will be passed as a string to the constructor.
    mock_rates_class.assert_called_once_with("USDCAD", "H12")

    # Check that the method on the instance was called correctly
    mock_rates_instance.get_historical_data.assert_called_once()
    call_args, call_kwargs = mock_rates_instance.get_historical_data.call_args
    assert call_kwargs["date_from"] == date_from
    assert call_kwargs["filter"] is True
