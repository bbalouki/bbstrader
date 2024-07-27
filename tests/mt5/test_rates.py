import pytest
import os
import pandas as pd
from datetime import datetime, timedelta
from mtrader5.rates import Rates
import MetaTrader5 as Mt5

# Make sure the symbols use have full acces for trading
# Make sure an instance of MT5 is running with a demo account

# Fixture to check MT5 terminal initialization
@pytest.fixture(scope="module", autouse=True)
def check_mt5_terminal():
    if not Mt5.initialize():
        pytest.skip("MT5 terminal not running", allow_module_level=True)

    yield

    Mt5.shutdown()

def test_rates_initialization():
    symbol = "SPY"
    time_frame = "D1"
    start_pos = 0
    count = 10
    rates_instance = Rates(symbol, time_frame, start_pos, count)
    assert rates_instance.symbol == symbol
    assert rates_instance.start_pos == start_pos
    assert rates_instance.count == count
    assert isinstance(rates_instance.data, pd.DataFrame)

def test_unsupported_time_frame():
    with pytest.raises(ValueError):
        Rates("SPY", "1y", 0, 10)  # '1y' is an unsupported time frame

def test_rate_retrieval():
    rates_instance = Rates("SPY", "D1", 0, 10)
    data = rates_instance.get_rate_frame()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

@pytest.mark.parametrize("date_from, date_to", [
    (datetime.now() - timedelta(days=1), datetime.now()),
    (datetime.now() - timedelta(weeks=1), datetime.now()),
])
def test_get_history(date_from, date_to):
    rates_instance = Rates("SPY", "1h", 0, 100)
    history = rates_instance.get_history(date_from, date_to)
    assert isinstance(history, pd.DataFrame)
    assert not history.empty

def test_get_history_with_save():
    symbol= "SPY"
    rates_instance = Rates(symbol, "D1", 0, 100)
    date_from = datetime.now() - timedelta(days=10)
    date_to = datetime.now()
    history = rates_instance.get_history(date_from, date_to, save=True)
    assert isinstance(history, pd.DataFrame)
    assert not history.empty
    # Construct the expected file name based on the symbol
    expected_file_name = f"{symbol}.csv"
    # Check if the file exists
    assert os.path.exists(expected_file_name)
    # Load the saved CSV file
    saved_data = pd.read_csv(
        expected_file_name, index_col='Date', parse_dates=True)
    assert len(saved_data) == len(history)
    # Check if the columns match
    assert all(saved_data.columns == history.columns)
    # Clean up: Remove the CSV file after the test
    os.remove(expected_file_name)
