import warnings
from datetime import datetime
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bbstrader.btengine.event import FillEvent, MarketEvent, OrderEvent, SignalEvent
from bbstrader.btengine.portfolio import Portfolio

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)


class MockDataHandler:
    """A mock DataHandler to control market data during tests."""

    def __init__(self, symbol_list, initial_data):
        self.symbols = symbol_list
        self.latest_symbol_data = initial_data

    def get_latest_bar_datetime(self, symbol):
        return self.latest_symbol_data[symbol]["datetime"]

    def get_latest_bar_value(self, symbol, val_type):
        # val_type can be 'adj_close' or 'close'
        return self.latest_symbol_data[symbol]["price"]

    def update_bar(self, symbol, dt, price):
        self.latest_symbol_data[symbol]["datetime"] = dt
        self.latest_symbol_data[symbol]["price"] = price


# PYTEST FIXTURES
@pytest.fixture
def basic_portfolio():
    """Fixture to create a fresh Portfolio instance for each test."""
    symbol_list = ["AAPL", "GOOG"]
    start_date = datetime(2023, 1, 1)
    initial_capital = 100000.0
    initial_data = {
        "AAPL": {"datetime": start_date, "price": 150.0},
        "GOOG": {"datetime": start_date, "price": 100.0},
    }

    events_queue = Queue()
    mock_bars = MockDataHandler(symbol_list, initial_data)

    portfolio = Portfolio(
        bars=mock_bars,
        events=events_queue,
        start_date=start_date,
        initial_capital=initial_capital,
        print_stats=False,  # Disable printing/plotting during tests
    )
    return portfolio


# --- TEST CASES ---
def test_initialization(basic_portfolio):
    """Tests that the portfolio is initialized with the correct state."""
    p = basic_portfolio
    assert p.initial_capital == 100000.0
    assert p.symbol_list == ["AAPL", "GOOG"]
    assert p.current_positions == {"AAPL": 0, "GOOG": 0}

    expected_holdings = {
        "AAPL": 0.0,
        "GOOG": 0.0,
        "Cash": 100000.0,
        "Commission": 0.0,
        "Total": 100000.0,
    }
    assert p.current_holdings == expected_holdings

    assert len(p.all_positions) == 1
    initial_positions = p.all_positions[0]
    assert initial_positions["Datetime"] == datetime(2023, 1, 1)

    assert initial_positions["AAPL"] == 0
    assert initial_positions["GOOG"] == 0
    assert "Total" not in initial_positions

    assert len(p.all_holdings) == 1
    assert p.all_holdings[0]["Total"] == 100000.0


def test_tf_mapping_and_initialization_error():
    """Tests the timeframe mapping and ensures invalid timeframes raise errors."""
    mock_bars = MagicMock()
    mock_bars.symbols = ["DUMMY"]

    # Test a valid timeframe
    p = Portfolio(
        mock_bars, Queue(), datetime.now(), time_frame="5m", session_duration=6.5
    )
    assert p.tf == int(252 * (60 / 5) * 6.5)

    # Test another valid timeframe
    p_d1 = Portfolio(mock_bars, Queue(), datetime.now(), time_frame="D1")
    assert p_d1.tf == 252

    # Test that an unsupported timeframe raises a ValueError
    with pytest.raises(ValueError, match="Timeframe not supported"):
        Portfolio(mock_bars, Queue(), datetime.now(), time_frame="UnsupportedTF")


def test_update_timeindex(basic_portfolio):
    """Tests that portfolio history is correctly updated on a new market event."""
    p = basic_portfolio
    new_date = datetime(2023, 1, 2)
    p.bars.update_bar("AAPL", new_date, 155.0)
    p.bars.update_bar("GOOG", new_date, 102.0)

    # Give the portfolio a position to make the test more meaningful
    p.current_positions["AAPL"] = 10

    p.update_timeindex(MarketEvent())

    assert len(p.all_positions) == 2
    assert p.all_positions[1]["Datetime"] == new_date
    assert p.all_positions[1]["AAPL"] == 10  # Records position from previous bar

    assert len(p.all_holdings) == 2
    market_value = 10 * 155.0
    expected_total = 100000.0 + market_value
    assert p.all_holdings[1]["Total"] == expected_total
    assert p.all_holdings[1]["AAPL"] == market_value


def test_update_fill_buy(basic_portfolio):
    """Tests that a BUY FillEvent correctly updates positions and holdings."""
    p = basic_portfolio
    fill_event = FillEvent(
        datetime.now(), "AAPL", "TEST_EXCHANGE", 10, "BUY", None, commission=5.0
    )

    p.update_fill(fill_event)

    assert p.current_positions["AAPL"] == 10
    cost = 10 * 150.0
    assert p.current_holdings["Cash"] == 100000.0 - cost - 5.0
    assert p.current_holdings["Commission"] == 5.0
    assert p.current_holdings["AAPL"] == cost


def test_update_fill_sell_short(basic_portfolio):
    """Tests that a SELL (short) FillEvent correctly updates state."""
    p = basic_portfolio
    fill_event = FillEvent(
        datetime.now(), "GOOG", "TEST_EXCHANGE", 20, "SELL", None, commission=7.0
    )

    p.update_fill(fill_event)

    assert p.current_positions["GOOG"] == -20
    proceeds = 20 * 100.0
    assert p.current_holdings["Cash"] == 100000.0 + proceeds - 7.0
    assert p.current_holdings["GOOG"] == -proceeds


@pytest.mark.parametrize(
    "signal_type, initial_pos, expected_direction, expected_quantity",
    [
        ("LONG", 0, "BUY", 50),
        ("SHORT", 0, "SELL", 30),
        ("EXIT", 100, "SELL", 100),
        ("EXIT", -75, "BUY", 75),
        ("EXIT", 0, None, 0),  # No order if exiting from a flat position
        ("LONG", 10, "BUY", 50),  # New LONG signal ignores existing long position
    ],
)
def test_generate_order(
    basic_portfolio, signal_type, initial_pos, expected_direction, expected_quantity
):
    """Tests order generation logic for various signal types and positions."""
    p = basic_portfolio
    p.current_positions["AAPL"] = initial_pos

    quantity = 50 if signal_type == "LONG" else 30
    signal = SignalEvent(
        1, "AAPL", datetime.now(), signal_type, quantity=quantity, strength=1.0
    )

    order = p.generate_order(signal)

    if expected_direction is None:
        assert order is None
    else:
        assert isinstance(order, OrderEvent)
        assert order.direction == expected_direction
        assert order.quantity == expected_quantity
        assert order.order_type == "MKT"


def test_update_signal_puts_order_on_queue(basic_portfolio):
    """Tests that update_signal correctly generates and queues an order."""
    p = basic_portfolio
    signal = SignalEvent(1, "GOOG", datetime.now(), "LONG", quantity=100, strength=0.5)

    assert p.events.qsize() == 0
    p.update_signal(signal)
    assert p.events.qsize() == 1

    order = p.events.get()
    assert isinstance(order, OrderEvent)
    assert order.symbol == "GOOG"
    assert order.direction == "BUY"
    assert order.quantity == 50  # 100 * 0.5


@pytest.mark.filterwarnings("ignore")
@patch("bbstrader.btengine.performance.plt.show")
@patch("bbstrader.btengine.portfolio.plot_performance")
@patch("bbstrader.btengine.portfolio.plot_returns_and_dd")
@patch("bbstrader.btengine.portfolio.plot_monthly_yearly_returns")
@patch("bbstrader.btengine.portfolio.show_qs_stats")
@patch("bbstrader.btengine.portfolio.qs.plots.monthly_heatmap")
@patch("pandas.DataFrame.to_csv")
def test_output_summary_stats(
    mock_to_csv,
    mock_qs_heatmap,
    mock_show_qs,
    mock_plot_monthly,
    mock_plot_ret_dd,
    mock_plot_perf,
    mock_plt_show,
    basic_portfolio,
):
    """Tests performance calculation and that reporting functions are called without side effects."""
    p = basic_portfolio
    p.print_stats = True
    p.strategy_name = "Test Strategy"
    p.output_dir = "test_results"

    # Manually create a simple history for the equity curve
    tz = pd.Timestamp.utcnow().tzinfo
    p.all_holdings = [
        {
            "Datetime": datetime(2023, 1, 1, tzinfo=tz),
            "Total": 100000.0,
            "Commission": 0.0,
            "Cash": 100000.0,
            "AAPL": 0,
            "GOOG": 0,
        },
        {
            "Datetime": datetime(2023, 1, 2, tzinfo=tz),
            "Total": 101000.0,
            "Commission": 0.0,
            "Cash": 100000.0,
            "AAPL": 0,
            "GOOG": 0,
        },
        {
            "Datetime": datetime(2023, 1, 3, tzinfo=tz),
            "Total": 100500.0,
            "Commission": 0.0,
            "Cash": 100000.0,
            "AAPL": 0,
            "GOOG": 0,
        },
        {
            "Datetime": datetime(2023, 1, 4, tzinfo=tz),
            "Total": 102000.0,
            "Commission": 0.0,
            "Cash": 100000.0,
            "AAPL": 0,
            "GOOG": 0,
        },
    ]

    p.create_equity_curve_dataframe()

    p.equity_curve["Returns"] = p.equity_curve["Returns"].fillna(0.0)
    p.equity_curve["Equity Curve"] = p.equity_curve["Equity Curve"].fillna(1.0)

    p.equity_curve["Drawdown"] = 0.0

    # Call the method under test
    stats = p.output_summary_stats()

    # Assertions
    stats_dict = dict(stats)
    assert stats_dict["Total Return"] == "2.00%"
    assert "Sharpe Ratio" in stats_dict
    assert "Max Drawdown" in stats_dict

    # Verify that our mocked (and problematic) functions were called
    mock_plot_perf.assert_called_once()
    mock_plot_ret_dd.assert_called_once()
    mock_plot_monthly.assert_called_once()
    mock_qs_heatmap.assert_called_once()
    mock_show_qs.assert_called_once()
    mock_to_csv.assert_called_once()

    # Verify plots are never actually shown
    mock_plt_show.assert_not_called()

    # 4. Verify the path for the saved CSV file
    call_args, _ = mock_to_csv.call_args
    file_path = call_args[0]
    assert isinstance(file_path, Path)
    assert "test_results" in str(file_path)
    assert "Test_Strategy" in str(file_path)
    assert str(file_path).endswith("_equities.csv")
