import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from bbstrader.metatrader.trade import Trade, EXPERT_ID


@pytest.fixture
def mock_account_trade():
    """Fixture to mock the Account class for trade tests."""
    with (
        patch(
            "bbstrader.metatrader.trade.check_mt5_connection"
        ) as mock_check_connection,
        patch("bbstrader.metatrader.trade.Account") as mock_account_class,
    ):
        mock_check_connection.return_value = True
        yield mock_account_class.return_value


@pytest.fixture
def mock_risk_management():
    """Fixture to mock the RiskManagement class for trade tests."""
    with patch("bbstrader.metatrader.trade.RiskManagement") as mock_rm_class:
        mock_rm_instance = mock_rm_class.return_value
        mock_rm_instance.get_lot.return_value = 0.1
        mock_rm_instance.get_stop_loss.return_value = 100
        mock_rm_instance.get_take_profit.return_value = 200
        mock_rm_instance.get_deviation.return_value = 5
        mock_rm_instance.is_risk_ok.return_value = True
        yield mock_rm_instance


@pytest.fixture
def mock_mt5_client_trade():
    """Fixture to mock the bbstrader.api.Mt5client for trade tests."""
    with (
        patch("bbstrader.metatrader.trade.client") as mock_client,
        patch("bbstrader.metatrader.trade.Mt5") as mock_mt5,
    ):
        mock_mt5.TRADE_RETCODE_DONE = 10009

        mock_symbol_info = MagicMock()
        mock_symbol_info.point = 0.00001
        mock_client.symbol_info.return_value = mock_symbol_info

        mock_tick = MagicMock(bid=1.1, ask=1.1002)
        mock_client.symbol_info_tick.return_value = mock_tick

        mock_order_result = MagicMock(retcode=10009)  # TRADE_RETCODE_DONE
        mock_client.order_send.return_value = mock_order_result
        yield mock_client


def test_trade_initialization(
    mock_account_trade, mock_risk_management, mock_mt5_client_trade
):
    """Test the initialization of the Trade class."""
    trade = Trade("EURUSD")
    assert trade.symbol == "EURUSD"


def test_open_buy_position(
    mock_account_trade, mock_risk_management, mock_mt5_client_trade
):
    """Test open_buy_position."""
    trade = Trade("EURUSD")
    with patch.object(trade, "trading_time", return_value=True):
        with patch.object(trade, "is_max_trades_reached", return_value=False):
            result = trade.open_buy_position()
            assert result is True
            mock_mt5_client_trade.order_send.assert_called()


@patch("bbstrader.metatrader.trade.tabulate")
def test_statistics(
    mock_tabulate, mock_account_trade, mock_risk_management, mock_mt5_client_trade
):
    """Test the statistics method."""
    trade = Trade("EURUSD", verbose=True)  # verbose to ensure it runs
    with patch.object(
        trade,
        "get_stats",
        return_value=(
            {
                "deals": 0,
                "profit": 0.0,
                "win_trades": 0,
                "loss_trades": 0,
                "total_fees": 0.0,
                "average_fee": 0.0,
                "win_rate": 0.0,
            },
            {"total_profit": 100, "profitability": "Yes"},
        ),
    ):
        with patch.object(trade, "sharpe", return_value=1.5):
            trade.statistics(save=False)
            mock_tabulate.assert_called()


def test_get_stats(mock_account_trade, mock_risk_management, mock_mt5_client_trade):
    """Test the get_stats method."""
    trade = Trade("EURUSD")
    mock_account_trade.get_today_deals.return_value = []
    mock_account_trade.get_trades_history.return_value = None

    session_stats, historical_stats = trade.get_stats()

    assert isinstance(session_stats, dict)
    assert isinstance(historical_stats, dict)


@patch("bbstrader.metatrader.trade.qs")
def test_sharpe(
    mock_qs, mock_account_trade, mock_risk_management, mock_mt5_client_trade
):
    """Test the sharpe method."""
    trade = Trade("EURUSD")

    mock_df = pd.DataFrame(
        {"profit": [100, -50], "commission": [-5, -5], "fee": [0, 0], "swap": [0, 0]}
    )
    mock_account_trade.get_trades_history.return_value = mock_df
    mock_qs.stats.sharpe.return_value = 1.5

    sharpe = trade.sharpe()

    mock_qs.stats.sharpe.assert_called()
    assert isinstance(sharpe, float)


def test_break_even(mock_account_trade, mock_risk_management, mock_mt5_client_trade):
    """Test the break_even method."""
    trade = Trade("EURUSD")
    mock_position = MagicMock(
        magic=trade.expert_id, ticket=1, profit=100.0, volume=0.1, price_open=1.1
    )
    mock_account_trade.get_positions.return_value = [mock_position]
    mock_risk_management.get_break_even.return_value = 10

    # Configure the symbol_info mock with the necessary attributes
    mock_symbol_info = MagicMock()
    mock_symbol_info.point = 0.00001
    mock_symbol_info.trade_tick_size = 0.00001
    mock_symbol_info.trade_tick_value = 1.0
    mock_mt5_client_trade.symbol_info.return_value = mock_symbol_info

    with patch.object(trade, "set_break_even") as mock_set_be:
        trade.break_even()
        mock_set_be.assert_called()


def test_set_break_even(
    mock_account_trade, mock_risk_management, mock_mt5_client_trade
):
    """Test the set_break_even method."""
    trade = Trade("EURUSD")
    mock_position = MagicMock(
        profit=100.0,
        price_open=1.1,
        type=0,  # Buy
    )

    with patch.object(
        trade, "break_even_request", return_value=True
    ) as mock_be_request:
        result = trade.set_break_even(mock_position, 10)
        assert result is True
        mock_be_request.assert_called()


def test_get_current_positions(
    mock_account_trade, mock_risk_management, mock_mt5_client_trade
):
    """Test get_current_positions."""
    trade = Trade("EURUSD")
    mock_account_trade.get_positions.return_value = [
        MagicMock(magic=trade.expert_id, ticket=1)
    ]
    positions = trade.get_current_positions()
    assert positions == [1]


def test_get_current_orders_filtered(
    mock_account_trade, mock_risk_management, mock_mt5_client_trade
):
    """Test get_current_orders with a filter."""
    trade = Trade("EURUSD")
    mock_order = MagicMock(magic=trade.expert_id, ticket=1, type=2)  # Buy Stop
    mock_account_trade.get_orders.return_value = [mock_order]

    with patch.object(
        trade, "get_filtered_tickets", return_value=[1]
    ) as mock_get_filtered:
        orders = trade.get_current_buy_stops(id=trade.expert_id)
        mock_get_filtered.assert_called_once_with(
            id=trade.expert_id, filter_type="buy_stops"
        )
        assert orders == [1]


def test_close_position(
    mock_account_trade, mock_risk_management, mock_mt5_client_trade
):
    """Test close_position."""
    mock_position = MagicMock(
        ticket=1, magic=EXPERT_ID, volume=0.1, type=0
    )  # Buy position
    mock_account_trade.get_positions.return_value = [mock_position]

    trade = Trade("EURUSD")
    result = trade.close_position(ticket=1)

    assert result is True
    mock_mt5_client_trade.order_send.assert_called()


def test_close_order(mock_account_trade, mock_risk_management, mock_mt5_client_trade):
    """Test close_order."""
    trade = Trade("EURUSD")
    result = trade.close_order(ticket=123)

    assert result is True
    mock_mt5_client_trade.order_send.assert_called()


def test_close_positions(
    mock_account_trade, mock_risk_management, mock_mt5_client_trade
):
    """Test close_positions."""
    trade = Trade("EURUSD")
    with patch.object(
        trade, "get_current_positions", return_value=[1, 2]
    ) as mock_get_positions:
        with patch.object(trade, "bulk_close") as mock_bulk_close:
            trade.close_positions("all")
            mock_get_positions.assert_called_once_with(id=trade.expert_id)
            mock_bulk_close.assert_called_once_with(
                [1, 2],
                "positions",
                trade.close_position,
                "all",
                id=trade.expert_id,
                comment=None,
            )


def test_close_orders(mock_account_trade, mock_risk_management, mock_mt5_client_trade):
    """Test close_orders."""
    trade = Trade("EURUSD")
    with patch.object(
        trade, "get_current_orders", return_value=[1, 2]
    ) as mock_get_orders:
        with patch.object(trade, "bulk_close") as mock_bulk_close:
            trade.close_orders("all")
            mock_get_orders.assert_called_once_with(id=trade.expert_id)
            mock_bulk_close.assert_called_once_with(
                [1, 2],
                "orders",
                trade.close_order,
                "all",
                id=trade.expert_id,
                comment=None,
            )


def test_open_sell_position(
    mock_account_trade, mock_risk_management, mock_mt5_client_trade
):
    """Test open_sell_position."""
    trade = Trade("EURUSD")
    with patch.object(trade, "trading_time", return_value=True):
        with patch.object(trade, "is_max_trades_reached", return_value=False):
            result = trade.open_sell_position()
            assert result is True
            mock_mt5_client_trade.order_send.assert_called()


def test_open_position_pending(
    mock_account_trade, mock_risk_management, mock_mt5_client_trade
):
    """Test open_position for a pending order."""
    trade = Trade("EURUSD")
    with patch.object(trade, "trading_time", return_value=True):
        with patch.object(trade, "is_max_trades_reached", return_value=False):
            result = trade.open_position(action="BLMT", price=1.0)
            assert result is True
            mock_mt5_client_trade.order_send.assert_called()
