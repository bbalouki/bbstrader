from unittest.mock import MagicMock, patch

import pytest

from bbstrader.metatrader.account import Account
from bbstrader.metatrader.broker import Broker


@pytest.fixture
def mock_mt5_client():
    """Fixture to mock the bbstrader.api.Mt5client."""
    with (
        patch(
            "bbstrader.metatrader.account.check_mt5_connection"
        ) as mock_check_connection,
        patch("bbstrader.metatrader.account.client") as mock_client,
        patch("bbstrader.metatrader.broker.client") as mock_broker_client,
    ):
        mock_check_connection.return_value = True
        mock_account_info = MagicMock()
        mock_account_info.name = "Test User"
        mock_account_info.login = 12345
        mock_account_info.server = "Test Server"
        mock_account_info.balance = 10000.0
        mock_account_info.leverage = 100
        mock_account_info.equity = 10500.0
        mock_account_info.currency = "USD"
        mock_client.account_info.return_value = mock_account_info

        mock_terminal_info = MagicMock()
        mock_terminal_info.company = "Test Broker"
        mock_client.terminal_info.return_value = mock_terminal_info

        mock_symbol = MagicMock()
        mock_symbol.name = "EURUSD"
        mock_client.symbols_get.return_value = [mock_symbol]
        mock_broker_client.symbols_get.return_value = [mock_symbol]

        mock_tick = MagicMock()
        mock_tick.time = 0
        mock_client.symbol_info_tick.return_value = mock_tick
        mock_broker_client.symbol_info_tick.return_value = mock_tick

        yield mock_client


def test_account_initialization(mock_mt5_client):
    """Test the initialization of the Account class."""
    account = Account()
    assert isinstance(account.broker, Broker)
    assert account.broker.name == "Test Broker"
    mock_mt5_client.account_info.assert_called_once()
    mock_mt5_client.terminal_info.assert_called_once()


def test_account_properties(mock_mt5_client):
    """Test the properties of the Account class."""
    account = Account()
    assert account.name == "Test User"
    assert account.number == 12345
    assert account.server == "Test Server"
    assert account.balance == 10000.0
    assert account.leverage == 100
    assert account.equity == 10500.0
    assert account.currency == "USD"
    assert account.info == mock_mt5_client.account_info.return_value


def test_get_account_info_default(mock_mt5_client):
    """Test get_account_info without arguments."""
    account = Account()
    info = account.get_account_info()
    assert info == mock_mt5_client.account_info.return_value
    assert (
        mock_mt5_client.account_info.call_count == 2
    )  # Once in __init__ and once here


def test_get_account_info_specific(mock_mt5_client):
    """Test get_account_info with specific account credentials."""
    account = Account()
    mock_mt5_client.login.return_value = True
    new_account_info = MagicMock()
    new_account_info.name = "New User"
    mock_mt5_client.account_info.return_value = new_account_info

    info = account.get_account_info(
        account=54321, password="password", server="NewServer"
    )

    mock_mt5_client.login.assert_called_with(
        54321, password="password", server="NewServer", timeout=60000
    )
    assert info == new_account_info


def test_get_terminal_info(mock_mt5_client):
    """Test get_terminal_info."""
    account = Account()
    info = account.get_terminal_info()
    assert info == mock_mt5_client.terminal_info.return_value
    assert (
        mock_mt5_client.terminal_info.call_count == 2
    )  # Once in __init__ and once here


def test_get_symbol_info(mock_mt5_client):
    """Test get_symbol_info."""
    mock_symbol_info = MagicMock()
    mock_symbol_info.name = "EURUSD"
    mock_mt5_client.symbol_info.return_value = mock_symbol_info

    account = Account()
    info = account.get_symbol_info("EURUSD")

    mock_mt5_client.symbol_info.assert_called_with("EURUSD")
    assert info == mock_symbol_info


def test_get_symbol_info_none(mock_mt5_client):
    """Test get_symbol_info when symbol does not exist."""
    mock_mt5_client.symbol_info.return_value = None

    account = Account()
    info = account.get_symbol_info("INVALID")

    mock_mt5_client.symbol_info.assert_called_with("INVALID")
    assert info is None


def test_get_tick_info(mock_mt5_client):
    """Test get_tick_info."""
    mock_tick_info = MagicMock()
    mock_tick_info.bid = 1.1
    mock_tick_info.ask = 1.1002
    mock_mt5_client.symbol_info_tick.return_value = mock_tick_info

    account = Account()
    info = account.get_tick_info("EURUSD")

    mock_mt5_client.symbol_info_tick.assert_called_with("EURUSD")
    assert info == mock_tick_info


def test_get_currency_rates(mock_mt5_client):
    """Test get_currency_rates."""
    mock_symbol_info = MagicMock()
    mock_symbol_info.currency_base = "EUR"
    mock_symbol_info.currency_profit = "USD"
    mock_symbol_info.currency_margin = "EUR"
    mock_mt5_client.symbol_info.return_value = mock_symbol_info

    account = Account()
    rates = account.get_currency_rates("EURUSD")

    assert rates == {"bc": "EUR", "mc": "EUR", "pc": "USD", "ac": "USD"}


def test_get_positions_all(mock_mt5_client):
    """Test get_positions to retrieve all open positions."""
    mock_position = MagicMock()
    mock_mt5_client.positions_get.return_value = [mock_position]

    account = Account()
    positions = account.get_positions()

    mock_mt5_client.positions_get.assert_called_once_with()
    assert positions == [mock_position]


def test_get_positions_by_symbol(mock_mt5_client):
    """Test get_positions filtered by symbol."""
    mock_position = MagicMock()
    mock_mt5_client.positions_get.return_value = [mock_position]

    account = Account()
    positions = account.get_positions(symbol="EURUSD")

    mock_mt5_client.positions_get.assert_called_with("EURUSD")
    assert positions == [mock_position]


def test_get_orders_by_ticket(mock_mt5_client):
    """Test get_orders filtered by ticket."""
    mock_order = MagicMock()
    mock_mt5_client.order_get_by_ticket.return_value = [mock_order]

    account = Account()
    orders = account.get_orders(ticket=123)

    mock_mt5_client.order_get_by_ticket.assert_called_with(123)
    assert orders == [mock_order]


@patch("bbstrader.metatrader.account.pd")
def test_get_trades_history(mock_pd, mock_mt5_client):
    """Test get_trades_history."""
    mock_deal = MagicMock()
    mock_mt5_client.history_deals_get.return_value = [mock_deal]

    account = Account()
    history = account.get_trades_history(to_df=True)

    assert history is not None
    mock_mt5_client.history_deals_get.assert_called()


@patch("bbstrader.metatrader.account.pd")
def test_get_orders_history(mock_pd, mock_mt5_client):
    """Test get_orders_history."""
    mock_order = MagicMock()
    mock_mt5_client.history_orders_get.return_value = [mock_order]

    account = Account()
    history = account.get_orders_history(to_df=True)

    assert history is not None
    mock_mt5_client.history_orders_get.assert_called()
