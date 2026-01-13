import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from bbstrader.metatrader.risk import RiskManagement


@pytest.fixture
def mock_account_risk():
    """Fixture to mock the Account class for risk management tests."""
    with patch("bbstrader.metatrader.risk.Account") as mock_account_class:
        mock_account_instance = mock_account_class.return_value
        mock_account_instance.get_account_info.return_value = MagicMock(
            balance=10000.0, equity=10500.0, margin_free=5000.0
        )
        mock_account_instance.get_trades_history.return_value = None
        mock_account_instance.broker.adjust_tick_values.return_value = (1.0, 1.0)
        yield mock_account_instance


@pytest.fixture
def mock_mt5_client_risk():
    """Fixture to mock the bbstrader.api.Mt5client for risk management tests."""
    with patch("bbstrader.metatrader.risk.client") as mock_client:
        mock_symbol_info = MagicMock()
        mock_symbol_info.spread = 10
        mock_symbol_info.point = 0.00001
        mock_client.symbol_info.return_value = mock_symbol_info

        mock_rates = np.array(
            [
                (
                    i,
                    1.1 + i * 0.01,
                    1.1 + i * 0.01,
                    1.1 + i * 0.01,
                    1.1 + i * 0.01,
                    1,
                    1,
                    1,
                )
                for i in range(100)
            ],
            dtype=[
                ("time", "<i8"),
                ("open", "<f8"),
                ("high", "<f8"),
                ("low", "<f8"),
                ("close", "<f8"),
                ("tick_volume", "<u8"),
                ("spread", "<i4"),
                ("real_volume", "<u8"),
            ],
        )
        mock_client.copy_rates_from_pos.return_value = mock_rates
        yield mock_client


def test_risk_management_initialization(mock_account_risk, mock_mt5_client_risk):
    """Test the initialization of the RiskManagement class."""
    rm = RiskManagement("EURUSD")
    assert rm.symbol == "EURUSD"
    assert rm.max_risk == 10.0


def test_risk_level(mock_account_risk, mock_mt5_client_risk):
    """Test the risk_level method."""
    rm = RiskManagement("EURUSD")
    risk = rm.risk_level()
    assert risk == 0.0  # Based on mock data


def test_get_trade_risk(mock_account_risk, mock_mt5_client_risk):
    """Test the get_trade_risk method."""
    rm = RiskManagement("EURUSD", max_risk=10.0, daily_risk=2.0)
    trade_risk = rm.get_trade_risk()
    assert trade_risk > 0


def test_calculate_var(mock_account_risk, mock_mt5_client_risk):
    """Test the calculate_var method."""
    rm = RiskManagement("EURUSD")
    var = rm.calculate_var()
    assert isinstance(var, float)


@patch("bbstrader.metatrader.risk.RiskManagement.validate_currency_risk")
def test_get_lot(mock_validate_currency_risk, mock_account_risk, mock_mt5_client_risk):
    """Test the get_lot method."""
    mock_validate_currency_risk.return_value = (0.1, 100)
    rm = RiskManagement("EURUSD")
    lot = rm.get_lot()
    assert lot == 0.1


def test_currency_risk(mock_account_risk, mock_mt5_client_risk):
    """Test the currency_risk method."""
    rm = RiskManagement("EURUSD")
    with patch.object(rm, "get_trade_risk", return_value=0.01):
        with patch.object(rm, "var_loss_value", return_value=50):
            risk = rm.currency_risk()
            assert isinstance(risk, dict)
            assert "currency_risk" in risk
            assert "lot" in risk


@patch("bbstrader.metatrader.risk.RiskManagement.validate_currency_risk")
def test_get_stop_loss(
    mock_validate_currency_risk, mock_account_risk, mock_mt5_client_risk
):
    """Test the get_stop_loss method."""
    mock_validate_currency_risk.return_value = (0.1, 100)
    rm = RiskManagement("EURUSD")
    sl = rm.get_stop_loss()
    assert sl == 100


def test_get_take_profit(mock_account_risk, mock_mt5_client_risk):
    """Test the get_take_profit method."""
    rm = RiskManagement("EURUSD", rr=2.0)
    with patch.object(rm, "get_stop_loss", return_value=100):
        tp = rm.get_take_profit()
        assert tp == 200
