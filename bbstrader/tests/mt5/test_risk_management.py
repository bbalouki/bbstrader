import pytest
from decimal import Decimal
import MetaTrader5 as Mt5

from mtrader5.risk import RiskManagement
from mtrader5.account import Account

@pytest.fixture(scope='session')
def setup_mt5():
    if not Mt5.initialize():
        pytest.fail("MT5 terminal is not running. Please ensure MT5 is active before running tests.")
    yield
    Mt5.shutdown()

# Make sure the symbols use have full acces for trading
# Make sure an instance of MT5 is running with a demo account
@pytest.fixture
def risk_management():
    kwargs = {
        "symbol": 'SPY',
        "max_risk": 30.0,
        "daily_risk": 5.0,
        "max_trades": None,
        "std_stop": False,
        "pchange_sl": 1.0,
        "account_leverage": False,
        "time_frame": "1h",
        "start_time": "09:00",
        "finishing_time": "17:00",
        "sl": 20,
        "tp": 40,
        "be": 10,
        "rr": 1.5
    }
    return RiskManagement(**kwargs)

def test_risk_level(setup_mt5, risk_management):
    risk_level = risk_management.risk_level()
    assert isinstance(risk_level, float)

def test_get_lot(setup_mt5, risk_management):
    lot = risk_management.get_lot()
    assert isinstance(lot, (float, int))

def test_max_trade(setup_mt5, risk_management):
    max_trades = risk_management.max_trade()
    assert isinstance(max_trades, int)

def test_get_minutes(setup_mt5, risk_management):
    minutes = risk_management.get_minutes()
    assert minutes > 0

def test_get_hours(setup_mt5, risk_management):
    hours = risk_management.get_hours()
    assert hours > 0

def test_get_stop_loss_with_standard_stop(setup_mt5, risk_management):
    stop_loss = risk_management.get_stop_loss()
    assert isinstance(stop_loss, int)

def test_get_take_profit(setup_mt5, risk_management):
    take_profit = risk_management.get_take_profit()
    assert isinstance(take_profit, int)

def test_get_currency_risk(setup_mt5, risk_management):
    currency_risk = risk_management.get_currency_risk()
    assert isinstance(currency_risk, float)

def test_expected_profit(setup_mt5, risk_management):
    expected_profit = risk_management.expected_profit()
    assert isinstance(expected_profit, float)

def test_is_risk_ok(setup_mt5, risk_management):
    risk_ok = risk_management.is_risk_ok()
    assert isinstance(risk_ok, bool)

def test_calculate_var(setup_mt5, risk_management):
    var = risk_management.calculate_var()
    assert isinstance(var, (float, Decimal))

def test_get_leverage(setup_mt5, risk_management):
    leverage = risk_management.get_leverage(True)
    assert isinstance(leverage, int)

def test_get_deviation(setup_mt5, risk_management):
    deviation = risk_management.get_deviation()
    assert isinstance(deviation, int)

def test_get_break_even(setup_mt5, risk_management):
    break_even = risk_management.get_break_even()
    assert isinstance(break_even, int)
