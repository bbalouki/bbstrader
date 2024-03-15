import pytest
import mtrader5
import time
from mtrader5.trade import Trade
import MetaTrader5
from unittest.mock import MagicMock

# Make sure the symbols use have full acces for trading
# Make sure an instance of MT5 is running with a demo account
SYMBOL = 'NAS100'
SYMBOL2 = 'SPX500'

# Fixture for the Trade instance
@pytest.fixture
def trade_instance():
    trade_params = {
        "symbol": SYMBOL,
        "expert_name": "TestEA",
        "expert_id": 12345,
        "version": "1.0",
        "target": 1.0,
        "start_time": "6:30",
        "time_frame": 'D1',
        "finishing_time": "19:30",
        "ending_time": "20:30",
    }
    trade = Trade(**trade_params)
    return trade

# Test initialization and attribute assignment
def test_initialization(trade_instance):
    assert trade_instance.symbol == SYMBOL
    assert trade_instance.expert_name == "TestEA"
    assert trade_instance.expert_id == 12345
    assert trade_instance.version == "1.0"
    assert isinstance(trade_instance.lot, (float, int))
    assert isinstance(trade_instance.stop_loss, int)
    assert isinstance(trade_instance.take_profit, int)
    assert isinstance(trade_instance.break_even_points, int)

# Test opening a buy position
def test_open_buy_position(monkeypatch, trade_instance):
    mock_result = MagicMock()
    mock_result.retcode = 10009  # TRADE_RETCODE_DONE
    monkeypatch.setattr(
        "MetaTrader5.order_send", MagicMock(return_value=mock_result))
    trade_instance.open_buy_position()
    MetaTrader5.order_send.assert_called_once()

# Test opening a sell position
def test_open_sell_position(monkeypatch, trade_instance):
    mock_result = MagicMock()
    mock_result.retcode = 10009  # TRADE_RETCODE_DONE
    monkeypatch.setattr(
        "MetaTrader5.order_send", MagicMock(return_value=mock_result))
    trade_instance.open_sell_position()
    MetaTrader5.order_send.assert_called_once()

# Test risk management check
def test_risk_management(trade_instance):
    assert trade_instance.is_risk_ok() == True or False  # Depending on the risk setup

# Test statistics gathering
def test_statistics(trade_instance):
    stats, additional_stats = trade_instance.get_stats()
    assert isinstance(stats, dict)
    assert isinstance(additional_stats, dict)

# Test the Sharpe ratio calculation
def test_sharpe_ratio(trade_instance):
    sharpe_ratio = trade_instance.sharpe()
    assert isinstance(sharpe_ratio, float)

# Test the trading time check
def test_trading_time(trade_instance):
    assert trade_instance.trading_time() in [True, False]

# Test profit target reached check
def test_profit_target(trade_instance):
    assert trade_instance.profit_target() in [True, False]

# Test calculating the volume for risk management
def test_volume_calculation(trade_instance):
    volume = trade_instance.volume()
    assert isinstance(volume, (int,float)) 

# Test checking if it's the end of the trading day
def test_days_end(trade_instance):
    assert trade_instance.days_end() in [True, False]

# Test getting current buy positions
def test_get_current_buys(trade_instance):
    buys = trade_instance.get_current_buys()
    assert isinstance(buys, list) or buys is None

# Test getting current sell positions
def test_get_current_sells(trade_instance):
    sells = trade_instance.get_current_sells()
    assert isinstance(sells, list) or sells is None


# Test closing a position
def test_close_position():
    trade_instance = Trade(symbol=SYMBOL)
    trade_instance.open_buy_position()
    trade_instance.open_sell_position()
    trades = trade_instance.get_current_open_positions()
    assert trades is not None
    trade_instance.close_position(trades[0])
    positions = trade_instance.get_current_open_positions()
    assert len(positions) == len(trades)- 1

# Test the functionality to close all positions
def test_close_all_positions():
    trade = Trade(symbol=SYMBOL2)
    for i in range(5):
        if i%2 == 0:
            trade.open_buy_position()
        else: trade.open_sell_position()
    buys = trade.get_buy_positions
    sells = trade.get_sell_positions
    assert buys is not None
    assert sells is not None
    assert len(buys) == 3
    assert len(sells) == 2
    trade.close_all_positions()
    current = trade.get_current_open_positions()
    assert current is None

# Test setting a break even
def test_set_break_even():
    # This test may take a time to run so be patient
    trade = Trade(symbol=SYMBOL, be=2)
    trade.open_buy_position()
    time.sleep(10)
    trade.open_sell_position()
    time.sleep(10)
    while True:
        trade.break_even()
        be = trade.get_be_positions
        time.sleep(10)
        try:
            assert be is not None
        except AssertionError:
            time.sleep(10)
            continue
        else: break

# Test the sleep time calculation
def test_sleep_time():
    trade = Trade(symbol=SYMBOL, be=3) 
    day_sleep = trade.sleep_time()
    week_sleep = trade.sleep_time(weekend=True)
    assert isinstance(day_sleep, (int, float))
    assert isinstance(week_sleep, (int, float))
    assert day_sleep != week_sleep