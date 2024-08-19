import os
import time
import unittest
import MetaTrader5
from unittest.mock import patch, MagicMock, mock_open
from bbstrader.metatrader.trade import Trade
from bbstrader.metatrader.utils import MT5TerminalError
from bbstrader.metatrader.utils import TradePosition

SYMBOL = 'GBPUSD'
SYMBOL2 = 'AUDUSD'
kwargs = dict(
    symbol=SYMBOL,
    target=5.0,
    expert_name="TestEA",
    expert_id=1234,
    time_frame='1h',
    max_risk=2.0,
    daily_risk=1.0,
    max_trades=50,
    rr=2.0,
    pcchange_sl=1.0,
)


class TestTrade(unittest.TestCase):
    def setUp(self):
        self.trade_instance = Trade(**kwargs)

    def test_initialization(self):
        self.assertEqual(self.trade_instance.symbol, SYMBOL)
        self.assertEqual(self.trade_instance.expert_name, "TestEA")
        self.assertEqual(self.trade_instance.expert_id, 1234)
        self.assertEqual(self.trade_instance.version, "1.0")
        self.assertIsInstance(self.trade_instance.lot, (float, int))
        self.assertIsInstance(self.trade_instance.stop_loss, int)
        self.assertIsInstance(self.trade_instance.take_profit, int)
        self.assertIsInstance(self.trade_instance.break_even_points, int)

    def test_risk_management(self):
        self.assertIn(self.trade_instance.is_risk_ok(), [True, False])

    def test_statistics(self):
        stats, additional_stats = self.trade_instance.get_stats()
        self.assertIsInstance(stats, dict)
        self.assertIsInstance(additional_stats, dict)

    def test_sharpe_ratio(self):
        sharpe_ratio = self.trade_instance.sharpe()
        self.assertIsInstance(sharpe_ratio, float)

    def test_trading_time(self):
        self.assertIn(self.trade_instance.trading_time(), [True, False])

    def test_profit_target(self):
        self.assertIn(self.trade_instance.profit_target(), [True, False])

    def test_volume_calculation(self):
        volume = self.trade_instance.volume()
        self.assertIsInstance(volume, (int, float))

    def test_days_end(self):
        self.assertIn(self.trade_instance.days_end(), [True, False])

    @patch('bbstrader.metatrader.trade.Trade.break_even')
    @patch('bbstrader.metatrader.trade.Trade.check')
    def test_open_buy_position(self, mock_check, mock_break_even):
        trade_instance = Trade(symbol="EURUSD")
        trade_instance.get_symbol_info = MagicMock(
            return_value=MagicMock(point=0.0001, digits=5))
        trade_instance.get_tick_info = MagicMock(
            return_value=MagicMock(ask=1.1000))
        trade_instance.get_lot = MagicMock(return_value=1.0)
        trade_instance.get_stop_loss = MagicMock(return_value=50)
        trade_instance.get_take_profit = MagicMock(return_value=100)
        trade_instance.get_deviation = MagicMock(return_value=10)
        mock_check.return_value = True
        trade_instance.open_buy_position(action='BMKT')
        mock_break_even.assert_called_once()
        mock_check.assert_called_once()

    @patch('bbstrader.metatrader.trade.Trade.break_even')
    @patch('bbstrader.metatrader.trade.Trade.check')
    def test_open_sell_position(self, mock_check, mock_break_even):
        trade_instance = Trade(symbol="EURUSD")
        trade_instance.get_symbol_info = MagicMock(
            return_value=MagicMock(point=0.0001, digits=5))
        trade_instance.get_tick_info = MagicMock(
            return_value=MagicMock(bid=1.1000))
        trade_instance.get_lot = MagicMock(return_value=1.0)
        trade_instance.get_stop_loss = MagicMock(return_value=50)
        trade_instance.get_take_profit = MagicMock(return_value=100)
        trade_instance.get_deviation = MagicMock(return_value=10)
        mock_check.return_value = True
        trade_instance.open_sell_position(action='SMKT')
        mock_break_even.assert_called_once()
        mock_check.assert_called_once()

    @patch('bbstrader.metatrader.trade.Trade.get_filtered_tickets')
    def test_get_current_open_orders(self, mock_get_filtered_tickets):
        trade_instance = Trade()
        trade_instance.get_current_open_orders(id=123)
        mock_get_filtered_tickets.assert_called_once_with(
            id=123, filter_type='orders')

    @patch('bbstrader.metatrader.trade.Trade.get_filtered_tickets')
    def test_get_current_open_positions(self, mock_get_filtered_tickets):
        trade_instance = Trade()
        trade_instance.get_current_open_positions(id=123)
        mock_get_filtered_tickets.assert_called_once_with(
            id=123, filter_type='positions')

    @patch('bbstrader.metatrader.trade.Trade.get_filtered_tickets')
    def test_get_current_win_trades(self, mock_get_filtered_tickets):
        trade_instance = Trade()
        trade_instance.get_current_win_trades(id=123)
        mock_get_filtered_tickets.assert_called_once_with(
            id=123, filter_type='win_trades', th=None)

    @patch('bbstrader.metatrader.trade.Trade.get_filtered_tickets')
    def test_get_current_buys(self, mock_get_filtered_tickets):
        trade_instance = Trade()
        trade_instance.get_current_buys(id=123)
        mock_get_filtered_tickets.assert_called_once_with(
            id=123, filter_type='buys')

    @patch('bbstrader.metatrader.trade.Trade.get_filtered_tickets')
    def test_get_current_sells(self, mock_get_filtered_tickets):
        trade_instance = Trade()
        trade_instance.get_current_sells(id=123)
        mock_get_filtered_tickets.assert_called_once_with(
            id=123, filter_type='sells')

    @patch('bbstrader.metatrader.Trade.open_buy_position')
    @patch('bbstrader.metatrader.Trade.open_sell_position')
    @patch('bbstrader.metatrader.Trade.get_current_open_positions')
    @patch('bbstrader.metatrader.Trade.close_position')
    def test_close_position(self, mock_close_position, 
                            mock_get_current_open_positions, 
                            mock_open_sell_position, mock_open_buy_position):
        mock_open_buy_position.return_value = None
        mock_open_sell_position.return_value = None
        mock_get_current_open_positions.return_value = [1, 2]
        self.trade_instance.close_position(1)
        self.trade_instance.get_current_open_positions()
        mock_close_position.assert_called_once_with(1)
        mock_get_current_open_positions.assert_called_once()

    @patch('bbstrader.metatrader.Trade.open_buy_position')
    @patch('bbstrader.metatrader.Trade.open_sell_position')
    @patch('bbstrader.metatrader.Trade.get_current_open_positions')
    @patch('bbstrader.metatrader.Trade.close_positions')
    def test_close_all_positions(self, mock_close_positions, 
                                 mock_get_current_open_positions, 
                                 mock_open_sell_position, mock_open_buy_position):
        kw_copy = kwargs.copy()
        kw_copy['symbol'] = SYMBOL2
        trade = Trade(**kw_copy)
        for i in range(5):
            if i % 2 == 0:
                trade.open_buy_position()
            else:
                trade.open_sell_position()
        trade.close_positions(position_type='all')
        mock_close_positions.assert_called_once_with(position_type='all')

    @patch('bbstrader.metatrader.Trade.open_buy_position')
    @patch('bbstrader.metatrader.Trade.open_sell_position')
    @patch('bbstrader.metatrader.Trade.break_even')
    @patch('bbstrader.metatrader.Trade.get_be_positions')
    def test_set_break_even(self, mock_get_be_positions, 
                            mock_break_even, mock_open_sell_position, 
                            mock_open_buy_position):
        kw_copy = kwargs.copy()
        kw_copy['be'] = 3
        trade = Trade(**kw_copy)
        mock_open_buy_position.return_value = None
        mock_open_sell_position.return_value = None
        mock_get_be_positions.return_value = [1]
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
            else:
                break
            mock_break_even.assert_called()
            mock_get_be_positions.assert_called()

    def test_sleep_time(self):
        kw_copy = kwargs.copy()
        kw_copy['be'] = 3
        trade = Trade(**kw_copy)
        day_sleep = trade.sleep_time()
        week_sleep = trade.sleep_time(weekend=True)
        self.assertIsInstance(day_sleep, (int, float))
        self.assertIsInstance(week_sleep, (int, float))
        self.assertNotEqual(day_sleep, week_sleep)


if __name__ == '__main__':
    unittest.main()
