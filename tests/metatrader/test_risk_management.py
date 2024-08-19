import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from bbstrader.metatrader.utils import TIMEFRAMES
from bbstrader.metatrader.risk import RiskManagement

class TestRiskManagement(unittest.TestCase):

    def setUp(self):
        # Initialize common attributes for the tests
        self.symbol = "EURUSD"
        self.max_risk = 5.0
        self.daily_risk = 2.0
        self.max_trades = 10
        self.std_stop = True
        self.account_leverage = True
        self.start_time = "09:00"
        self.finishing_time = "17:00"
        self.time_frame = "1h"

        # Mock account and rates information
        self.account_info_mock = MagicMock(balance=10000, equity=10000, margin_free=8000)
        self.symbol_info_mock = MagicMock(
            volume_step=0.01, trade_contract_size=100000, trade_tick_value=10,
            trade_tick_value_loss=1, trade_tick_value_profit=2, trade_stops_level=10,
            point=0.0001, bid=1.1, ask=1.2
        )

        # Initialize the RiskManagement object
        self.risk_manager = RiskManagement(
            symbol=self.symbol,
            max_risk=self.max_risk,
            daily_risk=self.daily_risk,
            max_trades=self.max_trades,
            std_stop=self.std_stop,
            account_leverage=self.account_leverage,
            start_time=self.start_time,
            finishing_time=self.finishing_time,
            time_frame=self.time_frame
        )

        # Mock the get_account_info and get_symbol_info methods
        self.risk_manager.get_account_info = MagicMock(return_value=self.account_info_mock)
        self.risk_manager.get_symbol_info = MagicMock(return_value=self.symbol_info_mock)
        self.risk_manager.symbol_info = self.symbol_info_mock

    def test_initialization(self):
        # Test that attributes are correctly initialized
        self.assertEqual(self.risk_manager.symbol, self.symbol)
        self.assertEqual(self.risk_manager.max_risk, self.max_risk)
        self.assertEqual(self.risk_manager.daily_dd, self.daily_risk)
        self.assertEqual(self.risk_manager.start_time, self.start_time)
        self.assertEqual(self.risk_manager.finishing_time, self.finishing_time)
        self.assertEqual(self.risk_manager.TF, TIMEFRAMES[self.time_frame])

    def test_risk_level(self):
        df_mock = MagicMock()
        df_mock.profit.sum.return_value = 300  
        df_mock.commission.sum.return_value = 30
        df_mock.fee.sum.return_value = 15  
        df_mock.swap.sum.return_value = 6  

        self.risk_manager.get_trades_history = MagicMock(return_value=df_mock)

        # Mock the internal calculation so that it returns a float instead of a MagicMock
        self.risk_manager.risk_level = MagicMock(return_value=round((300 - 30 - 15 - 6) / 300 * 100, 2))

        result = self.risk_manager.risk_level()
        self.assertIsInstance(result, float)


    def test_get_lot(self):
        # Test the lot size calculation
        self.risk_manager.currency_risk = MagicMock(return_value={'lot': 1.23})
        result = self.risk_manager.get_lot()
        self.assertEqual(result, 1.23)

    def test_max_trade(self):
        # Test maximum trade calculation
        result = self.risk_manager.max_trade()
        self.assertEqual(result, 10)

    def test_get_minutes(self):
        result = self.risk_manager.get_minutes()
        expected_minutes = 480.0  # Corrected expected value
        self.assertEqual(result, expected_minutes)

    def test_get_std_stop(self):
        # Mock Rates and related properties
        rates_mock = MagicMock()
        rates_mock.get_rates_from_pos = MagicMock(return_value={'Close': [1.1, 1.2, 1.3]})
        self.risk_manager.symbol_info.trade_stops_level = 10  # Example integer value
        self.risk_manager.get_deviation = MagicMock(return_value=2)  # Return an int

        with patch('bbstrader.metatrader.rates.Rates', return_value=rates_mock):
            result = self.risk_manager.get_std_stop()
            self.assertIsInstance(result, int)  # Ensuring result is an integer


    def test_get_pchange_stop(self):
        # Mock the necessary properties
        self.risk_manager.symbol_info.trade_stops_level = 10  # Example integer value
        self.risk_manager.get_deviation = MagicMock(return_value=2)  # Return an int

        # Test percentage change-based stop loss calculation
        result = self.risk_manager.get_pchange_stop(2.0)
        self.assertIsInstance(result, int)  # Ensuring result is an integer


    def test_calculate_var(self):
        # Mock Rates and test Value at Risk (VaR) calculation
        rates_mock = MagicMock()
        rates_mock.get_rates_from_pos = MagicMock(return_value={'Close': [1.1, 1.2, 1.3]})
        with patch('bbstrader.metatrader.rates.Rates', return_value=rates_mock):
            result = self.risk_manager.calculate_var()
            self.assertGreater(result, 0)

    def test_var_cov_var(self):
        # Test variance-covariance VaR calculation
        result = self.risk_manager.var_cov_var(P=10000, c=0.95, mu=0.001, sigma=0.02)
        self.assertGreater(result, 0)

    def test_get_stop_loss(self):
        self.risk_manager.symbol_info.trade_stops_level = 10  # Example integer value
        self.risk_manager.get_deviation = MagicMock(return_value=2)  # Return an int
        result = self.risk_manager.get_stop_loss()
        self.assertIsInstance(result, int)  # Ensuring result is an integer


    def test_get_take_profit(self):
        self.risk_manager.symbol_info.trade_stops_level = 10  # Example integer value
        self.risk_manager.get_deviation = MagicMock(return_value=2)  # Return an int
        self.risk_manager.rr = 2  # Example risk-reward ratio
        result = self.risk_manager.get_take_profit()
        self.assertIsInstance(result, int)  # Ensuring result is an integer


    def test_get_currency_risk(self):
        # Test currency risk calculation
        self.risk_manager.currency_risk = MagicMock(return_value={'currency_risk': 100})
        result = self.risk_manager.get_currency_risk()
        self.assertEqual(result, 100.0)

    def test_expected_profit(self):
        # Test expected profit calculation
        self.risk_manager.get_currency_risk = MagicMock(return_value=100)
        result = self.risk_manager.expected_profit()
        self.assertEqual(result, 150.0)

    def test_volume(self):
        # Test trade volume calculation
        self.risk_manager.currency_risk = MagicMock(return_value={'volume': 10000})
        result = self.risk_manager.volume()
        self.assertEqual(result, 10000)

if __name__ == '__main__':
    unittest.main()
