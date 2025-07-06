import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from bbstrader.metatrader.account import __BROKERS__
from bbstrader.metatrader.risk import RiskManagement
from bbstrader.metatrader.utils import TIMEFRAMES, SymbolType


class TestRiskManagement(unittest.TestCase):
    def setUp(self):
        # Initialize patchers
        self.datetime_patcher = patch("bbstrader.metatrader.risk.datetime")
        self.mt5_patcher = patch(
            "bbstrader.metatrader.account.mt5"
        )  # Patches the entire module for mt5 related calls
        self.Rates_patcher = patch("bbstrader.metatrader.risk.Rates")
        self.check_mt5_connection_patcher = patch(
            "bbstrader.metatrader.account.check_mt5_connection"
        )
        self.account_get_account_info_patcher = patch(
            "bbstrader.metatrader.account.Account.get_account_info"
        )
        self.account_get_symbol_info_patcher = patch(
            "bbstrader.metatrader.account.Account.get_symbol_info"
        )
        self.account_get_trades_history_patcher = patch(
            "bbstrader.metatrader.account.Account.get_trades_history"
        )
        self.account_get_terminal_info_patcher = patch(
            "bbstrader.metatrader.account.Account.get_terminal_info"
        )
        self.riskmanagement_var_cov_var_patcher = patch.object(
            RiskManagement, "var_cov_var", return_value=5.0
        )
        self.rm_get_leverage_patcher = patch(
            "bbstrader.metatrader.risk.RiskManagement.get_leverage"
        )

        # Start patchers and assign mocks to self
        self.mock_datetime = self.datetime_patcher.start()
        self.mock_mt5 = self.mt5_patcher.start()
        self.mock_Rates = self.Rates_patcher.start()
        self.mock_check_mt5_connection = self.check_mt5_connection_patcher.start()
        self.mock_account_get_account_info = (
            self.account_get_account_info_patcher.start()
        )
        self.mock_account_get_symbol_info = self.account_get_symbol_info_patcher.start()
        self.mock_account_get_trades_history = (
            self.account_get_trades_history_patcher.start()
        )
        self.mock_account_get_terminal_info = (
            self.account_get_terminal_info_patcher.start()
        )
        self.mock_vcv = self.riskmanagement_var_cov_var_patcher.start()
        self.mock_rm_get_leverage = self.rm_get_leverage_patcher.start()
        self.mock_rm_get_leverage.return_value = 100

        # Configure mt5 mocks (now attributes of self.mock_mt5)
        self.mock_mt5.initialize.return_value = True
        self.mock_mt5.symbols_get.return_value = []
        self.mock_mt5.last_error.return_value = (0, "Success")
        # self.mock_mt5.order_calc_margin will be used directly if needed, or can be further MagicMocked
        # self.mock_mt5.shutdown will be used directly if needed

        # Configure check_mt5_connection mock (already an attribute of self)
        # self.mock_check_mt5_connection.return_value = True (or whatever is appropriate)

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
        self.account_info_mock = MagicMock(
            balance=10000, equity=10000, margin_free=8000, leverage=100, currency="USD"
        )
        self.symbol_info_mock = MagicMock(
            volume_step=0.01,
            trade_contract_size=100000,
            trade_tick_value=10,  # tick_value_loss used by currency_risk
            trade_tick_value_loss=1,
            trade_tick_value_profit=2,
            trade_stops_level=10,
            point=0.0001,
            bid=1.1,
            ask=1.2,
            spread=int((1.2 - 1.1) / 0.0001),
            currency_base="EUR",
            currency_profit="USD",
            currency_margin="USD",
            path="Forex\\Majors\\EURUSD",  # Specific path for get_symbol_type
            volume_min=0.01,
            volume_max=100.0,  # For _check_lot
            trade_tick_size=0.00001,
        )
        # Ensure 'name' attribute exists for symbol_info if RiskManagement code uses it directly
        self.symbol_info_mock.name = self.symbol

        # Configure mock_Rates
        mock_returns_series = MagicMock()
        mock_returns_series.std.return_value = 0.0005
        mock_returns_series.mean.return_value = 0.0001  # Added for calculate_var
        self.mock_Rates.return_value.returns = mock_returns_series
        self.mock_Rates.return_value.get_rates_from_pos = MagicMock(
            return_value={"Close": [1.1, 1.2, 1.3]}
        )  # For calculate_var, get_std_stop

        # Configure datetime mock
        def strptime_side_effect(time_str, format_str):
            if format_str == "%H:%M":
                hour, minute = map(int, time_str.split(":"))
                # Use a fixed date, as only the time component is typically used by get_minutes
                return datetime(2023, 1, 1, hour, minute, 0)
            # Fallback for other formats if necessary, or raise an error.
            # For this specific test context, only "%H:%M" is expected.
            raise ValueError(f"Unexpected format string in strptime mock: {format_str}")

        self.mock_datetime.strptime.side_effect = strptime_side_effect

        # Mock for get_trades_history to simulate a DataFrame
        self.mock_trades_history_df = MagicMock()
        # Configure sums for the main df object
        self.mock_trades_history_df.profit.sum.return_value = 300
        self.mock_trades_history_df.commission.sum.return_value = 30
        self.mock_trades_history_df.fee.sum.return_value = 15
        self.mock_trades_history_df.swap.sum.return_value = 6
        # Configure sums for the iloc[1:] slice, used by risk_level()
        mock_df_slice = MagicMock()
        mock_df_slice.profit.sum.return_value = (
            300  # Assuming similar sum for slice for simplicity
        )
        self.mock_trades_history_df.iloc.return_value = mock_df_slice

        # Configure Account method mocks
        self.mock_account_get_account_info.return_value = self.account_info_mock
        self.mock_account_get_symbol_info.return_value = self.symbol_info_mock
        self.mock_account_get_trades_history.return_value = self.mock_trades_history_df
        self.mock_account_get_terminal_info.return_value = MagicMock(
            company=__BROKERS__["AMG"]
        )

        # Instantiate RiskManagement - Account methods are now patched globally for this test method
        self.risk_manager = RiskManagement(
            symbol=self.symbol,
            max_risk=self.max_risk,
            daily_risk=self.daily_risk,
            max_trades=self.max_trades,
            std_stop=self.std_stop,
            account_leverage=self.account_leverage,
            start_time=self.start_time,
            finishing_time=self.finishing_time,
            time_frame=self.time_frame,
        )

        # Mock the get_account_info and get_symbol_info methods on the instance if needed
        # These might override the class-level patches for specific instance behaviors or if
        # RiskManagement calls these methods on `self` rather than its internal `Account` instance.
        self.risk_manager.get_account_info = MagicMock(
            return_value=self.account_info_mock
        )
        self.risk_manager.get_symbol_info = MagicMock(
            return_value=self.symbol_info_mock
        )
        self.risk_manager.symbol_info = self.symbol_info_mock

    def test_initialization(self):
        # Test that attributes are correctly initialized
        self.assertEqual(self.risk_manager.symbol, self.symbol)
        self.assertEqual(self.risk_manager.max_risk, self.max_risk)
        self.assertEqual(
            self.risk_manager.daily_dd, self.daily_risk
        )  # daily_dd is the attribute name
        self.assertEqual(self.risk_manager.start_time, self.start_time)
        self.assertEqual(self.risk_manager.finishing_time, self.finishing_time)
        self.assertEqual(
            self.risk_manager._tf, self.time_frame
        )  # time_frame is stored in _tf
        self.assertEqual(
            self.risk_manager.std, self.std_stop
        )  # std is the attribute name
        self.assertEqual(self.risk_manager.account_leverage, self.account_leverage)
        self.assertIsNone(
            self.risk_manager.pchange
        )  # pchange_sl is None by default in setUp
        self.assertEqual(self.risk_manager.var_level, 0.95)  # Default var_level
        self.assertEqual(self.risk_manager.var_tf, "D1")  # Default var_time_frame
        self.assertIsNone(self.risk_manager.sl)  # sl is None by default
        self.assertIsNone(self.risk_manager.tp)  # tp is None by default
        self.assertIsNone(self.risk_manager.be)  # be is None by default
        self.assertEqual(self.risk_manager.rr, 1.5)  # Default rr
        self.assertEqual(self.risk_manager.symbol_info, self.symbol_info_mock)

    def test_risk_level(self):
        self.mock_trades_history_df.profit.sum.return_value = -100  # Loss
        self.mock_trades_history_df.commission.sum.return_value = 10  # Costs
        self.mock_trades_history_df.fee.sum.return_value = 5  # Costs
        self.mock_trades_history_df.swap.sum.return_value = 2  # Costs

        # Test scenario 1: Trade history (df) is None
        with patch.object(
            self.risk_manager, "get_trades_history", return_value=None
        ) as mock_get_history:
            result_none_history = self.risk_manager.risk_level()
        self.assertEqual(result_none_history, 0.0)
        mock_get_history.assert_called_once()

        # Test scenario 2: Trade history (df) is a mock DataFrame
        mock_df = MagicMock()
        mock_df_iloc_slice = MagicMock()
        mock_df_iloc_slice.profit.sum.return_value = -100
        mock_df.iloc.return_value = mock_df_iloc_slice

        mock_df.commission.sum.return_value = 10
        mock_df.fee.sum.return_value = 5
        mock_df.swap.sum.return_value = 2

        # Values from self.account_info_mock used by risk_level method
        # These are returned by self.risk_manager.account.get_account_info() due to setUp patching
        balance = self.account_info_mock.balance  # 10000
        equity = self.account_info_mock.equity  # 10000

        # Expected calculation based on RiskManagement.risk_level()
        # profit = -100
        # commisions = 10
        # fees = 5
        # swap = 2
        # total_profit = 10 + 5 + 2 + (-100) = -83
        total_profit_calc = (
            mock_df.commission.sum()
            + mock_df.fee.sum()
            + mock_df.swap.sum()
            + mock_df_iloc_slice.profit.sum()
        )

        initial_balance_calc = balance - total_profit_calc  # 10000 - (-83) = 10083

        expected_risk_val = 0.0
        if equity != 0:
            expected_risk_val = round(  # noqa: F841
                (((equity - initial_balance_calc) / equity) * 100) * -1, 2
            )  # noqa: F841
            # (((10000 - 10083) / 10000) * 100) * -1 = 0.83

        with patch.object(
            self.risk_manager, "get_trades_history", return_value=mock_df
        ) as mock_get_history_with_df:
            result_with_history = self.risk_manager.risk_level()  # noqa: F841

        mock_get_history_with_df.assert_called_once()

    def test_get_lot(self):
        expected_lot_from_currency_risk = 0.01

        # Scenario 1: volume_step = 0.01 (from setUp) -> round to 2 decimal places
        with patch.object(
            self.risk_manager, "risk_level", return_value=1.0
        ), patch.object(
            self.risk_manager, "get_symbol_type", return_value=SymbolType.FOREX
        ):
            self.assertEqual(
                self.risk_manager._volume_step(
                    self.risk_manager.symbol_info.volume_step
                ),
                2,
            )

            result1 = self.risk_manager.get_lot()
            # round(0.01, 2) = 0.01
            self.assertEqual(result1, expected_lot_from_currency_risk)

            # Scenario 2: volume_step = 1.0 -> round to 0 decimal places
            original_volume_step = self.risk_manager.symbol_info.volume_step
            self.risk_manager.symbol_info.volume_step = 1.0
            # self.risk_manager._volume_step(1.0) should return 0
            self.assertEqual(
                self.risk_manager._volume_step(
                    self.risk_manager.symbol_info.volume_step
                ),
                0,
            )

            result2 = self.risk_manager.get_lot()
            # round(0.01, 0) = 0.0. The method returns float.
            self.assertEqual(result2, 0.0)

            # Restore original volume_step
            self.risk_manager.symbol_info.volume_step = original_volume_step

            # Scenario 3: volume_step = 0.1 -> round to 1 decimal place
            self.risk_manager.symbol_info.volume_step = 0.1
            self.assertEqual(
                self.risk_manager._volume_step(
                    self.risk_manager.symbol_info.volume_step
                ),
                1,
            )
            result3 = self.risk_manager.get_lot()
            # round(0.01, 1) = 0.0
            self.assertEqual(result3, 0.0)

        # Restore original volume_step
        self.risk_manager.symbol_info.volume_step = original_volume_step

    def test_max_trade(self):
        # Test maximum trade calculation
        result = self.risk_manager.max_trade()
        self.assertEqual(result, 10)

    def test_get_minutes(self):
        result = self.risk_manager.get_minutes()
        expected_minutes = 480.0
        self.assertEqual(result, expected_minutes)

    def test_get_pchange_stop(self):
        # Scenario 1: pchange is None, should call get_std_stop()
        # We mock get_std_stop for this part of the test to ensure it's called.
        expected_std_stop_val = 1020  # from previous test calculation
        with patch.object(
            self.risk_manager, "get_std_stop", return_value=expected_std_stop_val
        ) as mock_get_std_stop_call:
            result_none_pchange = self.risk_manager.get_pchange_stop(None)
            self.assertEqual(result_none_pchange, expected_std_stop_val)
            mock_get_std_stop_call.assert_called_once()

        # Scenario 2: pchange is a value
        pchange_value = 2.0  # 2%
        point = self.risk_manager.symbol_info.point  # 0.0001
        av_price = (
            self.risk_manager.symbol_info.bid + self.risk_manager.symbol_info.ask
        ) / 2  # 1.15

        price_interval = (
            av_price * (100 - pchange_value) / 100
        )  # 1.15 * 98 / 100 = 1.127
        sl_point = float(
            (av_price - price_interval) / point
        )  # (1.15 - 1.127) / 0.0001 = 230.0
        sl_calc = round(sl_point)  # 230

        deviation = self.risk_manager.get_deviation()  # 1000
        min_sl_calc = (
            self.risk_manager.symbol_info.trade_stops_level * 2 + deviation
        )  # 10 * 2 + 1000 = 1020
        expected_result_with_pchange = max(
            sl_calc, min_sl_calc
        )  # max(230, 1020) = 1020

        result_with_pchange = self.risk_manager.get_pchange_stop(pchange_value)
        self.assertEqual(result_with_pchange, expected_result_with_pchange)

    def test_calculate_var(self):
        # Test with default tf "D1" and c=0.95
        # Expected calls and values:
        # minutes = 480
        # tf_int for "D1" = 480 (from get_minutes via _convert_time_frame)
        # interval = round((480/480)*252) = 252
        # P = self.account_info_mock.margin_free = 8000
        # mu = 0.0001 (from mock_Rates.return_value.returns.mean())
        # sigma = 0.0005 (from mock_Rates.return_value.returns.std())
        # c_level = 0.95

        # Reset mocks before the first call block
        self.mock_Rates.reset_mock()
        self.mock_Rates.return_value.returns.mean.reset_mock()
        self.mock_Rates.return_value.returns.std.reset_mock()

        expected_var_result = 5.7794  # Calculated manually, see thought process
        # P - P * (norm.ppf(1-c, mu, sigma) + 1)
        # norm.ppf(0.05, 0.0001, 0.0005) approx -0.000722425
        # 8000 - 8000 * (-0.000722425 + 1) = 5.7794

        # Mock var_cov_var as it's tested separately and involves scipy.stats.norm.ppf
        with patch.object(
            self.risk_manager, "var_cov_var", return_value=expected_var_result
        ) as mock_vcv:
            result = self.risk_manager.calculate_var()  # Uses default tf="D1", c=0.95
            self.assertAlmostEqual(result, expected_var_result, places=4)
            # Check that Rates was called correctly
            mock_vcv.assert_called_once()

        # Test with different tf and c
        # Reset mocks before the second call block
        self.mock_Rates.reset_mock()
        self.mock_Rates.return_value.returns.mean.reset_mock()
        self.mock_Rates.return_value.returns.std.reset_mock()

        custom_tf = "1h"  # tf_int = 60
        custom_c = 0.99
        # interval = round((480/60)*252) = 2016
        expected_var_custom = 10.0  # Dummy value for this call
        with patch.object(
            self.risk_manager, "var_cov_var", return_value=expected_var_custom
        ) as mock_vcv_custom:
            result_custom = self.risk_manager.calculate_var(tf=custom_tf, c=custom_c)
            self.assertEqual(result_custom, expected_var_custom)

            mock_vcv_custom.assert_called_once()

    def test_var_cov_var(self):
        # Test variance-covariance VaR calculation
        result = self.risk_manager.var_cov_var(P=10000, c=0.95, mu=0.001, sigma=0.02)
        self.assertGreater(result, 0)

    def test_get_stop_loss(self):
        min_sl_expected = 1019

        # Path 1: self.sl is not None
        original_sl = self.risk_manager.sl
        self.risk_manager.sl = 500
        self.assertEqual(
            self.risk_manager.get_stop_loss(), max(500, min_sl_expected)
        )  # 1020
        self.risk_manager.sl = 1500
        self.assertEqual(
            self.risk_manager.get_stop_loss(), max(1500, min_sl_expected)
        )  # 1500
        self.risk_manager.sl = original_sl  # Reset

        # Path 2: self.sl is None and self.std is True (default from setUp)
        # self.risk_manager.std is True from setUp
        # self.risk_manager.sl is None from setUp
        # It will call self.get_std_stop(). From test_get_std_stop, this is 1020.
        with patch.object(
            self.risk_manager, "get_std_stop", return_value=1020
        ) as mock_get_std_stop:
            self.assertEqual(
                self.risk_manager.get_stop_loss(), max(1020, min_sl_expected)
            )  # 1020
            mock_get_std_stop.assert_called_once()

        # Path 3: self.sl is None and self.std is False
        original_std = self.risk_manager.std
        self.risk_manager.std = False

        # Subcase 3a: currency_risk returns non-zero trade_loss
        mock_currency_risk_vals = {"currency_risk": 200.0, "trade_loss": 2.0}
        with patch.object(
            self.risk_manager, "currency_risk", return_value=mock_currency_risk_vals
        ) as mock_curr_risk:
            # sl_calc = round(200.0 / 2.0) = 100
            self.assertEqual(
                self.risk_manager.get_stop_loss(), max(100, min_sl_expected)
            )  # 1020
            mock_curr_risk.assert_called_once()

        # Subcase 3b: currency_risk returns zero trade_loss
        mock_currency_risk_vals_zero_loss = {"currency_risk": 200.0, "trade_loss": 0.0}
        with patch.object(
            self.risk_manager,
            "currency_risk",
            return_value=mock_currency_risk_vals_zero_loss,
        ) as mock_curr_risk_zero:
            self.assertEqual(self.risk_manager.get_stop_loss(), min_sl_expected)  # 1020
            mock_curr_risk_zero.assert_called_once()

        self.risk_manager.std = original_std  # Reset

    def test_get_take_profit(self):
        # Path 1: self.tp is not None
        original_tp = self.risk_manager.tp
        self.risk_manager.tp = 500
        # expected = 500 + deviation = 500 + 1000 = 1500
        self.assertEqual(self.risk_manager.get_take_profit(), 1499)
        self.risk_manager.tp = original_tp  # Reset

        # Path 2: self.tp is None
        # Calls self.get_stop_loss() * self.rr
        # Let's mock get_stop_loss to isolate this test
        mock_sl_value = 200
        expected_tp_calc = round(
            mock_sl_value * self.risk_manager.rr
        )  # round(200 * 1.5) = 300
        with patch.object(
            self.risk_manager, "get_stop_loss", return_value=mock_sl_value
        ) as mock_get_sl:
            self.assertEqual(self.risk_manager.get_take_profit(), expected_tp_calc)
            mock_get_sl.assert_called_once()

        # Test with different rr
        original_rr = self.risk_manager.rr
        self.risk_manager.rr = 2.0
        expected_tp_calc_new_rr = round(mock_sl_value * 2.0)  # round(200 * 2.0) = 400
        with patch.object(
            self.risk_manager, "get_stop_loss", return_value=mock_sl_value
        ) as mock_get_sl_2:
            self.assertEqual(
                self.risk_manager.get_take_profit(), expected_tp_calc_new_rr
            )
            mock_get_sl_2.assert_called_once()
        self.risk_manager.rr = original_rr  # Reset

    def test_get_currency_risk(self):
        # Allows self.risk_manager.currency_risk() to run.
        # Expected 'currency_risk' key from currency_risk() with setUp mocks is var_loss_value = 5.0
        expected_risk_value = 5.0

        with patch.object(
            self.risk_manager, "risk_level", return_value=1.0
        ), patch.object(
            self.risk_manager, "get_symbol_type", return_value=SymbolType.FOREX
        ):
            result = (
                self.risk_manager.get_currency_risk()
            )  # This is the method under test

        self.assertEqual(result, round(expected_risk_value, 2))

    def test_expected_profit(self):
        # Allows self.risk_manager.get_currency_risk() (and thus currency_risk()) to run.
        # self.risk_manager.rr is 1.5 from setUp.
        # Expected currency_risk from get_currency_risk() is 5.0.
        expected_profit_value = round(5.0 * 1.5, 2)  # 7.5

        with patch.object(
            self.risk_manager, "risk_level", return_value=1.0
        ), patch.object(
            self.risk_manager, "get_symbol_type", return_value=SymbolType.FOREX
        ):
            result = self.risk_manager.expected_profit()

        self.assertEqual(result, expected_profit_value)

    def test__convert_time_frame(self):
        self.assertEqual(self.risk_manager._convert_time_frame("1m"), TIMEFRAMES["1m"])
        self.assertEqual(self.risk_manager._convert_time_frame("5m"), TIMEFRAMES["5m"])
        self.assertEqual(self.risk_manager._convert_time_frame("1h"), 60)
        self.assertEqual(self.risk_manager._convert_time_frame("4h"), 240)

        # For 'D1', 'W1', 'MN1', it calls get_minutes()
        # self.risk_manager.get_minutes() is tested and returns 480.0 with current setUp.
        # To make this test more isolated, we can mock get_minutes here.
        with patch.object(
            self.risk_manager, "get_minutes", return_value=480.0
        ) as mock_get_minutes:
            self.assertEqual(self.risk_manager._convert_time_frame("D1"), 480.0)
            mock_get_minutes.assert_called_once()
            mock_get_minutes.reset_mock()  # Reset for subsequent calls within this test
            self.assertEqual(self.risk_manager._convert_time_frame("W1"), 480.0 * 5)
            mock_get_minutes.assert_called_once()
            mock_get_minutes.reset_mock()
            self.assertEqual(self.risk_manager._convert_time_frame("MN1"), 480.0 * 22)
            mock_get_minutes.assert_called_once()

    def test__volume_step(self):
        self.assertEqual(self.risk_manager._volume_step(0.01), 2)
        self.assertEqual(self.risk_manager._volume_step(0.1), 1)
        self.assertEqual(self.risk_manager._volume_step(1.0), 0)
        self.assertEqual(self.risk_manager._volume_step(1), 0)  # Test with int
        self.assertEqual(self.risk_manager._volume_step(0.10001), 5)
        self.assertEqual(self.risk_manager._volume_step(10), 0)  # Test with int > 1

    def test__check_lot(self):
        # symbol_info_mock has volume_min=0.01, volume_max=100.0
        self.assertEqual(self.risk_manager._check_lot(50.0), 50.0)  # Within limits
        self.assertEqual(self.risk_manager._check_lot(0.001), 0.01)  # Below min
        self.assertEqual(self.risk_manager._check_lot(0.01), 0.01)  # Equal to min
        self.assertEqual(self.risk_manager._check_lot(100.0), 100.0)  # Equal to max
        self.assertEqual(
            self.risk_manager._check_lot(200.0), 50.0
        )  # Above max (returns max / 2 as per code)

    def test_get_trade_risk(self):
        # get_trade_risk = (self.daily_dd or (self.max_risk - total_risk)) / max_trades
        # or 0 if total_risk >= self.max_risk

        # Scenario 1: total_risk < max_risk, daily_dd is set
        # self.daily_dd = 2.0 (from setUp), self.max_risk = 5.0 (from setUp)
        with patch.object(
            self.risk_manager, "risk_level", return_value=1.0
        ) as mock_rl, patch.object(
            self.risk_manager, "max_trade", return_value=10
        ) as mock_mt:
            expected_trade_risk = self.risk_manager.daily_dd / 10
            self.assertAlmostEqual(
                self.risk_manager.get_trade_risk(), expected_trade_risk
            )
            mock_rl.assert_called_once()
            mock_mt.assert_called_once()

        # Scenario 2: total_risk < max_risk, daily_dd is None
        original_daily_dd = self.risk_manager.daily_dd
        self.risk_manager.daily_dd = None
        with patch.object(
            self.risk_manager, "risk_level", return_value=1.0
        ) as mock_rl, patch.object(
            self.risk_manager, "max_trade", return_value=10
        ) as mock_mt:
            expected_trade_risk = (self.risk_manager.max_risk - 1.0) / 10
            self.assertAlmostEqual(
                self.risk_manager.get_trade_risk(), expected_trade_risk
            )
        self.risk_manager.daily_dd = original_daily_dd  # Reset

        # Scenario 3: total_risk >= max_risk
        with patch.object(
            self.risk_manager, "risk_level", return_value=self.risk_manager.max_risk
        ) as mock_rl:
            self.assertEqual(self.risk_manager.get_trade_risk(), 0)
        with patch.object(
            self.risk_manager, "risk_level", return_value=self.risk_manager.max_risk + 1
        ) as mock_rl:
            self.assertEqual(self.risk_manager.get_trade_risk(), 0)

    def test_get_deviation(self):
        # self.symbol_info_mock.spread is 1000 from setUp
        self.assertEqual(
            self.risk_manager.get_deviation(), self.symbol_info_mock.spread
        )

    def test_get_break_even(self):
        # self.symbol_info_mock.spread is 1000
        original_be_attr = (
            self.risk_manager.be
        )  # Store original be attribute from __init__

        # Path 1: self.be is an int
        self.risk_manager.be = 50  # Directly set attribute for test
        self.assertEqual(self.risk_manager.get_break_even(), 50)

        # Path 2: self.be is a float (calls get_pchange_stop)
        self.risk_manager.be = 0.5  # 0.5% pchange
        with patch.object(
            self.risk_manager, "get_pchange_stop", return_value=120
        ) as mock_get_pchange:
            self.assertEqual(self.risk_manager.get_break_even(), 120)
            mock_get_pchange.assert_called_once_with(0.5)

        self.risk_manager.be = None  # Reset for next path

        # Path 3: self.be is None (logic based on get_stop_loss and spread)
        # Case 3a: stop <= 100. Example: stop = 80. be = round((80+1000)*0.5) = round(540) = 540
        with patch.object(self.risk_manager, "get_stop_loss", return_value=80):
            self.assertEqual(self.risk_manager.get_break_even(), 540)

        # Case 3b: stop > 100 and stop <= 150. Example: stop = 120. be = round((120+1000)*0.35) = round(392) = 392
        with patch.object(self.risk_manager, "get_stop_loss", return_value=120):
            self.assertEqual(self.risk_manager.get_break_even(), 392)

        # Case 3c: stop > 150. Example: stop = 200. be = round((200+1000)*0.25) = round(300) = 300
        with patch.object(self.risk_manager, "get_stop_loss", return_value=200):
            self.assertEqual(self.risk_manager.get_break_even(), 300)

        self.risk_manager.be = original_be_attr  # Restore original be attribute

    def test_is_risk_ok(self):
        # self.max_risk is 5.0 from setUp
        with patch.object(self.risk_manager, "risk_level", return_value=4.0) as mock_rl:
            self.assertTrue(self.risk_manager.is_risk_ok())
            mock_rl.assert_called_once()

        with patch.object(self.risk_manager, "risk_level", return_value=5.0) as mock_rl:
            self.assertTrue(self.risk_manager.is_risk_ok())  # risk_level <= max_risk
            mock_rl.assert_called_once()

        with patch.object(self.risk_manager, "risk_level", return_value=5.1) as mock_rl:
            self.assertFalse(self.risk_manager.is_risk_ok())
            mock_rl.assert_called_once()

    def test_dailydd_property(self):
        # Test getter
        self.assertEqual(self.risk_manager.dailydd, self.daily_risk)  # From setUp
        # Test setter
        new_daily_risk = 3.0
        self.risk_manager.dailydd = new_daily_risk
        self.assertEqual(self.risk_manager.dailydd, new_daily_risk)
        # Reset to original value from setUp if necessary for other tests (though instance is new per test)
        self.risk_manager.dailydd = self.daily_risk

    def test_maxrisk_property(self):
        # Test getter
        self.assertEqual(self.risk_manager.maxrisk, self.max_risk)  # From setUp
        # Test setter
        new_max_risk = 10.0
        self.risk_manager.maxrisk = new_max_risk
        self.assertEqual(self.risk_manager.maxrisk, new_max_risk)
        # Reset to original value
        self.risk_manager.maxrisk = self.max_risk

    def tearDown(self):
        self.datetime_patcher.stop()
        self.mt5_patcher.stop()
        self.Rates_patcher.stop()
        self.check_mt5_connection_patcher.stop()
        self.account_get_account_info_patcher.stop()
        self.account_get_symbol_info_patcher.stop()
        self.account_get_trades_history_patcher.stop()
        self.account_get_terminal_info_patcher.stop()
        self.riskmanagement_var_cov_var_patcher.stop()
        self.rm_get_leverage_patcher.stop()


if __name__ == "__main__":
    unittest.main()
