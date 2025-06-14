import unittest
from datetime import datetime
from io import StringIO
from unittest.mock import MagicMock, patch

import pandas as pd

from bbstrader.metatrader.account import (
    __BROKERS__,
    FTMO,
    SUPPORTED_BROKERS,
    Account,
    AdmiralMarktsGroup,
    Broker,
    JustGlobalMarkets,
    PepperstoneGroupLimited,
)
from bbstrader.metatrader.utils import (
    AccountInfo,
    BookInfo,
    InvalidBroker,
    OrderCheckResult,
    OrderSentResult,
    SymbolInfo,
    SymbolType,
    TerminalInfo,
    TickInfo,
    TradeDeal,
    TradeOrder,
    TradePosition,
)

class TestAccount(unittest.TestCase):
    def setUp(self):
        # Patch the MetaTrader5 module
        self.mt5_patcher = patch("bbstrader.metatrader.account.mt5")
        self.mock_mt5 = self.mt5_patcher.start()

        # Configure the mock mt5 object
        # Add relevant MT5 constants for error codes
        self.mock_mt5.RES_S_OK = 0
        self.mock_mt5.RES_E_FAIL = 1
        self.mock_mt5.RES_E_INVALID_PARAMS = 2
        self.mock_mt5.RES_E_NOT_FOUND = 3
        self.mock_mt5.RES_E_INVALID_VERSION = 4
        self.mock_mt5.RES_E_AUTH_FAILED = 5
        self.mock_mt5.RES_E_UNSUPPORTED = 6
        self.mock_mt5.RES_E_AUTO_TRADING_DISABLED = 7
        self.mock_mt5.RES_E_INTERNAL_FAIL_SEND = 8
        self.mock_mt5.RES_E_INTERNAL_FAIL_RECEIVE = 9
        self.mock_mt5.RES_E_INTERNAL_FAIL_INIT = 10
        self.mock_mt5.RES_E_INTERNAL_FAIL_CONNECT = 11
        self.mock_mt5.RES_E_INTERNAL_FAIL_TIMEOUT = 12
        # Add trade action constants that might be used in tests
        self.mock_mt5.TRADE_ACTION_DEAL = 1
        self.mock_mt5.ORDER_TYPE_BUY = 0
        self.mock_mt5.ORDER_TYPE_SELL = 1
        self.mock_mt5.ORDER_FILLING_FOK = 1
        self.mock_mt5.ORDER_FILLING_IOC = 2
        self.mock_mt5.ORDER_FILLING_RETURN = 3
        self.mock_mt5.ORDER_TIME_GTC = 0
        self.mock_mt5.ORDER_TIME_DAY = 1
        self.mock_mt5.ORDER_TIME_SPECIFIED = 2
        self.mock_mt5.ORDER_TIME_SPECIFIED_DAY = 3

        self.mock_mt5.initialize.return_value = True
        self.mock_mt5.account_info.return_value = AccountInfo(
            login=12345,
            trade_mode=0,
            leverage=100,
            limit_orders=0,
            margin_so_mode=0,
            trade_allowed=True,
            trade_expert=True,
            margin_mode=0,
            currency_digits=2,
            fifo_close=False,
            balance=10000.0,
            credit=0.0,
            profit=0.0,
            equity=10000.0,
            margin=0.0,
            margin_free=10000.0,
            margin_level=0.0,
            margin_so_call=0.0,
            margin_so_so=0.0,
            margin_initial=0.0,
            margin_maintenance=0.0,
            assets=0.0,
            liabilities=0.0,
            commission_blocked=0.0,
            name="Test Account",
            server="Test Server",
            currency="USD",
            company=__BROKERS__["AMG"],
        )
        self.mock_mt5.terminal_info.return_value = TerminalInfo(
            community_account=False,
            community_connection=False,
            connected=True,
            dlls_allowed=True,
            trade_allowed=True,
            tradeapi_disabled=False,
            email_enabled=False,
            ftp_enabled=False,
            notifications_enabled=False,
            mqid=False,
            build=1355,
            maxbars=100000,
            codepage=0,
            ping_last=0,
            community_balance=0.0,
            retransmission=0.0,
            company=__BROKERS__["AMG"],
            name="MetaTrader 5",
            language="en",
            path="",
            data_path="",
            commondata_path="",
        )

        # Instantiate the Account class
        self.account = Account()

    def tearDown(self):
        # Stop the patcher
        self.mt5_patcher.stop()

    def test_get_account_info_default(self):
        # Test get_account_info with no arguments
        account_info = self.account.get_account_info()
        self.assertIsNotNone(account_info)
        self.assertEqual(account_info.login, 12345)
        self.assertEqual(account_info.name, "Test Account")

    def test_get_account_info_with_credentials(self):
        # Test get_account_info with account, password, and server
        mock_specific_account_info = AccountInfo(
            login=67890,
            trade_mode=0,
            leverage=50,
            limit_orders=0,
            margin_so_mode=0,
            trade_allowed=True,
            trade_expert=True,
            margin_mode=0,
            currency_digits=2,
            fifo_close=False,
            balance=5000.0,
            credit=0.0,
            profit=0.0,
            equity=5000.0,
            margin=0.0,
            margin_free=5000.0,
            margin_level=0.0,
            margin_so_call=0.0,
            margin_so_so=0.0,
            margin_initial=0.0,
            margin_maintenance=0.0,
            assets=0.0,
            liabilities=0.0,
            commission_blocked=0.0,
            name="Specific Account",
            server="Specific Server",
            currency="EUR",
            company="Specific Company",
        )
        self.mock_mt5.login.return_value = True

        # Set the return_value directly for the mt5.account_info() call
        # that will occur within self.account.get_account_info() for this specific test.
        original_account_info_mock = (
            self.mock_mt5.account_info.return_value
        )  # Preserve from setUp  # noqa: F841
        self.mock_mt5.account_info.return_value = mock_specific_account_info

        account_info_returned = self.account.get_account_info(
            account=67890, password="password", server="Specific Server"
        )

        # Restore mock for other tests if this wasn't the last action (though setUp handles isolation)
        self.mock_mt5.account_info.return_value = (
            original_account_info_mock  # Not strictly needed due to test isolation
        )

        self.mock_mt5.login.assert_called_once_with(
            67890, password="password", server="Specific Server", timeout=60000
        )
        self.assertIsNotNone(account_info_returned)
        self.assertEqual(account_info_returned.login, 67890)
        self.assertEqual(account_info_returned.name, "Specific Account")
        self.assertEqual(account_info_returned.currency, "EUR")

    def test_get_account_info_login_fails(self):
        self.mock_mt5.login.return_value = False
        # RES_E_AUTH_FAILED is a common code for login failures.
        self.mock_mt5.last_error.return_value = (
            self.mock_mt5.RES_E_AUTH_FAILED,
            "Login authorization failed",
        )
        with self.assertRaises(Exception):
            self.account.get_account_info(
                account=67890, password="password", server="Specific Server"
            )

    def test_get_account_info_returns_none(self):
        self.mock_mt5.account_info.return_value = None
        # Reset side_effect if it was set in another test
        self.mock_mt5.account_info.side_effect = None
        self.mock_mt5.last_error.return_value = (2, "Account info not found")

        # Call get_account_info without credentials first to reset internal state if necessary
        ret_val = self.account.get_account_info()
        self.assertIsNone(ret_val)

    @patch("bbstrader.metatrader.account.print")  # Patched print in the account module
    def test_show_account_info_success(
        self, mock_print
    ):  # mock_print is now the mock for the print function
        # Reset side_effect for account_info to ensure it returns the default mock value
        self.mock_mt5.account_info.side_effect = None
        # Ensure a valid AccountInfo object is returned by the mock
        self.mock_mt5.account_info.return_value = AccountInfo(
            login=12345,
            trade_mode=0,
            leverage=100,
            limit_orders=0,
            margin_so_mode=0,
            trade_allowed=True,
            trade_expert=True,
            margin_mode=0,
            currency_digits=2,
            fifo_close=False,
            balance=10000.0,
            credit=0.0,
            profit=0.0,
            equity=10000.0,
            margin=0.0,
            margin_free=10000.0,
            margin_level=0.0,
            margin_so_call=0.0,
            margin_so_so=0.0,
            margin_initial=0.0,
            margin_maintenance=0.0,
            assets=0.0,
            liabilities=0.0,
            commission_blocked=0.0,
            name="Test Account",
            server="Test Server",
            currency="USD",
            company=__BROKERS__["AMG"], 
        )

        self.account.show_account_info()

        # Check that print was called with the header
        # The f-string in source is print(f"
        header_found = False
        dataframe_output_found = False
        for call_args in mock_print.call_args_list:
            args, _ = call_args
            if args:  # Ensure there are positional arguments
                printed_text = str(args[0])
                if "ACCOUNT INFORMATIONS:" in printed_text:
                    header_found = True
                if (
                    "Test Account" in printed_text
                    and "12345" in printed_text
                    and "PROPERTY" in printed_text
                ):  # Assuming PROPERTY is a column name in df.to_string()
                    dataframe_output_found = True

        self.assertTrue(header_found, "Header 'ACCOUNT INFORMATIONS:' not printed.")
        self.assertTrue(
            dataframe_output_found,
            "DataFrame content (Test Account, 12345, PROPERTY) not found in print calls.",
        )

    @patch("sys.stdout", new_callable=StringIO)
    def test_show_account_info_failure(
        self, mock_stdout
    ):  # Added mock_stdout from original
        self.mock_mt5.account_info.return_value = None
        self.mock_mt5.account_info.side_effect = None  # Clear side effect
        self.mock_mt5.last_error.return_value = (
            self.mock_mt5.RES_E_NOT_FOUND,
            "Account info not found",
        )
        with self.assertRaises(Exception):
            self.account.show_account_info()

    def test_get_terminal_info_success(self):
        terminal_info = self.account.get_terminal_info()
        self.assertIsNotNone(terminal_info)
        # Assert against the company name explicitly set in setUp's mock TerminalInfo
        self.assertEqual(terminal_info.company, __BROKERS__["AMG"])
        self.assertEqual(terminal_info.name, "MetaTrader 5")

    def test_get_terminal_info_failure(self):
        self.mock_mt5.terminal_info.return_value = None
        self.mock_mt5.last_error.return_value = (3, "Terminal info not found")
        ret_val = self.account.get_terminal_info()
        self.assertIsNone(ret_val)

    @patch("bbstrader.metatrader.account.print")  # Patched print in the account module
    def test_get_terminal_info_show_success(
        self, mock_print
    ):  # mock_print for the print function
        # Ensure terminal_info mock is set up (done in setUp)
        # self.mock_mt5.terminal_info.return_value is already set in setUp.

        self.account.get_terminal_info(show=True)

        # Check that print was called with the DataFrame's string representation
        # This output will contain column names like "PROPERTY" and values.
        dataframe_output_found = False
        for call_args in mock_print.call_args_list:
            args, _ = call_args
            if args:
                printed_text = str(args[0])
                if (
                    "PROPERTY" in printed_text
                    and __BROKERS__["AMG"] in printed_text
                    and "MetaTrader 5" in printed_text
                ):
                    dataframe_output_found = True

        self.assertTrue(
            dataframe_output_found,
            "DataFrame content (PROPERTY, broker company, terminal name) not found in print calls.",
        )

    @patch("bbstrader.metatrader.account.urllib.request.urlretrieve")
    @patch("bbstrader.metatrader.account.CurrencyConverter")
    @patch("bbstrader.metatrader.account.os.path.isfile")
    @patch("bbstrader.metatrader.account.os.remove")
    def test_convert_currencies(
        self,
        mock_os_remove,
        mock_os_path_isfile,
        mock_currency_converter,
        mock_urlretrieve,
    ):
        # Mock that the file doesn't exist so it tries to download
        mock_os_path_isfile.return_value = False

        # Mock the CurrencyConverter
        mock_converter_instance = MagicMock()
        mock_converter_instance.currencies = {"USD", "EUR", "JPY"}
        mock_converter_instance.convert.return_value = 110.0  # Example conversion rate
        mock_currency_converter.return_value = mock_converter_instance

        # Test conversion
        result = self.account.convert_currencies(100, "USD", "JPY")
        self.assertEqual(result, 110.0)
        mock_urlretrieve.assert_called_once()  # Ensure download was attempted
        mock_currency_converter.assert_called_once()
        mock_converter_instance.convert.assert_called_once_with(
            amount=100, currency="USD", new_currency="JPY"
        )
        mock_os_remove.assert_called_once()  # Ensure cleanup was attempted

    @patch("bbstrader.metatrader.account.urllib.request.urlretrieve")
    @patch("bbstrader.metatrader.account.CurrencyConverter")
    @patch("bbstrader.metatrader.account.os.path.isfile")
    @patch("bbstrader.metatrader.account.os.remove")
    def test_convert_currencies_unsupported(
        self,
        mock_os_remove,
        mock_os_path_isfile,
        mock_currency_converter,
        mock_urlretrieve,
    ):
        # To avoid UnboundLocalError for 'c' with current source code, ensure 'c' is defined.
        # This means the block 'if not os.path.isfile(filename):' must be entered.
        mock_os_path_isfile.return_value = False  # Changed from True

        mock_converter_instance = MagicMock()
        mock_converter_instance.currencies = {
            "USD",
            "EUR",
        }  # JPY is not supported for conversion
        mock_currency_converter.return_value = mock_converter_instance

        result = self.account.convert_currencies(100, "USD", "JPY")
        self.assertEqual(
            result, 100
        )  # Should return original amount due to unsupported target currency

        # Assertions adjusted for the new path
        mock_urlretrieve.assert_called_once()  # Download is now attempted
        mock_currency_converter.assert_called_once()  # CurrencyConverter is initialized
        # os.remove is called in the source code after 'c = CurrencyConverter(filename)'
        # and before 'supported = c.currencies'
        mock_os_remove.assert_called_once()

    def test_get_currency_rates(self):
        mock_symbol_info = SymbolInfo(
            custom=False,
            chart_mode=0,
            select=True,
            visible=True,
            session_deals=0,
            session_buy_orders=0,
            session_sell_orders=0,
            volume=0,
            volumehigh=0,
            volumelow=0,
            time=datetime.now(),
            digits=5,
            spread=0,
            spread_float=True,
            ticks_bookdepth=0,
            trade_calc_mode=0,
            trade_mode=0,
            start_time=0,
            expiration_time=0,
            trade_stops_level=0,
            trade_freeze_level=0,
            trade_exemode=0,
            swap_mode=0,
            swap_rollover3days=0,
            margin_hedged_use_leg=False,
            expiration_mode=0,
            filling_mode=0,
            order_mode=0,
            order_gtc_mode=0,
            option_mode=0,
            option_right=0,
            bid=1.0,
            bidhigh=1.0,
            bidlow=1.0,
            ask=1.0,
            askhigh=1.0,
            asklow=1.0,
            last=1.0,
            lasthigh=1.0,
            lastlow=1.0,
            volume_real=0,
            volumehigh_real=0,
            volumelow_real=0,
            option_strike=0,
            point=0.00001,
            trade_tick_value=0,
            trade_tick_value_profit=0,
            trade_tick_value_loss=0,
            trade_tick_size=0,
            trade_contract_size=100000,
            trade_accrued_interest=0,
            trade_face_value=0,
            trade_liquidity_rate=0,
            volume_min=0.01,
            volume_max=100,
            volume_step=0.01,
            volume_limit=0,
            swap_long=0,
            swap_short=0,
            margin_initial=0,
            margin_maintenance=0,
            session_volume=0,
            session_turnover=0,
            session_interest=0,
            session_buy_orders_volume=0,
            session_sell_orders_volume=0,
            session_open=0,
            session_close=0,
            session_aw=0,
            session_price_settlement=0,
            session_price_limit_min=0,
            session_price_limit_max=0,
            margin_hedged=0,
            price_change=0,
            price_volatility=0,
            price_theoretical=0,
            price_greeks_delta=0,
            price_greeks_theta=0,
            price_greeks_gamma=0,
            price_greeks_vega=0,
            price_greeks_rho=0,
            price_greeks_omega=0,
            price_sensitivity=0,
            basis="",
            category="",
            currency_base="EUR",
            currency_profit="USD",
            currency_margin="EUR",
            bank="",
            description="Euro vs US Dollar",
            exchange="",
            formula="",
            isin="",
            name="EURUSD",
            page="",
            path="Forex\\Majors\\EURUSD",
        )
        self.mock_mt5.symbol_info.return_value = mock_symbol_info
        # Default account currency is USD from setUp
        rates = self.account.get_currency_rates("EURUSD")
        expected_rates = {"bc": "EUR", "mc": "EUR", "pc": "USD", "ac": "USD"}
        self.assertEqual(rates, expected_rates)
        self.mock_mt5.symbol_info.assert_called_once_with("EURUSD")

    def _get_mock_symbol_info(
        self,
        name="EURUSD",
        path="Forex\\Majors\\EURUSD",
        description="Euro vs US Dollar",
        currency_base="EUR",
        currency_profit="USD",
        currency_margin="EUR",
        time_val=1678886400,
    ):
        return SymbolInfo(
            custom=False,
            chart_mode=0,
            select=True,
            visible=True,
            session_deals=0,
            session_buy_orders=0,
            session_sell_orders=0,
            volume=0,
            volumehigh=0,
            volumelow=0,
            time=datetime.fromtimestamp(time_val),
            digits=5,
            spread=0,
            spread_float=True,
            ticks_bookdepth=0,
            trade_calc_mode=0,
            trade_mode=0,
            start_time=0,
            expiration_time=0,
            trade_stops_level=0,
            trade_freeze_level=0,
            trade_exemode=0,
            swap_mode=0,
            swap_rollover3days=0,
            margin_hedged_use_leg=False,
            expiration_mode=0,
            filling_mode=0,
            order_mode=0,
            order_gtc_mode=0,
            option_mode=0,
            option_right=0,
            bid=1.0,
            bidhigh=1.0,
            bidlow=1.0,
            ask=1.0,
            askhigh=1.0,
            asklow=1.0,
            last=1.0,
            lasthigh=1.0,
            lastlow=1.0,
            volume_real=0,
            volumehigh_real=0,
            volumelow_real=0,
            option_strike=0,
            point=0.00001,
            trade_tick_value=0,
            trade_tick_value_profit=0,
            trade_tick_value_loss=0,
            trade_tick_size=0,
            trade_contract_size=100000,
            trade_accrued_interest=0,
            trade_face_value=0,
            trade_liquidity_rate=0,
            volume_min=0.01,
            volume_max=100,
            volume_step=0.01,
            volume_limit=0,
            swap_long=0,
            swap_short=0,
            margin_initial=0,
            margin_maintenance=0,
            session_volume=0,
            session_turnover=0,
            session_interest=0,
            session_buy_orders_volume=0,
            session_sell_orders_volume=0,
            session_open=0,
            session_close=0,
            session_aw=0,
            session_price_settlement=0,
            session_price_limit_min=0,
            session_price_limit_max=0,
            margin_hedged=0,
            price_change=0,
            price_volatility=0,
            price_theoretical=0,
            price_greeks_delta=0,
            price_greeks_theta=0,
            price_greeks_gamma=0,
            price_greeks_vega=0,
            price_greeks_rho=0,
            price_greeks_omega=0,
            price_sensitivity=0,
            basis="",
            category="",
            currency_base=currency_base,
            currency_profit=currency_profit,
            currency_margin=currency_margin,
            bank="",
            description=description,
            exchange="",
            formula="",
            isin="",
            name=name,
            page="",
            path=path,
        )

    def test_get_symbols_all(self):
        mock_symbols_data = []
        for symbol_name in ["EURUSD", "AAPL", "[ES]"]:
            mock = MagicMock()
            mock.name = symbol_name
            mock_symbols_data.append(mock)

        self.mock_mt5.symbols_get.return_value = mock_symbols_data
        self.mock_mt5.symbol_info.side_effect = [
            self._get_mock_symbol_info(name="EURUSD", path="Forex\\Majors\\EURUSD"),
            self._get_mock_symbol_info(
                name="AAPL", path="Stocks\\US\\AAPL", description="Apple Inc."
            ),
            self._get_mock_symbol_info(
                name="[ES]", path="Futures\\Indices\\ES", description="E-mini S&P 500"
            ),
        ]

        symbols = self.account.get_symbols(symbol_type="ALL")
        self.assertEqual(len(symbols), 3)
        self.assertIn("EURUSD", symbols)
        self.assertIn("AAPL", symbols)
        self.assertIn("[ES]", symbols)
        self.mock_mt5.symbols_get.assert_called_once()

    def test_get_symbols_filtered_forex(self):
        mock_symbols_data = []
        for symbol_name in ["EURUSD", "USDJPY", "USDJPY"]:
            mock = MagicMock()
            mock.name = symbol_name
            mock_symbols_data.append(mock)

        self.mock_mt5.symbols_get.return_value = mock_symbols_data
        self.mock_mt5.symbol_info.side_effect = [
            self._get_mock_symbol_info(name="EURUSD", path="Forex\\Majors\\EURUSD"),
            self._get_mock_symbol_info(name="USDJPY", path="Forex\\Majors\\USDJPY"),
            self._get_mock_symbol_info(name="AAPL", path="Stocks\\US\\AAPL"),
        ]
        symbols = self.account.get_symbols(symbol_type=SymbolType.FOREX)
        self.assertEqual(len(symbols), 2)
        self.assertIn("EURUSD", symbols)
        self.assertIn("USDJPY", symbols)
        self.assertNotIn("AAPL", symbols)

    def test_get_symbols_filtered_etf_check_description(self):
        mock_symbols_data = []
        for symbol_name in ["SPY", "GLD"]:
            mock = MagicMock()
            mock.name = symbol_name
            mock_symbols_data.append(mock)

        self.mock_mt5.symbols_get.return_value = mock_symbols_data
        self.mock_mt5.symbol_info.side_effect = [
            self._get_mock_symbol_info(
                name="SPY", path="ETFs\\US\\SPY", description="SPDR S&P 500 ETF Trust"
            ),
            self._get_mock_symbol_info(
                name="GLD", path="ETFs\\US\\GLD", description="SPDR Gold Shares"
            ),  # This one should fail the check
        ]
        with self.assertRaises(ValueError) as context:
            self.account.get_symbols(symbol_type=SymbolType.ETFs, check_etf=True)
        self.assertIn("doesn't have 'ETF' in its description", str(context.exception))

    def test_get_symbols_save_to_file(self):
        mock_symbols_data = [MagicMock(name="EURUSD")]
        self.mock_mt5.symbols_get.return_value = mock_symbols_data
        self.mock_mt5.symbol_info.return_value = self._get_mock_symbol_info(
            name="EURUSD"
        )

        with patch("builtins.open", unittest.mock.mock_open()) as mock_file:
            self.account.get_symbols(
                save=True, file_name="test_symbols", include_desc=True
            )
            mock_file.assert_called_once_with(
                "test_symbols.txt", mode="w", encoding="utf-8"
            )
            # Check if content was written (simplified check)
            handle = mock_file()
            handle.write.assert_any_call(
                "EURUSD|Euro vs US Dollar\n"
            )  # Max length dependent

    def test_get_symbols_no_symbols_found(self):
        self.mock_mt5.symbols_get.return_value = []
        self.mock_mt5.last_error.return_value = (
            self.mock_mt5.RES_E_NOT_FOUND,
            "No symbols available",
        )
        with self.assertRaises(Exception):
            self.account.get_symbols()

    def test_get_symbol_type(self):
        self.mock_mt5.symbol_info.return_value = self._get_mock_symbol_info(
            path="Forex\\Majors\\EURUSD"
        )
        self.assertEqual(self.account.get_symbol_type("EURUSD"), SymbolType.FOREX)

        self.mock_mt5.symbol_info.return_value = self._get_mock_symbol_info(
            path="Stocks\\US\\AAPL"
        )
        self.assertEqual(self.account.get_symbol_type("AAPL"), SymbolType.STOCKS)

        self.mock_mt5.symbol_info.return_value = self._get_mock_symbol_info(
            path="Futures\\Energies\\CL"
        )
        self.assertEqual(self.account.get_symbol_type("CL"), SymbolType.FUTURES)

    def test_get_fx_symbols_majors_default_broker(self):
        # Default broker is AdmiralMarkets which supports specific FX categories
        mock_symbols_data = []
        for symbol_name in ["EURUSD", "AUDCAD", "EURTRY"]:
            mock = MagicMock()
            mock.name = symbol_name
            mock_symbols_data.append(mock)

        self.mock_mt5.symbols_get.return_value = mock_symbols_data
        self.mock_mt5.symbol_info.side_effect = [
            self._get_mock_symbol_info(name="EURUSD", path="Forex\\Majors\\EURUSD"),
            self._get_mock_symbol_info(name="AUDCAD", path="Forex\\Crosses\\AUDCAD"),
            self._get_mock_symbol_info(name="EURTRY", path="Forex\\Exotics\\EURTRY"),
        ]
        symbols = self.account.get_fx_symbols(category="majors")
        self.assertEqual(len(symbols), 3)
        self.assertIn("EURUSD", symbols)

    def test_get_fx_symbols_unsupported_broker_raises_invalidbroker_on_init(
        self,
    ):  # Renamed for clarity
        # Store original mock configurations to restore if needed, though setUp handles isolation
        original_terminal_company = self.mock_mt5.terminal_info.return_value.company
        original_account_company = self.mock_mt5.account_info.return_value.company

        unsupported_broker_name = "SomeOtherBroker"

        # Configure mocks to simulate an unsupported broker
        # Both terminal_info().company and account_info().company might be checked by Account or Broker class
        self.mock_mt5.terminal_info.return_value = (
            self.mock_mt5.terminal_info.return_value._replace(
                company=unsupported_broker_name
            )
        )
        self.mock_mt5.account_info.return_value = (
            self.mock_mt5.account_info.return_value._replace(
                company=unsupported_broker_name
            )
        )

        with self.assertRaises(InvalidBroker) as context:
            # Instantiating Account with an unsupported broker (and default copy=False, backtest=False)
            # should raise InvalidBroker.
            Account()

        self.assertIn(
            f"{unsupported_broker_name} is not currently supported broker",
            str(context.exception),
        )

        # Restore original mock configurations if these mocks are used by other tests in a specific sequence
        # (Not strictly necessary due to test isolation by setUp/tearDown for instance mocks,
        # but good practice if class-level mocks or shared state were involved)
        self.mock_mt5.terminal_info.return_value = (
            self.mock_mt5.terminal_info.return_value._replace(
                company=original_terminal_company
            )
        )
        self.mock_mt5.account_info.return_value = (
            self.mock_mt5.account_info.return_value._replace(
                company=original_account_company
            )
        )

    def test_get_stocks_from_country_default_broker(self):
        mock_symbols_data = []
        for symbol_name in ["AAPL", "MSFT", "VOW3.DE"]:
            mock = MagicMock()
            mock.name = symbol_name
            mock_symbols_data.append(mock)

        self.mock_mt5.symbols_get.return_value = mock_symbols_data
        self.mock_mt5.symbol_info.side_effect = [
            self._get_mock_symbol_info(name="AAPL", path="Stocks\\US\\AAPL"),
            self._get_mock_symbol_info(name="MSFT", path="Stocks\\US\\MSFT"),
            self._get_mock_symbol_info(name="VOW3.DE", path="Stocks\\Germany\\VOW3.DE"),
        ]
        symbols = self.account.get_stocks_from_country(country_code="USA")
        self.assertEqual(len(symbols), 3)
        self.assertIn("AAPL", symbols)
        self.assertIn("MSFT", symbols)


    def test_get_future_symbols_default_broker_metals(self):
        # AdmiralMarkets specific logic for futures categories
        mock_symbols_data = []
        for symbol_name in ["_XAUUSD", "_OILUSD", "COCOA", "#USTBond"]:
            mock = MagicMock()
            mock.name = symbol_name
            mock_symbols_data.append(mock)

        commodities_symbols_data = []
        for symbol_name in ["XAUUSD", "OILUSD", "COCOA"]:
            mock = MagicMock()
            mock.name = symbol_name
            commodities_symbols_data.append(mock)

        # This setup is a bit complex due to nested calls to get_symbols and get_symbol_info
        def symbol_info_side_effect_futures(symbol_name):
            if symbol_name == "_XAUUSD":
                return self._get_mock_symbol_info(
                    name="_XAUUSD", path="Futures\\Metals\\_XAUUSD"
                )
            if symbol_name == "_OILUSD":
                return self._get_mock_symbol_info(
                    name="_OILUSD", path="Futures\\Energies\\_OILUSD"
                )
            if symbol_name == "COCOA":
                return self._get_mock_symbol_info(
                    name="COCOA", path="Commodities\\Agricultures\\COCOA"
                )  # For the commodity check
            if symbol_name == "XAUUSD":
                return self._get_mock_symbol_info(
                    name="XAUUSD", path="Commodities\\Metals\\XAUUSD"
                )
            if symbol_name == "OILUSD":
                return self._get_mock_symbol_info(
                    name="OILUSD", path="Commodities\\Energies\\OILUSD"
                )
            if symbol_name == "#USTBond":
                return self._get_mock_symbol_info(
                    name="#USTBond", path="Futures\\Bonds\\#USTBond"
                )
            return self._get_mock_symbol_info(name=symbol_name)

        self.mock_mt5.symbols_get.side_effect = [
            commodities_symbols_data,  # First call from get_symbols(SymbolType.COMMODITIES)
            mock_symbols_data,  # Second call from get_symbols(SymbolType.FUTURES)
        ]
        self.mock_mt5.symbol_info.side_effect = symbol_info_side_effect_futures

        symbols = self.account.get_future_symbols(category="metals")
        self.assertIn("_XAUUSD", symbols)
        self.assertNotIn("_OILUSD", symbols)

    def test_get_symbol_info_success(self):
        mock_info = self._get_mock_symbol_info(name="EURUSD", time_val=1678886400)
        self.mock_mt5.symbol_info.return_value = mock_info
        info = self.account.get_symbol_info("EURUSD")
        self.assertIsNotNone(info)
        self.assertEqual(info.name, "EURUSD")
        self.assertEqual(info.time, datetime.fromtimestamp(1678886400))
        self.mock_mt5.symbol_info.assert_called_once_with("EURUSD")

    def test_get_symbol_info_not_found(self):
        self.mock_mt5.symbol_info.return_value = None
        # RES_E_NOT_FOUND for symbol not found
        self.mock_mt5.last_error.return_value = (
            self.mock_mt5.RES_E_NOT_FOUND,
            "Symbol not found in Market Watch",
        )

        ret_val = self.account.get_symbol_info("UNKNOWN")
        self.assertIsNone(ret_val)  # This part should still be true

        # Test that show_symbol_info raises correctly
        with self.assertRaises(
            Exception
        ) as context:  # Expecting specific HistoryNotFound
            self.account.show_symbol_info("UNKNOWN")
        self.assertTrue(
            "No history found for UNKNOWN" in str(context.exception)
            or "Symbol not found" in str(context.exception)
        )

    @patch("sys.stdout", new_callable=StringIO)
    def test_show_symbol_info_success(self, mock_stdout):
        mock_info = self._get_mock_symbol_info(
            name="GBPUSD", description="Great Britain Pound vs US Dollar"
        )
        self.mock_mt5.symbol_info.return_value = mock_info
        self.account.show_symbol_info("GBPUSD")
        output = mock_stdout.getvalue()
        self.assertIn(
            "SYMBOL INFO FOR GBPUSD (Great Britain Pound vs US Dollar)", output
        )
        self.assertIn("currency_base", output)
        self.assertIn("GBP", output)

    def test_get_tick_info_success(self):
        mock_tick = TickInfo(
            time=datetime.fromtimestamp(1678886500),
            bid=1.05,
            ask=1.06,
            last=1.055,
            volume=10,
            time_msc=1678886500000,
            flags=0,
            volume_real=10.0,
        )
        self.mock_mt5.symbol_info_tick.return_value = mock_tick
        tick = self.account.get_tick_info("EURUSD")
        self.assertIsNotNone(tick)
        self.assertEqual(tick.bid, 1.05)
        self.assertEqual(tick.time, datetime.fromtimestamp(1678886500))
        self.mock_mt5.symbol_info_tick.assert_called_once_with("EURUSD")

    def test_get_tick_info_not_found(self):
        self.mock_mt5.symbol_info_tick.return_value = None
        self.mock_mt5.last_error.return_value = (6, "Tick not found")
        ret_val = self.account.get_tick_info("UNKNOWN_TICK")
        self.assertIsNone(ret_val)

    @patch("sys.stdout", new_callable=StringIO)
    def test_show_tick_info_success(self, mock_stdout):
        mock_tick = TickInfo(
            time=datetime.fromtimestamp(1678886500),
            bid=1.05,
            ask=1.06,
            last=1.055,
            volume=10,
            time_msc=1678886500000,
            flags=0,
            volume_real=10.0,
        )
        self.mock_mt5.symbol_info_tick.return_value = mock_tick
        # Also need to mock symbol_info for the description part in _show_info
        self.mock_mt5.symbol_info.return_value = self._get_mock_symbol_info(
            name="EURUSD", description="Euro vs US Dollar"
        )

        self.account.show_tick_info("EURUSD")
        output = mock_stdout.getvalue()
        self.assertIn(
            "TICK INFO FOR EURUSD", output
        )  # Description might or might not be there based on how _show_info handles TickInfo
        self.assertIn("bid", output)
        self.assertIn("1.05", output)

    def test_get_market_book_success(self):
        mock_book_data = (
            BookInfo(type=0, price=1.1, volume=10.0, volume_dbl=10.0),  # TYPE_BUY
            BookInfo(type=1, price=1.2, volume=5.0, volume_dbl=5.0),  # TYPE_SELL
        )
        self.mock_mt5.market_book_get.return_value = mock_book_data
        book = self.account.get_market_book("EURUSD")
        self.assertIsNotNone(book)
        self.assertEqual(len(book), 2)
        self.assertEqual(book[0].price, 1.1)
        self.assertEqual(book[1].volume, 5.0)
        self.mock_mt5.market_book_get.assert_called_once_with("EURUSD")

    def test_get_market_book_empty(self):
        self.mock_mt5.market_book_get.return_value = None
        self.mock_mt5.last_error.return_value = (
            7,
            "Market book empty",
        )  # Example error
        book = self.account.get_market_book("EMPTYBOOK")
        self.assertIsNone(book)

    def test_calculate_margin_success(self):
        self.mock_mt5.order_calc_margin.return_value = 150.75
        margin = self.account.calculate_margin(
            action="buy", symbol="EURUSD", lot=0.1, price=1.1000
        )
        self.assertEqual(margin, 150.75)
        self.mock_mt5.order_calc_margin.assert_called_once_with(
            self.mock_mt5.ORDER_TYPE_BUY, "EURUSD", 0.1, 1.1000
        )

    def test_calculate_margin_error(self):
        self.mock_mt5.order_calc_margin.side_effect = Exception("Calculation error")
        self.mock_mt5.last_error.return_value = (
            self.mock_mt5.RES_E_FAIL,
            "Calc error detail",
        )  # 1 is often generic MT5.RES_E_FAIL
        with self.assertRaises(Exception) as context:
            self.account.calculate_margin(
                action="sell", symbol="GBPUSD", lot=0.5, price=1.2500
            )
        self.assertTrue(
            "Calc error detail" in str(context.exception)
            or "Calculation error" in str(context.exception)
        )

    def test_check_order_success(self):
        # Mock the TradeRequest object that would be part of OrderCheckResult.request
        # This should simulate the MqlTradeRequest structure returned by MT5
        mock_mql_trade_request = MagicMock()
        mock_mql_trade_request.action = self.mock_mt5.TRADE_ACTION_DEAL
        mock_mql_trade_request.symbol = "EURUSD"
        mock_mql_trade_request.volume = 0.1
        mock_mql_trade_request.price = 1.1
        mock_mql_trade_request.type = self.mock_mt5.ORDER_TYPE_BUY
        mock_mql_trade_request.magic = 123
        mock_mql_trade_request.order = 0
        mock_mql_trade_request.stoplimit = 0.0
        mock_mql_trade_request.sl = 0.0
        mock_mql_trade_request.tp = 0.0
        mock_mql_trade_request.deviation = 0
        mock_mql_trade_request.type_filling = self.mock_mt5.ORDER_FILLING_FOK
        mock_mql_trade_request.type_time = self.mock_mt5.ORDER_TIME_GTC
        mock_mql_trade_request.expiration = 0
        mock_mql_trade_request.comment = "test check"
        mock_mql_trade_request.position = 0
        mock_mql_trade_request.position_by = 0

        # The _asdict() method is crucial for named tuples
        mock_mql_trade_request._asdict.return_value = {
            "action": mock_mql_trade_request.action,
            "symbol": mock_mql_trade_request.symbol,
            "volume": mock_mql_trade_request.volume,
            "price": mock_mql_trade_request.price,
            "type": mock_mql_trade_request.type,
            "magic": mock_mql_trade_request.magic,
            "order": mock_mql_trade_request.order,
            "stoplimit": mock_mql_trade_request.stoplimit,
            "sl": mock_mql_trade_request.sl,
            "tp": mock_mql_trade_request.tp,
            "deviation": mock_mql_trade_request.deviation,
            "type_filling": mock_mql_trade_request.type_filling,
            "type_time": mock_mql_trade_request.type_time,
            "expiration": mock_mql_trade_request.expiration,
            "comment": mock_mql_trade_request.comment,
            "position": mock_mql_trade_request.position,
            "position_by": mock_mql_trade_request.position_by,
        }

        mock_result = MagicMock(spec=OrderCheckResult)  # Use spec for better mocking
        mock_result.retcode = 0
        mock_result.balance = 10000.0
        mock_result.equity = 10000.0
        mock_result.profit = 0.0
        mock_result.margin = 50.0
        mock_result.margin_free = 9950.0
        mock_result.margin_level = 20000.0
        mock_result.comment = "Order check OK"
        mock_result.request = mock_mql_trade_request  # Assign the detailed mock here

        # The _asdict() for the main OrderCheckResult object
        mock_result._asdict.return_value = {
            "retcode": mock_result.retcode,
            "balance": mock_result.balance,
            "equity": mock_result.equity,
            "profit": mock_result.profit,
            "margin": mock_result.margin,
            "margin_free": mock_result.margin_free,
            "margin_level": mock_result.margin_level,
            "comment": mock_result.comment,
            "request": mock_mql_trade_request,
        }
        self.mock_mt5.order_check.return_value = mock_result

        # Reconstruct the request to pass to the method
        check_request = {
            "action": self.mock_mt5.TRADE_ACTION_DEAL,
            "symbol": "EURUSD",
            "volume": 0.1,
            "price": 1.1,
            "type": self.mock_mt5.ORDER_TYPE_BUY,
            "magic": 123,
            "order": 0,
            "stoplimit": 0.0,
            "sl": 0.0,
            "tp": 0.0,
            "deviation": 0,
            "type_filling": self.mock_mt5.ORDER_FILLING_FOK,
            "type_time": self.mock_mt5.ORDER_TIME_GTC,
            "expiration": 0,
            "comment": "test check",
            "position": 0,
            "position_by": 0,
        }

        result = self.account.check_order(check_request)
        self.assertIsNotNone(result)
        self.assertEqual(result.retcode, 0)
        self.assertEqual(result.comment, "Order check OK")
        self.mock_mt5.order_check.assert_called_once_with(check_request)
        self.assertEqual(
            result.request.symbol, "EURUSD"
        )  # Accessing the TradeRequest object

    def test_send_order_success(self):
        mock_mql_trade_request = MagicMock()
        mock_mql_trade_request.action = self.mock_mt5.TRADE_ACTION_DEAL
        mock_mql_trade_request.symbol = "EURUSD"
        mock_mql_trade_request.volume = 0.1
        mock_mql_trade_request.price = 1.1
        mock_mql_trade_request.type = self.mock_mt5.ORDER_TYPE_BUY
        mock_mql_trade_request.magic = 123
        mock_mql_trade_request.order = 0  # For new orders, order ticket is 0
        mock_mql_trade_request.stoplimit = 0.0
        mock_mql_trade_request.sl = 0.0
        mock_mql_trade_request.tp = 0.0
        mock_mql_trade_request.deviation = 10  # Example deviation
        mock_mql_trade_request.type_filling = self.mock_mt5.ORDER_FILLING_FOK
        mock_mql_trade_request.type_time = self.mock_mt5.ORDER_TIME_GTC
        mock_mql_trade_request.expiration = 0
        mock_mql_trade_request.comment = "test send"
        mock_mql_trade_request.position = 0
        mock_mql_trade_request.position_by = 0

        mock_mql_trade_request._asdict.return_value = {
            "action": mock_mql_trade_request.action,
            "symbol": mock_mql_trade_request.symbol,
            "volume": mock_mql_trade_request.volume,
            "price": mock_mql_trade_request.price,
            "type": mock_mql_trade_request.type,
            "magic": mock_mql_trade_request.magic,
            "order": mock_mql_trade_request.order,
            "stoplimit": mock_mql_trade_request.stoplimit,
            "sl": mock_mql_trade_request.sl,
            "tp": mock_mql_trade_request.tp,
            "deviation": mock_mql_trade_request.deviation,
            "type_filling": mock_mql_trade_request.type_filling,
            "type_time": mock_mql_trade_request.type_time,
            "expiration": mock_mql_trade_request.expiration,
            "comment": mock_mql_trade_request.comment,
            "position": mock_mql_trade_request.position,
            "position_by": mock_mql_trade_request.position_by,
        }

        mock_result = MagicMock(spec=OrderSentResult)
        mock_result.retcode = 10009  # Request completed
        mock_result.deal = 12345
        mock_result.order = 54321  # Actual order ticket assigned by server
        mock_result.volume = 0.1
        mock_result.price = 1.1
        mock_result.bid = 1.0990
        mock_result.ask = 1.1010
        mock_result.comment = "Request completed"
        mock_result.request_id = 0
        mock_result.retcode_external = 0
        mock_result.request = mock_mql_trade_request

        mock_result._asdict.return_value = {
            "retcode": mock_result.retcode,
            "deal": mock_result.deal,
            "order": mock_result.order,
            "volume": mock_result.volume,
            "price": mock_result.price,
            "bid": mock_result.bid,
            "ask": mock_result.ask,
            "comment": mock_result.comment,
            "request_id": mock_result.request_id,
            "retcode_external": mock_result.retcode_external,
            "request": mock_mql_trade_request,
        }
        self.mock_mt5.order_send.return_value = mock_result

        send_request = {
            "action": self.mock_mt5.TRADE_ACTION_DEAL,
            "symbol": "EURUSD",
            "volume": 0.1,
            "price": 1.1,
            "type": self.mock_mt5.ORDER_TYPE_BUY,
            "magic": 123,
            "order": 0,
            "stoplimit": 0.0,
            "sl": 0.0,
            "tp": 0.0,
            "deviation": 10,
            "type_filling": self.mock_mt5.ORDER_FILLING_FOK,
            "type_time": self.mock_mt5.ORDER_TIME_GTC,
            "expiration": 0,
            "comment": "test send",
            "position": 0,
            "position_by": 0,
        }
        result = self.account.send_order(send_request)
        self.assertIsNotNone(result)
        self.assertEqual(result.retcode, 10009)
        self.assertEqual(result.order, 54321)
        self.mock_mt5.order_send.assert_called_once_with(send_request)
        self.assertEqual(
            result.request.symbol, "EURUSD"
        )  # Accessing the TradeRequest object

    def _get_mock_position(
        self, ticket=1, symbol="EURUSD", volume=0.1, price_open=1.1, type_val=0
    ):  # type_val=0 for buy
        return TradePosition(
            ticket=ticket,
            time=int(datetime.now().timestamp()),
            time_msc=0,
            time_update=0,
            time_update_msc=0,
            type=type_val,
            magic=0,
            identifier=0,
            reason=0,
            volume=volume,
            price_open=price_open,
            sl=0,
            tp=0,
            price_current=price_open + 0.001,
            swap=0,
            profit=10.0,
            symbol=symbol,
            comment="test",
            external_id="",
        )

    def test_get_positions_all_as_tuple(self):
        mock_positions_data = [
            self._get_mock_position(ticket=101, symbol="EURUSD"),
            self._get_mock_position(
                ticket=102, symbol="GBPUSD", type_val=1
            ),  # type_val=1 for sell
        ]
        self.mock_mt5.positions_get.return_value = mock_positions_data
        positions = self.account.get_positions(to_df=False)
        self.assertIsNotNone(positions)
        self.assertIsInstance(positions, tuple)
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions[0].ticket, 101)
        self.assertEqual(positions[1].symbol, "GBPUSD")
        self.mock_mt5.positions_get.assert_called_once_with()

    def test_get_positions_by_symbol_as_df(self):
        mock_positions_data = [self._get_mock_position(symbol="AAPL")]
        self.mock_mt5.positions_get.return_value = mock_positions_data
        positions_df = self.account.get_positions(symbol="AAPL", to_df=True)
        self.assertIsNotNone(positions_df)
        self.assertIsInstance(positions_df, pd.DataFrame)
        self.assertEqual(len(positions_df), 1)
        self.assertEqual(positions_df.iloc[0]["symbol"], "AAPL")
        self.mock_mt5.positions_get.assert_called_once_with(symbol="AAPL")

    def test_get_positions_no_positions(self):
        self.mock_mt5.positions_get.return_value = []  # Empty list
        positions = self.account.get_positions()
        self.assertIsNone(positions)

        self.mock_mt5.positions_get.return_value = None  # None
        positions = self.account.get_positions()
        self.assertIsNone(positions)

    def _get_mock_deal(
        self,
        ticket=201,
        order=54321,
        symbol="EURUSD",
        volume=0.1,
        price=1.1,
        type_val=0,
        entry=0,
        time=None,
        position_id=101,
    ):
        deal_time = int(time if time is not None else datetime.now().timestamp())
        return TradeDeal(
            ticket=ticket,
            order=order,
            time=deal_time,
            time_msc=deal_time * 1000,  # Match time_msc with time
            type=type_val,
            entry=entry,
            magic=0,
            position_id=position_id,
            reason=0,
            volume=volume,
            price=price,
            commission=0,
            swap=0,
            profit=10.0,
            fee=0,
            symbol=symbol,
            comment="test deal",
            external_id="",
        )

    def test_get_trades_history_date_range_as_df(self):
        mock_deals_data = [
            self._get_mock_deal(
                ticket=201, symbol="EURUSD", time=int(datetime(2023, 1, 15).timestamp())
            ),
            self._get_mock_deal(
                ticket=202, symbol="GBPUSD", time=int(datetime(2023, 1, 16).timestamp())
            ),
        ]
        self.mock_mt5.history_deals_get.return_value = mock_deals_data
        from_date = datetime(2023, 1, 1)
        to_date = datetime(2023, 1, 31)
        history_df = self.account.get_trades_history(
            date_from=from_date, date_to=to_date, to_df=True
        )

        self.assertIsNotNone(history_df)
        self.assertIsInstance(history_df, pd.DataFrame)
        self.assertEqual(len(history_df), 2)
        self.assertEqual(history_df.iloc[0]["symbol"], "EURUSD")
        self.mock_mt5.history_deals_get.assert_called_once_with(from_date, to_date)

    def test_get_trades_history_by_ticket_as_tuple(self):
        mock_deals_data = [self._get_mock_deal(ticket=205, order=50001)]
        self.mock_mt5.history_deals_get.return_value = mock_deals_data
        history_tuple = self.account.get_trades_history(
            ticket=50001, to_df=False
        )  # Filter by order ticket

        self.assertIsNotNone(history_tuple)
        self.assertIsInstance(history_tuple, tuple)
        self.assertEqual(len(history_tuple), 1)
        self.assertEqual(history_tuple[0].ticket, 205)
        self.mock_mt5.history_deals_get.assert_called_once_with(ticket=50001)

    def test_get_trades_history_no_deals(self):
        self.mock_mt5.history_deals_get.return_value = []
        history = self.account.get_trades_history()
        self.assertIsNone(history)

        self.mock_mt5.history_deals_get.return_value = None
        history = self.account.get_trades_history()
        self.assertIsNone(history)

    def _get_mock_order(
        self,
        ticket=301,
        symbol="EURUSD",
        price_open=1.1,
        volume_initial=0.1,
        type_val=0,
        time_setup=None,
        position_id=0,
    ):
        order_time_setup = int(
            time_setup if time_setup is not None else datetime.now().timestamp()
        )
        return TradeOrder(
            ticket=ticket,
            time_setup=order_time_setup,
            time_setup_msc=order_time_setup * 1000,  # Match time_setup_msc
            time_done=0,
            time_done_msc=0,
            time_expiration=0,
            type=type_val,
            type_time=0,
            type_filling=0,
            state=0,
            magic=0,
            position_id=position_id,
            position_by_id=0,
            reason=0,
            volume_initial=volume_initial,
            volume_current=volume_initial,
            price_open=price_open,
            sl=0,
            tp=0,
            price_current=price_open,
            price_stoplimit=0,
            symbol=symbol,
            comment="test order",
            external_id="",
        )

    def test_get_orders_all_as_tuple(self):
        mock_orders_data = [
            self._get_mock_order(ticket=301, symbol="EURUSD"),
            self._get_mock_order(
                ticket=302, symbol="GBPUSD", type_val=1
            ),  # type=1 SELL
        ]
        self.mock_mt5.orders_get.return_value = mock_orders_data
        orders = self.account.get_orders(to_df=False)
        self.assertIsNotNone(orders)
        self.assertIsInstance(orders, tuple)
        self.assertEqual(len(orders), 2)
        self.assertEqual(orders[0].ticket, 301)
        self.mock_mt5.orders_get.assert_called_once_with()

    def test_get_orders_by_symbol_as_df(self):
        mock_orders_data = [self._get_mock_order(symbol="AAPL")]
        self.mock_mt5.orders_get.return_value = mock_orders_data
        orders_df = self.account.get_orders(symbol="AAPL", to_df=True)
        self.assertIsNotNone(orders_df)
        self.assertIsInstance(orders_df, pd.DataFrame)
        self.assertEqual(len(orders_df), 1)
        self.assertEqual(orders_df.iloc[0]["symbol"], "AAPL")
        self.mock_mt5.orders_get.assert_called_once_with(symbol="AAPL")

    def test_get_orders_no_orders(self):
        self.mock_mt5.orders_get.return_value = []
        orders = self.account.get_orders()
        self.assertIsNone(orders)

        self.mock_mt5.orders_get.return_value = None
        orders = self.account.get_orders()
        self.assertIsNone(orders)

    def test_get_orders_history_date_range_as_df(self):
        mock_orders_hist_data = [
            self._get_mock_order(
                ticket=401,
                symbol="XAUUSD",
                time_setup=int(datetime(2023, 2, 10).timestamp()),
            ),
            self._get_mock_order(
                ticket=402,
                symbol="USOIL",
                time_setup=int(datetime(2023, 2, 12).timestamp()),
            ),
        ]
        self.mock_mt5.history_orders_get.return_value = mock_orders_hist_data
        from_date = datetime(2023, 2, 1)
        to_date = datetime(2023, 2, 28)
        history_df = self.account.get_orders_history(
            date_from=from_date, date_to=to_date, to_df=True
        )

        self.assertIsNotNone(history_df)
        self.assertIsInstance(history_df, pd.DataFrame)
        self.assertEqual(len(history_df), 2)
        self.assertEqual(history_df.iloc[0]["symbol"], "XAUUSD")
        self.mock_mt5.history_orders_get.assert_called_once_with(from_date, to_date)

    def test_get_orders_history_by_position_as_tuple(self):
        mock_orders_hist_data = [self._get_mock_order(ticket=405, position_id=1001)]
        self.mock_mt5.history_orders_get.return_value = mock_orders_hist_data
        history_tuple = self.account.get_orders_history(position=1001, to_df=False)

        self.assertIsNotNone(history_tuple)
        self.assertIsInstance(history_tuple, tuple)
        self.assertEqual(len(history_tuple), 1)
        self.assertEqual(history_tuple[0].position_id, 1001)
        self.mock_mt5.history_orders_get.assert_called_once_with(position=1001)

    def test_get_orders_history_no_orders(self):
        self.mock_mt5.history_orders_get.return_value = []
        history = self.account.get_orders_history()
        self.assertIsNone(history)

        self.mock_mt5.history_orders_get.return_value = None
        history = self.account.get_orders_history()
        self.assertIsNone(history)

    def test_shutdown(self):
        self.account.shutdown()
        self.mock_mt5.shutdown.assert_called_once()

    def test_check_brokers_supported(self):
        # This is implicitly tested by setUp, but an explicit test can be added
        # Ensure no InvalidBroker exception is raised for a supported broker
        try:
            # Re-initialize with a known supported broker (already done in setUp)
            self.mock_mt5.account_info.return_value = (
                self.mock_mt5.account_info.return_value._replace(
                    company=SUPPORTED_BROKERS[0]
                )
            )
            self.mock_mt5.terminal_info.return_value = (
                self.mock_mt5.terminal_info.return_value._replace(
                    company=SUPPORTED_BROKERS[0]
                )
            )
            Account()
        except InvalidBroker:
            self.fail("InvalidBroker raised unexpectedly for a supported broker")

    def test_check_brokers_unsupported(self):
        self.mock_mt5.account_info.return_value = (
            self.mock_mt5.account_info.return_value._replace(
                company="Unsupported Broker Inc."
            )
        )
        self.mock_mt5.terminal_info.return_value = (
            self.mock_mt5.terminal_info.return_value._replace(
                company="Unsupported Broker Inc."
            )
        )
        with self.assertRaises(InvalidBroker) as context:
            Account()
        self.assertIn("is not currently supported broker", str(context.exception))

    def test_check_brokers_copy_flag(self):
        self.mock_mt5.account_info.return_value = (
            self.mock_mt5.account_info.return_value._replace(
                company="Unsupported Broker Inc."
            )
        )
        self.mock_mt5.terminal_info.return_value = (
            self.mock_mt5.terminal_info.return_value._replace(
                company="Unsupported Broker Inc."
            )
        )
        try:
            Account(copy=True)  # Should not raise InvalidBroker
        except InvalidBroker:
            self.fail("InvalidBroker raised unexpectedly when copy=True")

    def test_check_brokers_backtest_flag(self):
        self.mock_mt5.account_info.return_value = (
            self.mock_mt5.account_info.return_value._replace(
                company="Unsupported Broker Inc."
            )
        )
        self.mock_mt5.terminal_info.return_value = (
            self.mock_mt5.terminal_info.return_value._replace(
                company="Unsupported Broker Inc."
            )
        )
        try:
            Account(backtest=True)  # Should not raise InvalidBroker
        except InvalidBroker:
            self.fail("InvalidBroker raised unexpectedly when backtest=True")

    def test_property_broker(self):
        # __BROKERS__ needs to be available in the test context.
        # Default company from setUp is __BROKERS__["AMG"]
        self.assertEqual(self.account.broker.name, __BROKERS__["AMG"])
        self.assertIsInstance(self.account.broker, Broker)

        # Test with a different broker
        self.mock_mt5.terminal_info.return_value = (
            self.mock_mt5.terminal_info.return_value._replace(
                company=__BROKERS__["JGM"]
            )
        )
        self.account = Account()
        self.assertEqual(self.account.broker.name, __BROKERS__["JGM"])
        self.assertIsInstance(
            self.account.broker, Broker
        )  # It's always a Broker instance

    def test_property_timezone(self):
        # Default broker from setUp is AdmiralMarktsGroup (AMG)
        self.assertEqual(self.account.timezone, AdmiralMarktsGroup().timezone)

        # Test with FTMO
        self.mock_mt5.account_info.return_value = (
            self.mock_mt5.account_info.return_value._replace(
                company=__BROKERS__["FTMO"]
            )
        )
        self.mock_mt5.terminal_info.return_value = (
            self.mock_mt5.terminal_info.return_value._replace(
                company=__BROKERS__["FTMO"]
            )
        )
        self.account = Account()
        self.assertEqual(self.account.timezone, FTMO().timezone)

        # Test with Pepperstone
        self.mock_mt5.account_info.return_value = (
            self.mock_mt5.account_info.return_value._replace(company=__BROKERS__["PGL"])
        )
        self.mock_mt5.terminal_info.return_value = (
            self.mock_mt5.terminal_info.return_value._replace(
                company=__BROKERS__["PGL"]
            )
        )
        self.account = Account()
        self.assertEqual(self.account.timezone, PepperstoneGroupLimited().timezone)

        # Test with JustGlobalMarkets
        self.mock_mt5.account_info.return_value = (
            self.mock_mt5.account_info.return_value._replace(company=__BROKERS__["JGM"])
        )
        self.mock_mt5.terminal_info.return_value = (
            self.mock_mt5.terminal_info.return_value._replace(
                company=__BROKERS__["JGM"]
            )
        )
        self.account = Account()
        self.assertEqual(self.account.timezone, JustGlobalMarkets().timezone)

    def test_property_name(self):
        # Relies on get_account_info().name
        self.assertEqual(self.account.name, "Test Account")
        self.mock_mt5.account_info.return_value = (
            self.mock_mt5.account_info.return_value._replace(name="Another Name")
        )
        self.assertEqual(self.account.name, "Another Name")

    def test_property_number(self):
        self.assertEqual(self.account.number, 12345)
        self.mock_mt5.account_info.return_value = (
            self.mock_mt5.account_info.return_value._replace(login=99999)
        )
        self.assertEqual(self.account.number, 99999)

    def test_property_server(self):
        self.assertEqual(self.account.server, "Test Server")
        self.mock_mt5.account_info.return_value = (
            self.mock_mt5.account_info.return_value._replace(server="Live Server")
        )
        self.assertEqual(self.account.server, "Live Server")

    def test_property_balance(self):
        self.assertEqual(self.account.balance, 10000.0)
        self.mock_mt5.account_info.return_value = (
            self.mock_mt5.account_info.return_value._replace(balance=12345.67)
        )
        self.assertEqual(self.account.balance, 12345.67)

    def test_property_leverage(self):
        self.assertEqual(self.account.leverage, 100)
        self.mock_mt5.account_info.return_value = (
            self.mock_mt5.account_info.return_value._replace(leverage=200)
        )
        self.assertEqual(self.account.leverage, 200)

    def test_property_equity(self):
        self.assertEqual(self.account.equity, 10000.0)
        self.mock_mt5.account_info.return_value = (
            self.mock_mt5.account_info.return_value._replace(equity=10500.50)
        )
        self.assertEqual(self.account.equity, 10500.50)

    def test_property_currency(self):
        self.assertEqual(self.account.currency, "USD")
        self.mock_mt5.account_info.return_value = (
            self.mock_mt5.account_info.return_value._replace(currency="EUR")
        )
        self.assertEqual(self.account.currency, "EUR")

    def test_property_language(self):
        # Relies on get_terminal_info().language
        self.assertEqual(self.account.language, "en")
        self.mock_mt5.terminal_info.return_value = (
            self.mock_mt5.terminal_info.return_value._replace(language="fr")
        )
        # Re-initialize account if terminal_info is cached by the property or its underlying calls upon Account init.
        # self.account = Account() # Or ensure property re-fetches.
        self.assertEqual(self.account.language, "fr")

    def test_property_maxbars(self):
        # Relies on get_terminal_info().maxbars
        self.assertEqual(self.account.maxbars, 100000)
        self.mock_mt5.terminal_info.return_value = (
            self.mock_mt5.terminal_info.return_value._replace(maxbars=50000)
        )
        self.assertEqual(self.account.maxbars, 50000)


if __name__ == "__main__":
    unittest.main()
