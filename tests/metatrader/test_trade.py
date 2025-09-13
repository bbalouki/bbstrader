import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from bbstrader.metatrader.trade import (
    Trade,
    TradeAction,
    TradeSignal,
    TradingMode,
    create_trade_instance,
    generate_signal,
)
from bbstrader.metatrader.utils import (
    AccountInfo,
    SymbolInfo,
    TickInfo,
    TradeOrder,
    TradePosition,
)


class TestTrade(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        self.mt5_patcher = patch("bbstrader.metatrader.trade.Mt5")
        self.mock_mt5 = self.mt5_patcher.start()
        self.addCleanup(self.mt5_patcher.stop)

        self.logger_patcher = patch("bbstrader.metatrader.trade.log")
        self.mock_logger = self.logger_patcher.start()
        self.addCleanup(self.logger_patcher.stop)

        patch(
            "bbstrader.metatrader.risk.RiskManagement.__init__", return_value=None
        ).start()
        self.addCleanup(patch.stopall)

        patch("bbstrader.metatrader.trade.check_mt5_connection").start()
        patch("bbstrader.metatrader.trade.Trade.select_symbol").start()
        patch("bbstrader.metatrader.trade.Trade.prepare_symbol").start()

        self.account_info = AccountInfo(
            login=12345,
            balance=10000.0,
            equity=10000.0,
            currency="USD",
            name="Test Account",
            server="Test Server",
            leverage=100,
            trade_mode=0,
            limit_orders=0,
            margin_so_mode=0,
            trade_allowed=True,
            trade_expert=True,
            margin_mode=0,
            currency_digits=2,
            fifo_close=False,
            credit=0.0,
            profit=0.0,
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
            company="MetaQuotes",
        )
        self.symbol_info = SymbolInfo(
            name="EURUSD",
            visible=True,
            point=0.00001,
            digits=5,
            spread=10,
            trade_tick_value=1.0,
            trade_tick_size=0.00001,
            trade_contract_size=100000,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            bid=1.10000,
            ask=1.10010,
            time=datetime.now(),
            custom=False,
            chart_mode=0,
            select=True,
            session_deals=0,
            session_buy_orders=0,
            session_sell_orders=0,
            volume=0,
            volumehigh=0,
            volumelow=0,
            bidhigh=0,
            bidlow=0,
            askhigh=0,
            asklow=0,
            last=0,
            lasthigh=0,
            lastlow=0,
            volume_real=0,
            volumehigh_real=0,
            volumelow_real=0,
            option_strike=0,
            trade_tick_value_profit=1.0,
            trade_tick_value_loss=1.0,
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
            page="",
            path="Forex\\Majors\\EURUSD",
            start_time=0,
            expiration_time=0,
            spread_float=True,
            ticks_bookdepth=0,
            trade_calc_mode=0,
            trade_mode=0,
            trade_accrued_interest=0.0,
            trade_face_value=0.0,
            trade_liquidity_rate=0.0,
            volume_limit=0.0,
            swap_long=0.0,
            swap_short=0.0,
        )
        self.tick_info = TickInfo(
            time=datetime.now(),
            bid=1.10000,
            ask=1.10010,
            last=1.10005,
            volume=100,
            time_msc=int(datetime.now().timestamp() * 1000),
            flags=6,
            volume_real=100.0,
        )

        self.trade = Trade(symbol="EURUSD", expert_id=98181105, verbose=False)

        # Manually set attributes from the patched RiskManagement parent class
        self.trade.symbol_info = self.symbol_info
        self.trade.account_leverage = True
        self.trade.be = 10
        self.trade.max_risk = 5.0
        self.trade.rr = 2.0
        self.trade.copy_mode = False

        # Patch instance methods to return mock data
        patch.object(
            self.trade, "get_account_info", return_value=self.account_info
        ).start()
        patch.object(
            self.trade, "get_symbol_info", return_value=self.symbol_info
        ).start()
        patch.object(self.trade, "get_tick_info", return_value=self.tick_info).start()
        patch.object(self.trade, "get_lot", return_value=0.1).start()
        patch.object(self.trade, "get_stop_loss", return_value=200).start()
        patch.object(self.trade, "get_take_profit", return_value=300).start()
        patch.object(
            self.trade,
            "send_order",
            return_value=MagicMock(retcode=self.mock_mt5.TRADE_RETCODE_DONE, order=123),
        ).start()
        # This patch is crucial to prevent the UnboundLocalError in close_request
        patch.object(self.trade, "check_order", return_value=True).start()

    def tearDown(self):
        """This method is no longer needed as addCleanup handles stopping patches."""
        pass

    def test_trade_signal_initialization(self):
        """Test TradeSignal dataclass initialization and validation."""
        signal = TradeSignal(id=1, symbol="EURUSD", action=TradeAction.BUY, price=1.2)
        self.assertEqual(signal.id, 1)
        self.assertEqual(signal.action, TradeAction.BUY)

        with self.assertRaises(TypeError):
            TradeSignal(id=1, symbol="EURUSD", action="BUY", price=1.2)

        with self.assertRaises(ValueError):
            TradeSignal(id=1, symbol="EURUSD", action=TradeAction.BUY, stoplimit=1.2)

    def test_generate_signal(self):
        """Test the generate_signal factory function."""
        signal = generate_signal(
            id=1, symbol="EURUSD", action=TradeAction.SELL, price=1.3
        )
        self.assertIsInstance(signal, TradeSignal)
        self.assertEqual(signal.price, 1.3)

    def test_trading_mode_enum(self):
        """Test the TradingMode enum."""
        self.assertTrue(TradingMode.BACKTEST.isbacktest())
        self.assertFalse(TradingMode.LIVE.isbacktest())
        self.assertTrue(TradingMode.LIVE.islive())
        self.assertFalse(TradingMode.BACKTEST.islive())

    @patch("bbstrader.metatrader.trade.tabulate")
    @patch("builtins.print")
    def test_summary(self, mock_print, mock_tabulate):
        """Test the summary method."""
        self.trade.summary()
        mock_tabulate.assert_called_once()
        mock_print.assert_called()

    @patch("bbstrader.metatrader.trade.tabulate")
    @patch("builtins.print")
    def test_risk_management_summary(self, mock_print, mock_tabulate):
        """Test the risk_managment method."""
        with (
            patch.object(
                self.trade, "get_stats", return_value=({}, {"total_profit": 100})
            ),
            patch.object(
                self.trade,
                "currency_risk",
                return_value={"trade_loss": 10, "trade_profit": 20},
            ),
            patch.object(self.trade, "get_currency_rates", return_value={"mc": "EUR"}),
            patch.object(self.trade, "is_risk_ok", return_value=True),
            patch.object(self.trade, "risk_level", return_value=1.0),
            patch.object(self.trade, "get_leverage", return_value="1:100"),
            patch.object(self.trade, "volume", return_value=0.1),
            patch.object(self.trade, "get_currency_risk", return_value=100),
            patch.object(self.trade, "expected_profit", return_value=200),
            patch.object(self.trade, "get_break_even", return_value=50),
            patch.object(self.trade, "get_deviation", return_value=20),
            patch.object(self.trade, "get_minutes", return_value=60),
            patch.object(self.trade, "max_trade", return_value=10),
        ):
            self.trade.risk_managment()
            mock_tabulate.assert_called_once()
            mock_print.assert_called()

    @patch("pandas.DataFrame.to_csv")
    @patch("os.makedirs")
    def test_statistics(self, mock_makedirs, mock_to_csv):
        """Test the statistics method."""
        stats1 = {
            "deals": 1,
            "profit": 100,
            "win_trades": 1,
            "loss_trades": 0,
            "total_fees": -10,
            "average_fee": -10,
            "win_rate": 100,
        }
        stats2 = {"total_profit": 90, "profitability": "Yes"}
        with (
            patch.object(self.trade, "get_stats", return_value=(stats1, stats2)),
            patch.object(self.trade, "sharpe", return_value=1.5),
            patch.object(self.trade, "get_currency_risk", return_value=100),
            patch.object(self.trade, "expected_profit", return_value=200),
        ):
            self.trade.statistics(save=True, dir="test_stats")
            mock_makedirs.assert_called_with("test_stats", exist_ok=True)
            mock_to_csv.assert_called_once()

    def test_open_buy_position(self):
        """Test opening a buy position."""
        with patch.object(self.trade, "check", return_value=True):
            result = self.trade.open_buy_position(action="BMKT")
            self.assertTrue(result)
            self.trade.send_order.assert_called()

    def test_open_sell_position(self):
        """Test opening a sell position."""
        with patch.object(self.trade, "check", return_value=True):
            result = self.trade.open_sell_position(action="SMKT")
            self.assertTrue(result)
            self.trade.send_order.assert_called()

    def test_open_position(self):
        """Test the generic open_position method."""
        with patch.object(self.trade, "open_buy_position") as mock_buy:
            self.trade.open_position(action="BMKT")
            mock_buy.assert_called_once()

        with patch.object(self.trade, "open_sell_position") as mock_sell:
            self.trade.open_position(action="SMKT")
            mock_sell.assert_called_once()

        with self.assertRaises(ValueError):
            self.trade.open_position(action="INVALID_ACTION")

    def test_close_position(self):
        """Test closing a position."""
        position = self._get_mock_position(ticket=123)
        with patch.object(self.trade, "get_positions", return_value=[position]):
            result = self.trade.close_position(ticket=123)
            self.assertTrue(result)
            self.trade.send_order.assert_called()

    def test_close_order(self):
        """Test closing an order."""
        with patch.object(
            self.trade, "close_request", return_value=True
        ) as mock_close_request:
            result = self.trade.close_order(ticket=456)
            self.assertTrue(result)
            mock_close_request.assert_called_once()

    def test_modify_order(self):
        """Test modifying an order."""
        order = self._get_mock_order(ticket=789)
        with (
            patch.object(self.trade, "get_orders", return_value=[order]),
            patch.object(self.trade, "check_order", return_value=True),
        ):
            self.trade.modify_order(ticket=789, price=1.15)
            self.trade.send_order.assert_called()
            call_args = self.trade.send_order.call_args[0][0]
            self.assertEqual(call_args["price"], 1.15)

    def test_create_trade_instance(self):
        """Test the create_trade_instance factory function."""
        with patch("bbstrader.metatrader.trade.Trade") as mock_trade:
            params = {"expert_id": 123}
            symbols = ["EURUSD", "GBPUSD"]
            instances = create_trade_instance(symbols, params)
            self.assertEqual(len(instances), 2)
            self.assertIn("EURUSD", instances)
            self.assertIn("GBPUSD", instances)
            self.assertEqual(mock_trade.call_count, 2)

    def _get_mock_position(
        self,
        ticket=1,
        symbol="EURUSD",
        volume=0.1,
        price_open=1.1,
        type=0,
        magic=98181105,
        profit=0.0,
    ):
        return TradePosition(
            ticket=ticket,
            time=int(datetime.now().timestamp()),
            time_msc=0,
            time_update=0,
            time_update_msc=0,
            type=type,
            magic=magic,
            identifier=0,
            reason=0,
            volume=volume,
            price_open=price_open,
            sl=0,
            tp=0,
            price_current=price_open + 0.001,
            swap=0,
            profit=profit,
            symbol=symbol,
            comment="test",
            external_id="",
        )

    def _get_mock_order(
        self,
        ticket=1,
        symbol="EURUSD",
        price_open=1.1,
        volume_initial=0.1,
        type=0,
        magic=98181105,
    ):
        return TradeOrder(
            ticket=ticket,
            time_setup=int(datetime.now().timestamp()),
            time_setup_msc=0,
            time_done=0,
            time_done_msc=0,
            time_expiration=0,
            type=type,
            type_time=0,
            type_filling=0,
            state=0,
            magic=magic,
            position_id=0,
            position_by_id=0,
            reason=0,
            volume_initial=volume_initial,
            volume_current=0.1,
            price_open=price_open,
            sl=0,
            tp=0,
            price_current=price_open,
            price_stoplimit=0,
            symbol=symbol,
            comment="test",
            external_id="",
        )


if __name__ == "__main__":
    unittest.main()
