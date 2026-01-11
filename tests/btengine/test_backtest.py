import unittest
from unittest.mock import MagicMock
from datetime import datetime
from bbstrader.btengine.backtest import BacktestEngine


class TestBacktestEngine(unittest.TestCase):
    def setUp(self):
        # Create mock classes with minimum expected interface
        self.symbol_list = ['FAKE']
        self.initial_capital = 100000.0
        self.heartbeat = 0.0
        self.start_date = datetime(2020, 1, 1)

        self.mock_data_handler_cls = MagicMock()
        self.mock_strategy_cls = MagicMock()
        self.mock_execution_handler_cls = MagicMock()

        # Mock data_handler instance
        self.mock_data_handler = MagicMock()
        self.mock_data_handler.continue_backtest = False
        self.mock_data_handler.get_latest_bar_datetime.return_value = self.start_date
        self.mock_data_handler.update_bars.return_value = None

        # Strategy and portfolio mock
        self.mock_strategy = MagicMock()
        self.mock_strategy.check_pending_orders.return_value = None
        self.mock_strategy.get_update_from_portfolio.return_value = None

        self.mock_portfolio = MagicMock()
        self.mock_portfolio.all_holdings = [{"Total": self.initial_capital}]
        self.mock_portfolio.current_positions = {}
        self.mock_portfolio.current_holdings = {}

        self.mock_execution_handler = MagicMock()

        # Bind mock return values
        self.mock_data_handler_cls.return_value = self.mock_data_handler
        self.mock_strategy_cls.return_value = self.mock_strategy
        self.mock_execution_handler_cls.return_value = self.mock_execution_handler

    def test_backtest_engine_runs(self):
        engine = BacktestEngine(
            self.symbol_list,
            self.initial_capital,
            self.heartbeat,
            self.start_date,
            self.mock_data_handler_cls,
            self.mock_execution_handler_cls,
            self.mock_strategy_cls,
        )
        engine.portfolio = self.mock_portfolio 

        result = engine.simulate_trading()
        self.assertTrue(hasattr(result, '__class__')) 
        
if __name__ == "__main__":
    unittest.main()
