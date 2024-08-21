import unittest
from unittest.mock import MagicMock, PropertyMock, create_autospec
from bbstrader.btengine.backtest import Backtest
from bbstrader.btengine.data import DataHandler
from bbstrader.btengine.execution import ExecutionHandler
from bbstrader.btengine.portfolio import Portfolio
from bbstrader.btengine.strategy import Strategy
from datetime import datetime


class TestBacktest(unittest.TestCase):

    def setUp(self):
        self.symbol_list = ["AAPL", "GOOG"]
        self.initial_capital = 100000
        self.heartbeat = 0
        self.start_date = datetime(2020, 1, 1)  
        # Creating mocks for classes
        self.data_handler_cls = create_autospec(DataHandler)
        self.execution_handler_cls = create_autospec(ExecutionHandler)
        self.portfolio_cls = create_autospec(Portfolio)
        self.strategy_cls = create_autospec(Strategy)

        # Mocking attributes and behaviors needed for the test
        mock_data_handler_instance = self.data_handler_cls.return_value
        mock_data_handler_instance.symbol_list = self.symbol_list
        mock_data_handler_instance.continue_backtest = PropertyMock(
            side_effect=[True, False])

        mock_portfolio_instance = self.portfolio_cls.return_value
        mock_portfolio_instance.equity_curve = PropertyMock(
            return_value=MagicMock())

        self.backtest = Backtest(
            self.symbol_list, self.initial_capital,
            self.heartbeat, self.start_date, self.data_handler_cls,
            self.execution_handler_cls, self.strategy_cls
        )

    def test_backtest_initialization(self):
        self.assertIsInstance(self.backtest.data_handler, DataHandler)
        self.assertIsInstance(
            self.backtest.execution_handler, ExecutionHandler)
        self.assertIsInstance(self.backtest.portfolio, Portfolio)
        self.assertIsInstance(self.backtest.strategy, Strategy)

    def test_simulate_trading(self):
        try:
            self.backtest.simulate_trading()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"simulate_trading raised an exception {e}")


if __name__ == '__main__':
    unittest.main()