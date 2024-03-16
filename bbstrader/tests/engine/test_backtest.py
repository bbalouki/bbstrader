import unittest
from unittest.mock import MagicMock, PropertyMock, create_autospec
from btengine.backtest import Backtest
from btengine.data import DataHandler
from btengine.execution import ExecutionHandler
from btengine.portfolio import Portfolio
from btengine.strategy import Strategy


class TestBacktest(unittest.TestCase):

    def setUp(self):
        self.csv_dir = "mock/csv/dir"
        self.symbol_list = ["AAPL", "GOOG"]
        self.initial_capital = 100000
        self.heartbeat = 0
        self.start_date = "2020-01-01"

        self.data_handler_cls = create_autospec(DataHandler)
        self.execution_handler_cls = create_autospec(ExecutionHandler)
        self.portfolio_cls = create_autospec(Portfolio)
        self.strategy_cls = create_autospec(Strategy)

        mock_equity_curve = MagicMock()
        type(self.portfolio_cls.return_value).equity_curve = PropertyMock(
            return_value=mock_equity_curve)
        type(self.data_handler_cls.return_value).continue_backtest = PropertyMock(
            side_effect=[True, False])

        self.backtest = Backtest(
            self.csv_dir, self.symbol_list, self.initial_capital,
            self.heartbeat, self.start_date, self.data_handler_cls,
            self.execution_handler_cls, self.portfolio_cls, self.strategy_cls
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
