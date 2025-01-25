import os
import time
from datetime import datetime
from logging import Logger
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import MetaTrader5 as Mt5
import pandas as pd
from tabulate import tabulate

from bbstrader.btengine.performance import create_sharpe_ratio
from bbstrader.config import config_logger
from bbstrader.metatrader.account import INIT_MSG, check_mt5_connection
from bbstrader.metatrader.risk import RiskManagement
from bbstrader.metatrader.utils import (
    TradePosition,
    raise_mt5_error,
    trade_retcode_message,
)

__all__ = [
    "Trade",
    "create_trade_instance",
]

FILLING_TYPE = [
    Mt5.ORDER_FILLING_IOC,
    Mt5.ORDER_FILLING_RETURN,
    Mt5.ORDER_FILLING_BOC,
]


class Trade(RiskManagement):
    """
    Extends the `RiskManagement` class to include specific trading operations,
    incorporating risk management strategies directly into trade executions.
    It offers functionalities to execute trades while managing risks
    according to the inherited RiskManagement parameters and methods.

    Exemple:
        >>> import time
        >>> # Initialize the Trade class with parameters
        >>> trade = Trade(
        ...     symbol="EURUSD",              # Symbol to trade
        ...     expert_name="bbstrader",      # Name of the expert advisor
        ...     expert_id=12345,              # Unique ID for the expert advisor
        ...     version="1.0",                # Version of the expert advisor
        ...     target=5.0,                   # Daily profit target in percentage
        ...     start_time="09:00",           # Start time for trading
        ...     finishing_time="17:00",       # Time to stop opening new positions
        ...     ending_time="17:30",          # Time to close any open positions
        ...     max_risk=2.0,                 # Maximum risk allowed on the account in percentage
        ...     daily_risk=1.0,               # Daily risk allowed in percentage
        ...     max_trades=5,                 # Maximum number of trades per session
        ...     rr=2.0,                       # Risk-reward ratio
        ...     account_leverage=True,        # Use account leverage in calculations
        ...     std_stop=True,                # Use standard deviation for stop loss calculation
        ...     sl=20,                        # Stop loss in points (optional)
        ...     tp=30,                        # Take profit in points (optional)
        ...     be=10                         # Break-even in points (optional)
        ... )

        >>> # Example to open a buy position
        >>> trade.open_buy_position(mm=True, comment="Opening Buy Position")

        >>> # Example to open a sell position
        >>> trade.open_sell_position(mm=True, comment="Opening Sell Position")

        >>> # Check current open positions
        >>> opened_positions = trade.get_opened_positions
        >>> if opened_positions is not None:
        ...     print(f"Current open positions: {opened_positions}")

        >>> # Close all open positions at the end of the trading session
        >>> if trade.days_end():
        ...    trade.close_all_positions(comment="Closing all positions at day's end")

        >>> # Print trading session statistics
        >>> trade.statistics(save=True, dir="my_trading_stats")

        >>> # Sleep until the next trading session if needed (example usage)
        >>> sleep_time = trade.sleep_time()
        >>> print(f"Sleeping for {sleep_time} minutes until the next trading session.")
        >>> time.sleep(sleep_time * 60)
    """

    def __init__(
        self,
        symbol: str = "EURUSD",
        expert_name: str = "bbstrader",
        expert_id: int = 9818,
        version: str = "1.0",
        target: float = 5.0,
        start_time: str = "1:00",
        finishing_time: str = "23:00",
        ending_time: str = "23:30",
        verbose: Optional[bool] = None,
        console_log: Optional[bool] = False,
        logger: Logger | str = "bbstrader.log",
        **kwargs,
    ):
        """
        Initializes the Trade class with the specified parameters.

        Args:
            symbol (str): The `symbol` that the expert advisor will trade.
            expert_name (str): The name of the `expert advisor`.
            expert_id (int): The `unique ID` used to identify the expert advisor
                or the strategy used on the symbol.
            version (str): The `version` of the expert advisor.
            target (float): `Trading period (day, week, month) profit target` in percentage.
            start_time (str): The` hour and minutes` that the expert advisor is able to start to run.
            finishing_time (str): The time after which no new position can be opened.
            ending_time (str): The time after which any open position will be closed.
            verbose (bool | None): If set to None (default), account summary and risk managment
                parameters are printed in the terminal.
            console_log (bool): If set to True, log messages are displayed in the console.
            logger (Logger | str): The logger object to use for logging messages could be a string or a logger object.

        Inherits:
            -   max_risk
            -   max_trades
            -   rr
            -   daily_risk
            -   time_frame
            -   account_leverage
            -   std_stop
            -   pchange_sl
            -   sl
            -   tp
            -   be
        See the ``bbstrader.metatrader.risk.RiskManagement`` class for more details on these parameters.
        See `bbstrader.metatrader.account.check_mt5_connection()` for more details on how to connect to MT5 terminal.
        """
        # Call the parent class constructor first
        super().__init__(
            symbol=symbol,
            start_time=start_time,
            finishing_time=finishing_time,
            **kwargs,  # Pass kwargs to the parent constructor
        )

        # Initialize Trade-specific attributes
        self.symbol = symbol
        self.expert_name = expert_name
        self.expert_id = expert_id
        self.version = version
        self.target = target
        self.verbose = verbose
        self.start = start_time
        self.end = ending_time
        self.finishing = finishing_time
        self.console_log = console_log
        self.logger = self._get_logger(logger, console_log)
        self.tf = kwargs.get("time_frame", "D1")
        self.kwargs = kwargs

        self.start_time_hour, self.start_time_minutes = self.start.split(":")
        self.finishing_time_hour, self.finishing_time_minutes = self.finishing.split(
            ":"
        )
        self.ending_time_hour, self.ending_time_minutes = self.end.split(":")

        self.buy_positions = []
        self.sell_positions = []
        self.opened_positions = []
        self.opened_orders = []
        self.break_even_status = []
        self.break_even_points = {}
        self.trail_after_points = []
        self._retcodes = []

        self.initialize(**kwargs)
        self.select_symbol(**kwargs)
        self.prepare_symbol()

        if self.verbose:
            self.summary()
            time.sleep(1)
            print()
            self.risk_managment()
            print(f">>> Everything is OK, @{self.expert_name} is Running ...>>>\n")

    @property
    def retcodes(self) -> List[int]:
        """Return all the retcodes"""
        return self._retcodes

    def _get_logger(self, logger: str | Logger, consol_log: bool) -> Logger:
        """Get the logger object"""
        if isinstance(logger, str):
            return config_logger(logger, consol_log)
        return logger

    def initialize(self, **kwargs):
        """
        Initializes the MetaTrader 5 (MT5) terminal for trading operations.
        This method attempts to establish a connection with the MT5 terminal.
        If the initial connection attempt fails due to a timeout, it retries after a specified delay.
        Successful initialization is crucial for the execution of trading operations.

        Raises:
            MT5TerminalError: If initialization fails.
        """
        try:
            if self.verbose:
                print("\nInitializing the basics.")
            check_mt5_connection(**kwargs)
            if self.verbose:
                print(
                    f"You are running the @{self.expert_name} Expert advisor,"
                    f" Version @{self.version}, on {self.symbol}."
                )
        except Exception as e:
            self.logger.error(f"During initialization: {e}")

    def select_symbol(self, **kwargs):
        """
        Selects the trading symbol in the MetaTrader 5 (MT5) terminal.
        This method ensures that the specified trading
        symbol is selected and visible in the MT5 terminal,
        allowing subsequent trading operations such as opening and
        closing positions on this symbol.

        Raises:
            MT5TerminalError: If symbole selection fails.
        """
        try:
            check_mt5_connection(**kwargs)
            if not Mt5.symbol_select(self.symbol, True):
                raise_mt5_error(message=INIT_MSG)
        except Exception as e:
            self.logger.error(f"Selecting symbol '{self.symbol}': {e}")

    def prepare_symbol(self):
        """
        Prepares the selected symbol for trading.
        This method checks if the symbol is available and visible in the
        MT5 terminal. If the symbol is not visible, it attempts to select the symbol again.
        This step ensures that trading operations can be performed on the selected symbol without issues.

        Raises:
            MT5TerminalError: If the symbol cannot be made visible for trading operations.
        """
        try:
            symbol_info = self.get_symbol_info(self.symbol)
            if symbol_info is None:
                raise_mt5_error(message=INIT_MSG)

            if not symbol_info.visible:
                raise_mt5_error(message=INIT_MSG)
            if self.verbose:
                print("Initialization successfully completed.")
        except Exception as e:
            self.logger.error(f"Preparing symbol '{self.symbol}': {e}")

    def summary(self):
        """Show a brief description about the trading program"""
        summary_data = [
            ["Expert Advisor Name", f"@{self.expert_name}"],
            ["Expert Advisor Version", f"@{self.version}"],
            ["Expert | Strategy ID", self.expert_id],
            ["Trading Symbol", self.symbol],
            ["Trading Time Frame", self.tf],
            ["Start Trading Time", f"{self.start_time_hour}:{self.start_time_minutes}"],
            [
                "Finishing Trading Time",
                f"{self.finishing_time_hour}:{self.finishing_time_minutes}",
            ],
            [
                "Closing Position After",
                f"{self.ending_time_hour}:{self.ending_time_minutes}",
            ],
        ]
        # Custom table format
        summary_table = tabulate(
            summary_data, headers=["Summary", "Values"], tablefmt="outline"
        )

        # Print the table
        print("\n[======= Trade Account Summary =======]")
        print(summary_table)

    def risk_managment(self):
        """Show the risk management parameters"""

        loss = self.currency_risk()["trade_loss"]
        profit = self.currency_risk()["trade_profit"]
        ok = "OK" if self.is_risk_ok() else "Not OK"
        account_info = self.get_account_info()
        _profit = round(self.get_stats()[1]["total_profit"], 2)
        currency = account_info.currency
        rates = self.get_currency_rates(self.symbol)
        marging_currency = rates["mc"]
        account_data = [
            ["Account Name", account_info.name],
            ["Account Number", account_info.login],
            ["Account Server", account_info.server],
            ["Account Balance", f"{account_info.balance} {currency}"],
            ["Account Profit", f"{_profit} {currency}"],
            ["Account Equity", f"{account_info.equity} {currency}"],
            ["Account Leverage", self.get_leverage(True)],
            ["Account Margin", f"{round(account_info.margin, 2)} {currency}"],
            ["Account Free Margin", f"{account_info.margin_free} {currency}"],
            ["Maximum Drawdown", f"{self.max_risk}%"],
            ["Risk Allowed", f"{round((self.max_risk - self.risk_level()), 2)}%"],
            ["Volume", f"{self.volume()} {marging_currency}"],
            ["Risk Per trade", f"{-self.get_currency_risk()} {currency}"],
            ["Profit Expected Per trade", f"{self.expected_profit()} {currency}"],
            ["Lot Size", f"{self.get_lot()} Lots"],
            ["Stop Loss", f"{self.get_stop_loss()} Points"],
            ["Loss Value Per Tick", f"{round(loss, 5)} {currency}"],
            ["Take Profit", f"{self.get_take_profit()} Points"],
            ["Profit Value Per Tick", f"{round(profit, 5)} {currency}"],
            ["Break Even", f"{self.get_break_even()} Points"],
            ["Deviation", f"{self.get_deviation()} Points"],
            ["Trading Time Interval", f"{self.get_minutes()} Minutes"],
            ["Risk Level", ok],
            ["Maximum Trades", self.max_trade()],
        ]
        # Custom table format
        print("\n[======= Account Risk Management Overview =======]")
        table = tabulate(
            account_data, headers=["Risk Metrics", "Values"], tablefmt="outline"
        )

        # Print the table
        print(table)

    def statistics(self, save=True, dir=None):
        """
        Print some statistics for the trading session and save to CSV if specified.

        Args:
            save (bool, optional): Whether to save the statistics to a CSV file.
            dir (str, optional): The directory to save the CSV file.
        """
        stats, additional_stats = self.get_stats()

        profit = round(stats["profit"], 2)
        win_rate = stats["win_rate"]
        total_fees = round(stats["total_fees"], 3)
        average_fee = round(stats["average_fee"], 3)
        currency = self.get_account_info().currency
        net_profit = round((profit + total_fees), 2)
        trade_risk = round(self.get_currency_risk() * -1, 2)

        # Formatting the statistics output
        session_data = [
            ["Total Trades", stats["deals"]],
            ["Winning Trades", stats["win_trades"]],
            ["Losing Trades", stats["loss_trades"]],
            ["Session Profit", f"{profit} {currency}"],
            ["Total Fees", f"{total_fees} {currency}"],
            ["Average Fees", f"{average_fee} {currency}"],
            ["Net Profit", f"{net_profit} {currency}"],
            ["Risk per Trade", f"{trade_risk} {currency}"],
            ["Expected Profit per Trade", f"{self.expected_profit()} {currency}"],
            ["Risk Reward Ratio", self.rr],
            ["Win Rate", f"{win_rate}%"],
            ["Sharpe Ratio", self.sharpe()],
            ["Trade Profitability", additional_stats["profitability"]],
        ]
        session_table = tabulate(
            session_data, headers=["Statistics", "Values"], tablefmt="outline"
        )

        # Print the formatted statistics
        if self.verbose:
            print("\n[======= Trading Session Statistics =======]")
            print(session_table)

        # Save to CSV if specified
        if save:
            today_date = datetime.now().strftime("%Y%m%d%H%M%S")
            # Create a dictionary with the statistics
            statistics_dict = {item[0]: item[1] for item in session_data}
            stats_df = pd.DataFrame(statistics_dict, index=[0])
            # Create the directory if it doesn't exist
            dir = dir or ".sessions"
            os.makedirs(dir, exist_ok=True)
            if "." in self.symbol:
                symbol = self.symbol.split(".")[0]
            else:
                symbol = self.symbol

            filename = f"{symbol}_{today_date}@{self.expert_id}.csv"
            filepath = os.path.join(dir, filename)
            stats_df.to_csv(filepath, index=False)
            self.logger.info(f"Session statistics saved to {filepath}")

    Buys = Literal["BMKT", "BLMT", "BSTP", "BSTPLMT"]

    def open_buy_position(
        self,
        action: Buys = "BMKT",
        price: Optional[float] = None,
        stoplimit: Optional[float] = None,
        mm: bool = True,
        id: Optional[int] = None,
        comment: Optional[str] = None,
    ):
        """
        Open a Buy positin

        Args:
            action (str): `BMKT` for Market orders or `BLMT`,
                `BSTP`,`BSTPLMT` for pending orders
            price (float): The price at which to open an order
            stoplimit (float): A price a pending Limit order is set at when the price reaches the 'price' value (this condition is mandatory).
                The pending order is not passed to the trading system until that moment
            id (int): The strategy id or expert Id
            mm (bool): Weither to put stop loss and tp or not
            comment (str): The comment for the opening position
        """
        Id = id if id is not None else self.expert_id
        point = self.get_symbol_info(self.symbol).point
        if action != "BMKT":
            if price is not None:
                _price = price
            else:
                raise ValueError("You need to set a price for pending orders")
        else:
            _price = self.get_tick_info(self.symbol).ask

        lot = self.get_lot()
        stop_loss = self.get_stop_loss()
        take_profit = self.get_take_profit()
        deviation = self.get_deviation()
        request = {
            "action": Mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(lot),
            "type": Mt5.ORDER_TYPE_BUY,
            "price": _price,
            "deviation": deviation,
            "magic": Id,
            "comment": f"@{self.expert_name}" if comment is None else comment,
            "type_time": Mt5.ORDER_TIME_GTC,
            "type_filling": Mt5.ORDER_FILLING_FOK,
        }
        mm_price = _price
        if action != "BMKT":
            request["action"] = Mt5.TRADE_ACTION_PENDING
            request["type"] = self._order_type()[action][0]
        if action == "BSTPLMT":
            if stoplimit is None:
                raise ValueError("You need to set a stoplimit price for BSTPLMT orders")
            if stoplimit > _price:
                raise ValueError(
                    "Stoplimit price must be less than the price and greater than the current price"
                )
            request["stoplimit"] = stoplimit
            mm_price = stoplimit
        if mm:
            request["sl"] = mm_price - stop_loss * point
            request["tp"] = mm_price + take_profit * point
        self.break_even(mm=mm, id=Id)
        if self.check(comment):
            self.request_result(_price, request, action)

    def _order_type(self):
        type = {
            "BMKT": (Mt5.ORDER_TYPE_BUY, "BUY"),
            "SMKT": (Mt5.ORDER_TYPE_BUY, "SELL"),
            "BLMT": (Mt5.ORDER_TYPE_BUY_LIMIT, "BUY_LIMIT"),
            "SLMT": (Mt5.ORDER_TYPE_SELL_LIMIT, "SELL_LIMIT"),
            "BSTP": (Mt5.ORDER_TYPE_BUY_STOP, "BUY_STOP"),
            "SSTP": (Mt5.ORDER_TYPE_SELL_STOP, "SELL_STOP"),
            "BSTPLMT": (Mt5.ORDER_TYPE_BUY_STOP_LIMIT, "BUY_STOP_LIMIT"),
            "SSTPLMT": (Mt5.ORDER_TYPE_SELL_STOP_LIMIT, "SELL_STOP_LIMIT"),
        }
        return type

    Sells = Literal["SMKT", "SLMT", "SSTP", "SSTPLMT"]

    def open_sell_position(
        self,
        action: Sells = "SMKT",
        price: Optional[float] = None,
        stoplimit: Optional[float] = None,
        mm: bool = True,
        id: Optional[int] = None,
        comment: Optional[str] = None,
    ):
        """
        Open a sell positin

        Args:
            action (str): `SMKT` for Market orders
                or ``SLMT``, ``SSTP``,``SSTPLMT`` for pending orders
            price (float): The price at which to open an order
            stoplimit (float): A price a pending Limit order is set at when the price reaches the 'price' value (this condition is mandatory).
                The pending order is not passed to the trading system until that moment
            id (int): The strategy id or expert Id
            mm (bool): Weither to put stop loss and tp or not
            comment (str): The comment for the closing position
        """
        Id = id if id is not None else self.expert_id
        point = self.get_symbol_info(self.symbol).point
        if action != "SMKT":
            if price is not None:
                _price = price
            else:
                raise ValueError("You need to set a price for pending orders")
        else:
            _price = self.get_tick_info(self.symbol).bid

        lot = self.get_lot()
        stop_loss = self.get_stop_loss()
        take_profit = self.get_take_profit()
        deviation = self.get_deviation()
        request = {
            "action": Mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(lot),
            "type": Mt5.ORDER_TYPE_SELL,
            "price": _price,
            "deviation": deviation,
            "magic": Id,
            "comment": f"@{self.expert_name}" if comment is None else comment,
            "type_time": Mt5.ORDER_TIME_GTC,
            "type_filling": Mt5.ORDER_FILLING_FOK,
        }
        mm_price = _price
        if action != "SMKT":
            request["action"] = Mt5.TRADE_ACTION_PENDING
            request["type"] = self._order_type()[action][0]
        if action == "SSTPLMT":
            if stoplimit is None:
                raise ValueError("You need to set a stoplimit price for SSTPLMT orders")
            if stoplimit < _price:
                raise ValueError(
                    "Stoplimit price must be greater than the price and less than the current price"
                )
            request["stoplimit"] = stoplimit
            mm_price = stoplimit
        if mm:
            request["sl"] = mm_price + stop_loss * point
            request["tp"] = mm_price - take_profit * point
        self.break_even(mm=mm, id=Id)
        if self.check(comment):
            self.request_result(_price, request, action)

    def check(self, comment):
        """
        Verify if all conditions for taking a position are valide,
        These conditions are based on the Maximum risk ,daily risk,
        the starting, the finishing, and ending trading time.

        Args:
            comment (str): The comment for the closing position
        """
        if self.days_end():
            return False
        elif not self.trading_time():
            self.logger.info(f"Not Trading time, SYMBOL={self.symbol}")
            return False
        elif not self.is_risk_ok():
            self.logger.error(f"Account Risk not allowed, SYMBOL={self.symbol}")
            self._check(comment)
            return False
        elif self.profit_target():
            self._check(f"Profit target Reached !!! SYMBOL={self.symbol}")
        return True

    def _check(self, txt: str = ""):
        if (
            self.positive_profit(id=self.expert_id)
            or self.get_current_positions() is None
        ):
            self.close_positions(position_type="all")
            self.logger.info(txt)
            time.sleep(5)
            self.statistics(save=True)

    def request_result(self, price: float, request: Dict[str, Any], type: Buys | Sells):
        """
        Check if a trading order has been sent correctly

        Args:
            price (float): Price for opening the position
            request (Dict[str, Any]): A trade request to sent to Mt5.order_sent()
            all detail in request can be found here https://www.mql5.com/en/docs/python_metatrader5/mt5ordersend_py

            type (str): The type of the order `(BMKT, SMKT, BLMT, SLMT, BSTP, SSTP, BSTPLMT, SSTPLMT)`
        """
        # Send a trading request
        # Check the execution result
        pos = self._order_type()[type][1]
        addtionnal = f", SYMBOL={self.symbol}"
        try:
            self.check_order(request)
            result = self.send_order(request)
        except Exception as e:
            print(f"{self.current_datetime()} -", end=" ")
            trade_retcode_message(
                result.retcode, display=True, add_msg=f"{e}{addtionnal}"
            )
        if result.retcode != Mt5.TRADE_RETCODE_DONE:
            if result.retcode == Mt5.TRADE_RETCODE_INVALID_FILL:  # 10030
                for fill in FILLING_TYPE:
                    request["type_filling"] = fill
                    result = self.send_order(request)
                    if result.retcode == Mt5.TRADE_RETCODE_DONE:
                        break
            elif result.retcode == Mt5.TRADE_RETCODE_INVALID_VOLUME: #10014
                new_volume = int(request["volume"])
                if new_volume >= 1:
                    request["volume"] = new_volume
                    result = self.send_order(request)
            elif result.retcode not in self._retcodes:
                self._retcodes.append(result.retcode)
                msg = trade_retcode_message(result.retcode)
                self.logger.error(
                    f"Trade Order Request, RETCODE={result.retcode}: {msg}{addtionnal}"
                )
            elif result.retcode in [
                Mt5.TRADE_RETCODE_CONNECTION,
                Mt5.TRADE_RETCODE_TIMEOUT,
            ]:
                tries = 0
                while result.retcode != Mt5.TRADE_RETCODE_DONE and tries < 5:
                    time.sleep(1)
                    try:
                        self.check_order(request)
                        result = self.send_order(request)
                    except Exception as e:
                        print(f"{self.current_datetime()} -", end=" ")
                        trade_retcode_message(
                            result.retcode, display=True, add_msg=f"{e}{addtionnal}"
                        )
                    if result.retcode == Mt5.TRADE_RETCODE_DONE:
                        break
                    tries += 1
        # Print the result
        if result.retcode == Mt5.TRADE_RETCODE_DONE:
            msg = trade_retcode_message(result.retcode)
            self.logger.info(f"Trade Order {msg}{addtionnal}")
            if type != "BMKT" or type != "SMKT":
                self.opened_orders.append(result.order)
            long_msg = (
                f"1. {pos} Order #{result.order} Sent, Symbol: {self.symbol}, Price: @{price}, "
                f"Lot(s): {result.volume}, Sl: {self.get_stop_loss()}, "
                f"Tp: {self.get_take_profit()}"
            )
            self.logger.info(long_msg)
            time.sleep(0.1)
            if type == "BMKT" or type == "SMKT":
                self.opened_positions.append(result.order)
                positions = self.get_positions(symbol=self.symbol)
                if positions is not None:
                    for position in positions:
                        if position.ticket == result.order:
                            if position.type == 0:
                                order_type = "BUY"
                                self.buy_positions.append(position.ticket)
                            else:
                                order_type = "SELL"
                                self.sell_positions.append(position.ticket)
                            profit = round(self.get_account_info().profit, 5)
                            order_info = (
                                f"2. {order_type} Position Opened, Symbol: {self.symbol}, Price: @{round(position.price_open,5)}, "
                                f"Sl: @{position.sl} Tp: @{position.tp}"
                            )
                            self.logger.info(order_info)
                            pos_info = (
                                f"3. [OPEN POSITIONS ON {self.symbol} = {len(positions)}, ACCOUNT OPEN PnL = {profit} "
                                f"{self.get_account_info().currency}]\n"
                            )
                            self.logger.info(pos_info)
        else:
            msg = trade_retcode_message(result.retcode)
            self.logger.error(
                f"Unable to Open Position, RETCODE={result.retcode}: {msg}{addtionnal}"
            )

    def open_position(
        self,
        action: Buys | Sells,
        price: Optional[float] = None,
        stoplimit: Optional[float] = None,
        id: Optional[int] = None,
        mm: bool = True,
        comment: Optional[str] = None,
    ):
        """
        Open a buy or sell position.

        Args:
            action (str): (`'BMKT'`, `'SMKT'`) for Market orders
                or (`'BLMT', 'SLMT', 'BSTP', 'SSTP', 'BSTPLMT', 'SSTPLMT'`) for pending orders
            price (float): The price at which to open an order
            stoplimit (float): A price a pending Limit order is set at when the price reaches the 'price' value (this condition is mandatory).
                The pending order is not passed to the trading system until that moment
            id (int): The strategy id or expert Id
            mm (bool): Weither to put stop loss and tp or not
            comment (str): The comment for the closing position
        """
        BUYS = ["BMKT", "BLMT", "BSTP", "BSTPLMT"]
        SELLS = ["SMKT", "SLMT", "SSTP", "SSTPLMT"]
        if action in BUYS:
            self.open_buy_position(
                action=action,
                price=price,
                stoplimit=stoplimit,
                id=id,
                mm=mm,
                comment=comment,
            )
        elif action in SELLS:
            self.open_sell_position(
                action=action,
                price=price,
                stoplimit=stoplimit,
                id=id,
                mm=mm,
                comment=comment,
            )
        else:
            raise ValueError(
                f"Invalid action type '{action}', must be {', '.join(BUYS + SELLS)}"
            )

    @property
    def orders(self):
        """Return all opened order's tickets"""
        if len(self.opened_orders) != 0:
            return self.opened_orders
        return None

    @property
    def positions(self):
        """Return all opened position's tickets"""
        if len(self.opened_positions) != 0:
            return self.opened_positions
        return None

    @property
    def buypos(self):
        """Return all buy  opened position's tickets"""
        if len(self.buy_positions) != 0:
            return self.buy_positions
        return None

    @property
    def sellpos(self):
        """Return all sell  opened position's tickets"""
        if len(self.sell_positions) != 0:
            return self.sell_positions
        return None

    @property
    def bepos(self):
        """Return All positon's tickets
        for which a break even has been set"""
        if len(self.break_even_status) != 0:
            return self.break_even_status
        return None

    def get_filtered_tickets(
        self, id: Optional[int] = None, filter_type: Optional[str] = None, th=None
    ) -> List[int] | None:
        """
        Get tickets for positions or orders based on filters.

        Args:
            id (int): The strategy id or expert Id
            filter_type (str): Filter type to apply on the tickets,
                - `orders` are current open orders
                - `buy_stops` are current buy stop orders
                - `sell_stops` are current sell stop orders
                - `buy_limits` are current buy limit orders
                - `sell_limits` are current sell limit orders
                - `buy_stop_limits` are current buy stop limit orders
                - `sell_stop_limits` are current sell stop limit orders
                - `positions` are all current open positions
                - `buys` and `sells` are current buy or sell open positions
                - `profitables` are current open position that have a profit greater than a threshold
                - `losings` are current open position that have a negative profit
            th (bool): the minimum treshold for winning position
                (only relevant when filter_type is 'profitables')

        Returns:
            List[int] | None: A list of filtered tickets
                or None if no tickets match the criteria.
        """
        Id = id if id is not None else self.expert_id
        POSITIONS = ["positions", "buys", "sells", "profitables", "losings"]

        if filter_type not in POSITIONS:
            items = self.get_orders(symbol=self.symbol)
        else:
            items = self.get_positions(symbol=self.symbol)

        filtered_tickets = []

        if items is not None:
            for item in items:
                if item.magic == Id:
                    if filter_type == "buys" and item.type != 0:
                        continue
                    if filter_type == "sells" and item.type != 1:
                        continue
                    if filter_type == "losings" and item.profit > 0:
                        continue
                    if filter_type == "profitables" and not self.win_trade(item, th=th):
                        continue
                    if (
                        filter_type == "buy_stops"
                        and item.type != self._order_type()["BSTP"][0]
                    ):
                        continue
                    if (
                        filter_type == "sell_stops"
                        and item.type != self._order_type()["SSTP"][0]
                    ):
                        continue
                    if (
                        filter_type == "buy_limits"
                        and item.type != self._order_type()["BLMT"][0]
                    ):
                        continue
                    if (
                        filter_type == "sell_limits"
                        and item.type != self._order_type()["SLMT"][0]
                    ):
                        continue
                    if (
                        filter_type == "buy_stop_limits"
                        and item.type != self._order_type()["BSTPLMT"][0]
                    ):
                        continue
                    if (
                        filter_type == "sell_stop_limits"
                        and item.type != self._order_type()["SSTPLMT"][0]
                    ):
                        continue
                    filtered_tickets.append(item.ticket)
            return filtered_tickets if filtered_tickets else None
        return None

    def get_current_orders(self, id: Optional[int] = None) -> List[int] | None:
        return self.get_filtered_tickets(id=id, filter_type="orders")

    def get_current_buy_stops(self, id: Optional[int] = None) -> List[int] | None:
        return self.get_filtered_tickets(id=id, filter_type="buy_stops")

    def get_current_sell_stops(self, id: Optional[int] = None) -> List[int] | None:
        return self.get_filtered_tickets(id=id, filter_type="sell_stops")

    def get_current_buy_limits(self, id: Optional[int] = None) -> List[int] | None:
        return self.get_filtered_tickets(id=id, filter_type="buy_limits")

    def get_current_sell_limits(self, id: Optional[int] = None) -> List[int] | None:
        return self.get_filtered_tickets(id=id, filter_type="sell_limits")

    def get_current_buy_stop_limits(self, id: Optional[int] = None) -> List[int] | None:
        return self.get_filtered_tickets(id=id, filter_type="buy_stop_limits")

    def get_current_sell_stop_limits(
        self, id: Optional[int] = None
    ) -> List[int] | None:
        return self.get_filtered_tickets(id=id, filter_type="sell_stop_limits")

    def get_current_positions(self, id: Optional[int] = None) -> List[int] | None:
        return self.get_filtered_tickets(id=id, filter_type="positions")

    def get_current_profitables(
        self, id: Optional[int] = None, th=None
    ) -> List[int] | None:
        return self.get_filtered_tickets(id=id, filter_type="profitables", th=th)

    def get_current_losings(self, id: Optional[int] = None) -> List[int] | None:
        return self.get_filtered_tickets(id=id, filter_type="losings")

    def get_current_buys(self, id: Optional[int] = None) -> List[int] | None:
        return self.get_filtered_tickets(id=id, filter_type="buys")

    def get_current_sells(self, id: Optional[int] = None) -> List[int] | None:
        return self.get_filtered_tickets(id=id, filter_type="sells")

    def positive_profit(
        self, th: Optional[float] = None, id: Optional[int] = None, account: bool = True
    ) -> bool:
        """
        Check is the total profit on current open positions
        Is greater than a minimum profit express as percentage
        of the profit target.

        Args:
            th (float): The minimum profit target on current positions
            id (int): The strategy id or expert Id
            account (bool): Weither to check positions on the account or on the symbol
        """
        if account and id is None:
            # All open positions no matter the symbol or strategy or expert
            positions = self.get_positions()
        elif account and id is not None:
            # All open positions for a specific strategy or expert no matter the symbol
            positions = self.get_positions()
            if positions is not None:
                positions = [position for position in positions if position.magic == id]
        elif not account and id is None:
            # All open positions for the current symbol no matter the strategy or expert
            positions = self.get_positions(symbol=self.symbol)
        elif not account and id is not None:
            # All open positions for the current symbol and a specific strategy or expert
            positions = self.get_positions(symbol=self.symbol)
            if positions is not None:
                positions = [position for position in positions if position.magic == id]

        if positions is not None:
            profit = 0.0
            balance = self.get_account_info().balance
            target = round((balance * self.target) / 100, 2)
            for position in positions:
                profit += position.profit
            fees = self.get_stats()[0]["average_fee"] * len(positions)
            current_profit = profit + fees
            th_profit = (target * th) / 100 if th is not None else (target * 0.01)
            return current_profit >= th_profit
        return False

    def break_even(
        self,
        mm=True,
        id: Optional[int] = None,
        trail: Optional[bool] = True,
        stop_trail: Optional[int] = None,
        trail_after_points: Optional[int] = None,
        be_plus_points: Optional[int] = None,
    ):
        """
        This function checks if it's time to set the break-even level for a trading position.
        If it is, it sets the break-even level. If the break-even level has already been set,
        it checks if the price has moved in a favorable direction.
        If it has, and the trail parameter is set to True, it updates
        the break-even level based on the trail_after_points and stop_trail parameters.

        Args:
            id (int): The strategy ID or expert ID.
            mm (bool): Whether to manage the position or not.
            trail (bool): Whether to trail the stop loss or not.
            stop_trail (int): Number of points to trail the stop loss by.
                It represent the distance from the current price to the stop loss.
            trail_after_points (int): Number of points in profit
                from where the strategy will start to trail the stop loss.
            be_plus_points (int): Number of points to add to the break-even level.
                Represents the minimum profit to secure.
        """
        time.sleep(0.1)
        if not mm:
            return
        Id = id if id is not None else self.expert_id
        positions = self.get_positions(symbol=self.symbol)
        be = self.get_break_even()
        if trail_after_points is not None:
            assert trail_after_points > be, (
                "trail_after_points must be greater than break even" " or set to None"
            )
        if positions is not None:
            for position in positions:
                if position.magic == Id:
                    size = self.get_symbol_info(self.symbol).trade_tick_size
                    value = self.get_symbol_info(self.symbol).trade_tick_value
                    point = self.get_symbol_info(self.symbol).point
                    digits = self.get_symbol_info(self.symbol).digits
                    points = position.profit * (size / value / position.volume)
                    break_even = float(points / point) >= be
                    if break_even:
                        # Check if break-even has already been set for this position
                        if position.ticket not in self.break_even_status:
                            price = None
                            if be_plus_points is not None:
                                price = position.price_open + (be_plus_points * point)
                            self.set_break_even(position, be, price=price)
                            self.break_even_status.append(position.ticket)
                            self.break_even_points[position.ticket] = be
                        else:
                            # Skip this if the trail is not set to True
                            if not trail:
                                continue
                            # Check if the price has moved favorably
                            new_be = (
                                round(be * 0.10)
                                if be_plus_points is None
                                else be_plus_points
                            )
                            if trail_after_points is not None:
                                if position.ticket not in self.trail_after_points:
                                    # This ensures that the position rich the minimum points required
                                    # before the trail can be set
                                    new_be = trail_after_points - be
                                    self.trail_after_points.append(position.ticket)
                            new_be_points = (
                                self.break_even_points[position.ticket] + new_be
                            )
                            favorable_move = float(points / point) >= new_be_points
                            if favorable_move:
                                # This allows the position to go to take profit in case of a swing trade
                                # If is a scalping position, we can set the stop_trail close to the current price.
                                trail_points = (
                                    round(be * 0.50)
                                    if stop_trail is None
                                    else stop_trail
                                )
                                # Calculate the new break-even level and price
                                if position.type == 0:
                                    # This level validate the favorable move of the price
                                    new_level = round(
                                        position.price_open + (new_be_points * point),
                                        digits,
                                    )
                                    # This price is set away from the current price by the trail_points
                                    new_price = round(
                                        position.price_current - (trail_points * point),
                                        digits,
                                    )
                                    if new_price < position.sl:
                                        new_price = position.sl
                                elif position.type == 1:
                                    new_level = round(
                                        position.price_open - (new_be_points * point),
                                        digits,
                                    )
                                    new_price = round(
                                        position.price_current + (trail_points * point),
                                        digits,
                                    )
                                    if new_price > position.sl:
                                        new_price = position.sl
                                self.set_break_even(
                                    position, be, price=new_price, level=new_level
                                )

    def set_break_even(
        self,
        position: TradePosition,
        be: int,
        price: Optional[float] = None,
        level: Optional[float] = None,
    ):
        """
        Sets the break-even level for a given trading position.

        Args:
            position (TradePosition): The trading position for which the break-even is to be set. This is the value return by `mt5.positions_get()`.
            be (int): The break-even level in points.
            level (float): The break-even level in price, if set to None , it will be calated automaticaly.
            price (float): The break-even price, if set to None , it will be calated automaticaly.
        """
        point = self.get_symbol_info(self.symbol).point
        digits = self.get_symbol_info(self.symbol).digits
        spread = self.get_symbol_info(self.symbol).spread
        fees = self.get_stats()[0]["average_fee"] * -1
        risk = self.currency_risk()["trade_profit"]
        try:
            fees_points = round((fees / risk), 3)
        except ZeroDivisionError:
            fees_points = 0
        # If Buy
        if position.type == 0 and position.price_current > position.price_open:
            # Calculate the break-even level and price
            break_even_level = position.price_open + (be * point)
            break_even_price = position.price_open + ((fees_points + spread) * point)
            # Check if the price specified is greater or lower than the calculated price
            _price = (
                break_even_price if price is None or price < break_even_price else price
            )
            _level = break_even_level if level is None else level

            if self.get_tick_info(self.symbol).ask > _level:
                # Set the stop loss to break even
                request = {
                    "action": Mt5.TRADE_ACTION_SLTP,
                    "type": Mt5.ORDER_TYPE_SELL_STOP,
                    "position": position.ticket,
                    "sl": round(_price, digits),
                    "tp": position.tp,
                }
                self.break_even_request(position.ticket, round(_price, digits), request)
        # If Sell
        elif position.type == 1 and position.price_current < position.price_open:
            break_even_level = position.price_open - (be * point)
            break_even_price = position.price_open - ((fees_points + spread) * point)
            _price = (
                break_even_price if price is None or price > break_even_price else price
            )
            _level = break_even_level if level is None else level

            if self.get_tick_info(self.symbol).bid < _level:
                # Set the stop loss to break even
                request = {
                    "action": Mt5.TRADE_ACTION_SLTP,
                    "type": Mt5.ORDER_TYPE_BUY_STOP,
                    "position": position.ticket,
                    "sl": round(_price, digits),
                    "tp": position.tp,
                }
                self.break_even_request(position.ticket, round(_price, digits), request)

    def break_even_request(self, tiket, price, request):
        """
        Send a request to set the stop loss to break even for a given trading position.

        Args:
            tiket (int): The ticket number of the trading position.
            price (float): The price at which the stop loss is to be set.
            request (dict): The request to set the stop loss to break even.
        """
        addtionnal = f", SYMBOL={self.symbol}"
        time.sleep(0.1)
        try:
            self.check_order(request)
            result = self.send_order(request)
        except Exception as e:
            print(f"{self.current_datetime()} -", end=" ")
            trade_retcode_message(
                result.retcode, display=True, add_msg=f"{e}{addtionnal}"
            )
        if result.retcode != Mt5.TRADE_RETCODE_DONE:
            msg = trade_retcode_message(result.retcode)
            if result.retcode != Mt5.TRADE_RETCODE_NO_CHANGES:
                self.logger.error(
                    f"Break-Even Order Request, Position: #{tiket}, RETCODE={result.retcode}: {msg}{addtionnal}"
                )
            tries = 0
            while result.retcode != Mt5.TRADE_RETCODE_DONE and tries < 10:
                if result.retcode == Mt5.TRADE_RETCODE_NO_CHANGES:
                    break
                else:
                    time.sleep(1)
                    try:
                        self.check_order(request)
                        result = self.send_order(request)
                    except Exception as e:
                        print(f"{self.current_datetime()} -", end=" ")
                        trade_retcode_message(
                            result.retcode, display=True, add_msg=f"{e}{addtionnal}"
                        )
                    if result.retcode == Mt5.TRADE_RETCODE_DONE:
                        break
                tries += 1
        if result.retcode == Mt5.TRADE_RETCODE_DONE:
            msg = trade_retcode_message(result.retcode)
            self.logger.info(f"Break-Even Order {msg}{addtionnal}")
            info = f"Stop loss set to Break-even, Position: #{tiket}, Symbol: {self.symbol}, Price: @{price}"
            self.logger.info(info)
            self.break_even_status.append(tiket)

    def win_trade(self, position: TradePosition, th: Optional[int] = None) -> bool:
        """
        Check if a positon is wining or looing
        wen it is closed before be level , tp or sl.

        Args:
            position (TradePosition): The trading position to check.
            th (int): The minimum profit for a position in point
        """
        size = self.get_symbol_info(self.symbol).trade_tick_size
        value = self.get_symbol_info(self.symbol).trade_tick_value
        points = position.profit * (size / value / position.volume)

        point = self.get_symbol_info(self.symbol).point
        fees = self.get_stats()[0]["average_fee"] * -1
        risk = self.currency_risk()["trade_profit"]
        try:
            min_be = round((fees / risk)) + 2
        except ZeroDivisionError:
            min_be = self.symbol_info(self.symbol).spread
        be = self.get_break_even()
        if th is not None:
            win_be = th
        else:
            win_be = max(min_be, round((0.1 * be)))
        win_trade = float(points / point) >= win_be
        # Check if the positon is in profit
        if win_trade:
            # Check if break-even has already been set for this position
            if position.ticket not in self.break_even_status:
                return True
        return False

    def profit_target(self):
        fee = 0.0
        swap = 0.0
        commission = 0.0
        profit = 0.0
        balance = self.get_account_info().balance
        target = round((balance * self.target) / 100, 2)
        if len(self.opened_positions) != 0:
            for position in self.opened_positions:
                time.sleep(0.1)
                # This return two TradeDeal Object,
                # The first one is the opening order
                # The second is the closing order
                history = self.get_trades_history(position=position, to_df=False)
                if history is not None and len(history) == 2:
                    profit += history[1].profit
                    commission += history[0].commission
                    swap += history[0].swap
                    fee += history[0].fee
            current_profit = profit + commission + fee + swap
            if current_profit >= target:
                return True
        return False

    def close_request(self, request: dict, type: str):
        """
        Close a trading order or position

        Args:
            request (dict): The request to close a trading order or position
            type (str): Type of the request ('order', 'position')
        """
        ticket = request[type]
        addtionnal = f", SYMBOL={self.symbol}"
        try:
            self.check_order(request)
            result = self.send_order(request)
        except Exception as e:
            print(f"{self.current_datetime()} -", end=" ")
            trade_retcode_message(
                result.retcode, display=True, add_msg=f"{e}{addtionnal}"
            )
        if result.retcode != Mt5.TRADE_RETCODE_DONE:
            if result.retcode == Mt5.TRADE_RETCODE_INVALID_FILL:  # 10030
                for fill in FILLING_TYPE:
                    request["type_filling"] = fill
                    result = self.send_order(request)
                    if result.retcode == Mt5.TRADE_RETCODE_DONE:
                        break
            elif result.retcode not in self._retcodes:
                self._retcodes.append(result.retcode)
                msg = trade_retcode_message(result.retcode)
                self.logger.error(
                    f"Closing Order Request, {type.capitalize()}: #{ticket}, RETCODE={result.retcode}: {msg}{addtionnal}"
                )
            else:
                tries = 0
                while result.retcode != Mt5.TRADE_RETCODE_DONE and tries < 5:
                    time.sleep(1)
                    try:
                        self.check_order(request)
                        result = self.send_order(request)
                    except Exception as e:
                        print(f"{self.current_datetime()} -", end=" ")
                        trade_retcode_message(
                            result.retcode, display=True, add_msg=f"{e}{addtionnal}"
                        )
                    if result.retcode == Mt5.TRADE_RETCODE_DONE:
                        break
                    tries += 1
        if result.retcode == Mt5.TRADE_RETCODE_DONE:
            msg = trade_retcode_message(result.retcode)
            self.logger.info(f"Closing Order {msg}{addtionnal}")
            info = f"{type.capitalize()} #{ticket} closed, Symbol: {self.symbol}, Price: @{request.get('price', 0.0)}"
            self.logger.info(info)
            return True
        else:
            return False

    def close_order(
        self, ticket: int, id: Optional[int] = None, comment: Optional[str] = None
    ):
        """
        Close an open order by it ticket

        Args:
            ticket (int): Order ticket to close (e.g TradeOrder.ticket)
            id (int): The unique ID of the Expert or Strategy
            comment (str): Comment for the closing position

        Returns:
        -   True if order closed, False otherwise
        """
        request = {
            "action": Mt5.TRADE_ACTION_REMOVE,
            "symbol": self.symbol,
            "order": ticket,
            "magic": id if id is not None else self.expert_id,
            "comment": f"@{self.expert_name}" if comment is None else comment,
        }
        return self.close_request(request, type="order")

    def close_position(
        self,
        ticket: int,
        id: Optional[int] = None,
        pct: Optional[float] = 1.0,
        comment: Optional[str] = None,
    ) -> bool:
        """
        Close an open position by it ticket

        Args:
            ticket (int): Positon ticket to close (e.g TradePosition.ticket)
            id (int): The unique ID of the Expert or Strategy
            pct (float): Percentage of the position to close
            comment (str): Comment for the closing position

        Returns:
        -   True if position closed, False otherwise
        """
        # get all Actives positions
        time.sleep(0.1)
        Id = id if id is not None else self.expert_id
        positions = self.get_positions(ticket=ticket)
        buy_price = self.get_tick_info(self.symbol).ask
        sell_price = self.get_tick_info(self.symbol).bid
        deviation = self.get_deviation()
        if positions is not None and len(positions) == 1:
            position = positions[0]
            if position.ticket == ticket and position.magic == Id:
                buy = position.type == 0
                request = {
                    "action": Mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": (position.volume * pct),
                    "type": Mt5.ORDER_TYPE_SELL if buy else Mt5.ORDER_TYPE_BUY,
                    "position": ticket,
                    "price": sell_price if buy else buy_price,
                    "deviation": deviation,
                    "magic": Id,
                    "comment": f"@{self.expert_name}" if comment is None else comment,
                    "type_time": Mt5.ORDER_TIME_GTC,
                    "type_filling": Mt5.ORDER_FILLING_FOK,
                }
                return self.close_request(request, type="position")

    def bulk_close(
        self,
        tickets: List,
        tikets_type: Literal["positions", "orders"],
        close_func: Callable,
        order_type: str,
        id: Optional[int] = None,
        comment: Optional[str] = None,
    ):
        """
        Close multiple orders or positions at once.

        Args:
            tickets (List): List of tickets to close
            tikets_type (str): Type of tickets to close ('positions', 'orders')
            close_func (Callable): The function to close the tickets
            order_type (str): Type of orders or positions to close
            id (int): The unique ID of the Expert or Strategy
            comment (str): Comment for the closing position
        """
        if order_type == "all":
            order_type = "open"
        if tickets is not None and len(tickets) > 0:
            for ticket in tickets.copy():
                if close_func(ticket, id=id, comment=comment):
                    tickets.remove(ticket)
                time.sleep(1)
            if tickets is not None and len(tickets) == 0:
                self.logger.info(
                    f"ALL {order_type.upper()} {tikets_type.upper()} closed, SYMBOL={self.symbol}."
                )
            else:
                self.logger.info(
                    f"{len(tickets)} {order_type.upper()} {tikets_type.upper()} not closed, SYMBOL={self.symbol}"
                )
        else:
            self.logger.info(
                f"No {order_type.upper()} {tikets_type.upper()} to close, SYMBOL={self.symbol}."
            )

    Orders = Literal[
        "all",
        "buy_stops",
        "sell_stops",
        "buy_limits",
        "sell_limits",
        "buy_stop_limits",
        "sell_stop_limits",
    ]

    def close_orders(
        self,
        order_type: Orders,
        id: Optional[int] = None,
        comment: Optional[str] = None,
    ):
        """
        Args:
            order_type (str): Type of orders to close ('all', 'buy_stops', 'sell_stops', 'buy_limits', 'sell_limits', 'buy_stop_limits', 'sell_stop_limits')
            id (int): The unique ID of the Expert or Strategy
            comment (str): Comment for the closing position
        """
        id = id if id is not None else self.expert_id
        if order_type == "all":
            orders = self.get_current_orders(id=id)
        elif order_type == "buy_stops":
            orders = self.get_current_buy_stops(id=id)
        elif order_type == "sell_stops":
            orders = self.get_current_sell_stops(id=id)
        elif order_type == "buy_limits":
            orders = self.get_current_buy_limits(id=id)
        elif order_type == "sell_limits":
            orders = self.get_current_sell_limits(id=id)
        elif order_type == "buy_stop_limits":
            orders = self.get_current_buy_stop_limits(id=id)
        elif order_type == "sell_stop_limits":
            orders = self.get_current_sell_stop_limits(id=id)
        else:
            self.logger.error(f"Invalid order type: {order_type}")
            return
        self.bulk_close(
            orders, "orders", self.close_order, order_type, id=id, comment=comment
        )

    Positions = Literal["all", "buy", "sell", "profitable", "losing"]

    def close_positions(
        self,
        position_type: Positions,
        id: Optional[int] = None,
        comment: Optional[str] = None,
    ):
        """
        Args:
            position_type (str): Type of positions to close ('all', 'buy', 'sell', 'profitable', 'losing')
            id (int): The unique ID of the Expert or Strategy
            comment (str): Comment for the closing position
        """
        id = id if id is not None else self.expert_id
        if position_type == "all":
            positions = self.get_current_positions(id=id)
        elif position_type == "buy":
            positions = self.get_current_buys(id=id)
        elif position_type == "sell":
            positions = self.get_current_sells(id=id)
        elif position_type == "profitable":
            positions = self.get_current_profitables(id=id)
        elif position_type == "losing":
            positions = self.get_current_losings(id=id)
        else:
            self.logger.error(f"Invalid position type: {position_type}")
            return
        self.bulk_close(
            positions,
            "positions",
            self.close_position,
            position_type,
            id=id,
            comment=comment,
        )

    def get_stats(self) -> Tuple[Dict[str, Any]]:
        """
        get some stats about the trading day and trading history

        :return: tuple[Dict[str, Any]]
        """
        # get history of deals for one trading session
        profit = 0.0
        total_fees = 0.0
        loss_trades = 0
        win_trades = 0
        balance = self.get_account_info().balance
        deals = len(self.opened_positions)
        if deals != 0:
            for position in self.opened_positions:
                time.sleep(0.1)
                history = self.get_trades_history(position=position, to_df=False)
                if history is not None and len(history) == 2:
                    result = history[1].profit
                    comm = history[0].commission
                    swap = history[0].swap
                    fee = history[0].fee
                    if (result + comm + swap + fee) <= 0:
                        loss_trades += 1
                    else:
                        win_trades += 1
                    profit += result
                    total_fees += comm + swap + fee
            average_fee = total_fees / deals
            win_rate = round((win_trades / deals) * 100, 2)
            stats1 = {
                "deals": deals,
                "profit": profit,
                "win_trades": win_trades,
                "loss_trades": loss_trades,
                "total_fees": total_fees,
                "average_fee": average_fee,
                "win_rate": win_rate,
            }
        else:
            stats1 = {
                "deals": 0,
                "profit": 0,
                "win_trades": 0,
                "loss_trades": 0,
                "total_fees": 0,
                "average_fee": 0,
                "win_rate": 0,
            }

        # Get total stats
        df = self.get_trades_history()
        if df is not None:
            df2 = df.iloc[1:]
            profit = df2["profit"].sum()
            commisions = df2["commission"].sum()
            _fees = df2["fee"].sum()
            _swap = df2["swap"].sum()
            total_profit = commisions + _fees + _swap + profit
            account_info = self.get_account_info()
            balance = account_info.balance
            initial_balance = balance - total_profit
            profittable = "Yes" if balance > initial_balance else "No"
            stats2 = {"total_profit": total_profit, "profitability": profittable}
        else:
            stats2 = {"total_profit": 0, "profitability": 0}
        return (stats1, stats2)

    def sharpe(self):
        """
        Calculate the Sharpe ratio of a returns stream
        based on a number of trading periods.
        The function assumes that the returns are the excess of
        those compared to a benchmark.
        """
        # Get total history
        df2 = self.get_trades_history()
        if df2 is None:
            return 0.0
        df = df2.iloc[1:]
        profit = df[["profit", "commission", "fee", "swap"]].sum(axis=1)
        returns = profit.pct_change(fill_method=None)
        periods = self.max_trade() * 252
        sharpe = create_sharpe_ratio(returns, periods=periods)

        return round(sharpe, 3)

    def days_end(self) -> bool:
        """Check if it is the end of the trading day."""
        current_hour = datetime.now().hour
        current_minute = datetime.now().minute

        ending_hour = int(self.ending_time_hour)
        ending_minute = int(self.ending_time_minutes)

        if current_hour > ending_hour or (
            current_hour == ending_hour and current_minute >= ending_minute
        ):
            return True
        else:
            return False

    def trading_time(self):
        """Check if it is time to trade."""
        if (
            int(self.start_time_hour)
            < datetime.now().hour
            < int(self.finishing_time_hour)
        ):
            return True
        elif datetime.now().hour == int(self.start_time_hour):
            if datetime.now().minute >= int(self.start_time_minutes):
                return True
        elif datetime.now().hour == int(self.finishing_time_hour):
            if datetime.now().minute < int(self.finishing_time_minutes):
                return True
        return False

    def sleep_time(self, weekend=False):
        if weekend:
            # claculate number of minute from the friday and to monday start
            friday_time = datetime.strptime(self.current_time(), "%H:%M")
            monday_time = datetime.strptime(self.start, "%H:%M")
            intra_day_diff = (monday_time - friday_time).total_seconds() // 60
            inter_day_diff = 3 * 24 * 60
            total_minutes = inter_day_diff + intra_day_diff
            return total_minutes
        else:
            # claculate number of minute from the end to the start
            start = datetime.strptime(self.start, "%H:%M")
            end = datetime.strptime(self.current_time(), "%H:%M")
            minutes = (end - start).total_seconds() // 60
            sleep_time = (24 * 60) - minutes
            return sleep_time

    def current_datetime(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def current_time(self, seconds=False):
        if seconds:
            return datetime.now().strftime("%H:%M:%S")
        return datetime.now().strftime("%H:%M")


def create_trade_instance(
    symbols: List[str],
    params: Dict[str, Any],
    daily_risk: Optional[Dict[str, float]] = None,
    max_risk: Optional[Dict[str, float]] = None,
    pchange_sl: Optional[Dict[str, float] | float] = None,
    **kwargs,
) -> Dict[str, Trade]:
    """
    Creates Trade instances for each symbol provided.

    Args:
        symbols: A list of trading symbols (e.g., ['AAPL', 'MSFT']).
        params: A dictionary containing parameters for the Trade instance.
        daily_risk: A dictionary containing daily risk weight for each symbol.
        max_risk: A dictionary containing maximum risk weight for each symbol.

    Returns:
        A dictionary where keys are symbols and values are corresponding Trade instances.

    Raises:
        ValueError: If the 'symbols' list is empty or the 'params' dictionary is missing required keys.

    Note:
        `daily_risk` and `max_risk`  can be used to manage the risk of each symbol
        based on the importance of the symbol in the portfolio or strategy.
    """
    logger = params.get("logger", None)
    ids = params.get("expert_id", None)
    trade_instances = {}
    if not symbols:
        raise ValueError("The 'symbols' list cannot be empty.")
    if not params:
        raise ValueError("The 'params' dictionary cannot be empty.")

    if daily_risk is not None:
        for symbol in symbols:
            if symbol not in daily_risk:
                raise ValueError(f"Missing daily risk weight for symbol '{symbol}'.")
    if max_risk is not None:
        for symbol in symbols:
            if symbol not in max_risk:
                raise ValueError(
                    f"Missing maximum risk percentage for symbol '{symbol}'."
                )
    if pchange_sl is not None:
        if isinstance(pchange_sl, dict):
            for symbol in symbols:
                if symbol not in pchange_sl:
                    raise ValueError(
                        f"Missing percentage change for symbol '{symbol}'."
                    )
    if isinstance(ids, dict):
        for symbol in symbols:
            if symbol not in ids:
                raise ValueError(f"Missing expert ID for symbol '{symbol}'.")

    for symbol in symbols:
        try:
            params["symbol"] = symbol
            params["expert_id"] = (
                ids[symbol]
                if ids is not None and isinstance(ids, dict)
                else ids
                if ids is not None and isinstance(ids, (int, float))
                else params["expert_id"]
                if "expert_id" in params
                else None
            )
            params["pchange_sl"] = (
                pchange_sl[symbol]
                if pchange_sl is not None and isinstance(pchange_sl, dict)
                else pchange_sl
                if pchange_sl is not None and isinstance(pchange_sl, (int, float))
                else params["pchange_sl"]
                if "pchange_sl" in params
                else None
            )
            params["daily_risk"] = (
                daily_risk[symbol]
                if daily_risk is not None
                else params["daily_risk"]
                if "daily_risk" in params
                else None
            )
            params["max_risk"] = (
                max_risk[symbol]
                if max_risk is not None
                else params["max_risk"]
                if "max_risk" in params
                else 10.0
            )
            trade_instances[symbol] = Trade(**params)
        except Exception as e:
            logger.error(f"Creating Trade instance, SYMBOL={symbol} {e}")

    if len(trade_instances) != len(symbols):
        for symbol in symbols:
            if symbol not in trade_instances:
                if logger is not None and isinstance(logger, Logger):
                    logger.error(f"Failed to create Trade instance for SYMBOL={symbol}")
                else:
                    raise ValueError(
                        f"Failed to create Trade instance for SYMBOL={symbol}"
                    )
    return trade_instances
