import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import pandas as pd
import quantstats as qs
from loguru import logger as log
from tabulate import tabulate

from bbstrader.api import Mt5client as client
from bbstrader.api.metatrader_client import TradePosition  # type: ignore
from bbstrader.config import BBSTRADER_DIR, config_logger
from bbstrader.metatrader.account import Account
from bbstrader.metatrader.broker import check_mt5_connection
from bbstrader.metatrader.risk import RiskManagement
from bbstrader.metatrader.utils import INIT_MSG, raise_mt5_error, trade_retcode_message

try:
    import MetaTrader5 as Mt5
except ImportError:
    import bbstrader.compat  # noqa: F401

__all__ = [
    "Trade",
    "create_trade_instance",
]

log.add(
    f"{BBSTRADER_DIR}/logs/trade.log",
    enqueue=True,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
)

global LOGGER
LOGGER = log


FILLING_TYPE = [
    Mt5.ORDER_FILLING_IOC,
    Mt5.ORDER_FILLING_RETURN,
    Mt5.ORDER_FILLING_BOC,
]


Buys = Literal["BMKT", "BLMT", "BSTP", "BSTPLMT"]
Sells = Literal["SMKT", "SLMT", "SSTP", "SSTPLMT"]
Positions = Literal["all", "buy", "sell", "profitable", "losing"]
Orders = Literal[
    "all",
    "buy_stops",
    "sell_stops",
    "buy_limits",
    "sell_limits",
    "buy_stop_limits",
    "sell_stop_limits",
]

EXPERT_ID = 181051


class Trade:
    """
    Extends the `RiskManagement` class to include specific trading operations,
    incorporating risk management strategies directly into trade executions.
    It offers functionalities to execute trades while managing risks.

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
        expert_id: int = EXPERT_ID,
        version: str = "3.0",
        target: float = 5.0,
        start_time: str = "1:00",
        finishing_time: str = "23:00",
        ending_time: str = "23:30",
        time_frame: str = "D1",
        broker_tz=False,
        verbose: bool = False,
        console_log: bool = False,
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
            **kwargs: Params for the RiskManagement and Account
                See the ``bbstrader.metatrader.risk.RiskManagement`` class for more details on these parameters.
                See `bbstrader.metatrader.broker.check_mt5_connection()` for more details on how to connect to MT5 terminal.
        """

        self.symbol = symbol
        self.expert_name = expert_name
        self.expert_id = expert_id
        self.version = version
        self.target = target
        self.verbose = verbose
        self.start = start_time
        self.end = ending_time
        self.finishing = finishing_time
        self.broker_tz = broker_tz
        self.console_log = console_log
        self.timeframe = time_frame
        self.kwargs = kwargs

        self.account = Account(**kwargs)
        self.rm = RiskManagement(
            symbol=symbol,
            start_time=start_time,
            finishing_time=finishing_time,
            time_frame=time_frame,
            broker_tz=broker_tz,
            **kwargs,
        )

        self.buy_positions = []
        self.sell_positions = []
        self.opened_positions = []
        self.opened_orders = []
        self.break_even_status = []
        self.break_even_points = {}
        self.trail_after_points = []
        self._retcodes = []

        self._get_logger(logger, console_log)
        self.initialize(**kwargs)
        self.select_symbol(**kwargs)
        self.prepare_symbol()

        if self.verbose:
            self.summary()
            print()
            self.risk_managment()
            print(f">>> Everything is OK, @{self.expert_name} is Running ...>>>\n")

    @property
    def retcodes(self) -> List[int]:
        """Return all the retcodes"""
        return self._retcodes

    @property
    def logger(self):
        return LOGGER

    @property
    def orders(self):
        """Return all opened order's tickets"""
        current_orders = self.get_current_orders() or []
        opened_orders = set(current_orders + self.opened_orders)
        return list(opened_orders) if len(opened_orders) != 0 else None

    @property
    def positions(self):
        """Return all opened position's tickets"""
        current_positions = self.get_current_positions() or []
        opened_positions = set(current_positions + self.opened_positions)
        return list(opened_positions) if len(opened_positions) != 0 else None

    @property
    def buypos(self):
        """Return all buy  opened position's tickets"""
        buy_positions = self.get_current_buys() or []
        buy_positions = set(buy_positions + self.buy_positions)
        return list(buy_positions) if len(buy_positions) != 0 else None

    @property
    def sellpos(self):
        """Return all sell  opened position's tickets"""
        sell_positions = self.get_current_sells() or []
        sell_positions = set(sell_positions + self.sell_positions)
        return list(sell_positions) if len(sell_positions) != 0 else None

    @property
    def bepos(self):
        """Return All positon's tickets
        for which a break even has been set"""
        if len(self.break_even_status) != 0:
            return self.break_even_status
        return None

    def _get_logger(self, loger: Any, consol_log: bool):
        """Get the logger object"""
        global LOGGER
        if loger is None:
            ...  # Do nothing
        elif isinstance(loger, (str, Path)):
            LOGGER = config_logger(f"{BBSTRADER_DIR}/logs/{loger}", consol_log)
        elif isinstance(loger, (Logger, type(log))):
            LOGGER = loger

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
            LOGGER.error(f"During initialization: {e}")

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
            if not client.symbol_select(self.symbol, True):
                raise_mt5_error(message=INIT_MSG)
        except Exception as e:
            LOGGER.error(f"Selecting symbol '{self.symbol}': {e}")

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
            symbol_info = client.symbol_info(self.symbol)
            if symbol_info is None:
                raise_mt5_error(message=INIT_MSG)

            if not symbol_info.visible:
                raise_mt5_error(message=INIT_MSG)
            if self.verbose:
                print("Initialization successfully completed.")
        except Exception as e:
            LOGGER.error(f"Preparing symbol '{self.symbol}': {e}")

    def summary(self):
        """Show a brief description about the trading program"""
        fmt = "%H:%M"
        start = datetime.strptime(self.start, fmt).time()
        finish = datetime.strptime(self.finishing, fmt).time()
        end = datetime.strptime(self.end, fmt).time()
        if self.broker_tz:
            start = self.account.broker.get_broker_time(self.start, fmt).time()
            finish = self.account.broker.get_broker_time(self.finishing, fmt).time()
            end = self.account.broker.get_broker_time(self.end, fmt).time()
        summary_data = [
            ["Expert Advisor Name", f"@{self.expert_name}"],
            ["Expert Advisor Version", f"@{self.version}"],
            ["Expert | Strategy ID", self.expert_id],
            ["Trading Symbol", self.symbol],
            ["Trading Time Frame", self.timeframe],
            ["Start Trading Time", f"{start}"],
            ["Finishing Trading Time", f"{finish}"],
            ["Closing Position After", f"{end}"],
        ]
        # Custom table format
        summary_table = tabulate(
            summary_data, headers=["Summary", "Values"], tablefmt="outline"
        )

        # Print the table
        print("\n[============ Trade Account Summary ==============]")
        print(summary_table)

    def risk_managment(self):
        """Show the risk management parameters"""

        loss = self.rm.currency_risk()["trade_loss"]
        trade_profit = self.rm.currency_risk()["trade_profit"]
        ok = "OK" if self.rm.is_risk_ok() else "Not OK"
        account_info = self.account.get_account_info()
        total_profit = round(self.get_stats()[1]["total_profit"], 2)
        currency = account_info.currency
        rates = self.account.get_currency_rates(self.symbol)

        account_data = [
            ["Account Name", account_info.name],
            ["Account Number", account_info.login],
            ["Account Server", account_info.server],
            ["Account Balance", f"{account_info.balance} {currency}"],
            ["Account Profit", f"{total_profit} {currency}"],
            ["Account Equity", f"{account_info.equity} {currency}"],
            ["Account Leverage", account_info.leverage],
            ["Account Margin", f"{round(account_info.margin, 2)} {currency}"],
            ["Account Free Margin", f"{account_info.margin_free} {currency}"],
            ["Maximum Drawdown", f"{self.rm.max_risk}%"],
            ["Risk Allowed", f"{round((self.rm.max_risk - self.rm.risk_level()), 2)}%"],
            ["Volume", f"{self.rm.volume()} {rates.get('pc')}"],
            ["Risk Per trade", f"{-self.rm.get_currency_risk()} {currency}"],
            ["Profit Expected Per trade", f"{self.rm.expected_profit()} {currency}"],
            ["Lot Size", f"{self.rm.get_lot()} Lots"],
            ["Stop Loss", f"{self.rm.get_stop_loss()} Points"],
            ["Loss Value Per Tick", f"{round(loss, 5)} {currency}"],
            ["Take Profit", f"{self.rm.get_take_profit()} Points"],
            ["Profit Value Per Tick", f"{round(trade_profit, 5)} {currency}"],
            ["Break Even", f"{self.rm.get_break_even()} Points"],
            ["Deviation", f"{self.rm.get_deviation()} Points"],
            ["Trading Time Interval", f"{self.rm.get_minutes()} Minutes"],
            ["Risk Level", ok],
            ["Maximum Trades", self.rm.max_trade()],
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
        currency = self.account.info.currency
        net_profit = round((profit + total_fees), 2)
        trade_risk = round(self.rm.get_currency_risk() * -1, 2)

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
            ["Expected Profit per Trade", f"{self.rm.expected_profit()} {currency}"],
            ["Risk Reward Ratio", self.rm.rr],
            ["Win Rate", f"{win_rate}%"],
            ["Sharpe Ratio", self.sharpe()],
            ["Trade Profitability", additional_stats["profitability"]],
        ]
        session_table = tabulate(
            session_data, headers=["Statistics", "Values"], tablefmt="outline"
        )

        if self.verbose:
            print("\n[========== Trading Session Statistics ===========]")
            print(session_table)

        if save and stats["deals"] > 0:
            today_date = datetime.now().strftime("%Y%m%d%H%M%S")
            statistics_dict = {item[0]: item[1] for item in session_data}
            stats_df = pd.DataFrame(statistics_dict, index=[0])

            dir = dir or ".sessions"
            os.makedirs(dir, exist_ok=True)
            symbol = self.symbol.split(".")[0] if "." in self.symbol else self.symbol

            filename = f"{symbol}_{today_date}@{self.expert_id}.csv"
            filepath = os.path.join(dir, filename)
            stats_df.to_csv(filepath, index=False)
            LOGGER.info(f"Session statistics saved to {filepath}")

    def _order_type(self):
        return {
            "BMKT": (Mt5.ORDER_TYPE_BUY, "BUY"),
            "SMKT": (Mt5.ORDER_TYPE_SELL, "SELL"),
            "BLMT": (Mt5.ORDER_TYPE_BUY_LIMIT, "BUY_LIMIT"),
            "SLMT": (Mt5.ORDER_TYPE_SELL_LIMIT, "SELL_LIMIT"),
            "BSTP": (Mt5.ORDER_TYPE_BUY_STOP, "BUY_STOP"),
            "SSTP": (Mt5.ORDER_TYPE_SELL_STOP, "SELL_STOP"),
            "BSTPLMT": (Mt5.ORDER_TYPE_BUY_STOP_LIMIT, "BUY_STOP_LIMIT"),
            "SSTPLMT": (Mt5.ORDER_TYPE_SELL_STOP_LIMIT, "SELL_STOP_LIMIT"),
        }

    def open_position(
        self,
        action: Buys | Sells,
        price: Optional[float] = None,
        stoplimit: Optional[float] = None,
        id: Optional[int] = None,
        mm: bool = True,
        trail: bool = True,
        comment: Optional[str] = None,
        symbol: Optional[str] = None,
        volume: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        """Opens a Buy or Sell position (Market or Pending).

        Args:
            action (str): (`'BMKT'`, `'SMKT'`) for Market orders
                or (`'BLMT', 'SLMT', 'BSTP', 'SSTP', 'BSTPLMT', 'SSTPLMT'`) for pending orders
            price (float): The price at which to open an order
            stoplimit (float): A price a pending Limit order is set at
                when the price reaches the 'price' value (this condition is mandatory).
                The pending order is not passed to the trading system until that moment
            id (int): The strategy id or expert Id
            mm (bool): Weither to put stop loss and tp or not
            trail (bool): Weither to trail the stop loss or not
            comment (str): The comment for the closing position
            symbol (str): The symbol to trade
            volume (float): The volume (lot) to trade
            sl (float): The stop loss price
            tp (float): The take profit price
        """
        is_buy = action.startswith("B")
        symbol = symbol or self.symbol
        expert_id = id if id is not None else self.expert_id
        point = client.symbol_info(symbol).point
        tick = client.symbol_info_tick(symbol)

        req_price = None
        if "MKT" in action:
            req_price = tick.bid if is_buy else tick.ask
        else:
            if price is None:
                raise ValueError(f"Price is required for pending order: {action}")
            req_price = price

        mm_price = req_price
        if "TPLMT" in action:
            if stoplimit is None:
                raise ValueError(f"StopLimit price required for {action}")
            if (is_buy and stoplimit > req_price) or (
                not is_buy and stoplimit < req_price
            ):
                raise ValueError("Invalid StopLimit relationship to Price.")
            mm_price = stoplimit

        order_type, _ = self._order_type()[action]
        trade_action = (
            Mt5.TRADE_ACTION_DEAL if "MKT" in action else Mt5.TRADE_ACTION_PENDING
        )
        request = {
            "action": trade_action,
            "symbol": symbol,
            "volume": float(volume or self.rm.get_lot()),
            "type": order_type,
            "price": req_price,
            "deviation": self.rm.get_deviation(),
            "magic": expert_id,
            "comment": comment or f"@{self.expert_name}",
            "type_time": Mt5.ORDER_TIME_GTC,
            "type_filling": Mt5.ORDER_FILLING_FOK,
        }

        if "TPLMT" in action:
            request["stoplimit"] = stoplimit

        if mm:
            direction = 1 if is_buy else -1
            request["sl"] = sl or (
                mm_price - (direction * self.rm.get_stop_loss() * point)
            )
            request["tp"] = tp or (
                mm_price + (direction * self.rm.get_take_profit() * point)
            )

        self.break_even(mm=mm, id=expert_id, trail=trail)

        if self.check(comment):
            final_price = stoplimit if "TPLMT" in action else req_price
            return self.request_result(final_price, request, action)

        return False

    def open_buy_position(self, **kwargs):
        """
        Open a buy position or order.

        See Trade.open_position for the ``kwargs`` parameters.
        """
        return self.open_position(action=kwargs.pop("action", "BMKT"), **kwargs)


    def open_sell_position(self, **kwargs):
        """
        Open a sell position or order.

        See Trade.open_position for the ``kwargs`` parameters.
        """
        return self.open_position(action=kwargs.pop("action", "SMKT"), **kwargs)

    def check(self, comment):
        """
        Verify if all conditions for taking a position are valide,
        These conditions are based on the Maximum risk ,daily risk,
        the starting, the finishing, and ending trading time.

        Args:
            comment (str): The comment for the closing position
        """

        def _check(txt: str = ""):
            if (
                self.positive_profit(id=self.expert_id)
                or self.get_current_positions() is None
            ):
                self.close_positions(position_type="all")
                LOGGER.info(txt)
                self.statistics(save=True)

        if self.days_end():
            LOGGER.warning(f"End of the trading Day, SYMBOL={self.symbol}")
            return False
        elif not self.trading_time():
            LOGGER.warning(f"Not Trading time, SYMBOL={self.symbol}")
            return False
        elif not self.rm.is_risk_ok():
            LOGGER.warning(f"Account Risk not allowed, SYMBOL={self.symbol}")
            _check(comment)
            return False
        elif self.is_max_trades_reached():
            LOGGER.warning(f"Maximum trades reached for Today, SYMBOL={self.symbol}")
            return False
        elif self.profit_target():
            _check(f"Profit target Reached !!! SYMBOL={self.symbol}")
        return True

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
        result = None
        try:
            client.order_check(request)
            result = client.order_send(request)
        except Exception as e:
            msg = trade_retcode_message(result.retcode) if result else "N/A"
            LOGGER.error(f"Trade Order Request, {msg}{addtionnal}, {e}")
        if result and result.retcode != Mt5.TRADE_RETCODE_DONE:
            if result.retcode == Mt5.TRADE_RETCODE_INVALID_FILL:  # 10030
                for fill in FILLING_TYPE:
                    request["type_filling"] = fill
                    result = client.order_send(request)
                    if result and result.retcode == Mt5.TRADE_RETCODE_DONE:
                        break
            elif result.retcode == Mt5.TRADE_RETCODE_INVALID_VOLUME:  # 10014
                new_volume = int(request["volume"])
                if new_volume >= 1:
                    request["volume"] = new_volume
                    result = client.order_send(request)
            elif result.retcode not in self._retcodes:
                self._retcodes.append(result.retcode)
                msg = trade_retcode_message(result.retcode)
                LOGGER.error(
                    f"Trade Order Request, RETCODE={result.retcode}: {msg}{addtionnal}"
                )
            elif result.retcode in [
                Mt5.TRADE_RETCODE_CONNECTION,
                Mt5.TRADE_RETCODE_TIMEOUT,
            ]:
                tries = 0
                while result.retcode != Mt5.TRADE_RETCODE_DONE and tries < 5:
                    try:
                        client.order_check(request)
                        result = client.order_send(request)
                    except Exception as e:
                        msg = trade_retcode_message(result.retcode) if result else "N/A"
                        LOGGER.error(f"Trade Order Request, {msg}{addtionnal}, {e}")
                    if result and result.retcode == Mt5.TRADE_RETCODE_DONE:
                        break
                    tries += 1
        # Print the result
        if result and result.retcode == Mt5.TRADE_RETCODE_DONE:
            msg = trade_retcode_message(result.retcode)
            LOGGER.info(f"Trade Order {msg}{addtionnal}")
            if type != "BMKT" or type != "SMKT":
                self.opened_orders.append(result.order)
            long_msg = (
                f"1. {pos} Order #{result.order} Sent, Symbol: {self.symbol}, Price: @{round(price, 5)}, "
                f"Lot(s): {result.volume}, Sl: {self.rm.get_stop_loss()}, "
                f"Tp: {self.rm.get_take_profit()}"
            )
            LOGGER.info(long_msg)
            if type == "BMKT" or type == "SMKT":
                self.opened_positions.append(result.order)
                positions = self.account.get_positions(symbol=self.symbol)
                if positions is not None:
                    for position in positions:
                        if position.ticket == result.order:
                            if position.type == 0:
                                order_type = "BUY"
                                self.buy_positions.append(position.ticket)
                            else:
                                order_type = "SELL"
                                self.sell_positions.append(position.ticket)
                            profit = round(client.account_info().profit, 5)
                            order_info = (
                                f"2. {order_type} Position Opened, Symbol: {self.symbol}, Price: @{round(position.price_open, 5)}, "
                                f"Sl: @{round(position.sl, 5)} Tp: @{round(position.tp, 5)}"
                            )
                            LOGGER.info(order_info)
                            pos_info = (
                                f"3. [OPEN POSITIONS ON {self.symbol} = {len(positions)}, ACCOUNT OPEN PnL = {profit} "
                                f"{client.account_info().currency}]\n"
                            )
                            LOGGER.info(pos_info)
            return True
        else:
            msg = trade_retcode_message(result.retcode) if result else "N/A"
            LOGGER.error(
                f"Unable to Open Position, RETCODE={result.retcode}: {msg}{addtionnal}"
            )
            return False

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
            items = self.account.get_orders(symbol=self.symbol)
        else:
            items = self.account.get_positions(symbol=self.symbol)

        filtered_tickets = []

        if items is None:
            return []
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
        return filtered_tickets

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
            positions = self.account.get_positions()
        elif account and id is not None:
            # All open positions for a specific strategy or expert no matter the symbol
            positions = self.account.get_positions()
            if positions is not None:
                positions = [position for position in positions if position.magic == id]
        elif not account and id is None:
            # All open positions for the current symbol no matter the strategy or expert
            positions = self.account.get_positions(symbol=self.symbol)
        elif not account and id is not None:
            # All open positions for the current symbol and a specific strategy or expert
            positions = self.account.get_positions(symbol=self.symbol)
            if positions is not None:
                positions = [position for position in positions if position.magic == id]

        if positions is not None:
            profit = 0.0
            balance = client.account_info().balance
            target = round((balance * self.target) / 100, 2)
            for position in positions:
                profit += position.profit
            fees = self.get_average_fees()
            current_profit = profit + fees
            th_profit = (target * th) / 100 if th is not None else (target * 0.01)
            return current_profit >= th_profit
        return False

    def _get_trail_after_points(self, trail_after_points: int | str) -> int:
        if isinstance(trail_after_points, str):
            if trail_after_points == "SL":
                return self.rm.get_stop_loss()
            elif trail_after_points == "TP":
                return self.rm.get_take_profit()
            elif trail_after_points == "BE":
                return self.rm.get_break_even()
        # TODO: Add other combinations (e.g. "SL+TP", "SL+BE", "TP+BE", "SL*N", etc.)
        return trail_after_points

    def break_even(
        self,
        mm=True,
        id: Optional[int] = None,
        trail: Optional[bool] = True,
        stop_trail: int | str = None,
        trail_after_points: int | str = None,
        be_plus_points: Optional[int] = None,
    ):
        """
        Manages the break-even level of a trading position.

        This function checks whether it is time to set a break-even stop loss for an open position.
        If the break-even level is already set, it monitors price movement and updates the stop loss
        accordingly if the `trail` parameter is enabled.

        When `trail` is enabled, the function dynamically adjusts the break-even level based on the
        `trail_after_points` and `stop_trail` parameters.

        Args:
            id (int): The strategy ID or expert ID.
            mm (bool): Whether to manage the position or not.
            trail (bool): Whether to trail the stop loss or not.
            stop_trail (int): Number of points to trail the stop loss by.
                It represent the distance from the current price to the stop loss.
            trail_after_points (int, str): Number of points in profit
                from where the strategy will start to trail the stop loss.
                If set to str, it must be one of the following values:
                - 'SL' to trail the stop loss after the profit reaches the stop loss level in points.
                - 'TP' to trail the stop loss after the profit reaches the take profit level in points.
                - 'BE' to trail the stop loss after the profit reaches the break-even level in points.
            be_plus_points (int): Number of points to add to the break-even level.
                Represents the minimum profit to secure.
        """

        if not mm:
            return False

        Id = id if id is not None else self.expert_id
        positions = self.account.get_positions(symbol=self.symbol)
        be = self.rm.get_break_even()
        if trail_after_points is not None:
            if isinstance(trail_after_points, int):
                assert trail_after_points > be, (
                    "trail_after_points must be greater than break even or set to None"
                )
            trail_after_points = self._get_trail_after_points(trail_after_points)

        if not positions:
            return False

        for position in positions:
            if position.magic == Id:
                symbol_info = client.symbol_info(self.symbol)

                point = symbol_info.point
                digits = symbol_info.digits

                points = position.profit * (
                    symbol_info.trade_tick_size
                    / symbol_info.trade_tick_value
                    / position.volume
                )
                break_even = float(points / point) >= be
                if not break_even:
                    continue
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
                        round(be * 0.10) if be_plus_points is None else be_plus_points
                    )
                    if trail_after_points is not None:
                        if position.ticket not in self.trail_after_points:
                            # This ensures that the position rich the minimum points required
                            # before the trail can be set
                            new_be = trail_after_points - be
                            self.trail_after_points.append(position.ticket)
                    new_be_points = self.break_even_points[position.ticket] + new_be
                    favorable_move = float(points / point) >= new_be_points
                    if not favorable_move:
                        continue
                    # This allows the position to go to take profit in case of a swing trade
                    # If is a scalping position, we can set the stop_trail close to the current price.
                    trail_points = (
                        round(be * 0.50) if stop_trail is None else stop_trail
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
                    return self.set_break_even(
                        position, be, price=new_price, level=new_level
                    )
        return False

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
            position (TradePosition): The trading position for which the break-even is to be set.
                This is the value return by `mt5.positions_get()`.
            be (int): The break-even level in points.
            level (float): The break-even level in price, if set to None , it will be calated automaticaly.
            price (float): The break-even price, if set to None , it will be calated automaticaly.
        """

        symbol_info = client.symbol_info(self.symbol)
        average_fee = abs(self.get_average_fees())
        point_value = self.rm.currency_risk().get("trade_profit", 1)
        fees_points = round((average_fee / point_value), 3) if point_value != 0 else 0

        is_buy = position.type == 0
        direction = 1 if is_buy else -1
        if not position.profit > 0:
            return False
        calc_be_level = position.price_open + (direction * be * symbol_info.point)
        calc_be_price = position.price_open + (
            direction * (fees_points + symbol_info.spread) * symbol_info.point
        )
        if price is None:
            be_price = calc_be_price
        else:
            be_price = (
                max(price, calc_be_price) if is_buy else min(price, calc_be_price)
            )
        be_level = calc_be_level if level is None else level
        tick = client.symbol_info_tick(self.symbol)
        send_request = (tick.ask > be_level) if is_buy else (tick.bid < be_level)

        if send_request:
            request = {
                "action": Mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "sl": round(be_price, symbol_info.digits),
                "tp": position.tp,
            }
            return self.break_even_request(
                position.ticket, round(be_price, symbol_info.digits), request
            )
        return False

    def break_even_request(self, tiket, price, request):
        """
        Send a request to set the stop loss to break even for a given trading position.

        Args:
            tiket (int): The ticket number of the trading position.
            price (float): The price at which the stop loss is to be set.
            request (dict): The request to set the stop loss to break even.
        """
        addtionnal = f", SYMBOL={self.symbol}"
        result = None
        try:
            client.order_check(request)
            result = client.order_send(request)
        except Exception as e:
            msg = trade_retcode_message(result.retcode) if result else "N/A"
            LOGGER.error(f"Break-Even Order Request, {msg}{addtionnal}, Error: {e}")
        if result and result.retcode != Mt5.TRADE_RETCODE_DONE:
            msg = trade_retcode_message(result.retcode)
            if result.retcode != Mt5.TRADE_RETCODE_NO_CHANGES:
                LOGGER.error(
                    f"Break-Even Order Request, Position: #{tiket}, RETCODE={result.retcode}: {msg}{addtionnal}"
                )
            tries = 0
            while result.retcode != Mt5.TRADE_RETCODE_DONE and tries < 10:
                if result.retcode == Mt5.TRADE_RETCODE_NO_CHANGES:
                    break
                else:
                    try:
                        client.order_check(request)
                        result = client.order_send(request)
                    except Exception as e:
                        msg = trade_retcode_message(result.retcode) if result else "N/A"
                        LOGGER.error(
                            f"Break-Even Order Request, {msg}{addtionnal}, Error: {e}"
                        )
                    if result and result.retcode == Mt5.TRADE_RETCODE_DONE:
                        break
                tries += 1
        if result and result.retcode == Mt5.TRADE_RETCODE_DONE:
            msg = trade_retcode_message(result.retcode)
            LOGGER.info(f"Break-Even Order {msg}{addtionnal}")
            info = f"Stop loss set to Break-even, Position: #{tiket}, Symbol: {self.symbol}, Price: @{round(price, 5)}"
            LOGGER.info(info)
            self.break_even_status.append(tiket)
            return True
        return False

    def win_trade(self, position: TradePosition, th: Optional[int] = None) -> bool:
        """
        Determines if a position has met the minimum 'win' threshold in points.
        """
        points = self._convert_profit_to_points(position)
        if th is not None:
            win_threshold = th
        else:
            win_threshold = self._calculate_dynamic_threshold()

        is_profitable = points >= win_threshold
        not_processed = position.ticket not in self.break_even_status

        return is_profitable and not_processed

    def _convert_profit_to_points(self, position: TradePosition) -> float:
        info = client.symbol_info(self.symbol)
        if position.volume == 0 or info.trade_tick_value == 0:
            return 0.0

        raw_points = position.profit * (
            info.trade_tick_size / info.trade_tick_value / position.volume
        )
        return raw_points / info.point

    def _calculate_dynamic_threshold(self) -> int:
        try:
            avg_fee = abs(self.get_average_fees())
            profit_per_tick = self.rm.currency_risk().get("trade_profit", 1)

            min_be = round(avg_fee / profit_per_tick) + 2
        except (ZeroDivisionError, KeyError):
            min_be = client.symbol_info(self.symbol).spread

        be_level = self.rm.get_break_even()
        return max(min_be, round(0.1 * be_level))

    def profit_target(self) -> bool:
        """Checks if the net profit for today's deals has reached the percentage target."""
        from bbstrader.api import trade_object_to_df

        balance = client.account_info().balance
        target_amount = (balance * self.target) / 100

        opened_positions = self.get_today_deals(group=self.symbol)
        history_df = trade_object_to_df(opened_positions)

        if history_df.empty:
            return False
        net_profit = history_df[["profit", "commission", "swap", "fee"]].sum().sum()

        return net_profit >= target_amount

    def close_request(self, request: dict, type: str):
        """
        Close a trading order or position

        Args:
            request (dict): The request to close a trading order or position
            type (str): Type of the request ('order', 'position')
        """
        ticket = request[type]
        addtionnal = f", SYMBOL={self.symbol}"
        result = None
        try:
            client.order_check(request)
            result = client.order_send(request)
        except Exception as e:
            msg = trade_retcode_message(result.retcode) if result else "N/A"
            LOGGER.error(
                f"Closing {type.capitalize()} Request, RETCODE={msg}{addtionnal}, Error: {e}"
            )

        if result and result.retcode != Mt5.TRADE_RETCODE_DONE:
            if result.retcode == Mt5.TRADE_RETCODE_INVALID_FILL:  # 10030
                for fill in FILLING_TYPE:
                    request["type_filling"] = fill
                    result = client.order_send(request)
                    if result.retcode == Mt5.TRADE_RETCODE_DONE:
                        break
            elif result.retcode not in self._retcodes:
                self._retcodes.append(result.retcode)
                msg = trade_retcode_message(result.retcode)
                LOGGER.error(
                    f"Closing Order Request, {type.capitalize()}: #{ticket}, "
                    f"RETCODE={result.retcode}: {msg}{addtionnal}"
                )
            else:
                tries = 0
                while result.retcode != Mt5.TRADE_RETCODE_DONE and tries < 5:
                    try:
                        client.order_check(request)
                        result = client.order_send(request)
                    except Exception as e:
                        msg = trade_retcode_message(result.retcode) if result else "N/A"
                        LOGGER.error(
                            f"Closing {type.capitalize()} Request, {msg}{addtionnal}, Error: {e}"
                        )
                    if result and result.retcode == Mt5.TRADE_RETCODE_DONE:
                        break
                    tries += 1
        if result and result.retcode == Mt5.TRADE_RETCODE_DONE:
            msg = trade_retcode_message(result.retcode)
            LOGGER.info(f"Closing Order {msg}{addtionnal}")
            info = (
                f"{type.capitalize()} #{ticket} closed, Symbol: {self.symbol},"
                f"Price: @{round(request.get('price', 0.0), 5)}"
            )
            LOGGER.info(info)
            return True
        else:
            return False

    def modify_order(
        self,
        ticket: int,
        price: Optional[float] = None,
        stoplimit: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ):
        """
        Modify an open order by it ticket

        Args:
            ticket (int): Order ticket to modify (e.g TradeOrder.ticket)
            price (float): The price at which to modify the order
            stoplimit (float): A price a pending Limit order is set at
                when the price reaches the 'price' value (this condition is mandatory).
                The pending order is not passed to the trading system until that moment
            sl (float): The stop loss in points
            tp (float): The take profit in points
        """
        orders = self.account.get_orders(ticket=ticket) or []
        if len(orders) == 0:
            LOGGER.error(
                f"Order #{ticket} not found, SYMBOL={self.symbol}, PRICE={round(price, 5) if price else 'N/A'}"
            )
            return False
        order = orders[0]
        request = {
            "action": Mt5.TRADE_ACTION_MODIFY,
            "order": ticket,
            "price": price or order.price_open,
            "sl": sl or order.sl,
            "tp": tp or order.tp,
            "stoplimit": stoplimit or order.price_stoplimit,
        }
        client.order_check(request)
        result = client.order_send(request)
        if result.retcode == Mt5.TRADE_RETCODE_DONE:
            LOGGER.info(
                f"Order #{ticket} modified, SYMBOL={self.symbol}, PRICE={round(request['price'], 5)},"
                f"SL={round(request['sl'], 5)}, TP={round(request['tp'], 5)}, STOP_LIMIT={round(request['stoplimit'], 5)}"
            )
            return True
        else:
            msg = trade_retcode_message(result.retcode)
            LOGGER.error(
                f"Unable to modify Order #{ticket}, RETCODE={result.retcode}: {msg}, SYMBOL={self.symbol}"
            )
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
        symbol: Optional[str] = None,
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
        symbol = symbol or self.symbol
        Id = id if id is not None else self.expert_id
        positions = self.account.get_positions(ticket=ticket)
        deviation = self.rm.get_deviation()
        if positions is not None and len(positions) == 1:
            position = positions[0]
            if position.ticket == ticket and position.magic == Id:
                buy = position.type == 0
                request = {
                    "action": Mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": (position.volume * pct),
                    "type": Mt5.ORDER_TYPE_SELL if buy else Mt5.ORDER_TYPE_BUY,
                    "position": ticket,
                    "price": position.price_current,
                    "deviation": deviation,
                    "comment": f"@{self.expert_name}" if comment is None else comment,
                    "type_time": Mt5.ORDER_TIME_GTC,
                    "type_filling": Mt5.ORDER_FILLING_FOK,
                }
                return self.close_request(request, type="position")
        return False

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

        if not tickets:
            LOGGER.info(
                f"No {order_type.upper()} {tikets_type.upper()} to close, SYMBOL={self.symbol}."
            )
            return
        failed_tickets = []
        with ThreadPoolExecutor(max_workers=min(len(tickets), 20)) as executor:
            future_to_ticket = {
                executor.submit(close_func, ticket, id=id, comment=comment): ticket
                for ticket in tickets
            }
            for future in as_completed(future_to_ticket):
                ticket = future_to_ticket[future]
                try:
                    success = future.result()
                    if not success:
                        failed_tickets.append(ticket)
                except Exception as exc:
                    LOGGER.error(f"Ticket {ticket} generated an exception: {exc}")
                    failed_tickets.append(ticket)
        if not failed_tickets:
            LOGGER.info(
                f"ALL {order_type.upper()} {tikets_type.upper()} closed, SYMBOL={self.symbol}."
            )
        else:
            LOGGER.info(
                f"{len(failed_tickets)}/{len(tickets)} {order_type.upper()} {tikets_type.upper()} NOT closed, SYMBOL={self.symbol}"
            )

    def close_orders(
        self,
        order_type: Orders,
        id: Optional[int] = None,
        comment: Optional[str] = None,
    ):
        """
        Args:
            order_type (str): Type of orders to close
                ('all', 'buy_stops', 'sell_stops', 'buy_limits', 'sell_limits', 'buy_stop_limits', 'sell_stop_limits')
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
            LOGGER.error(f"Invalid order type: {order_type}")
            return
        self.bulk_close(
            orders, "orders", self.close_order, order_type, id=id, comment=comment
        )

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
            LOGGER.error(f"Invalid position type: {position_type}")
            return
        self.bulk_close(
            positions,
            "positions",
            self.close_position,
            position_type,
            id=id,
            comment=comment,
        )

    def get_today_deals(self, group=None):
        return self.account.get_today_deals(self.expert_id, group=group)

    def is_max_trades_reached(self) -> bool:
        """
        Check if the maximum number of trades for the day has been reached.

        :return: bool
        """
        negative_deals = 0
        max_trades = self.rm.max_trade()
        today_deals = self.get_today_deals(group=self.symbol)
        for deal in today_deals:
            if deal.profit < 0:
                negative_deals += 1
        if negative_deals >= max_trades:
            return True
        return False

    def get_stats(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Retrieves aggregated session and historical trading performance."""
        today_deals = self.get_today_deals(group=self.symbol)
        stats1 = self._calculate_session_stats(today_deals)
        stats2 = self._calculate_historical_stats()

        return stats1, stats2

    def _calculate_session_stats(self, deals_list) -> Dict[str, Any]:
        stats = {
            "deals": len(deals_list),
            "profit": 0.0,
            "win_trades": 0,
            "loss_trades": 0,
            "total_fees": 0.0,
            "average_fee": 0.0,
            "win_rate": 0.0,
        }

        if not deals_list:
            return stats

        for pos in deals_list:
            history = self.account.get_trades_history(
                position=pos.position_id, to_df=False
            )

            if not history or len(history) < 2:
                continue

            fee_sum = sum([history[0].commission, history[0].swap, history[0].fee])
            net_result = history[1].profit + fee_sum

            stats["profit"] += history[1].profit
            stats["total_fees"] += fee_sum

            if net_result <= 0:
                stats["loss_trades"] += 1
            else:
                stats["win_trades"] += 1

        if stats["deals"] > 0:
            stats["average_fee"] = stats["total_fees"] / stats["deals"]
            stats["win_rate"] = round((stats["win_trades"] / stats["deals"]) * 100, 2)
        return stats

    def get_average_fees(self) -> float:
        positions = self.account.get_trades_history(to_df=False)
        if not positions:
            return 0.0
        fees = sum([p.swap + p.fee + p.commission for p in positions])
        return fees / len(positions) if len(positions) > 0 else 0.0

    def _calculate_historical_stats(self) -> Dict[str, Any]:
        history = self.account.get_trades_history()

        if history is None or len(history) <= 1:
            return {"total_profit": 0, "profitability": "No"}

        df = history.iloc[1:]
        total_net = df[["profit", "commission", "fee", "swap"]].sum().sum()
        return {
            "total_profit": total_net,
            "profitability": "Yes" if total_net > 0 else "No",
        }

    def sharpe(self):
        """
        Calculate the Sharpe ratio of a returns stream
        based on a number of trading periods.
        The function assumes that the returns are the excess of
        those compared to a benchmark.
        """
        import warnings

        warnings.filterwarnings("ignore")
        history = self.account.get_trades_history()
        if history is None or len(history) < 2:
            return 0.0
        df = history.iloc[1:]
        profit = df[["profit", "commission", "fee", "swap"]].sum(axis=1)
        returns = profit.pct_change(fill_method=None)
        periods = self.rm.max_trade() * 252
        sharpe = qs.stats.sharpe(returns, periods=periods)

        return round(sharpe, 3)

    def days_end(self) -> bool:
        """Check if it is the end of the trading day."""
        fmt = "%H:%M"
        now = datetime.now().time()
        end = datetime.strptime(self.end, fmt).time()
        if self.broker_tz:
            now = self.account.broker.get_broker_time(self.current_time(), fmt).time()
            end = self.account.broker.get_broker_time(self.end, fmt).time()
        if now >= end:
            return True
        return False

    def trading_time(self):
        """Check if it is time to trade."""
        fmt = "%H:%M"
        now = datetime.now()
        start = datetime.strptime(self.start, fmt).time()
        end = datetime.strptime(self.finishing, fmt).time()
        if self.broker_tz:
            now = self.account.broker.get_broker_time(self.current_time(), fmt).time()
            start = self.account.broker.get_broker_time(self.start, fmt).time()
            now = self.account.broker.get_broker_time(self.finishing, fmt).time()
        if start <= now.time() <= end:
            return True
        return False

    def sleep_time(self, weekend=False):
        fmt = "%H:%M"
        now = datetime.now()
        if weekend:
            # calculate number of minute from now and monday start
            multiplyer = {"friday": 3, "saturday": 2, "sunday": 1}
            current_time = datetime.strptime(self.current_time(), fmt)
            monday_time = datetime.strptime(self.start, fmt)
            if self.broker_tz:
                now = self.account.broker.get_broker_time(self.current_time(), fmt)
                current_time = self.account.broker.get_broker_time(
                    self.current_time(), fmt
                )
                monday_time = self.account.broker.get_broker_time(self.start, fmt)
            intra_day_diff = (monday_time - current_time).total_seconds() // 60
            inter_day_diff = multiplyer[now.strftime("%A").lower()] * 24 * 60
            total_minutes = inter_day_diff + intra_day_diff
            return total_minutes
        else:
            # calculate number of minute from the end to the start
            start = datetime.strptime(self.start, fmt)
            end = datetime.strptime(self.current_time(), fmt)
            if self.broker_tz:
                start = self.account.broker.get_broker_time(self.start, fmt)
                end = self.account.broker.get_broker_time(self.current_time(), fmt)
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
        See bbstrader.metatrader.risk.RiskManagement for more details.
    """
    if not symbols or not params:
        raise ValueError("Symbols and params are required.")

    logger = params.get("logger") if isinstance(params.get("logger"), Logger) else log
    base_id = params.get("expert_id", EXPERT_ID)

    def get_val(source, symbol, default=None):
        if isinstance(source, dict):
            if symbol not in source:
                raise ValueError(f"Missing key '{symbol}' in configuration.")
            return source[symbol]
        return source if source is not None else default

    trades = {}
    for sym in symbols:
        try:
            conf = {
                **params,
                "symbol": sym,
                "expert_id": get_val(base_id, sym),
                "daily_risk": get_val(daily_risk, sym, params.get("daily_risk")),
                "max_risk": get_val(max_risk, sym, params.get("max_risk", 10.0)),
                "pchange_sl": get_val(pchange_sl, sym, params.get("pchange_sl")),
            }
            trades[sym] = Trade(**conf)

        except Exception as e:
            logger.error(f"Failed trade init: SYMBOL={sym} | ERR={e}")

    # Final Audit
    if len(trades) < len(symbols):
        missing = set(symbols) - set(trades.keys())
        logger.warning(f"Partial success. Missing symbols: {missing}")

    logger.info(f"Initialized {len(trades)} trade instances.")
    return trades
