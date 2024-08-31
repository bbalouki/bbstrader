import os
import csv
import time
import logging
import numpy as np
from datetime import datetime
import MetaTrader5 as Mt5
from typing import List, Tuple, Dict, Any, Optional, Literal
from bbstrader.metatrader.risk import RiskManagement
from bbstrader.metatrader.account import INIT_MSG
from bbstrader.metatrader.utils import (
    TimeFrame, TradePosition, TickInfo,
    raise_mt5_error, trade_retcode_message, config_logger
)

# Configure the logger
logger = config_logger('trade.log', console_log=True)

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
        ...     symbol="#AAPL",               # Symbol to trade
        ...     expert_name="MyExpertAdvisor",# Name of the expert advisor
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
        symbol: str = 'EURUSD',
        expert_name: str = 'bbstrader',
        expert_id: int = 9818,
        version: str = '1.0',
        target: float = 5.0,
        start_time: str = "1:00",
        finishing_time: str = "23:00",
        ending_time: str = "23:30",
        verbose: Optional[bool] = None,
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
            target (float): `Trading period (day, week, month) profit target` in percentage
            start_time (str): The` hour and minutes` that the expert advisor is able to start to run.
            finishing_time (str): The time after which no new position can be opened.
            ending_time (str): The time after which any open position will be closed.
            verbose (bool | None): If set to None (default), account summary and risk managment
                parameters are printed in the terminal.

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
        See the RiskManagement class for more details on these parameters.
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
        self.tf = kwargs.get("time_frame", 'D1')

        self.lot = self.get_lot()
        self.stop_loss = self.get_stop_loss()
        self.take_profit = self.get_take_profit()
        self.break_even_points = self.get_break_even()
        self.deviation = self.get_deviation()

        self.start_time_hour, self.start_time_minutes = self.start.split(":")
        self.finishing_time_hour, self.finishing_time_minutes = self.finishing.split(
            ":")
        self.ending_time_hour, self.ending_time_minutes = self.end.split(":")

        self.buy_positions = []
        self.sell_positions = []
        self.opened_positions = []
        self.opened_orders = []
        self.break_even_status = []

        self.initialize()
        self.select_symbol()
        self.prepare_symbol()

        if self.verbose:
            self.summary()
            time.sleep(1)
            print()
            self.risk_managment()
            print(
                f">>> Everything is OK, @{self.expert_name} is Running ....>>>\n")

    def initialize(self):
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
            if not Mt5.initialize():
                raise_mt5_error(message=INIT_MSG)
            if self.verbose:
                print(
                    f"You are running the @{self.expert_name} Expert advisor,"
                    f" Version @{self.version}, on {self.symbol}."
                )
        except Exception as e:
            logger.error(f"During initialization: {e}")

    def select_symbol(self):
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
            if not Mt5.symbol_select(self.symbol, True):
                raise_mt5_error(message=INIT_MSG)
        except Exception as e:
            logger.error(f"Selecting symbol '{self.symbol}': {e}")

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
            logger.error(f"Preparing symbol '{self.symbol}': {e}")

    def summary(self):
        """Show a brief description about the trading program"""
        print(
            "╔═════════════════ Summary  ════════════════════╗\n"
            f"║ Expert Advisor Name             @{self.expert_name}\n"
            f"║ Expert Advisor Version          @{self.version}\n"
            f"║ Expert | Strategy ID            {self.expert_id}\n"
            f"║ Trading Symbol                  {self.symbol}\n"
            f"║ Trading Time Frame              {self.tf}\n"
            f"║ Start Trading Time              {self.start_time_hour}:{self.start_time_minutes}\n"
            f"║ Finishing Trading Time          {self.finishing_time_hour}:{self.finishing_time_minutes}\n"
            f"║ Closing Position After          {self.ending_time_hour}:{self.ending_time_minutes}\n"
            "╚═══════════════════════════════════════════════╝\n"
        )

    def risk_managment(self):
        """Show the risk management parameters"""

        loss = self.currency_risk()["trade_loss"]
        profit = self.currency_risk()["trade_profit"]
        ok = "OK" if self.is_risk_ok() else "Not OK"
        account_info = self.get_account_info()
        _profit = round(self.get_stats()[1]["total_profit"], 2)
        currency = account_info.currency
        rates = self.get_currency_rates(self.symbol)
        marging_currency = rates['mc']
        print(
            "╔═════════════════  Risk Management ═════════════════════╗\n"
            f"║ Account Name                     {account_info.name}\n"
            f"║ Account Number                   {account_info.login}\n"
            f"║ Account Server                   {account_info.server}\n"
            f"║ Account Balance                  {account_info.balance} {currency}\n"
            f"║ Account Profit                   {_profit} {currency}\n"
            f"║ Account Equity                   {account_info.equity} {currency}\n"
            f"║ Account Leverage                 {self.get_leverage(True)}\n"
            f"║ Account Margin                   {round(account_info.margin, 2)} {currency}\n"
            f"║ Account Free Margin              {account_info.margin_free} {currency}\n"
            f"║ Maximum Drawdown                 {self.max_risk}%\n"
            f"║ Risk Allowed                     {round((self.max_risk - self.risk_level()), 2)}%\n"
            f"║ Volume                           {self.volume()} {marging_currency}\n"
            f"║ Risk Per trade                   {-self.get_currency_risk()} {currency}\n"
            f"║ Profit Expected Per trade        {self.expected_profit()} {currency}\n"
            f"║ Lot Size                         {self.lot} Lots\n"
            f"║ Stop Loss                        {self.stop_loss} Points\n"
            f"║ Loss Value Per Tick              {round(loss, 5)} {currency}\n"
            f"║ Take Profit                      {self.take_profit} Points\n"
            f"║ Profit Value Per Tick            {round(profit, 5)} {currency}\n"
            f"║ Break Even                       {self.break_even_points} Points\n"
            f"║ Deviation                        {self.deviation} Points\n"
            f"║ Trading Time Interval            {self.get_minutes()} Minutes\n"
            f"║ Risk Level                       {ok}\n"
            f"║ Maximum Trades                   {self.max_trade()}\n"
            "╚══════════════════════════════════════════════════════╝\n"
        )

    def statistics(self, save=True, dir="stats"):
        """
        Print some statistics for the trading session and save to CSV if specified.

        Args:
            save (bool, optional): Whether to save the statistics to a CSV file.
            dir (str, optional): The directory to save the CSV file. 
        """
        stats, additional_stats = self.get_stats()

        deals = stats["deals"]
        wins = stats["win_trades"]
        losses = stats["loss_trades"]
        profit = round(stats["profit"], 2)
        win_rate = stats["win_rate"]
        total_fees = round(stats["total_fees"], 3)
        average_fee = round(stats["average_fee"], 3)
        profitability = additional_stats["profitability"]
        currency = self.get_account_info().currency
        net_profit = round((profit + total_fees), 2)
        trade_risk = round(self.get_currency_risk() * -1, 2)
        expected_profit = round((trade_risk * self.rr * -1), 2)

        # Formatting the statistics output
        stats_output = (
            f"╔═══════════════ Session Statistics ═════════════╗\n"
            f"║ Total Trades                    {deals}\n"
            f"║ Winning Trades                  {wins}\n"
            f"║ Losing Trades                   {losses}\n"
            f"║ Session Profit                  {profit} {currency}\n"
            f"║ Total Fees                      {total_fees} {currency}\n"
            f"║ Average Fees                    {average_fee} {currency}\n"
            f"║ Net Profit                      {net_profit} {currency}\n"
            f"║ Risk per Trade                  {trade_risk} {currency}\n"
            f"║ Expected Profit per Trade       {self.expected_profit()} {currency}\n"
            f"║ Risk Reward Ratio               {self.rr}\n"
            f"║ Win Rate                        {win_rate}%\n"
            f"║ Sharpe Ratio                    {self.sharpe()}\n"
            f"║ Trade Profitability             {profitability}\n"
            "╚═════════════════════════════════════════════════╝\n"
        )

        # Print the formatted statistics
        if self.verbose:
            print(stats_output)

        # Save to CSV if specified
        if save:
            today_date = datetime.now().strftime("%Y-%m-%d")
            # Create a dictionary with the statistics
            statistics_dict = {
                "Total Trades": deals,
                "Winning Trades": wins,
                "Losing Trades": losses,
                "Session Profit": f"{profit} {currency}",
                "Total Fees": f"{total_fees} {currency}",
                "Average Fees": f"{average_fee} {currency}",
                "Net Profit": f"{net_profit} {currency}",
                "Risk per Trade": f"{trade_risk} {currency}",
                "Expected Profit per Trade": f"{expected_profit} {currency}",
                "Risk Reward Ratio": self.rr,
                "Win Rate": f"{win_rate}%",
                "Sharpe Ratio": self.sharpe(),
                "Trade Profitability": profitability,
            }
            # Create the directory if it doesn't exist
            os.makedirs(dir, exist_ok=True)
            if '.' in self.symbol:
                symbol = self.symbol.split('.')[0]
            else:
                symbol = self.symbol

            filename = f"{symbol}_{today_date}_session.csv"
            filepath = os.path.join(dir, filename)

            # Updated code to write to CSV
            with open(filepath, mode="w", newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(
                    csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                writer.writerow(["Statistic", "Value"])
                for stat, value in statistics_dict.items():
                    writer.writerow([stat, value])
            logger.info(f"Session statistics saved to {filepath}")

    def open_buy_position(
        self,
        action: Literal['BMKT', 'BLMT', 'BSTP', 'BSTPLMT'] = 'BMKT',
        price: Optional[float] = None,
        mm: bool = True,
        id: Optional[int] = None,
        comment: Optional[str] = None
    ):
        """
        Open a Buy positin

        Args:
            action (str): `'BMKT'` for Market orders or `'BLMT', 
                'BSTP','BSTPLMT'` for pending orders
            price (float): The price at which to open an order
            id (int): The strategy id or expert Id
            mm (bool): Weither to put stop loss and tp or not
            comment (str): The comment for the opening position
        """
        Id = id if id is not None else self.expert_id
        point = self.get_symbol_info(self.symbol).point
        if action != 'BMKT':
            assert price is not None, \
                "You need to set a price for pending orders"
            _price = price
        else:
            _price = self.get_tick_info(self.symbol).ask
        digits = self.get_symbol_info(self.symbol).digits

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
        if mm:
            request['sl'] = (_price - stop_loss * point)
            request['tp'] = (_price + take_profit * point)
        if action != 'BMKT':
            request["action"] = Mt5.TRADE_ACTION_PENDING
            request["type"] = self._order_type()[action][0]

        self.break_even(comment)
        if self.check(comment):
            self.request_result(_price, request, action),

    def _order_type(self):
        type = {
            'BMKT': (Mt5.ORDER_TYPE_BUY, 'BUY'),
            'SMKT': (Mt5.ORDER_TYPE_BUY, 'SELL'),
            'BLMT': (Mt5.ORDER_TYPE_BUY_LIMIT, 'BUY_LIMIT'),
            'SLMT': (Mt5.ORDER_TYPE_SELL_LIMIT, 'SELL_LIMIT'),
            'BSTP': (Mt5.ORDER_TYPE_BUY_STOP, 'BUY_STOP'),
            'SSTP': (Mt5.ORDER_TYPE_SELL_STOP, 'SELL_STOP'),
            'BSTPLMT': (Mt5.ORDER_TYPE_BUY_STOP_LIMIT, 'BUY_STOP_LIMIT'),
            'SSTPLMT': (Mt5.ORDER_TYPE_SELL_STOP_LIMIT, 'SELL_STOP_LIMIT')
        }
        return type

    def open_sell_position(
        self,
        action: Literal['SMKT', 'SLMT', 'SSTP', 'SSTPLMT'] = 'SMKT',
        price: Optional[float] = None,
        mm: bool = True,
        id: Optional[int] = None,
        comment: Optional[str] = None
    ):
        """
        Open a sell positin

        Args:
            action (str): `'SMKT'` for Market orders
                or `'SLMT', 'SSTP','SSTPLMT'` for pending orders
            price (float): The price at which to open an order
            id (int): The strategy id or expert Id
            mm (bool): Weither to put stop loss and tp or not
            comment (str): The comment for the closing position
        """
        Id = id if id is not None else self.expert_id
        point = self.get_symbol_info(self.symbol).point
        if action != 'SMKT':
            assert price is not None, \
                "You need to set a price for pending orders"
            _price = price
        else:
            _price = self.get_tick_info(self.symbol).bid
        digits = self.get_symbol_info(self.symbol).digits

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
        if mm:
            request["sl"] = (_price + stop_loss * point)
            request["tp"] = (_price - take_profit * point)
        if action != 'SMKT':
            request["action"] = Mt5.TRADE_ACTION_PENDING
            request["type"] = self._order_type()[action][0]

        self.break_even(comment)
        if self.check(comment):
            self.request_result(_price, request, action)

    def _risk_free(self):
        max_trade = self.max_trade()
        loss_trades = self.get_stats()[0]['loss_trades']
        if loss_trades >= max_trade:
            return False
        return True

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
            logger.info(f"Not Trading time, SYMBOL={self.symbol}")
            return False
        elif not self.is_risk_ok():
            logger.error(f"Risk not allowed, SYMBOL={self.symbol}")
            self._check(comment)
            return False
        elif not self._risk_free():
            logger.error(f"Maximum trades Reached, SYMBOL={self.symbol}")
            self._check(comment)
            return False
        elif self.profit_target():
            self._check(f'Profit target Reached !!! SYMBOL={self.symbol}')
        return True

    def _check(self, txt: str = ""):
        if self.positive_profit() or self.get_current_open_positions() is None:
            self.close_positions(position_type='all')
            logger.info(txt)
            time.sleep(5)
            self.statistics(save=True)

    def request_result(
        self,
        price: float,
        request: Dict[str, Any],
        type: Literal['BMKT', 'BLMT', 'BSTP', 'BSTPLMT',
                      'SMKT', 'SLMT', 'SSTP', 'SSTPLMT']
    ):
        """
        Check if a trading order has been sent correctly

        Args:
            price (float): Price for opening the position
            request (Dict[str, Any]): A trade request to sent to Mt5.order_sent() 
            all detail in request can be found here https://www.mql5.com/en/docs/python_metatrader5/mt5ordersend_py
            
            type (str): The type of the order 
                `(BMKT, SMKT, BLMT, SLMT, BSTP, SSTP, BSTPLMT, SSTPLMT)` 
        """
        # Send a trading request
        # Check the execution result
        pos = self._order_type()[type][1]
        addtionnal = f", SYMBOL={self.symbol}"
        try:
            check_result = self.check_order(request)
            result = self.send_order(request)
        except Exception as e:
            print(f"{self.current_datetime()} -", end=' ')
            trade_retcode_message(
                result.retcode, display=True, add_msg=f"{e}{addtionnal}")
        if result.retcode != Mt5.TRADE_RETCODE_DONE:
            msg = trade_retcode_message(result.retcode)
            logger.error(
                f"Trade Order Request, RETCODE={result.retcode}: {msg}{addtionnal}")
            if result.retcode in [
                    Mt5.TRADE_RETCODE_CONNECTION, Mt5.TRADE_RETCODE_TIMEOUT]:
                tries = 0
                while result.retcode != Mt5.TRADE_RETCODE_DONE and tries < 5:
                    time.sleep(1)
                    try:
                        check_result = self.check_order(request)
                        result = self.send_order(request)
                    except Exception as e:
                        print(f"{self.current_datetime()} -", end=' ')
                        trade_retcode_message(
                            result.retcode, display=True, add_msg=f"{e}{addtionnal}")
                    if result.retcode == Mt5.TRADE_RETCODE_DONE:
                        break
                    tries += 1
        # Print the result
        if result.retcode == Mt5.TRADE_RETCODE_DONE:
            msg = trade_retcode_message(result.retcode)
            logger.info(f"Trade Order {msg}{addtionnal}")
            if type != "BMKT" or type != "SMKT":
                self.opened_orders.append(result.order)
            long_msg = (
                f"1. {pos} Order #{result.order} Sent, Symbol: {self.symbol}, Price: @{price}, "
                f"Lot(s): {result.volume}, Sl: {self.get_stop_loss()}, "
                f"Tp: {self.get_take_profit()}"
            )
            logger.info(long_msg)
            time.sleep(0.1)
            if type == "BMKT" or type == "SMKT":
                self.opened_positions.append(result.order)
                positions = self.get_positions(symbol=self.symbol)
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
                        logger.info(order_info)
                        pos_info = (
                            f"3. [OPEN POSITIONS ON {self.symbol} = {len(positions)}, ACCOUNT OPEN PnL = {profit} "
                            f"{self.get_account_info().currency}]\n"
                        )
                        logger.info(pos_info)

    def open_position(
        self,
        action: Literal[
            'BMKT', 'BLMT', 'BSTP', 'BSTPLMT',
            'SMKT', 'SLMT', 'SSTP', 'SSTPLMT'],
        buy: bool = False,
        sell: bool = False,
        price: Optional[float] = None,
        id: Optional[int] = None,
        mm: bool = True,
        comment: Optional[str] = None
    ):
        """
        Open a buy or sell position.

        Args:
            action (str): (`'BMKT'`, `'SMKT'`) for Market orders
                or (`'BLMT', 'SLMT', 'BSTP', 'SSTP', 'BSTPLMT', 'SSTPLMT'`) for pending orders
            buy (bool): A boolean True or False
            sell (bool): A boolean True or False
            id (int): The strategy id or expert Id
            mm (bool): Weither to put stop loss and tp or not
            comment (str): The comment for the closing position
        """
        if buy:
            self.open_buy_position(
                action=action, price=price, id=id, mm=mm, comment=comment)
        if sell:
            self.open_sell_position(
                action=action, price=price, id=id, mm=mm, comment=comment)

    @property
    def get_opened_orders(self):
        """ Return all opened order's tickets"""
        if len(self.opened_orders) != 0:
            return self.opened_orders
        return None

    @property
    def get_opened_positions(self):
        """Return all opened position's tickets"""
        if len(self.opened_positions) != 0:
            return self.opened_positions
        return None

    @property
    def get_buy_positions(self):
        """Return all buy  opened position's tickets"""
        if len(self.buy_positions) != 0:
            return self.buy_positions
        return None

    @property
    def get_sell_positions(self):
        """Return all sell  opened position's tickets"""
        if len(self.sell_positions) != 0:
            return self.sell_positions
        return None

    @property
    def get_be_positions(self):
        """Return All positon's tickets 
            for which a break even has been set"""
        if len(self.break_even_status) != 0:
            return self.break_even_status
        return None

    def get_filtered_tickets(self,
                             id: Optional[int] = None,
                             filter_type: Optional[str] = None,
                             th=None
                             ) -> List[int] | None:
        """
        Get tickets for positions or orders based on filters.

        Args:
            id (int): The strategy id or expert Id
            filter_type (str): Filter type ('orders', 'positions', 'buys', 'sells', 'win_trades')
                - `orders` are current open orders
                - `positions` are all current open positions
                - `buys` and `sells` are current buy or sell open positions 
                - `win_trades` are current open position that have a profit greater than a threshold
            th (bool): the minimum treshold for winning position
                (only relevant when filter_type is 'win_trades')

        Returns:
            List[int] | None: A list of filtered tickets 
                or None if no tickets match the criteria.
        """
        Id = id if id is not None else self.expert_id

        if filter_type == 'orders':
            items = self.get_orders(symbol=self.symbol)
        else:
            items = self.get_positions(symbol=self.symbol)

        filtered_tickets = []

        if items is not None:
            for item in items:
                if item.magic == Id:
                    if filter_type == 'buys' and item.type != 0:
                        continue
                    if filter_type == 'sells' and item.type != 1:
                        continue
                    if filter_type == 'win_trades' and not self.win_trade(item, th=th):
                        continue
                    filtered_tickets.append(item.ticket)
            return filtered_tickets if filtered_tickets else None
        return None

    def get_current_open_orders(self, id: Optional[int] = None) -> List[int] | None:
        return self.get_filtered_tickets(id=id, filter_type='orders')

    def get_current_open_positions(self, id: Optional[int] = None) -> List[int] | None:
        return self.get_filtered_tickets(id=id, filter_type='positions')

    def get_current_win_trades(self, id: Optional[int] = None, th=None) -> List[int] | None:
        return self.get_filtered_tickets(id=id, filter_type='win_trades', th=th)

    def get_current_buys(self, id: Optional[int] = None) -> List[int] | None:
        return self.get_filtered_tickets(id=id, filter_type='buys')

    def get_current_sells(self, id: Optional[int] = None) -> List[int] | None:
        return self.get_filtered_tickets(id=id, filter_type='sells')

    def positive_profit(self, th: Optional[float] = None
                        ) -> bool:
        """
        Check is the total profit on current open positions
        Is greater than a minimum profit express as percentage 
        of the profit target.

        Args:
            th (float): The minimum profit target on current positions
        """
        positions = self.get_current_open_positions()
        profit = 0.0
        balance = self.get_account_info().balance
        target = round((balance * self.target)/100, 2)
        if positions is not None:
            for position in positions:
                time.sleep(0.1)
                history = self.get_positions(
                    ticket=position
                )
                profit += history[0].profit
            fees = self.get_stats()[0]["average_fee"] * len(positions)
            current_profit = profit + fees
            th_profit = (target*th)/100 if th is not None else (target*0.01)
            if current_profit > th_profit:
                return True
        return False

    def break_even(self, id: Optional[int] = None):
        """
        Checks if it's time to put the break even,
        if so , it will sets the break even ,and if the break even was already set,
        it checks if the price has moved in favorable direction,
        if so , it set the new break even.

        Args:
            id (int): The strategy Id or Expert Id
        """
        time.sleep(0.1)
        Id = id if id is not None else self.expert_id
        positions = self.get_positions(symbol=self.symbol)
        be = self.get_break_even()
        if positions is not None:
            for position in positions:
                if position.magic == Id:
                    size = self.get_symbol_info(self.symbol).trade_tick_size
                    value = self.get_symbol_info(self.symbol).trade_tick_value
                    point = self.get_symbol_info(self.symbol).point
                    digits = self.get_symbol_info(self.symbol).digits
                    points = position.profit * (size / value / position.volume)
                    break_even = float(points/point) >= be
                    if break_even:
                        # Check if break-even has already been set for this position
                        if position.ticket not in self.break_even_status:
                            self.set_break_even(position, be)
                            self.break_even_status.append(position.ticket)
                        else:
                            # Check if the price has moved favorably
                            new_be = be * 0.50
                            favorable_move = (
                                (position.type == 0 and (
                                    (position.price_current - position.sl) / point) > new_be)
                                or
                                (position.type == 1 and (
                                    (position.sl - position.price_current) / point) > new_be)
                            )
                            if favorable_move:
                                # Calculate the new break-even level and price
                                if position.type == 0:
                                    new_level = round(
                                        position.sl + (new_be * point), digits)
                                    new_price = round(
                                        position.sl + ((0.25 * be) * point), digits)
                                else:
                                    new_level = round(
                                        position.sl - (new_be * point), digits)
                                    new_price = round(
                                        position.sl - ((0.25 * be) * point), digits)
                                self.set_break_even(
                                    position, be, price=new_price, level=new_level
                                )

    def set_break_even(self,
                       position: TradePosition,
                       be: int,
                       price: Optional[float] = None,
                       level: Optional[float] = None):
        """
        Sets the break-even level for a given trading position.

        Args:
            position (TradePosition):
                The trading position for which the break-even is to be set
                This is the value return by `mt5.positions_get()`
            be (int): The break-even level in points.
            level (float): The break-even level in price
                if set to None , it will be calated automaticaly.
            price (float): The break-even price
                if set to None , it will be calated automaticaly.
        """
        point = self.get_symbol_info(self.symbol).point
        digits = self.get_symbol_info(self.symbol).digits
        spread = self.get_symbol_info(self.symbol).spread
        fees = self.get_stats()[0]["average_fee"] * -1
        risk = self.currency_risk()["trade_profit"]
        fees_points = round((fees / risk), 3)
        # If Buy
        if position.type == 0 and position.price_current > position.price_open:
            # Calculate the break-even level and price
            break_even_level = position.price_open + (be * point)
            break_even_price = position.price_open + \
                ((fees_points + spread) * point)
            _price = break_even_price if price is None else price
            _level = break_even_level if level is None else level

            if self.get_tick_info(self.symbol).ask > _level:
                # Set the stop loss to break even
                request = {
                    "action": Mt5.TRADE_ACTION_SLTP,
                    "type": Mt5.ORDER_TYPE_SELL_STOP,
                    "position": position.ticket,
                    "sl": round(_price, digits),
                    "tp": position.tp
                }
                self._break_even_request(
                    position.ticket, round(_price, digits), request)
        # If Sell
        elif position.type == 1 and position.price_current < position.price_open:
            break_even_level = position.price_open - (be * point)
            break_even_price = position.price_open - \
                ((fees_points + spread) * point)
            _price = break_even_price if price is None else price
            _level = break_even_level if level is None else level

            if self.get_tick_info(self.symbol).bid < _level:
                # Set the stop loss to break even
                request = {
                    "action": Mt5.TRADE_ACTION_SLTP,
                    "type": Mt5.ORDER_TYPE_BUY_STOP,
                    "position": position.ticket,
                    "sl": round(_price, digits),
                    "tp": position.tp
                }
                self._break_even_request(
                    position.ticket, round(_price, digits), request)

    def _break_even_request(self, tiket, price, request):
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
            check_result = self.check_order(request)
            result = self.send_order(request)
        except Exception as e:
            print(f"{self.current_datetime()} -", end=' ')
            trade_retcode_message(
                result.retcode, display=True, add_msg=f"{e}{addtionnal}")
        if result.retcode != Mt5.TRADE_RETCODE_DONE:
            msg = trade_retcode_message(result.retcode)
            logger.error(
                f"Break-Even Order Request, Position: #{tiket}, RETCODE={result.retcode}: {msg}{addtionnal}")
            tries = 0
            while result.retcode != Mt5.TRADE_RETCODE_DONE and tries < 10:
                if result.retcode == Mt5.TRADE_RETCODE_NO_CHANGES:
                    break
                else:
                    time.sleep(1)
                    try:
                        check_result = self.check_order(request)
                        result = self.send_order(request)
                    except Exception as e:
                        print(f"{self.current_datetime()} -", end=' ')
                        trade_retcode_message(
                            result.retcode, display=True, add_msg=f"{e}{addtionnal}")
                    if result.retcode == Mt5.TRADE_RETCODE_DONE:
                        break
                tries += 1
        if result.retcode == Mt5.TRADE_RETCODE_DONE:
            msg = trade_retcode_message(result.retcode)
            logger.info(f"Break-Even Order {msg}{addtionnal}")
            info = (
                f"Stop loss set to Break-even, Position: #{tiket}, Symbol: {self.symbol}, Price: @{price}"
            )
            logger.info(info)
            self.break_even_status.append(tiket)

    def win_trade(self,
                  position: TradePosition,
                  th: Optional[int] = None) -> bool:
        """
        Check if a positon is wining or looing
        wen it is closed before be level , tp or sl.

        Args:
            th (int): The minimum profit for a position in point
        """
        size = self.get_symbol_info(self.symbol).trade_tick_size
        value = self.get_symbol_info(self.symbol).trade_tick_value
        points = position.profit * (size / value / position.volume)

        spread = self.get_symbol_info(self.symbol).spread
        point = self.get_symbol_info(self.symbol).point
        fees = self.get_stats()[0]["average_fee"] * -1
        risk = self.currency_risk()["trade_profit"]
        min_be = round((fees / risk)) + 2
        be = self.get_break_even()
        if th is not None:
            win_be = th
        else:
            win_be = max(min_be, round((0.1 * be)))
        win_trade = float(points/point) >= be
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
        target = round((balance * self.target)/100, 2)
        if len(self.opened_positions) != 0:
            for position in self.opened_positions:
                time.sleep(0.1)
                # This return two TradeDeal Object,
                # The first one is the one the opening order
                # The second is the closing order
                history = self.get_trades_history(
                    position=position, to_df=False
                )
                if len(history) == 2:
                    profit += history[1].profit
                    commission += history[0].commission
                    swap += history[0].swap
                    fee += history[0].fee
            current_profit = profit + commission + fee + swap
            if current_profit >= target:
                return True
        return False

    def close_position(self,
                       ticket: int,
                       id: Optional[int] = None,
                       pct: Optional[float] = 1.0,
                       comment: Optional[str] = None
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
        positions = self.get_positions(symbol=self.symbol)
        buy_price = self.get_tick_info(self.symbol).ask
        sell_price = self.get_tick_info(self.symbol).bid
        digits = self.get_symbol_info(self.symbol).digits
        deviation = self.get_deviation()
        if positions is not None:
            for position in positions:
                if (position.ticket == ticket
                        and position.magic == Id
                    ):
                    buy = position.type == 0
                    sell = position.type == 1
                    request = {
                        "action": Mt5.TRADE_ACTION_DEAL,
                        "symbol": self.symbol,
                        "volume": (position.volume*pct),
                        "type": Mt5.ORDER_TYPE_SELL if buy else Mt5.ORDER_TYPE_BUY,
                        "position": ticket,
                        "price": sell_price if buy else buy_price,
                        "deviation": deviation,
                        "magic": Id,
                        "comment": f"@{self.expert_name}" if comment is None else comment,
                        "type_time": Mt5.ORDER_TIME_GTC,
                        "type_filling": Mt5.ORDER_FILLING_FOK,
                    }
                    addtionnal = f", SYMBOL={self.symbol}"
                    try:
                        check_result = self.check_order(request)
                        result = self.send_order(request)
                    except Exception as e:
                        print(f"{self.current_datetime()} -", end=' ')
                        trade_retcode_message(
                            result.retcode, display=True, add_msg=f"{e}{addtionnal}")
                    if result.retcode != Mt5.TRADE_RETCODE_DONE:
                        msg = trade_retcode_message(result.retcode)
                        logger.error(
                            f"Closing Order Request, Position: #{ticket}, RETCODE={result.retcode}: {msg}{addtionnal}")
                        tries = 0
                        while result.retcode != Mt5.TRADE_RETCODE_DONE and tries < 5:
                            time.sleep(1)
                            try:
                                check_result = self.check_order(request)
                                result = self.send_order(request)
                            except Exception as e:
                                print(f"{self.current_datetime()} -", end=' ')
                                trade_retcode_message(
                                    result.retcode, display=True, add_msg=f"{e}{addtionnal}")
                            if result.retcode == Mt5.TRADE_RETCODE_DONE:
                                break
                            tries += 1
                    if result.retcode == Mt5.TRADE_RETCODE_DONE:
                        msg = trade_retcode_message(result.retcode)
                        logger.info(
                            f"Closing Order {msg}{addtionnal}")
                        info = (
                            f"Position #{ticket} closed, Symbol: {self.symbol}, Price: @{request['price']}")
                        logger.info(info)
                        return True
                    else:
                        return False

    def close_positions(
            self,
            position_type: Literal["all", "buy", "sell"] = "all",
            id: Optional[int] = None,
            comment: Optional[str] = None):
        """
        Args:
            position_type (str): Type of positions to close ("all", "buy", "sell")
            id (int): The unique ID of the Expert or Strategy
            comment (str): Comment for the closing position
        """
        if position_type == "all":
            positions = self.get_positions(symbol=self.symbol)
        elif position_type == "buy":
            positions = self.get_current_buys()
        elif position_type == "sell":
            positions = self.get_current_sells()
        else:
            logger.error(f"Invalid position type: {position_type}")
            return

        if positions is not None:
            if position_type == 'all':
                pos_type = ""
                tickets = [position.ticket for position in positions]
            else:
                tickets = positions
                pos_type = position_type
        else:
            tickets = []
            
        if len(tickets) != 0:
            for ticket in tickets.copy():
                if self.close_position(ticket, id=id, comment=comment):
                    tickets.remove(ticket)
                time.sleep(1)

            if len(tickets) == 0:
                logger.info(
                    f"ALL {position_type.upper()} Positions closed, SYMBOL={self.symbol}.")
            else:
                logger.info(
                    f"{len(tickets)} {position_type.upper()} Positions not closed, SYMBOL={self.symbol}")
        else:
            logger.info(
                f"No {position_type.upper()} Positions to close, SYMBOL={self.symbol}.")

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
        target = round((balance * self.target)/100)
        deals = len(self.opened_positions)
        if deals != 0:
            for position in self.opened_positions:
                time.sleep(0.1)
                history = self.get_trades_history(
                    position=position, to_df=False
                )
                if len(history) == 2:
                    result = history[1].profit
                    comm = history[0].commission
                    swap = history[0].swap
                    fee = history[0].fee
                    if (result + comm + swap + fee) <= 0:
                        loss_trades += 1
                    else:
                        win_trades += 1
                    profit += result
                    total_fees += (comm + swap + fee)
            average_fee = total_fees / deals
            win_rate = round((win_trades / deals) * 100, 2)
            stats1 = {
                "deals": deals,
                "profit": profit,
                "win_trades": win_trades,
                "loss_trades": loss_trades,
                "total_fees": total_fees,
                "average_fee": average_fee,
                "win_rate": win_rate
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
            stats2 = {
                "total_profit": total_profit,
                "profitability": profittable
            }
        else:
            stats2 = {
                "total_profit": 0,
                "profitability": 0
            }
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
        returns = profit.values
        returns = np.diff(returns, prepend=0.0)
        N = self.max_trade() * 252
        sharp = np.sqrt(N) * np.mean(returns) / (np.std(returns) + 1e-10)

        return round(sharp, 3)

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
            friday_time = datetime.strptime(self.current_time(), '%H:%M')
            monday_time = datetime.strptime(self.start, '%H:%M')
            intra_day_diff = (monday_time - friday_time).total_seconds() // 60
            inter_day_diff = 3 * 24 * 60
            total_minutes = inter_day_diff + intra_day_diff
            return total_minutes
        else:
            # claculate number of minute from the end to the start
            start = datetime.strptime(self.start, '%H:%M')
            end =  datetime.strptime(self.current_time(), '%H:%M')
            minutes = (end - start).total_seconds() // 60
            sleep_time = (24*60) - minutes
            return sleep_time

    def current_datetime(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def current_time(self, seconds=False):
        if seconds:
            return datetime.now().strftime("%H:%M:%S")
        return datetime.now().strftime("%H:%M")

def create_trade_instance(
        symbols: List[str],
        params: Dict[str, Any]) -> Dict[str, Trade]:
    """
    Creates Trade instances for each symbol provided.

    Args:
        symbols: A list of trading symbols (e.g., ['AAPL', 'MSFT']).
        params: A dictionary containing parameters for the Trade instance.

    Returns:
        A dictionary where keys are symbols and values are corresponding Trade instances.

    Raises:
        ValueError: If the 'symbols' list is empty or the 'params' dictionary is missing required keys.
    """
    instances = {}

    if not symbols:
        raise ValueError("The 'symbols' list cannot be empty.")
    for symbol in symbols:
        try:
            instances[symbol] = Trade(**params, symbol=symbol)
        except Exception as e:
            logger.error(f"Creating Trade instance, SYMBOL={symbol} {e}")
    return instances
