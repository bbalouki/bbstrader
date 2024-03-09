import os
import csv
from datetime import datetime
from mtrader5.risk import RiskManagement
import MetaTrader5 as Mt5
import numpy as np
import time


class Trade(RiskManagement):
    """
    Extends the RiskManagement class to include specific trading operations, 
    incorporating risk management strategies directly into trade executions.
    It offers functionalities to execute trades while managing risks 
    according to the inherited RiskManagement parameters and methods.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Trade class with the specified parameters.

        Parameters
        ==========
        :param symbol (str): The `symbol` that the expert advisor will trade.
        :param expert_name (str): The name of the `expert advisor`.
        :param expert_id (int): The `unique ID` used to identify the expert advisor 
            or the strategy used on the symbol.
        :param version (str): The `version` of the expert advisor.
        :param target (float): `Trading period (day, week, month) profit target` in percentage
        :param start_time (str): The` hour and minutes` that the expert advisor is able to start to run.
        :param finishing_time (str): The time after which no new position can be opened.
        :param ending_time (str): The time after which any open position will be closed.

        Inherits
        --------
            - max_risk, max_trades, rr, daily_risk, account_leverage, std_stop, pchange_sl, sl, tp, be
            See the RiskManagement class for more details on these parameters.

        """
        super().__init__(**kwargs)

        self.symbol = kwargs.get("symbol", "MSFT")
        self.expert_name = kwargs.get("expert_name", "bbstrader")
        self.expert_id = kwargs.get("expert_id", 9818)
        self.version = kwargs.get("version", "1.0")
        self.target = kwargs.get("target", 1.0)

        self.start = kwargs.get("start_time", "6:30")
        self.finishing = kwargs.get("finishing_time", "19:30")
        self.end = kwargs.get("ending_time", "20:30")

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
        self.break_even_status = []

        print("\nInitializing the basics.")
        self.initialize()
        self.select_symbol()
        self.prepare_symbol()
        print("Initialization successfully completed.")

        print()
        self.summary()
        time.sleep(1)
        print()
        self.risk_managment()
        self.statistics()
        print("║═════════════ Everything is oK : Running ....  ═══════║")

    def initialize(self):
        """
        Initializes the MetaTrader 5 (MT5) terminal for trading operations. 
        This method attempts to establish a connection with the MT5 terminal. 
        If the initial connection attempt fails due to a timeout, it retries after a specified delay. 
        Successful initialization is crucial for the execution of trading operations.

        Raises:
            Exception: If initialization fails even after retrying, 
            an exception is raised, and the terminal is shut down.
        """
        if not Mt5.initialize():
            if Mt5.last_error() == Mt5.RES_E_INTERNAL_FAIL_TIMEOUT:
                print("initialize() failed, error code =", Mt5.last_error())
                print("Trying again ....")
                time.sleep(60 * 3)
                if not Mt5.initialize():
                    print("initialize() failed, error code =", Mt5.last_error())
                    Mt5.shutdown()
        else:
            print(
                f"You are running the @{self.expert_name} Expert advisor,"
                f" Version @{self.version}, on {self.symbol}."
            )

    def select_symbol(self):
        """
        Selects the trading symbol in the MetaTrader 5 (MT5) terminal. 
        This method ensures that the specified trading 
        symbol is selected and visible in the MT5 terminal, 
        llowing subsequent trading operations such as opening and 
        closing positions on this symbol.

        Raises:
            Exception: If the symbol cannot be found or made visible in the MT5 terminal, 
            an exception is raised, and the terminal is shut down.
        """
        Mt5.symbol_select(self.symbol, True)

    def prepare_symbol(self):
        """
        Prepares the selected symbol for trading. 
        This method checks if the symbol is available and visible in the 
        MT5 terminal. If the symbol is not visible, it attempts to select the symbol again. 
        This step ensures that trading operations can be performed on the selected symbol without issues.

        Raises:
            Exception: If the symbol cannot be made visible for trading operations, 
            an exception is raised, and the terminal is shut down.
        """
        symbol_info = Mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"It was not possible to find {self.symbol}")
            Mt5.shutdown()
            print("Turned off")
            quit()

        if not symbol_info.visible:
            print(f"The {self.symbol} is not visible, needed to be switched on.")
            if not Mt5.symbol_select(self.symbol, True):
                print(
                    f"The expert advisor {self.expert_name} failed in select the symbol {self.symbol}, turning off."
                )
                Mt5.shutdown()
                print("Turned off")
                quit()

    def summary(self):
        """Show a brief description about the trading program"""
        print(
            "╔═════════════════ Summary  ════════════════════╗\n"
            f"║ Expert Advisor Name             @{self.expert_name}\n"
            f"║ Expert Advisor Version          @{self.version}\n"
            f"║ Magic Number                    {self.expert_id}\n"
            f"║ Running on Symbol               {self.symbol}\n"
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
            f"║ Volume                           {self.volume()} {currency}\n"
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

    def statistics(self, save=False, dir="stats"):
        """
        Print some statistics for the trading session and save to CSV if specified.

        Parameters
        ==========
        :param save (bool, optional): Whether to save the statistics to a CSV file.
                Defaults to False.
        :param dir (str, optional): The directory to save the CSV file. 
            Defaults to "stats".

        Returns
        =======
            None
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
            print(f"Session statistics saved to {filepath}")

    def open_buy_position(self, id=None, mm=True, comment=""):
        """
        Open a Buy positin

        Parameters
        ==========
        :param id (int) : The strategy id or expert Id
        :param mm (bool) : Weither to put stop loss and tp or not
        :param comment (str) : The comment for the closing position
        """
        Id = id if id is not None else self.expert_id
        point = Mt5.symbol_info(self.symbol).point
        price = Mt5.symbol_info_tick(self.symbol).ask
        digits = Mt5.symbol_info(self.symbol).digits

        lot = self.get_lot()
        stop_loss = self.get_stop_loss()
        take_profit = self.get_take_profit()
        deviation = self.get_deviation()
        if mm:
            request = {
                "action": Mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": float(lot),
                "type": Mt5.ORDER_TYPE_BUY,
                "price": price,
                "sl": price - stop_loss * point,
                "tp": price + take_profit * point,
                "deviation": deviation,
                "magic": Id,
                "comment": comment,
                "type_time": Mt5.ORDER_TIME_GTC,
                "type_filling": Mt5.ORDER_FILLING_FOK,
            }
        else:
            request = {
                "action": Mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": float(lot),
                "type": Mt5.ORDER_TYPE_BUY,
                "price": price,
                "tp": (price + take_profit * point),
                "deviation": deviation,
                "magic": Id,
                "comment": comment,
                "type_time": Mt5.ORDER_TIME_GTC,
                "type_filling": Mt5.ORDER_FILLING_FOK,
            }

        self.break_even(comment)
        if self.check(comment):
            self.request_result(price, request)

    def open_sell_position(self, id=None, mm=True, comment=""):
        """
        Open a sell positin

        Parameters
        ==========
        :param id (int) : The strategy id or expert Id
        :param mm (bool) : Weither to put stop loss and tp or not
        :param comment (str) : The comment for the closing position
        """
        Id = id if id is not None else self.expert_id
        point = Mt5.symbol_info(self.symbol).point
        price = Mt5.symbol_info_tick(self.symbol).bid
        digits = Mt5.symbol_info(self.symbol).digits

        lot = self.get_lot()
        print(lot)
        stop_loss = self.get_stop_loss()
        take_profit = self.get_take_profit()
        deviation = self.get_deviation()
        if mm:
            request = {
                "action": Mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": float(lot),
                "type": Mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": (price + stop_loss * point),
                "tp": (price - take_profit * point),
                "deviation": deviation,
                "magic": Id,
                "comment": comment,
                "type_time": Mt5.ORDER_TIME_GTC,
                "type_filling": Mt5.ORDER_FILLING_FOK,
            }
        else:
            request = {
                "action": Mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": float(lot),
                "type": Mt5.ORDER_TYPE_SELL,
                "price": price,
                "tp": (price - take_profit * point),
                "deviation": deviation,
                "magic": Id,
                "comment": comment,
                "type_time": Mt5.ORDER_TIME_GTC,
                "type_filling": Mt5.ORDER_FILLING_FOK,
            }
        self.break_even(comment)
        if self.check(comment):
            self.request_result(price, request)

    def request_result(self, price, request):
        """
        Check if a trading order has been sent correctly

        Paramters
        =========
        :param price (float) : Price for opening the position
        :param request (dict()): A trade request to sent to Mt5.order_sent()
            all detail in request can be found on:
              https://www.mql5.com/en/docs/python_metatrader5/mt5ordersend_py
        """
        # Send a trading request
        # Check the execution result
        try:
            result = Mt5.order_send(request)
        except Exception as e:
            print(e)
        tries = 0
        while result.retcode != Mt5.TRADE_RETCODE_DONE and tries < 5:
            if result.retcode == Mt5.TRADE_RETCODE_CONNECTION:
                print(
                    f"Something went wrong while opening positon #{result.order},"
                    f"error: {result.retcode} trying again"
                )
            time.sleep(1)
            try:
                result = Mt5.order_send(request)
            except Exception as e:
                print(e)
            tries += 1
        # Print the result
        if result.retcode == Mt5.TRADE_RETCODE_DONE:
            self.opened_positions.append(result.order)
            now = datetime.now().strftime("%H:%M:%S")
            print(
                f"\nOrder Sent On {self.symbol}, Lot(s): {result.volume}, "
                f"Sl: {self.get_stop_loss()}, Tp: {self.get_take_profit()} at {now}"
            )
            time.sleep(0.1)
            positions = Mt5.positions_get(symbol=self.symbol)
            for position in positions:
                if position.ticket == result.order:
                    if position.type == 0:
                        order_type = "Buy"
                        self.buy_positions.append(position.ticket)
                    else:
                        order_type = "Sell"
                        self.sell_positions.append(position.ticket)
                    profit = round(self.get_account_info().profit, 5)
                    print(
                        f"{order_type}, Position Opened at @{round(position.price_open,5)}, "
                        f"Sl: at @{position.sl} Tp: at @{position.tp}\n"
                    )
                    print(
                        f"== Open Positions on {self.symbol}: {len(positions)} == Open PnL: {profit} "
                        f"{self.get_account_info().currency}\n"
                    )

    def open_position(
        self,
        buy: bool = False,
        sell: bool = False,
        id: int = None,
        mm: bool = True,
        comment: str = ""
    ):
        """Open a buy or sell position.

        Parameters
        ==========
        :param buy (bool) : A boolean True or False
        :param sell (bool): A boolean True or False
        :param id (int) : The strategy id or expert Id
        :param mm (bool) : Weither to put stop loss and tp or not
        :param comment (str) : The comment for the closing position
        """
        if buy:
            self.open_buy_position(id=id, mm=mm, comment=comment)
        if sell:
            self.open_sell_position(id=id, mm=mm, comment=comment)

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

    def get_current_open_positions(self, id=None):
        """Get All current open position's tickets

        Parameters
        ==========
        :param id (int) : The strategy id or expert Id
        """
        positions = Mt5.positions_get(symbol=self.symbol)
        Id = id if id is not None else self.expert_id
        current_positions = []
        if len(positions) != 0:
            for position in positions:
                if position.magic == Id:
                    current_positions.append(position.ticket)
            if len(current_positions) != 0:
                return current_positions
            else:
                return None
        else:
            return None

    def get_current_win_trades(self, id=None):
        """Get all active profitable trades

        Parameters
        ==========
        :param id (int) : The strategy id or expert Id
        """
        positions = Mt5.positions_get(symbol=self.symbol)
        Id = id if id is not None else self.expert_id
        be_positions = []
        if len(positions) != 0:
            for position in positions:
                if position.magic == Id:
                    if self.win_trade(position, th=self.stop_loss):
                        be_positions.append(position.ticket)
            if len(be_positions) != 0:
                return be_positions
            else:
                return None
        else:
            return None

    def positive_profit(self, th: float = None):
        """
        Check is the total profit on current open positions
        Is greater than a minimum profit express as percentage 
        of the profit target.

        Parameters
        ==========
        :param th (float) : The minimum profit target on current positions
        """
        positions = self.get_current_open_positions()
        profit = 0.0
        balance = self.get_account_info().balance
        target = round((balance * self.target)/100, 2)
        if positions is not None:
            for position in positions:
                time.sleep(0.1)
                history = Mt5.positions_get(
                    ticket=position
                )
                profit += history[0].profit
            fees = self.get_stats()[0]["average_fee"] * len(positions)
            current_profit = profit + fees
            th_profit = (target*th)/100 if th is not None else (target*0.01)
            if current_profit > th_profit:
                return True
        return False

    def get_current_buys(self, id=None):
        """Get current buy positions open

        Parameters
        ==========
        :param id (int) : The strategy id or expert Id
        """
        positions = Mt5.positions_get(symbol=self.symbol)
        Id = id if id is not None else self.expert_id
        buys = []
        if len(positions) != 0:
            for position in positions:
                if (position.type == 0
                        and position.magic == Id
                        ):
                    buys.append(position.ticket)
            if len(buys) != 0:
                return buys
            else:
                return None
        else:
            return None

    def get_current_sells(self, id=None):
        """Get current sell positions open

        Parameters
        ==========
        :param id (int) : The strategy id or expert Id
        """
        positions = Mt5.positions_get(symbol=self.symbol)
        Id = id if id is not None else self.expert_id
        sells = []
        if len(positions) != 0:
            for position in positions:
                if (position.type == 1
                        and position.magic == Id
                    ):
                    sells.append(position.ticket)
            if len(sells) != 0:
                return sells
            else:
                return None
        else:
            return None

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

        Parameters
        ==========
        :param comment (str) : The comment for the closing position
        """
        if self.days_end():
            return False
        elif not self.trading_time():
            print("\nSorry It is not time to trade yet.")
            return False
        elif not self.is_risk_ok():
            print(f"\nSorry you can't trade, Please Check the Account balance")
            self._check()
            return False
        elif not self._risk_free():
            print("Sorry you can't take another trade for today")
            print("Check you strategy and come back tomorrow !!!!")
            self._check()
            return False
        elif self.profit_target():
            self._check(
                f'Congratulations, the profit target is reached for today !!!')
        return True

    def _check(self, txt: str = ""):
        if self.positive_profit() or self.get_current_open_positions() is None:
            self.close_all_positions()
            print(txt)
            time.sleep(5)
            self.statistics(save=True)
            time.sleep(5)
            quit()

    def break_even(self, id=None, comment=""):
        """
        Checks if it's time to put the break even,
        if so , it will sets the break even ,and if the break even was already set,
        it checks if the price has moved in favorable direction,
        if so , it set the new break even.

        Parameters
        ==========
        :param id(int) : The strategy Id or Expert Id
        :param comment (str) : The comment for the closing position
        """
        time.sleep(0.1)
        Id = id if id is not None else self.expert_id
        positions = Mt5.positions_get(symbol=self.symbol)
        be = self.get_break_even()
        if len(positions) != 0:
            for position in positions:
                if position.magic == Id:
                    size = Mt5.symbol_info(self.symbol).trade_tick_size
                    value = Mt5.symbol_info(self.symbol).trade_tick_value
                    point = Mt5.symbol_info(self.symbol).point
                    digits = Mt5.symbol_info(self.symbol).digits
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

    def set_break_even(self, position, be, price=None, level=None):
        """
        Sets the break-even level for a given trading position.

        Parameters
        ==========
        :param position (namedtuple):
            The trading position for which the break-even is to be set.
        :param be (int): The break-even level in points.
        :param level (float): The break-even level in price
            if set to None , it will be calated automaticaly.
        :param price (float): The break-even price
            if set to None , it will be calated automaticaly.
        """
        point = Mt5.symbol_info(self.symbol).point
        digits = Mt5.symbol_info(self.symbol).digits
        spread = Mt5.symbol_info(self.symbol).spread
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

            if Mt5.symbol_info_tick(self.symbol).ask > level:
                # Set the stop loss to break even
                request = {
                    "action": Mt5.TRADE_ACTION_SLTP,
                    "type": Mt5.ORDER_TYPE_SELL_STOP,
                    "position": position.ticket,
                    "sl": round(_price, digits)
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

            if Mt5.symbol_info_tick(self.symbol).bid < level:
                # Set the stop loss to break even
                request = {
                    "action": Mt5.TRADE_ACTION_SLTP,
                    "type": Mt5.ORDER_TYPE_BUY_STOP,
                    "position": position.ticket,
                    "sl": round(_price, digits)
                }
                self._break_even_request(
                    position.ticket, round(_price, digits), request)

    def _break_even_request(self, tiket, price, request):
        """
        Send a request to set the stop loss to break even for a given trading position.

        Parameters
        ==========
        :param tiket (int): The ticket number of the trading position.
        :param price (float): The price at which the stop loss is to be set.
        :param request (dict): The request to set the stop loss to break even.
        """
        time.sleep(0.1)
        try:
            result = Mt5.order_send(request)
        except Exception as e:
            print(e)
        tries = 0
        while result.retcode != Mt5.TRADE_RETCODE_DONE and tries < 10:
            if result.retcode == Mt5.TRADE_RETCODE_NO_CHANGES:
                break
            elif result.retcode == Mt5.TRADE_RETCODE_CONNECTION:
                print(
                    "Unable to set break-even for Position: "
                    f"#{tiket}, error: {result.retcode} trying again"
                )
                time.sleep(1)
                try:
                    result = Mt5.order_send(request)
                except Exception as e:
                    print(e)
                tries += 1
        if result.retcode == Mt5.TRADE_RETCODE_DONE:
            print(
                f"Stop loss set to break even for Position: #{tiket} at @{price}"
            )
            self.break_even_status.append(tiket)

    def win_trade(self, position, th: int = None) -> bool:
        """
        Check if a positon is wining or looing
        wen it is closed before be level , tp or sl.

        Parameters
        ==========
        :param th (int) : The minimum profit for a position in point
        """
        size = Mt5.symbol_info(self.symbol).trade_tick_size
        value = Mt5.symbol_info(self.symbol).trade_tick_value
        points = position.profit * (size / value / position.volume)

        spread = Mt5.symbol_info(self.symbol).spread
        point = Mt5.symbol_info(self.symbol).point
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
                history = Mt5.history_deals_get(
                    position=position
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

    def close_position(self, ticket, id=None, pct=1.0, comment=""):
        """
        Close an open position by it ticket

        Parameters
        ==========
        :param ticket (int) : Positon ticket to close
        :param id (int) : The unique ID of the Expert or Strategy
        :param pct (float) : Percentage of the position to close
        :param comment (str) : Comment for the closing position
        """
        # get all Actives positions
        time.sleep(0.1)
        Id = id if id is not None else self.expert_id
        positions = Mt5.positions_get(symbol=self.symbol)
        buy_price = Mt5.symbol_info_tick(self.symbol).ask
        sell_price = Mt5.symbol_info_tick(self.symbol).bid
        digits = Mt5.symbol_info(self.symbol).digits
        deviation = self.get_deviation()
        if len(positions) != 0:
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
                        "comment": comment,
                        "type_time": Mt5.ORDER_TIME_GTC,
                        "type_filling": Mt5.ORDER_FILLING_FOK,
                    }
                    result = Mt5.order_send(request)
                    tries = 0
                    while result.retcode != Mt5.TRADE_RETCODE_DONE and tries < 5:
                        if result.retcode == Mt5.TRADE_RETCODE_CONNECTION:
                            print(
                                f"Unable to close position: #{ticket}, "
                                f"error: {result.retcode} trying again"
                            )
                        time.sleep(1)
                        result = Mt5.order_send(request)
                        tries += 1
                    if result.retcode == Mt5.TRADE_RETCODE_DONE:
                        print(
                            f"Position #{ticket} closed at @{request['price']}")

    def close_all_positions(self, id=None, comment=""):
        positions = Mt5.positions_get(symbol=self.symbol)
        while len(positions) != 0:
            positions = Mt5.positions_get(symbol=self.symbol)
            for position in positions:
                self.close_position(position.ticket, id=id, comment=comment)
        if len(positions) == 0:
            print(f"\nAll positions closed on {self.symbol}.\n")

    def close_all_buys(self, id=None, comment=""):
        positions = self.get_current_buys()
        if positions is not None:
            for position in positions:
                self.close_position(position, id=id, comment=comment)
                positions.remove(position)
        if positions is None or len(positions) == 0:
            print(f"\nAll Buy positions closed on {self.symbol}.\n")
        else:
            print(f"\nUnable to close all Buy positons on {self.symbol}\n")

    def close_all_sells(self, id=None, comment=""):
        positions = self.get_current_sells()
        if positions is not None:
            for position in positions:
                self.close_position(position, id=id, comment=comment)
                positions.remove(position)
        if positions is None or len(positions) == 0:
            print(f"\nAll Sell positions closed on {self.symbol}.\n")
        else:
            print(f"\nUnable to close all Sell positons on {self.symbol}\n")

    def get_stats(self) -> tuple:
        """
        get some stats about the trading day and trading history

        :return: tuple[dict()]
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
                history = Mt5.history_deals_get(
                    position=position
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
        df = self.get_trade_history()
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
        df2 = self.get_trade_history()
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
            friday_time = datetime.strptime(self.end, '%H:%M')
            monday_time = datetime.strptime(self.start, '%H:%M')
            intra_day_diff = (monday_time - friday_time).total_seconds() // 60
            inter_day_diff = 3 * 24 * 60
            total_minutes = inter_day_diff + intra_day_diff
            return total_minutes
        else: 
            # claculate number of minute from the end and to the start 
            start = datetime.strptime(self.start, '%H:%M')
            end = datetime.strptime(self.end, '%H:%M')
            minutes = (end - start).total_seconds() // 60
            sleep_time = (24*60) - minutes
            return sleep_time


