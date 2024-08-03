from bbstrader.metatrader.account import Account
from bbstrader.metatrader.rates import Rates
import MetaTrader5 as Mt5
import numpy as np
from scipy.stats import norm

from datetime import datetime
import time
import random
import re


TF_MAPPING = {
    '1m':  1,    # 1 minute intervals
    '3m':  3,    # 3 minute intervals
    '5m':  5,    # 5 minute intervals
    '10m': 10,   # 10 minute intervals
    '15m': 15,   # 15 minute intervals
    '30m': 30,   # 30 minute intervals
    '1h':  60,   # 1 hour intervals
    '2h':  120,  # 2hour intervals
    '4h':  240,  # 4 hour intervals
    'D1':  1440  # 1 day intervals
}


class RiskManagement(Account):
    """
    The RiskManagement class provides foundational 
    risk management functionalities for trading activities.
    It calculates risk levels, determines stop loss and take profit levels, 
    and ensures trading activities align with predefined risk parameters.

    Exemple:
    >>> risk_manager = RiskManagement(
        symbol="EURUSD", 
        max_risk=5.0, 
        daily_risk=2.0, 
        max_trades=10, 
        std_stop=True, 
        account_leverage=True, 
        start_time="09:00", 
        finishing_time="17:00", 
        time_frame="1h"
    )
    >>> # Calculate risk level
    >>> risk_level = risk_manager.risk_level()

    >>> # Get appropriate lot size for a trade
    >>> lot_size = risk_manager.get_lot()

    >>> # Determine stop loss and take profit levels
    >>> stop_loss = risk_manager.get_stop_loss()
    >>> take_profit = risk_manager.get_take_profit()

    >>> # Check if current risk is acceptable
    >>> is_risk_acceptable = risk_manager.is_risk_ok()
    """

    def __init__(self, **kwargs):
        """
        Initialize the RiskManagement class to manage risk in trading activities.

        Args:
            symbol (str): The symbol of the financial instrument to trade.
            max_risk (float): The `maximum risk allowed` on the trading account.
            aily_risk (float): `Daily Max risk allowed`
                If Set to None it will be determine based on Maximum risk.
            max_trades (int): Maximum number of trades in a trading session
                If set to None it will be determine based on the timeframe of trading. 
            std_stop (bool): If set to True , the Stop loss is claculated based
                On `historical volatility` of the trading instrument.
            pchange_sl (float): If set , the Stop loss is claculated based
                On `percentage change` of the trading instrument.
            account_leverage (bool): If set to True the account leverage will be used
                In risk managment setting.
            ime_frame (str): The time frame on which the program is working
                `(1m, 3m, 5m, 10m, 15m, 30m, 1h, 2h, 4h, D1)`.
            start_time (str): The starting time for the trading strategy 
                `(HH:MM, H an M do not star with 0)`.
            finishing_time (str): The finishing time for the trading strategy
                `(HH:MM, H an M do not star with 0)`.
            sl (int, optional): Stop Loss in points, Must be a positive number.
            tp (int, optional): Take Profit in points, Must be a positive number.
            be (int, optional): Break Even in points, Must be a positive number.
            rr (float, optional): Risk reward ratio, Must be a positive number.
        """
        super().__init__()
        self.symbol = kwargs.get("symbol")
        self.start_time = kwargs.get("start_time", "6:30")
        self.finishing_time = kwargs.get("finishing_time", "19:30")
        self.max_trades = kwargs.get("max_trades")
        self.std = kwargs.get("std_stop", False)
        self.pchange = kwargs.get("pchange_sl")
        self.daily_dd = kwargs.get("daily_risk")
        self.max_risk = kwargs.get("max_risk", 10.0)
        self.rr = kwargs.get("rr", 1.5)
        self.sl = kwargs.get('sl')
        self.tp = kwargs.get('tp')
        self.be = kwargs.get('be')
        self.account_leverage = kwargs.get("account_leverage", False)
        self.symbol_info = super().get_symbol_info(self.symbol)
        if self.daily_dd is not None:
            if self.daily_dd < 0:
                raise ValueError("daily_risk must be positive number")
        if self.max_risk <= 0:
            raise ValueError("max_risk must be a positive number")

        # Validation for sl, tp, and be could be added here
        if self.sl is not None and not isinstance(self.sl, int):
            raise ValueError("sl must be an integer number")
        if self.tp is not None and not isinstance(self.tp, int):
            raise ValueError("tp must be an integer number")
        if self.be is not None:
            if not isinstance(self.be, int) or self.be <= 0:
                raise ValueError("be must be a positive integer number")

        self.timeframe = kwargs.get("time_frame", 'D1')
        if self.timeframe not in TF_MAPPING:
            raise ValueError("Unsupported time frame")
        elif self.timeframe == 'D1':
            tf = self.get_minutes()
        else:
            tf = TF_MAPPING[self.timeframe]
        self.TF = tf

    def risk_level(self) -> float:
        """
        Calculates the risk level of a trade

        Returns:
        -   Risk level in the form of a float percentage.
        """
        account_info = self.get_account_info()
        balance = account_info.balance
        equity = account_info.equity
        df = self.get_trade_history()
        if df is None:
            profit = 0
        else:
            profit_df = df.iloc[1:]
            profit = profit_df['profit'].sum()
            commisions = df['commission'].sum()
            fees = df['fee'].sum()
            swap = df['swap'].sum()
            total_profit = commisions + fees + swap + profit
            initial_balance = balance - total_profit
            if balance != 0:
                risk_alowed = (((equity-initial_balance)/equity)*100) * -1
                return round(risk_alowed, 2)
        return 0.0

    def get_lot(self) -> float:
        """"Get the approprite lot size for a trade"""
        s_info = self.symbol_info
        volume_step = s_info.volume_step
        lot = self.currency_risk()['lot']
        steps = self._volume_step(volume_step)
        if steps >= 2:
            return round(lot, steps)
        else:
            return round(lot)

    def _volume_step(self, value):
        """Get the number of decimal places in a number"""

        value_str = str(value)

        if '.' in value_str:
            decimal_index = value_str.index('.')
            num_digits = len(value_str) - decimal_index - 1

            return num_digits
        elif value_str == '1':
            return 1
        else:
            return 0

    def max_trade(self) -> int:
        """calculates the maximum number of trades allowed"""
        minutes = self.get_minutes()
        if self.max_trades is not None:
            max_trades = self.max_trades
        else:
            max_trades = round(minutes / self.TF)
        return max(max_trades, 1)

    def get_minutes(self) -> int:
        """calculates the number of minutes between two times"""

        start = datetime.strptime(self.start_time, '%H:%M')
        end = datetime.strptime(self.finishing_time, '%H:%M')
        return (end - start).total_seconds() // 60

    def get_hours(self) -> int:
        """Calculates the number of hours between two times"""

        start = datetime.strptime(self.start_time, '%H:%M')
        end = datetime.strptime(self.finishing_time, '%H:%M')
        # Calculate the difference in hours
        hours = (end - start).total_seconds() // 3600  # 1 hour = 3600 seconds

        return hours

    def get_stop_loss(self) -> int:
        """calculates the stop loss of a trade in points"""
        if self.sl is not None:
            return self.sl
        elif self.sl is None and self.std:
            sl = self.get_std_stop()
            return sl
        elif self.sl is None and not self.std:
            risk = self.currency_risk()
            if risk['trade_loss'] != 0:
                sl = round((risk['currency_risk'] / risk['trade_loss']))
                return (sl) if sl > 0 else 0
            return 0

    def get_std_stop(self, tf: str = 'D1', interval: int = 252):
        """
        Calculate the standard deviation-based stop loss level 
        for a given financial instrument.

        Args:
            tf (str): Timeframe for data, default is 'D1' (Daily).
            interval (int): Number of historical data points to consider 
                for calculating standard deviation, default is 252.

        Returns:
        -   Standard deviation-based stop loss level, rounded to the nearest point. 
        -   0 if the calculated stop loss is less than or equal to 0.
        """
        rate = Rates(self.symbol, tf, 0, interval)
        data = rate.get_rates()
        returns = np.diff(data['Close'])
        std = np.std(returns)
        point = Mt5.symbol_info(self.symbol).point
        av_price = (self.symbol_info.bid + self.symbol_info.ask)/2
        price_interval = av_price * ((100-std))/100
        sl_point = float((av_price - price_interval) / point)
        sl = round(sl_point)

        return sl if sl > 0 else 0

    def get_pchange_stop(self, pchange):
        """
        Calculate the percentage change-based stop loss level 
        for a given financial instrument.

        Args:
            pchange (float): Percentage change in price to use for calculating stop loss level.
                If pchange is set to None, the stop loss is calculate using std.

        Returns:
        -   Percentage change-based stop loss level, rounded to the nearest point. 
        -   0 if the calculated stop loss is less than or equal to 0.
        """
        if pchange is not None:
            av_price = (self.symbol_info.bid + self.symbol_info.ask)/2
            price_interval = av_price*((100-pchange))/100
            point = Mt5.symbol_info(self.symbol).point
            sl_point = float((av_price - price_interval) / point)
            sl = round(sl_point)
            return sl if sl > 0 else 0
        else:
            # Use std as default pchange
            return self.get_std_stop()

    def calculate_var(self, tf='D1', interval=252, c=0.95):
        """
        Calculate Value at Risk (VaR) for a given portfolio.

        Args:
            tf (str): Time frame to use to calculate volatility.
            interval (int): How many periods to use based on time frame.
            c (float): Confidence level for VaR calculation (default is 95%).

        Returns:
        -   VaR value
        """
        rate = Rates(self.symbol, tf, 0, interval)
        prices = rate.get_rates()
        prices['return'] = prices['Close'].pct_change()
        prices.dropna(inplace=True)
        P = self.get_account_info().margin_free
        mu = np.mean(prices['return'])
        sigma = np.std(prices['return'])
        var = self.var_cov_var(P, c, mu, sigma)
        return var

    def var_cov_var(self, P, c, mu, sigma):
        """
        Variance-Covariance calculation of daily Value-at-Risk.

        Args:
            P (float): Portfolio value in USD.
            c (float): Confidence level for Value-at-Risk,e.g., 0.99 for 99% confidence interval.
            mu (float): Mean of the returns of the portfolio.
            sigma (float): Standard deviation of the returns of the portfolio.

        Returns:
        -   float: Value-at-Risk for the given portfolio.
        """
        alpha = norm.ppf(1 - c, mu, sigma)
        return P - P * (alpha + 1)

    def var_loss_value(self):
        """
        Calculate the stop-loss level based on VaR.
        """
        P = self.get_account_info().margin_free
        trade_risk = self.get_trade_risk()
        loss_allowed = P * trade_risk
        var = self.calculate_var()
        return min(var, loss_allowed)

    def get_take_profit(self) -> int:
        """calculates the take profit of a trade in points"""
        if self.tp is not None:
            return self.tp + 10
        else:
            risk = self.currency_risk()
            if risk['trade_profit'] != 0:
                tp = round((risk['currency_risk'] /
                            risk['trade_profit']) * self.rr)
                return (tp+10) if tp > 0 else 0
            return 0

    def get_currency_risk(self) -> float:
        """calculates the currency risk of a trade"""
        return round(self.currency_risk()['currency_risk'], 2)

    def expected_profit(self):
        """Calculate the expected profit per trade"""
        tp = self.get_take_profit()
        trade_profit = self.currency_risk()['trade_profit']
        return round(tp*trade_profit, 2)

    def volume(self):
        """Volume per trade"""

        return self.currency_risk()['volume']

    def currency_risk(self) -> dict:
        """
        calculates the currency risk of a trade

        Returns:
            A dictionary containing the following keys:
                `'currency_risk'`: dollars amount risk on a single trade,
                `'trade_loss'`: Loss value per tick in dollars,
                `'trade_profit'`: Profit value per tick in dollars,
                `'volume'`: Contract size * average price,
                `'lot'`: Lot size per trade
        """
        account_info = self.get_account_info()
        s_info = self.symbol_info

        laverage = self.get_leverage(self.account_leverage)
        contract_size = s_info.trade_contract_size
        av_price = (s_info.bid + s_info.ask)/2

        trade_risk = self.get_trade_risk()
        if trade_risk > 0:
            currency_risk: float = round(self.var_loss_value(), 5)
            volume: float = currency_risk*laverage
            _lot: float = round((volume / (contract_size * av_price)), 2)
            lot = self._check_lot(_lot)

            tick_value = float(s_info.trade_tick_value)
            tick_value_loss = float(s_info.trade_tick_value_loss)
            tick_value_profit = float(s_info.trade_tick_value_profit)
            point = float(s_info.point)
            if self.sl is not None:
                trade_loss = (currency_risk/self.sl)
                if self.tp is not None:
                    trade_profit = (currency_risk*(self.tp//self.sl))/self.tp
                else:
                    trade_profit = (currency_risk*self.rr)/(self.sl*self.rr)
                lot_ = round(trade_loss/(contract_size*tick_value_loss), 2)
                lot = self._check_lot(lot_)
                volume = round(lot * contract_size * av_price)

            elif self.std and self.pchange is None and self.sl is None:
                sl = self.get_std_stop()
                infos = self._std_pchange_stop(
                    currency_risk, sl, contract_size, tick_value_loss)
                trade_loss, trade_profit, lot, volume = infos

            elif self.pchange is not None and not self.std and self.sl is None:
                sl = self.get_pchange_stop(self.pchange)
                infos = self._std_pchange_stop(
                    currency_risk, sl, contract_size, tick_value_loss)
                trade_loss, trade_profit, lot, volume = infos

            else:
                trade_loss = (lot * contract_size) * tick_value_loss
                trade_profit = (lot * contract_size) * tick_value_profit

            return {'currency_risk': currency_risk,
                    'trade_loss': trade_loss,
                    'trade_profit': trade_profit,
                    'volume': round(volume),
                    'lot': lot
                    }
        else:
            return {'currency_risk': 0,
                    'trade_loss': 0,
                    'trade_profit': 0,
                    'volume': 0,
                    'lot': 0.01
                    }

    def _std_pchange_stop(self, currency_risk, sl, size, loss):
        """
        Calculate the stop loss level based on standard deviation or percentage change.

        Args:
            currency_risk (float): The amount of risk in dollars.
            sl (int): Stop loss level in points.
            size (int): Contract size.
            loss (float): Loss value per tick in dollars.

        """
        trade_loss = currency_risk/sl
        trade_profit = (currency_risk*self.rr)/(sl*self.rr)
        av_price = (self.symbol_info.bid + self.symbol_info.ask)/2
        _lot = round(trade_loss/(size*loss), 2)
        lot = self._check_lot(_lot)
        volume = round(lot * size*av_price)

        return trade_loss, trade_profit, lot, volume

    def _check_lot(self, lot: float) -> float:
        """
        Check if the lot size is within the allowed range
        and return the appropriate lot size
        """
        s_info = self.symbol_info
        volume_min = s_info.volume_min
        volume_max = s_info.volume_max
        if lot >= volume_max:
            return volume_max
        elif lot < volume_min:
            return volume_min
        else:
            return lot

    def get_trade_risk(self):
        """Calculate risk per trade as percentage"""
        total_risk = self.risk_level()
        max_trades = self.max_trade()
        if total_risk < self.max_risk:
            if self.daily_dd is not None:
                trade_risk = (self.daily_dd / max_trades)
            else:
                trade_risk = ((self.max_risk - total_risk) / max_trades)
            return trade_risk
        else:
            return 0

    def get_leverage(self, account: bool) -> int:
        """
        get the Laverage for each symbol

        Args:
            account (bool): If set to True, the account leverage will be used
                in risk managment setting
        """
        if account:
            account_info = self.get_account_info()
            return account_info.leverage
        else:
            s_info = self.symbol_info
            volume_min = s_info.volume_min
            contract_size = s_info.trade_contract_size
            av_price = (s_info.bid + s_info.ask)//2
            action = random.choice(
                [Mt5.ORDER_TYPE_BUY, Mt5.ORDER_TYPE_SELL]
            )
            margin = Mt5.order_calc_margin(
                action, self.symbol, volume_min, av_price
            )
            leverage = (
                volume_min * contract_size * av_price
            ) // margin
            return round(leverage)

    def get_deviation(self) -> int:
        return self.symbol_info.spread

    def get_break_even(self) -> int:
        if self.be is not None:
            return self.be
        else:
            stop = self.get_stop_loss()
            spread = self.get_symbol_info(self.symbol).spread
            if stop <= 100:
                be = round((stop + spread) * 0.5)
            elif stop > 100 and stop <= 150:
                be = round((stop + spread) * 0.35)
            elif stop > 150:
                be = round((stop + spread) * 0.25)
            return be

    def is_risk_ok(self) -> bool:
        if self.risk_level() <= self.max_risk:
            return True
        else:
            return False
