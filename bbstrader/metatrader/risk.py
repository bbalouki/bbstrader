import random
import numpy as np
from scipy.stats import norm
from datetime import datetime
import MetaTrader5 as Mt5
from bbstrader.metatrader.account import Account
from bbstrader.metatrader.rates import Rates
from bbstrader.metatrader.utils import (
    TIMEFRAMES, raise_mt5_error, TimeFrame,
    _ADMIRAL_MARKETS_FUTURES_, _COMMD_SUPPORTED_
)
from typing import List, Dict, Optional, Literal, Union, Any


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

    def __init__(
        self,
        symbol: str,
        max_risk: float,
        daily_risk: Optional[float] = None,
        max_trades: Optional[int] = None,
        std_stop: bool = False,
        pchange_sl: Optional[float] = None,
        account_leverage: bool = True,
        time_frame: TimeFrame = 'D1',
        start_time: str = "6:30",
        finishing_time: str = "19:30",
        sl: Optional[int] = None,
        tp: Optional[int] = None,
        be: Optional[int] = None,
        rr: float = 1.5,
        **kwargs
    ):
        """
        Initialize the RiskManagement class to manage risk in trading activities.

        Args:
            symbol (str): The symbol of the financial instrument to trade.
            max_risk (float): The `maximum risk allowed` on the trading account.
            daily_risk (float, optional): `Daily Max risk allowed`.
                If Set to None it will be determine based on Maximum risk.
            max_trades (int, optional): Maximum number of trades in a trading session.
                If set to None it will be determine based on the timeframe of trading. 
            std_stop (bool, optional): If set to True, the Stop loss is calculated based
                On `historical volatility` of the trading instrument. Defaults to False.
            pchange_sl (float, optional): If set, the Stop loss is calculated based
                On `percentage change` of the trading instrument.
            account_leverage (bool, optional): If set to True the account leverage will be used
                In risk management setting. Defaults to False.
            time_frame (str, optional): The time frame on which the program is working
                `(1m, 3m, 5m, 10m, 15m, 30m, 1h, 2h, 4h, D1)`. Defaults to 'D1'.
            start_time (str, optional): The starting time for the trading strategy 
                `(HH:MM, H an M do not star with 0)`. Defaults to "6:30".
            finishing_time (str, optional): The finishing time for the trading strategy
                `(HH:MM, H an M do not star with 0)`. Defaults to "19:30".
            sl (int, optional): Stop Loss in points, Must be a positive number.
            tp (int, optional): Take Profit in points, Must be a positive number.
            be (int, optional): Break Even in points, Must be a positive number.
            rr (float, optional): Risk reward ratio, Must be a positive number. Defaults to 1.5.
        """
        super().__init__()

        # Validation
        if daily_risk is not None and daily_risk < 0:
            raise ValueError("daily_risk must be a positive number")
        if max_risk <= 0:
            raise ValueError("max_risk must be a positive number")
        if sl is not None and not isinstance(sl, int):
            raise ValueError("sl must be an integer number")
        if tp is not None and not isinstance(tp, int):
            raise ValueError("tp must be an integer number")
        if be is not None and (not isinstance(be, int) or be <= 0):
            raise ValueError("be must be a positive integer number")
        if time_frame not in TIMEFRAMES:
            raise ValueError("Unsupported time frame")

        self.symbol = symbol
        self.start_time = start_time
        self.finishing_time = finishing_time
        self.max_trades = max_trades
        self.std = std_stop
        self.pchange = pchange_sl
        self.daily_dd = daily_risk
        self.max_risk = max_risk
        self.rr = rr
        self.sl = sl
        self.tp = tp
        self.be = be

        self.account_leverage = account_leverage
        self.symbol_info = super().get_symbol_info(self.symbol)

        self.TF = self.get_minutes(
        ) if time_frame == 'D1' else TF_MAPPING[time_frame]

    def risk_level(self) -> float:
        """
        Calculates the risk level of a trade

        Returns:
        -   Risk level in the form of a float percentage.
        """
        account_info = self.get_account_info()
        balance = account_info.balance
        equity = account_info.equity
        df = self.get_trades_history()
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

    def get_std_stop(self, tf: TimeFrame = 'D1', interval: int = 252):
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
        data = rate.get_rates_from_pos()
        returns = np.diff(data['Close'])
        std = np.std(returns)
        point = Mt5.symbol_info(self.symbol).point
        av_price = (self.symbol_info.bid + self.symbol_info.ask)/2
        price_interval = av_price * ((100-std))/100
        sl_point = float((av_price - price_interval) / point)
        sl = round(sl_point)
        min_sl = self.symbol_info.trade_stops_level * 2 \
            + self.get_deviation()

        return max(sl, min_sl)

    def get_pchange_stop(self, pchange: Optional[float]):
        """
        Calculate the percentage change-based stop loss level 
        for a given financial instrument.

        Args:
            pchange (float): Percentage change in price to use for calculating stop loss level.
                If pchange is set to None, the stop loss is calculate using std.

        Returns:
        -   Percentage change-based stop loss level, rounded to the nearest point. 
        -   0 if the calculated stop loss is <= 0.
        """
        if pchange is not None:
            av_price = (self.symbol_info.bid + self.symbol_info.ask)/2
            price_interval = av_price*((100-pchange))/100
            point = Mt5.symbol_info(self.symbol).point
            sl_point = float((av_price - price_interval) / point)
            sl = round(sl_point)
            min_sl = self.symbol_info.trade_stops_level * 2 \
                + self.get_deviation()
            return max(sl, min_sl)
        else:
            # Use std as default pchange
            return self.get_std_stop()

    def calculate_var(self, tf: TimeFrame = 'D1', interval=252, c=0.95):
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
        prices = rate.get_rates_from_pos()
        prices['return'] = prices['Close'].pct_change()
        prices.dropna(inplace=True)
        P = self.get_account_info().margin_free
        mu = np.mean(prices['return'])
        sigma = np.std(prices['return'])
        var = self.var_cov_var(P, c, mu, sigma)
        return var

    def var_cov_var(self, P: float, c: float, mu: float, sigma: float):
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
        deviation = self.get_deviation()
        if self.tp is not None:
            return self.tp + deviation
        else:
            return self.get_stop_loss()*self.rr

    def get_stop_loss(self) -> int:
        """calculates the stop loss of a trade in points"""
        min_sl = self.symbol_info.trade_stops_level * 2 \
            + self.get_deviation()
        if self.sl is not None:
            return max(self.sl, min_sl)
        elif self.sl is None and self.std:
            sl = self.get_std_stop()
            return max(sl, min_sl)
        elif self.sl is None and not self.std:
            risk = self.currency_risk()
            if risk['trade_loss'] != 0:
                sl = round((risk['currency_risk'] / risk['trade_loss']))
                return max(sl, min_sl)
            return min_sl

    def get_currency_risk(self) -> float:
        """calculates the currency risk of a trade"""
        return round(self.currency_risk()['currency_risk'], 2)

    def expected_profit(self):
        """Calculate the expected profit per trade"""
        risk = self.get_currency_risk()
        return round(risk*self.rr, 2)

    def volume(self):
        """Volume per trade"""

        return self.currency_risk()['volume']

    def currency_risk(self) -> Dict[str, Union[int, float, Any]]:
        """
        calculates the currency risk of a trade

        Returns:
        -   A dictionary containing the following keys:
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
        FX = self.get_symbol_type(self.symbol) == 'FX'
        COMD = self.get_symbol_type(self.symbol) == 'COMD'
        FUT = self.get_symbol_type(self.symbol) == 'FUT'
        CRYPTO = self.get_symbol_type(self.symbol) == 'CRYPTO'
        if COMD:
            supported = _COMMD_SUPPORTED_
            if self.symbol.split('.')[0] not in supported:
                raise ValueError(
                    f"Currency risk calculation for '{self.symbol}' is not a currently supported. \n"
                    f"Supported commodity symbols are: {', '.join(supported)}"
                )
        if FUT:
            supported = _ADMIRAL_MARKETS_FUTURES_
            if self.symbol[:-2] not in supported:
                raise ValueError(
                    f"Currency risk calculation for '{self.symbol}' is not a currently supported. \n"
                    f"Supported future symbols are: {', '.join(supported)}"
                )
        if trade_risk > 0:
            currency_risk = round(self.var_loss_value(), 5)
            volume = round(currency_risk*laverage)
            _lot = round((volume / (contract_size * av_price)), 2)
            lot = self._check_lot(_lot)
            if COMD and contract_size > 1:
                # lot = volume / av_price / contract_size
                lot = volume / av_price / contract_size
                lot = self._check_lot(_lot)
            if FX:
                __lot = round((volume / contract_size), 2)
                lot = self._check_lot(__lot)

            tick_value = s_info.trade_tick_value
            tick_value_loss = s_info.trade_tick_value_loss
            tick_value_profit = s_info.trade_tick_value_profit

            if COMD or FUT or CRYPTO and contract_size > 1:
                tick_value_loss = tick_value_loss / contract_size
                tick_value_profit = tick_value_profit / contract_size
            if (tick_value == 0
                    or tick_value_loss == 0
                    or tick_value_profit == 0
                    ):
                raise ValueError(
                    f"""The Tick Values for {self.symbol} is 0.0
                    We can not procced with currency risk calculation  
                    Please check your Broker trade conditions
                    and symbol specifications for {self.symbol}"""
                )
            point = float(s_info.point)

            # Case where the stop loss is given
            if self.sl is not None:
                trade_loss = currency_risk/self.sl
                if self.tp is not None:
                    trade_profit = (currency_risk*(self.tp//self.sl))/self.tp
                else:
                    trade_profit = (currency_risk*self.rr)/(self.sl*self.rr)
                lot_ = round(trade_loss / (contract_size*tick_value_loss), 2)
                lot = self._check_lot(lot_)
                volume = round(lot * contract_size * av_price)

                if COMD or CRYPTO and contract_size > 1:
                    # trade_risk = points * tick_value_loss * lot
                    lot = currency_risk / \
                        (self.sl * tick_value_loss*contract_size)
                    lot = self._check_lot(lot)
                    trade_loss = lot * contract_size * tick_value_loss

                if FX:
                    volume = round((trade_loss * contract_size) / tick_value_loss)
                    __lot = round((volume / contract_size), 2)
                    lot = self._check_lot(__lot)

            # Case where the stantard deviation is used
            elif self.std and self.pchange is None and self.sl is None:
                sl = self.get_std_stop()
                infos = self._std_pchange_stop(
                    currency_risk, sl, contract_size, tick_value_loss)
                trade_loss, trade_profit, lot, volume = infos

            # Case where the stop loss is based on a percentage change
            elif self.pchange is not None and not self.std and self.sl is None:
                sl = self.get_pchange_stop(self.pchange)
                infos = self._std_pchange_stop(
                    currency_risk, sl, contract_size, tick_value_loss)
                trade_loss, trade_profit, lot, volume = infos

            # Default cases
            else:
                # Handle FX
                if FX:
                    trade_loss = tick_value_loss * (volume / contract_size)
                    trade_profit = tick_value_profit * (volume / contract_size)
                else:
                    trade_loss = (lot * contract_size) * tick_value_loss
                    trade_profit = (lot * contract_size) * tick_value_profit


            if self.get_symbol_type(self.symbol) == 'IDX':
                rates = self.get_currency_rates(self.symbol)
                if rates['mc'] == rates['pc'] == 'JPY':
                    if self.std:
                        raise ValueError(
                            f"""Please Set std=False or use pchange_sl=True
                            or set sl=value or use the default method calculation for {self.symbol}
                            Currency risk"""
                        )
                    lot = lot * contract_size
                    volume = round(lot * av_price * contract_size)
            if contract_size == 1:
                volume = round(lot * av_price)

            return {
                'currency_risk': currency_risk,
                'trade_loss': trade_loss,
                'trade_profit': trade_profit,
                'volume': round(volume),
                'lot': lot
            }
        else:
            return {
                'currency_risk': 0.0,
                'trade_loss': 0.0,
                'trade_profit': 0.0,
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
        laverage = self.get_account_info().leverage
        av_price = (self.symbol_info.bid + self.symbol_info.ask)/2
        _lot = round(trade_loss/(size*loss), 2)
        lot = self._check_lot(_lot)

        volume = round(lot * size * av_price)
        if self.get_symbol_type(self.symbol) == 'FX':
            volume = round((trade_loss * size) / loss)
            __lot = round((volume / size), 2)
            lot = self._check_lot(__lot)

        if (
            self.get_symbol_type(self.symbol) == 'COMD'
            or self.get_symbol_type(self.symbol) == 'CRYPTO'
            and size > 1
        ):
            lot = currency_risk / (sl * loss * size)
            lot = self._check_lot(lot)
            trade_loss = lot * size * loss
            volume = round(lot * size * av_price)

        return trade_loss, trade_profit, lot, volume

    def _check_lot(self, lot: float) -> float:
        if lot > self.symbol_info.volume_max:
            return self.symbol_info.volume_max / 2
        elif lot < self.symbol_info.volume_min:
            return self.symbol_info.volume_min
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
        Notes:
            For FX symbols, account leverage is used by default.
            For Other instruments the account leverage is used if any error 
            occurs in leverage calculation.
        """
        AL = self.get_account_info().leverage
        if account:
            return AL

        if self.get_symbol_type(self.symbol) == 'FX':
            return AL
        else:
            s_info = self.symbol_info
            volume_min = s_info.volume_min
            contract_size = s_info.trade_contract_size
            av_price = (s_info.bid + s_info.ask)/2
            action = random.choice(
                [Mt5.ORDER_TYPE_BUY, Mt5.ORDER_TYPE_SELL]
            )
            margin = Mt5.order_calc_margin(
                action, self.symbol, volume_min, av_price
            )
            if margin == None:
                return AL
            try:
                leverage = (
                    volume_min * contract_size * av_price
                ) / margin
                return round(leverage)
            except ZeroDivisionError:
                return AL

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
