import random
from datetime import datetime
from typing import Any, Dict, Optional, Union

from scipy.stats import norm

from bbstrader.metatrader.account import Account
from bbstrader.metatrader.rates import Rates
from bbstrader.metatrader.utils import TIMEFRAMES, TimeFrame, SymbolType

try:
    import MetaTrader5 as Mt5
except ImportError:
    import bbstrader.compat  # noqa: F401


_COMMD_SUPPORTED_ = [
    "GOLD",
    "SILVER",
    "BRENT",
    "CRUDOIL",
    "WTI",
    "UKOIL",
    "XAGEUR",
    "XAGUSD",
    "XAGAUD",
    "XAGGBP",
    "XAUAUD",
    "XAUEUR",
    "XAUUSD",
    "XAUGBP",
    "USOIL",
    "SpotCrude",
    "SpotBrent",
    "Soybeans",
    "Wheat",
    "SoyOil",
    "LeanHogs",
    "LDSugar",
    "Coffee",
    "OJ",
    "Cocoa",
    "Cattle",
    "Copper",
    "XCUUSD",
    "NatGas",
    "NATGAS",
    "Gasoline",
]

_ADMIRAL_MARKETS_FUTURES_ = [
    "#USTNote_",
    "#Bund_",
    "#USDX_",
    "_AUS200_",
    "_Canada60_",
    "_SouthAfrica40_",
    "_STXE600_",
    "_EURO50_",
    "_GER40_",
    "_GermanyTech30_",
    "_MidCapGER50_",
    "_SWISS20_",
    "_UK100_",
    "_USNASDAQ100_",
    "_YM_",
    "_ES_",
    "_CrudeOilUS_",
    "_DUTCH25_",
    "_FRANCE40_",
    "_NORWAY25_",
    "_SPAIN35_",
    "_CrudeOilUK_",
    "_XAU_",
    "_HK50_",
    "_HSCEI50_",
]

__PEPPERSTONE_FUTURES__ = [
    "AUS200-F",
    "GER40-F",
    "HK50-F",
    "JPN225-F",
    "UK100-F",
    "US30-F",
    "NAS100-F",
    "US500-F",
    "Crude-F",
    "Brent-F",
    "XAUUSD-F",
    "XAGUSD-F",
    "USDX-F",
    "EUSTX50-F",
    "FRA40-F",
    "GERTEC30-F",
    "SPA35-F",
    "SWI20-F",
]

__all__ = ["RiskManagement"]


class RiskManagement(Account):
    """
    The RiskManagement class provides foundational
    risk management functionalities for trading activities.
    It calculates risk levels, determines stop loss and take profit levels,
    and ensures trading activities align with predefined risk parameters.

    Exemple:
        >>> risk_manager = RiskManagement(
        ...    symbol="EURUSD",
        ...    max_risk=5.0,
        ...    daily_risk=2.0,
        ...    max_trades=10,
        ...    std_stop=True,
        ...    account_leverage=True,
        ...    start_time="09:00",
        ...    finishing_time="17:00",
        ...    time_frame="1h"
        ... )
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
        max_risk: float = 10.0,
        daily_risk: Optional[float] = None,
        max_trades: Optional[int] = None,
        std_stop: bool = False,
        pchange_sl: Optional[float] = None,
        var_level: float = 0.95,
        var_time_frame: TimeFrame = "D1",
        account_leverage: bool = True,
        time_frame: TimeFrame = "D1",
        start_time: str = "1:00",
        finishing_time: str = "23:00",
        sl: Optional[int] = None,
        tp: Optional[int] = None,
        be: Optional[int] = None,
        rr: float = 1.5,
        **kwargs,
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
            var_level (float, optional): Confidence level for Value-at-Risk,e.g., 0.99 for 99% confidence interval.
                The default is 0.95.
            var_time_frame (str, optional): Time frame to use to calculate the VaR.
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

        See `bbstrader.metatrader.account.check_mt5_connection()` for more details on how to connect to MT5 terminal.
        """
        super().__init__(**kwargs)

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
            raise ValueError("Unsupported time frame {}".format(time_frame))
        if var_time_frame not in TIMEFRAMES:
            raise ValueError("Unsupported time frame {}".format(var_time_frame))

        self.kwargs = kwargs
        self.symbol = symbol
        self.start_time = start_time
        self.finishing_time = finishing_time
        self.max_trades = max_trades
        self.std = std_stop
        self.pchange = pchange_sl
        self.var_level = var_level
        self.var_tf = var_time_frame
        self.daily_dd = round(daily_risk, 5) if daily_risk is not None else None
        self.max_risk = max_risk
        self.rr = rr
        self.sl = sl
        self.tp = tp
        self.be = be

        self.account_leverage = account_leverage
        self.symbol_info = super().get_symbol_info(self.symbol)
        self.copy_mode = kwargs.get("copy", False)

        self._tf = time_frame

    @property
    def dailydd(self) -> float:
        return self.daily_dd

    @dailydd.setter
    def dailydd(self, value: float):
        self.daily_dd = value

    @property
    def maxrisk(self) -> float:
        return self.max_risk

    @maxrisk.setter
    def maxrisk(self, value: float):
        self.max_risk = value

    def _convert_time_frame(self, tf: str) -> int:
        """Convert time frame to minutes"""
        if tf == "D1":
            tf_int = self.get_minutes()
        elif "m" in tf:
            tf_int = TIMEFRAMES[tf]
        elif "h" in tf:
            tf_int = int(tf[0]) * 60
        elif tf == "W1":
            tf_int = self.get_minutes() * 5
        elif tf == "MN1":
            tf_int = self.get_minutes() * 22
        return tf_int

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
            profit = profit_df["profit"].sum()
            commisions = df["commission"].sum()
            fees = df["fee"].sum()
            swap = df["swap"].sum()
            total_profit = commisions + fees + swap + profit
            initial_balance = balance - total_profit
            if equity != 0:
                risk_alowed = (((equity - initial_balance) / equity) * 100) * -1
                return round(risk_alowed, 2)
            else: # Handle equity is zero
                return 0.0
        return 0.0 # This is for the case where df is None

    def get_lot(self) -> float:
        """ "Get the approprite lot size for a trade"""
        s_info = self.symbol_info
        volume_step = s_info.volume_step
        lot = self.currency_risk()["lot"]
        steps = self._volume_step(volume_step)
        if float(steps) >= float(1):
            return round(lot, steps)
        else:
            return round(lot)

    def _volume_step(self, value):
        """Get the number of decimal places in a number"""

        value_str = str(value)

        if "." in value_str and value_str != "1.0":
            decimal_index = value_str.index(".")
            num_digits = len(value_str) - decimal_index - 1
            return num_digits

        elif value_str == "1.0":
            return 0
        else:
            return 0

    def max_trade(self) -> int:
        """calculates the maximum number of trades allowed"""
        minutes = self.get_minutes()
        tf_int = self._convert_time_frame(self._tf)
        if self.max_trades is not None:
            max_trades = self.max_trades
        else:
            max_trades = round(minutes / tf_int)
        return max(max_trades, 1)

    def get_minutes(self) -> int:
        """calculates the number of minutes between two times"""

        start = datetime.strptime(self.start_time, "%H:%M")
        end = datetime.strptime(self.finishing_time, "%H:%M")
        return (end - start).total_seconds() // 60

    def get_hours(self) -> int:
        """Calculates the number of hours between two times"""

        start = datetime.strptime(self.start_time, "%H:%M")
        end = datetime.strptime(self.finishing_time, "%H:%M")
        # Calculate the difference in hours
        hours = (end - start).total_seconds() // 3600  # 1 hour = 3600 seconds

        return hours

    def get_std_stop(self):
        """
        Calculate the standard deviation-based stop loss level
        for a given financial instrument.

        Returns:
        -   Standard deviation-based stop loss level, rounded to the nearest point.
        -   0 if the calculated stop loss is less than or equal to 0.
        """
        minutes = self.get_minutes()
        tf_int = self._convert_time_frame(self._tf)
        interval = round((minutes / tf_int) * 252)

        rate = Rates(
            self.symbol, timeframe=self._tf, start_pos=0, count=interval, **self.kwargs
        )
        returns = rate.returns * 100
        std = returns.std()
        point = self.get_symbol_info(self.symbol).point
        av_price = (self.symbol_info.bid + self.symbol_info.ask) / 2
        price_interval = av_price * (100 - std) / 100
        sl_point = float((av_price - price_interval) / point)
        sl = round(sl_point)
        min_sl = self.symbol_info.trade_stops_level * 2 + self.get_deviation()

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
            av_price = (self.symbol_info.bid + self.symbol_info.ask) / 2
            price_interval = av_price * (100 - pchange) / 100
            point = self.get_symbol_info(self.symbol).point
            sl_point = float((av_price - price_interval) / point)
            sl = round(sl_point)
            min_sl = self.symbol_info.trade_stops_level * 2 + self.get_deviation()
            return max(sl, min_sl)
        else:
            # Use std as default pchange
            return self.get_std_stop()

    def calculate_var(self, tf: TimeFrame = "D1", c=0.95):
        """
        Calculate Value at Risk (VaR) for a given portfolio.

        Args:
            tf (str): Time frame to use to calculate volatility.
            c (float): Confidence level for VaR calculation (default is 95%).

        Returns:
        -   VaR value
        """
        minutes = self.get_minutes()
        tf_int = self._convert_time_frame(tf)
        interval = round((minutes / tf_int) * 252)

        rate = Rates(
            self.symbol, timeframe=tf, start_pos=0, count=interval, **self.kwargs
        )
        returns = rate.returns * 100
        p = self.get_account_info().margin_free
        mu = returns.mean()
        sigma = returns.std()
        var = self.var_cov_var(p, c, mu, sigma)
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

        Notes:
            The Var is Estimated using the Variance-Covariance method on the daily returns.
            If you want to use the VaR for a different time frame .
        """
        P = self.get_account_info().margin_free
        trade_risk = self.get_trade_risk()
        loss_allowed = P * trade_risk / 100
        var = self.calculate_var(c=self.var_level, tf=self.var_tf)
        return min(var, loss_allowed)

    def get_take_profit(self) -> int:
        """calculates the take profit of a trade in points"""
        deviation = self.get_deviation()
        if self.tp is not None:
            return self.tp + deviation
        else:
            return round(self.get_stop_loss() * self.rr)

    def get_stop_loss(self) -> int:
        """calculates the stop loss of a trade in points"""
        min_sl = self.symbol_info.trade_stops_level * 2 + self.get_deviation()
        if self.sl is not None:
            return max(self.sl, min_sl)
        elif self.sl is None and self.std:
            sl = self.get_std_stop()
            return max(sl, min_sl)
        elif self.sl is None and not self.std:
            risk = self.currency_risk()
            if risk["trade_loss"] != 0:
                sl = round((risk["currency_risk"] / risk["trade_loss"]))
                return max(sl, min_sl)
            return min_sl

    def get_currency_risk(self) -> float:
        """calculates the currency risk of a trade"""
        return round(self.currency_risk()["currency_risk"], 2)

    def expected_profit(self):
        """Calculate the expected profit per trade"""
        risk = self.get_currency_risk()
        return round(risk * self.rr, 2)

    def volume(self):
        """Volume per trade"""

        return self.currency_risk()["volume"]

    def currency_risk(self) -> Dict[str, Union[int, float, Any]]:
        """
        Calculates the currency risk of a trade.

        Returns:
            Dict[str, Union[int, float, Any]]: A dictionary containing the following keys:

            - `'currency_risk'`: Dollar amount risk on a single trade.
            - `'trade_loss'`: Loss value per tick in dollars.
            - `'trade_profit'`: Profit value per tick in dollars.
            - `'volume'`: Contract size multiplied by the average price.
            - `'lot'`: Lot size per trade.
        """
        s_info = self.symbol_info

        laverage = self.get_leverage(self.account_leverage)
        contract_size = s_info.trade_contract_size

        av_price = (s_info.bid + s_info.ask) / 2
        trade_risk = self.get_trade_risk()
        symbol_type = self.get_symbol_type(self.symbol)
        FX = symbol_type == SymbolType.FOREX
        COMD = symbol_type == SymbolType.COMMODITIES
        FUT = symbol_type == SymbolType.FUTURES
        CRYPTO = symbol_type == SymbolType.CRYPTO
        if COMD:
            supported = _COMMD_SUPPORTED_
            if "." in self.symbol:
                symbol = self.symbol.split(".")[0]
            else:
                symbol = self.symbol
            if not self.copy_mode and str(symbol) not in supported:
                raise ValueError(
                    f"Currency risk calculation for '{self.symbol}' is not a currently supported. \n"
                    f"Supported commodity symbols are: {', '.join(supported)}"
                )
        if FUT:
            if "_" in self.symbol:
                symbol = self.symbol[:-2]
            else:
                symbol = self.symbol
            supported = _ADMIRAL_MARKETS_FUTURES_ + __PEPPERSTONE_FUTURES__
            if not self.copy_mode and str(symbol) not in supported:
                raise ValueError(
                    f"Currency risk calculation for '{self.symbol}' is not a currently supported. \n"
                    f"Supported future symbols are: {', '.join(supported)}"
                )
        if trade_risk > 0:
            currency_risk = round(self.var_loss_value(), 5)
            volume = round(currency_risk * laverage)
            try:
                _lot = round((volume / (contract_size * av_price)), 2)
            except ZeroDivisionError:
                _lot = 0.0
            lot = self._check_lot(_lot)
            if COMD and contract_size > 1:
                # lot = volume / av_price / contract_size
                try:
                    lot = volume / av_price / contract_size
                except ZeroDivisionError:
                    lot = 0.0
                lot = self._check_lot(_lot)
            if FX:
                try:
                    __lot = round((volume / contract_size), 2)
                except ZeroDivisionError:
                    __lot = 0.0
                lot = self._check_lot(__lot)

            tick_value = s_info.trade_tick_value
            tick_value_loss = s_info.trade_tick_value_loss
            tick_value_profit = s_info.trade_tick_value_profit

            if COMD or FUT or CRYPTO and contract_size > 1:
                tick_value_loss = tick_value_loss / contract_size
                tick_value_profit = tick_value_profit / contract_size
            if tick_value == 0 or tick_value_loss == 0 or tick_value_profit == 0:
                raise ValueError(
                    f"""The Tick Values for {self.symbol} is 0.0
                    We can not procced with currency risk calculation  
                    Please check your Broker trade conditions
                    and symbol specifications for {self.symbol}"""
                )

            # Case where the stop loss is given
            if self.sl is not None:
                trade_loss = currency_risk / self.sl
                if self.tp is not None:
                    trade_profit = (currency_risk * (self.tp // self.sl)) / self.tp
                else:
                    trade_profit = (currency_risk * self.rr) / (self.sl * self.rr)
                lot_ = round(trade_loss / (contract_size * tick_value_loss), 2)
                lot = self._check_lot(lot_)
                volume = round(lot * contract_size * av_price)

                if COMD or CRYPTO and contract_size > 1:
                    # trade_risk = points * tick_value_loss * lot
                    lot = currency_risk / (self.sl * tick_value_loss * contract_size)
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
                    currency_risk, sl, contract_size, tick_value_loss
                )
                trade_loss, trade_profit, lot, volume = infos

            # Case where the stop loss is based on a percentage change
            elif self.pchange is not None and not self.std and self.sl is None:
                sl = self.get_pchange_stop(self.pchange)
                infos = self._std_pchange_stop(
                    currency_risk, sl, contract_size, tick_value_loss
                )
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

            # if self.get_symbol_type(self.symbol) == 'IDX':
            #     rates = self.get_currency_rates(self.symbol)
            #     if rates['mc'] == rates['pc'] == 'JPY':
            #         lot = lot * contract_size
            #         lot = self._check_lot(lot)
            #         volume = round(lot * av_price * contract_size)
            # if contract_size == 1:
            #     volume = round(lot * av_price)

            return {
                "currency_risk": currency_risk,
                "trade_loss": trade_loss,
                "trade_profit": trade_profit,
                "volume": round(volume),
                "lot": lot,
            }
        else:
            return {
                "currency_risk": 0.0,
                "trade_loss": 0.0,
                "trade_profit": 0.0,
                "volume": 0,
                "lot": 0.01,
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
        trade_loss = currency_risk / sl
        trade_profit = (currency_risk * self.rr) / (sl * self.rr)
        av_price = (self.symbol_info.bid + self.symbol_info.ask) / 2
        _lot = round(trade_loss / (size * loss), 2)
        lot = self._check_lot(_lot)

        volume = round(lot * size * av_price)
        if self.get_symbol_type(self.symbol) == SymbolType.FOREX:
            volume = round((trade_loss * size) / loss)
            __lot = round((volume / size), 2)
            lot = self._check_lot(__lot)

        if (
            self.get_symbol_type(self.symbol) == SymbolType.COMMODITIES
            or self.get_symbol_type(self.symbol) == SymbolType.CRYPTO
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
                trade_risk = self.daily_dd / max_trades
            else:
                trade_risk = (self.max_risk - total_risk) / max_trades
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

        if self.get_symbol_type(self.symbol) == SymbolType.FOREX:
            return AL
        else:
            s_info = self.symbol_info
            volume_min = s_info.volume_min
            contract_size = s_info.trade_contract_size
            av_price = (s_info.bid + s_info.ask) / 2
            action = random.choice([Mt5.ORDER_TYPE_BUY, Mt5.ORDER_TYPE_SELL])
            margin = Mt5.order_calc_margin(action, self.symbol, volume_min, av_price)
            if margin is None:
                return AL
            try:
                leverage = (volume_min * contract_size * av_price) / margin
                return round(leverage)
            except ZeroDivisionError:
                return AL

    def get_deviation(self) -> int:
        return self.symbol_info.spread

    def get_break_even(self) -> int:
        if self.be is not None:
            if isinstance(self.be, int):
                return self.be
            elif isinstance(self.be, float):
                return self.get_pchange_stop(self.be)
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
