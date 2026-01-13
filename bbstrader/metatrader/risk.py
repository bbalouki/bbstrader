from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from loguru import logger
from scipy.stats import norm

from bbstrader.api import Mt5client as client
from bbstrader.config import BBSTRADER_DIR
from bbstrader.metatrader.account import Account
from bbstrader.metatrader.utils import TIMEFRAMES, SymbolType, TimeFrame

try:
    import MetaTrader5 as mt5
except ImportError:
    import bbstrader.compat  # noqa: F401

logger.add(
    f"{BBSTRADER_DIR}/logs/trade.log",
    enqueue=True,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
)


__all__ = ["RiskManagement"]


class RiskManagement:
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
        ...    act_leverage=True,
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
        accountt_leverage: bool = True,
        time_frame: TimeFrame = "D1",
        start_time: str = "1:00",
        finishing_time: str = "23:00",
        broker_tz: bool = False,
        sl: Optional[int] = None,
        tp: Optional[int] = None,
        be: Optional[int] = None,
        rr: float = 3.0,
        **kwargs,
    ):
        """
        Initialize the RiskManagement class to manage risk in trading activities.

        Args:
            symbol (str): The symbol of the financial instrument to trade.
            max_risk (float): The `maximum risk allowed` on the trading account.
            daily_risk (float, optional): `Daily Max risk allowed`.
                If Set to None it will be determine based on Maximum risk.
                The day is based on the start and the ending time
            max_trades (int, optional): Maximum number of trades at any point in time.
                If set to None it will be determine based on the timeframe of trading.
            std_stop (bool, optional): If set to True, the Stop loss is calculated based
                On `historical volatility` of the trading instrument. Defaults to False.
            pchange_sl (float, optional): If set, the Stop loss is calculated based
                On `percentage change` of the trading instrument.
            act_leverage (bool, optional): If set to True the account leverage will be used
                In risk management setting. Defaults to False.
            time_frame (str, optional): The time frame on which the program is working
                `(1m, 3m, 5m, 10m, 15m, 30m, 1h, 2h, 4h, D1)`. Defaults to 'D1'.
            start_time (str, optional): The starting time for the trading session
                `(HH:MM, H and M do not star with 0)`. Defaults to "1:00".
            finishing_time (str, optional): The finishing time for the trading strategy
                `(HH:MM, H and M do not star with 0)`. Defaults to "23:00".
            sl (int, optional): Stop Loss in points, Must be a positive number.
            tp (int, optional): Take Profit in points, Must be a positive number.
            be (int, optional): Break Even in points, Must be a positive number.
            rr (float, optional): Risk reward ratio, Must be a positive number. Defaults to 1.5.
        """

        assert max_risk > 0
        assert daily_risk > 0 if daily_risk is not None else ...
        daily_risk = round(daily_risk, 5) if daily_risk is not None else None
        assert all(isinstance(v, int) and v > 0 for v in [sl, tp] if v is not None)
        assert isinstance(be, (int, float)) and be > 0 if be else ...
        assert time_frame in TIMEFRAMES

        self.kwargs = kwargs
        self.symbol = symbol
        self.timeframe = time_frame
        self.start_time = start_time
        self.finishing_time = finishing_time
        self.max_trades = max_trades
        self.std_stop = std_stop
        self.pchange = pchange_sl
        self.act_leverage = accountt_leverage
        self.daily_dd = daily_risk
        self.max_risk = max_risk
        self.broker_tz = broker_tz
        self.rr = rr
        self.sl = sl
        self.tp = tp
        self.be = be

        self.account = Account(**kwargs)
        self.symbol_info = client.symbol_info(self.symbol)

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

    def _convert_time_frame(self, timeframe: str) -> int:
        """Convert time frame to minutes"""
        if timeframe == "D1":
            return self.get_minutes()
        elif "m" in timeframe:
            return TIMEFRAMES[timeframe]
        elif "h" in timeframe:
            return int(timeframe[0]) * 60
        elif timeframe == "W1":
            return self.get_minutes() * 5
        elif timeframe == "MN1":
            return self.get_minutes() * 22

    def get_minutes(self) -> int:
        """calculates the number of minutes between
        the starting of the session and the end of the session"""

        fmt = "%H:%M"
        start = datetime.strptime(self.start_time, fmt)
        end = datetime.strptime(self.finishing_time, fmt)
        if self.broker_tz:
            start = self.account.broker.get_broker_time(self.start_time, fmt)
            end = self.account.broker.get_broker_time(self.finishing_time, fmt)
        diff = (end - start).total_seconds()
        diff += 86400 if diff < 0 else diff
        return int(diff // 60)

    def get_hours(self) -> int:
        """Calculates the number of hours between
        the starting of the session and the end of the session"""
        return self.get_minutes() // 60

    def risk_level(self, balance_value=False) -> float | Tuple[float, float]:
        """
        Calculates the risk level of a trade

        Returns:
        -   Risk level in the form of a float percentage.
        """
        account_info = self.account.get_account_info()
        balance = account_info.balance
        equity = account_info.equity
        if equity == 0:
            return 0.0
        trades_history = self.account.get_trades_history()

        realized_profit = None
        if trades_history is None or len(trades_history) == 1:
            realized_profit = 0
        else:
            profit_df = trades_history.iloc[1:]
            profit = profit_df["profit"].sum()
            commisions = trades_history["commission"].sum()
            fees = trades_history["fee"].sum()
            swap = trades_history["swap"].sum()
            realized_profit = commisions + fees + swap + profit

        initial_balance = balance - realized_profit
        dd_percent = ((equity - initial_balance) / equity) * 100
        dd_percent = round(abs(dd_percent) if dd_percent < 0 else 0.0, 2)
        if balance_value:
            return (initial_balance, equity)
        return dd_percent

    def _get_lot(self) -> float:
        lot = self.currency_risk()["lot"]
        return self.account.broker.validate_lot_size(self.symbol, lot)

    def get_lot(self) -> float:
        return self.validate_currency_risk()[0]

    def max_trade(self) -> int:
        """calculates the maximum number of trades allowed"""
        minutes = self.get_minutes()
        tf_int = self._convert_time_frame(self.timeframe)
        max_trades = self.max_trades or round(minutes / tf_int)
        return max(max_trades, 1)

    def get_deviation(self) -> int:
        return client.symbol_info(self.symbol).spread

    def _get_stop(self, pchange: float) -> int:
        tick = client.symbol_info_tick(self.symbol)
        av_price = (tick.bid + tick.ask) / 2
        price_interval = av_price * (100 - pchange) / 100
        point = self.symbol_info.point
        sl = round(float((av_price - price_interval) / point))
        min_sl = (
            self.account.broker.get_min_stop_level(self.symbol) * 2
            + self.get_deviation()
        )
        return max(sl, min_sl)

    def _get_returns(self):
        minutes = self.get_minutes()
        tf_int = self._convert_time_frame(self.timeframe)
        interval = round((minutes / tf_int) * 252)
        rates = client.copy_rates_from_pos(
            self.symbol, TIMEFRAMES[self.timeframe], 0, interval
        )
        returns = (np.diff(rates["close"]) / rates["close"][:-1]) * 100
        return returns

    def get_std_stop(self) -> int:
        """
        Calculate the standard deviation-based stop loss level
        for a given financial instrument.

        Returns:
        -   Standard deviation-based stop loss level, rounded to the nearest point.
        -   0 if the calculated stop loss is less than or equal to 0.
        """
        std = np.std(self._get_returns())
        return self._get_stop(std)

    def get_pchange_stop(self, pchange: Optional[float]) -> int:
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
            return self._get_stop(pchange)
        else:
            # Use std as default pchange
            return self.get_std_stop()

    def calculate_var(self, tf: TimeFrame = "D1", c=0.95) -> float:
        """
        Calculate Value at Risk (VaR) for a given portfolio.

        Args:
            tf (str): Time frame to use to calculate volatility.
            c (float): Confidence level for VaR calculation (default is 95%).

        Returns:
        -   VaR value
        """
        returns = self._get_returns()
        P = self.account.get_account_info().margin_free
        mu = returns.mean()
        sigma = returns.std()
        alpha = norm.ppf(1 - c, mu, sigma)
        return P - P * (alpha + 1)

    def get_trade_risk(self) -> float:
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

    def var_loss_value(self) -> float:
        """
        Calculate the stop-loss level based on VaR.

        Notes:
            The Var is Estimated using the Variance-Covariance method on the daily returns.
            If you want to use the VaR for a different time frame .
        """
        P = self.account.get_account_info().margin_free
        trade_risk = self.get_trade_risk()
        loss_allowed = P * trade_risk / 100
        var = self.calculate_var()
        return min(var, loss_allowed)

    def get_take_profit(self) -> int:
        """calculates the take profit of a trade in points"""
        deviation = self.get_deviation()
        if self.tp is not None:
            return self.tp + deviation
        else:
            return round(self.get_stop_loss() * self.rr)

    def _get_stop_loss(self) -> int:
        """calculates the stop loss of a trade in points"""
        min_sl = (
            self.account.broker.get_min_stop_level(self.symbol) * 2
            + self.get_deviation()
        )
        if self.sl is not None:
            return max(self.sl, min_sl)
        if self.std_stop:
            sl = self.get_std_stop()
            return max(sl, min_sl)
        if self.pchange is not None:
            sl = self.get_pchange_stop(self.pchange)
            return max(sl, min_sl)
        risk = self.currency_risk()
        if risk["trade_loss"] != 0:
            sl = round(risk["currency_risk"] / risk["trade_loss"])
            return max(sl, min_sl)
        return min_sl

    def get_stop_loss(self) -> float:
        return self.validate_currency_risk()[1]

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

    def _std_pchange_stop(self, currency_risk, sl, size, loss):
        trade_loss = currency_risk / sl if sl != 0 else 0.0
        trade_profit = (currency_risk * self.rr) / (sl * self.rr) if sl != 0 else 0.0
        av_price = (self.symbol_info.bid + self.symbol_info.ask) / 2
        lot = round(trade_loss / (size * loss), 2) if size * loss != 0 else 0.0
        lot = self.account.broker.validate_lot_size(self.symbol, lot)
        volume = round(lot * size * av_price)
        if self.account.get_symbol_type(self.symbol) == SymbolType.FOREX:
            volume = round(trade_loss * size / loss) if loss != 0 else 0
            lot = round(volume / size, 2) if size != 0 else 0.0
            lot = self.account.broker.validate_lot_size(self.symbol, lot)
        if (
            self.account.get_symbol_type(self.symbol)
            in [SymbolType.COMMODITIES, SymbolType.CRYPTO]
            and size > 1
        ):
            lot = currency_risk / (sl * loss * size) if sl * loss * size != 0 else 0.0
            lot = self.account.broker.validate_lot_size(self.symbol, lot)
            trade_loss = lot * size * loss
            volume = round(lot * size * av_price)
        return trade_loss, trade_profit, lot, volume

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
        s_info = self.account.get_symbol_info(self.symbol)
        leverage = self.account.broker.get_leverage_for_symbol(
            self.symbol, self.act_leverage
        )
        contract_size = s_info.trade_contract_size
        av_price = (s_info.bid + s_info.ask) / 2
        trade_risk = self.get_trade_risk()
        symbol_type = self.account.get_symbol_type(self.symbol)

        tick_value_loss, tick_value_profit = self.account.broker.adjust_tick_values(
            self.symbol,
            s_info.trade_tick_value_loss,
            s_info.trade_tick_value_profit,
            contract_size,
        )
        tick_value = s_info.trade_tick_value  # For checks

        if tick_value == 0 or tick_value_loss == 0 or tick_value_profit == 0:
            logger.error(
                f"The Tick Values for {self.symbol} is 0.0. Check broker conditions for {self.symbol}."
            )
            return {
                "currency_risk": 0.0,
                "trade_loss": 0.0,
                "trade_profit": 0.0,
                "volume": 0,
                "lot": 0.01,
            }

        if trade_risk > 0:
            currency_risk = round(self.var_loss_value(), 5)
            volume = round(currency_risk * leverage)
            lot = (
                round(volume / (contract_size * av_price), 2)
                if contract_size * av_price != 0
                else 0.0
            )
            lot = self.account.broker.validate_lot_size(self.symbol, lot)

            if symbol_type == SymbolType.COMMODITIES and contract_size > 1:
                lot = (
                    volume / (av_price * contract_size)
                    if av_price * contract_size != 0
                    else 0.0
                )
                lot = self.account.broker.validate_lot_size(self.symbol, lot)
            if symbol_type == SymbolType.FOREX:
                lot = round(volume / contract_size, 2) if contract_size != 0 else 0.0
                lot = self.account.broker.validate_lot_size(self.symbol, lot)

            if self.sl is not None:
                trade_loss = currency_risk / self.sl if self.sl != 0 else 0.0
                trade_profit = (
                    (currency_risk * (self.tp // self.sl if self.tp else self.rr))
                    / (self.tp or (self.sl * self.rr))
                    if self.sl != 0
                    else 0.0
                )
                lot = (
                    round(trade_loss / (contract_size * tick_value_loss), 2)
                    if contract_size * tick_value_loss != 0
                    else 0.0
                )
                lot = self.account.broker.validate_lot_size(self.symbol, lot)
                volume = round(lot * contract_size * av_price)

                if (
                    symbol_type in [SymbolType.COMMODITIES, SymbolType.CRYPTO]
                ) and contract_size > 1:
                    lot = (
                        currency_risk / (self.sl * tick_value_loss * contract_size)
                        if self.sl * tick_value_loss * contract_size != 0
                        else 0.0
                    )
                    lot = self.account.broker.validate_lot_size(self.symbol, lot)
                    trade_loss = lot * contract_size * tick_value_loss

                if symbol_type == SymbolType.FOREX:
                    volume = (
                        round(trade_loss * contract_size / tick_value_loss)
                        if tick_value_loss != 0
                        else 0
                    )
                    lot = (
                        round(volume / contract_size, 2) if contract_size != 0 else 0.0
                    )
                    lot = self.account.broker.validate_lot_size(self.symbol, lot)

            elif self.std_stop and self.pchange is None and self.sl is None:
                sl = self.get_std_stop()
                trade_loss, trade_profit, lot, volume = self._std_pchange_stop(
                    currency_risk, sl, contract_size, tick_value_loss
                )

            elif self.pchange is not None and not self.std_stop and self.sl is None:
                sl = self.get_pchange_stop(self.pchange)
                trade_loss, trade_profit, lot, volume = self._std_pchange_stop(
                    currency_risk, sl, contract_size, tick_value_loss
                )

            else:
                if symbol_type == SymbolType.FOREX:
                    trade_loss = (
                        tick_value_loss * (volume / contract_size)
                        if contract_size != 0
                        else 0.0
                    )
                    trade_profit = (
                        tick_value_profit * (volume / contract_size)
                        if contract_size != 0
                        else 0.0
                    )
                else:
                    trade_loss = (lot * contract_size) * tick_value_loss
                    trade_profit = (lot * contract_size) * tick_value_profit

            # Apply currency conversion
            rates = self.account.get_currency_rates(self.symbol)
            factor = self.account.broker.get_currency_conversion_factor(
                self.symbol, rates.get("pc", ""), self.account.currency
            )
            trade_profit *= factor
            trade_loss *= factor
            currency_risk *= factor

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

    def validate_currency_risk(self):
        target_risk = self.get_currency_risk()
        sl_points = self._get_stop_loss()
        start_lot = self._get_lot()

        tick = client.symbol_info_tick(self.symbol)
        if tick is None:
            logger.error(f"No tick for {self.symbol}. Validation failed.")
            return 0.0, sl_points

        ask = tick.ask
        sl_price = ask - (sl_points * self.symbol_info.point)

        balance, equity = self.risk_level(balance_value=True)
        margin_free = self.account.get_account_info().margin_free
        allowed_drawdown = margin_free * (self.max_risk / 100)
        min_equity = balance - allowed_drawdown
        max_safe_loss = equity - min_equity

        if max_safe_loss <= 0:
            logger.warning(
                f"Equity ({equity}$) below safety threshold ({min_equity}$)."
            )
            return 0.0, sl_points
        min_vol = self.symbol_info.volume_min
        min_loss = client.order_calc_profit(
            mt5.ORDER_TYPE_BUY, self.symbol, min_vol, ask, sl_price
        )
        if min_loss is None or min_loss >= 0:
            logger.warning(
                f"Invalid min loss calculation for {self.symbol}: {min_loss}"
            )
            return 0.0, sl_points
        min_loss = abs(min_loss)
        if min_loss > max_safe_loss:
            logger.error(
                f"CRITICAL: Min vol loss ({min_loss:.2f}$) exceeds max safe loss ({max_safe_loss:.2f}$)."
            )
            return 0.0, sl_points
        effective_risk = min(target_risk, max_safe_loss)
        start_loss = client.order_calc_profit(
            mt5.ORDER_TYPE_BUY, self.symbol, start_lot, ask, sl_price
        )
        if start_loss is None or start_loss >= 0:
            base_vol, base_loss = min_vol, min_loss
        else:
            base_vol, base_loss = start_lot, abs(start_loss)
        if base_loss == 0:
            vol = min_vol
        else:
            ratio = effective_risk / base_loss
            calc_vol = base_vol * ratio

            step = self.symbol_info.volume_step
            vol = round(calc_vol / step) * step
            vol = self.account.broker.validate_lot_size(self.symbol, vol)
        final_loss = client.order_calc_profit(
            mt5.ORDER_TYPE_BUY, self.symbol, vol, ask, sl_price
        )
        if final_loss is None or final_loss >= 0:
            return 0.0, sl_points
        final_loss = abs(final_loss)

        if final_loss > max_safe_loss:
            vol_down = max(min_vol, vol - step)
            loss_down = client.order_calc_profit(
                mt5.ORDER_TYPE_BUY, self.symbol, vol_down, ask, sl_price
            )
            if loss_down is not None and abs(loss_down) <= max_safe_loss:
                vol = vol_down
            else:
                return 0.0, sl_points
        return (vol, round(sl_points)) if vol > 0 else (0.0, sl_points)

    def get_break_even(self, thresholds: list[tuple[int, float]] = None) -> int:
        """
        Calculates the break-even price level based on stop-loss tiers.

        The function determines the break-even point by applying a multiplier to the
        sum of the current stop-loss and market spread. If an explicit break-even
        value (`self.be`) is already set, it returns that value (converting
        percentage-based floats to absolute points if necessary).

        Args:
            thresholds (list[tuple[int, float]], optional): A list of tiers defined
                as (threshold_limit, multiplier).
                Example: [(150, 0.25), (100, 0.35), (0, 0.5)].
                If None, defaults to standard conservative tiers.

        Returns:
            int: The calculated break-even value in points/pips.

        Note:
            The function automatically sorts thresholds in descending order to
            ensure the 'stop' value is matched against the highest possible tier first.
        """
        if self.be is not None:
            return (
                self.be if isinstance(self.be, int) else self.get_pchange_stop(self.be)
            )

        if thresholds is None:
            thresholds = [(150, 0.25), (100, 0.35), (0, 0.50)]

        stop = self.get_stop_loss()
        spread = client.symbol_info(self.symbol).spread
        sorted_thresholds = sorted(thresholds, key=lambda x: x[0], reverse=True)

        for limit, multiplier in sorted_thresholds:
            if stop > limit:
                return round((stop + spread) * multiplier)
        return 0

    def is_risk_ok(self) -> bool:
        return self.risk_level() <= self.max_risk
