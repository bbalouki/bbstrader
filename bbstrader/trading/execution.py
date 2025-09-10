import concurrent.futures
import functools
import multiprocessing as mp
import sys
import time
from datetime import date, datetime
from multiprocessing.synchronize import Event
from typing import Callable, Dict, List, Literal, Optional

import pandas as pd
from loguru import logger as log

from bbstrader.btengine.strategy import MT5Strategy, Strategy
from bbstrader.config import BBSTRADER_DIR
from bbstrader.metatrader.account import check_mt5_connection
from bbstrader.metatrader.trade import Trade, TradeAction, TradingMode
from bbstrader.trading.utils import send_message

try:
    import MetaTrader5 as MT5
except ImportError:
    import bbstrader.compat  # noqa: F401


__all__ = ["Mt5ExecutionEngine", "RunMt5Engine", "RunMt5Engines", "TWSExecutionEngine"]

_TF_MAPPING = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "10m": 10,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "D1": 1440,
}

MT5_ENGINE_TIMEFRAMES = list(_TF_MAPPING.keys())

TradingDays = ["monday", "tuesday", "wednesday", "thursday", "friday"]
WEEK_DAYS = TradingDays + ["saturday", "sunday"]
FRIDAY = "friday"

BUYS = ["BMKT", "BLMT", "BSTP", "BSTPLMT"]
SELLS = ["SMKT", "SLMT", "SSTP", "SSTPLMT"]

ORDERS_TYPES = [
    "orders",
    "buy_stops",
    "sell_stops",
    "buy_limits",
    "sell_limits",
    "buy_stop_limits",
    "sell_stop_limits",
]
POSITIONS_TYPES = ["positions", "buys", "sells", "profitables", "losings"]

ACTIONS = ["buys", "sells"]
STOPS = ["buy_stops", "sell_stops"]
LIMITS = ["buy_limits", "sell_limits"]
STOP_LIMITS = ["buy_stop_limits", "sell_stop_limits"]

EXIT_SIGNAL_ACTIONS = {
    "EXIT": {a: a[:-1] for a in ACTIONS},
    "EXIT_LONG": {"buys": "buy"},
    "EXIT_SHORT": {"sells": "sell"},
    "EXIT_STOP": {stop: stop for stop in STOPS},
    "EXIT_LONG_STOP": {"buy_stops": "buy_stops"},
    "EXIT_SHORT_STOP": {"sell_stops": "sell_stops"},
    "EXIT_LIMIT": {limit: limit for limit in LIMITS},
    "EXIT_LONG_LIMIT": {"buy_limits": "buy_limits"},
    "EXIT_SHORT_LIMIT": {"sell_limits": "sell_limits"},
    "EXIT_STOP_LIMIT": {sl: sl for sl in STOP_LIMITS},
    "EXIT_LONG_STOP_LIMIT": {STOP_LIMITS[0]: STOP_LIMITS[0]},
    "EXIT_SHORT_STOP_LIMIT": {STOP_LIMITS[1]: STOP_LIMITS[1]},
    "EXIT_PROFITABLES": {"profitables": "profitable"},
    "EXIT_LOSINGS": {"losings": "losing"},
    "EXIT_ALL_POSITIONS": {"positions": "all"},
    "EXIT_ALL_ORDERS": {"orders": "all"},
}

COMMON_RETCODES = [MT5.TRADE_RETCODE_MARKET_CLOSED, MT5.TRADE_RETCODE_CLOSE_ONLY]

NON_EXEC_RETCODES = {
    "BMKT": [MT5.TRADE_RETCODE_SHORT_ONLY] + COMMON_RETCODES,
    "SMKT": [MT5.TRADE_RETCODE_LONG_ONLY] + COMMON_RETCODES,
}

log.add(
    f"{BBSTRADER_DIR}/logs/execution.log",
    enqueue=True,
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
)


class Mt5ExecutionEngine:
    """
    The `Mt5ExecutionEngine` class serves as the central hub for executing your trading strategies within the `bbstrader` framework.
    It orchestrates the entire trading process, ensuring seamless interaction between your strategies, market data, and your chosen
    trading platform.

    Key Features
    ------------

    - **Strategy Execution:** The `Mt5ExecutionEngine` is responsible for running your strategy, retrieving signals, and executing trades based on those signals.
    - **Time Management:** You can define a specific time frame for your trades and set the frequency with which the engine checks for signals and manages trades.
    - **Trade Period Control:** Define whether your strategy runs for a day, a week, or a month, allowing for flexible trading durations.
    - **Money Management:** The engine supports optional money management features, allowing you to control risk and optimize your trading performance.
    - **Trading Day Configuration:** You can customize the days of the week your strategy will execute, providing granular control over your trading schedule.
    - **Platform Integration:** The `Mt5ExecutionEngine` is currently designed to work with MT5.

    Examples
    --------

    >>> from bbstrader.metatrader import create_trade_instance
    >>> from bbstrader.trading.execution import Mt5ExecutionEngine
    >>> from bbstrader.trading.strategies import StockIndexSTBOTrading
    >>> from bbstrader.config import config_logger
    >>>
    >>> if __name__ == '__main__':
    >>>     logger = config_logger(index_trade.log, console_log=True)
    >>>     # Define symbols
    >>>     ndx = '[NQ100]'
    >>>     spx = '[SP500]'
    >>>     dji = '[DJI30]'
    >>>     dax = 'GERMANY40'
    >>>
    >>>     symbol_list = [spx, dax, dji,  ndx]
    >>>
    >>>     trade_kwargs = {
    ...        'expert_id': 5134,
    ...        'version': 2.0,
    ...        'time_frame': '15m',
    ...        'var_level': 0.99,
    ...        'start_time': '8:30',
    ...        'finishing_time': '19:30',
    ...        'ending_time': '21:30',
    ...        'max_risk': 5.0,
    ...        'daily_risk': 0.10,
    ...        'pchange_sl': 1.5,
    ...        'rr': 3.0,
    ...        'logger': logger
    ...    }
    >>>     strategy_kwargs = {
    ...        'max_trades': {ndx: 3, spx: 3, dji: 3, dax: 3},
    ...        'expected_returns': {ndx: 1.5, spx: 1.5, dji: 1.0, dax: 1.0},
    ...        'strategy_name': 'SISTBO',
    ...        'logger': logger,
    ...        'expert_id': 5134
    ...    }
    >>>     trades_instances = create_trade_instance(
    ...        symbol_list, trade_kwargs,
    ...        logger=logger,
    ...    )
    >>>
    >>>     engine = Mt5ExecutionEngine(
    ...        symbol_list,
    ...        trades_instances,
    ...        StockIndexCFDTrading,
    ...        time_frame='15m',
    ...        iter_time=5,
    ...        mm=True,
    ...        period='week',
    ...        comment='bbs_SISTBO_@2.0',
    ...        **strategy_kwargs
    ...    )
    >>>     engine.run()
    """

    def __init__(
        self,
        symbol_list: List[str],
        trades_instances: Dict[str, Trade],
        strategy_cls: Strategy | MT5Strategy,
        /,
        mm: bool = True,
        auto_trade: bool = True,
        prompt_callback: Callable = None,
        multithread: bool = False,
        shutdown_event: Event = None,
        optimizer: str = "equal",
        trail: bool = True,
        stop_trail: Optional[int] = None,
        trail_after_points: int | str = None,
        be_plus_points: Optional[int] = None,
        show_positions_orders: bool = False,
        iter_time: int | float = 5,
        use_trade_time: bool = True,
        period: Literal["24/7", "day", "week", "month"] = "month",
        period_end_action: Literal["break", "sleep"] = "sleep",
        closing_pnl: Optional[float] = None,
        trading_days: Optional[List[str]] = None,
        comment: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            symbol_list : List of symbols to trade
            trades_instances : Dictionary of Trade instances
            strategy_cls : Strategy class to use for trading
            mm : Enable Money Management. Defaults to True.
            optimizer : Risk management optimizer. Defaults to 'equal'.
                See `bbstrader.models.optimization` module for more information.
            auto_trade :  If set to true, when signal are generated by the strategy class,
                the Execution engine will automaticaly open position in other whise it will prompt
                the user for confimation.
            prompt_callback : Callback function to prompt the user for confirmation.
                This is useful when integrating with GUI applications.
             multithread : If True, use a thread pool to process signals in parallel.
                If False, process them sequentially. Set this to True only if the engine
                is running in a separate process. Default to False.
            shutdown_event : Use to terminate the copy process when runs in a custum environment like web App or GUI.
            show_positions_orders : Print open positions and orders. Defaults to False.
            iter_time : Interval to check for signals and `mm`. Defaults to 5.
            use_trade_time : Open trades after the time is completed. Defaults to True.
            period : Period to trade ("24/7", "day", "week", "month"). Defaults to 'week'.
            period_end_action : Action to take at the end of the period ("break", "sleep"). Defaults to 'break',
                this only applies when period is 'day', 'week'.
            closing_pnl : Minimum profit in percentage of target profit to close positions. Defaults to -0.001.
            trading_days : Trading days in a week. Defaults to monday to friday.
            comment: Comment for trades. Defaults to None.
            **kwargs: Additional keyword arguments
                _ time_frame : Time frame to trade. Defaults to '15m'.
                - strategy_name (Optional[str]): Strategy name. Defaults to None.
                - max_trades (Dict[str, int]): Maximum trades per symbol. Defaults to None.
                - notify (bool): Enable notifications. Defaults to False.
                - telegram (bool): Enable telegram notifications. Defaults to False.
                - bot_token (str): Telegram bot token. Defaults to None.
                - chat_id (Union[int, str, List] ): Telegram chat id. Defaults to None.
                - MT5 connection arguments.

        Note:
            1. For `trail` , `stop_trail` , `trail_after_points` , `be_plus_points` see `bbstrader.metatrader.trade.Trade.break_even()` .
            2. All Strategies must inherit from `bbstrader.btengine.strategy.MT5Strategy` class
            and have a `calculate_signals` method that returns a List of ``bbstrader.metatrader.trade.TradingSignal``.

            3. All strategies must have the following arguments in their `__init__` method:
                - bars (DataHandler): DataHandler instance default to None
                - events (Queue): Queue instance default to None
                - symbol_list (List[str]): List of symbols to trade can be none for backtesting
                - mode (str): Mode of the strategy. Must be either 'live' or 'backtest'
                - **kwargs: Additional keyword arguments
                    The keyword arguments are all the additional arguments passed to the `Mt5ExecutionEngine` class,
                    the `Strategy` class, the `DataHandler` class, the `Portfolio` class and the `ExecutionHandler` class.
                - The `bars` and `events` arguments are used for backtesting only.

            4. All strategies must generate signals for backtesting and live trading.
            See the `bbstrader.trading.strategies` module for more information on how to create custom strategies.
            See `bbstrader.metatrader.account.check_mt5_connection()` for more details on how to connect to MT5 terminal.
        """
        self.symbols = symbol_list.copy()
        self.trades_instances = trades_instances
        self.strategy_cls = strategy_cls
        self.mm = mm
        self.auto_trade = auto_trade
        self.prompt_callback = prompt_callback
        self.multithread = multithread
        self.optimizer = optimizer
        self.trail = trail
        self.stop_trail = stop_trail
        self.trail_after_points = trail_after_points
        self.be_plus_points = be_plus_points
        self.show_positions_orders = show_positions_orders
        self.iter_time = iter_time
        self.use_trade_time = use_trade_time
        self.period = period.strip()
        self.period_end_action = period_end_action
        self.closing_pnl = closing_pnl
        self.comment = comment
        self.kwargs = kwargs

        self.time_intervals = 0
        self.time_frame = kwargs.get("time_frame", "15m")
        self.trade_time = _TF_MAPPING[self.time_frame]

        self.long_market = {symbol: False for symbol in self.symbols}
        self.short_market = {symbol: False for symbol in self.symbols}

        self._initialize_engine(**kwargs)
        self.strategy = self._init_strategy(**kwargs)
        self.shutdown_event = (
            shutdown_event if shutdown_event is not None else mp.Event()
        )
        self._running = True

    def __repr__(self):
        trades = self.trades_instances.keys()
        strategy = self.strategy_cls.__name__
        return f"{self.__class__.__name__}(Symbols={list(trades)}, Strategy={strategy})"

    def _initialize_engine(self, **kwargs):
        global logger
        logger = kwargs.get("logger", log)
        try:
            self.daily_risk = kwargs.get("daily_risk")
            self.notify = kwargs.get("notify", False)
            self.debug_mode = kwargs.get("debug_mode", False)
            self.delay = kwargs.get("delay", 0)

            self.STRATEGY = kwargs.get("strategy_name")
            self.ACCOUNT = kwargs.get("account", "MT5 Account")
            self.signal_tickers = kwargs.get("signal_tickers", self.symbols)

            self.expert_ids = self._expert_ids(kwargs.get("expert_ids"))
            self.max_trades = self._max_trades(kwargs.get("max_trades"))
            if self.comment is None:
                trade = self.trades_instances[self.symbols[0]]
                self.comment = f"{trade.expert_name}@{trade.version}"
            if kwargs.get("trading_days") is None:
                if self.period.lower() == "24/7":
                    self.trading_days = WEEK_DAYS
                else:
                    self.trading_days = TradingDays
            else:
                self.trading_days = kwargs.get("trading_days")
        except Exception as e:
            self._print_exc(
                f"Initializing Execution Engine, STRATEGY={self.STRATEGY}, ACCOUNT={self.ACCOUNT}",
                e,
            )
            return

    def _print_exc(self, msg: str, e: Exception):
        if isinstance(e, KeyboardInterrupt):
            logger.info("Stopping the Execution Engine ...")
            self.stop()
            sys.exit(0)
        if self.debug_mode:
            raise ValueError(msg).with_traceback(e.__traceback__)
        else:
            logger.error(f"{msg}: {type(e).__name__}: {str(e)}")

    def _max_trades(self, mtrades):
        max_trades = {
            symbol: mtrades[symbol]
            if mtrades is not None and isinstance(mtrades, dict) and symbol in mtrades
            else self.trades_instances[symbol].max_trade()
            for symbol in self.symbols
        }
        return max_trades

    def _expert_ids(self, expert_ids):
        if expert_ids is None:
            expert_ids = list(
                set([trade.expert_id for trade in self.trades_instances.values()])
            )
        elif isinstance(expert_ids, int):
            expert_ids = [expert_ids]
        return expert_ids

    def _init_strategy(self, **kwargs) -> MT5Strategy:
        try:
            check_mt5_connection(**kwargs)
            strategy = self.strategy_cls(
                symbol_list=self.symbols, mode=TradingMode.LIVE, **kwargs
            )
        except Exception as e:
            self._print_exc(
                f"Initializing strategy, STRATEGY={self.STRATEGY}, ACCOUNT={self.ACCOUNT}",
                e,
            )
            return
        logger.info(
            f"Running {self.STRATEGY} Strategy in {self.time_frame} Interval ..., ACCOUNT={self.ACCOUNT}"
        )
        return strategy

    def _get_signal_info(self, signal, symbol, price, stoplimit):
        account = self.strategy.account
        symbol_info = account.get_symbol_info(symbol)

        common_data = {
            "signal": signal,
            "symbol": symbol,
            "strategy": self.STRATEGY,
            "timeframe": self.time_frame,
            "account": self.ACCOUNT,
        }

        info = (
            "SIGNAL={signal}, SYMBOL={symbol}, STRATEGY={strategy}, "
            "TIMEFRAME={timeframe}, ACCOUNT={account}"
        ).format(**common_data)

        sigmsg = (
            "SIGNAL={signal}\n"
            "SYMBOL={symbol}\n"
            "TYPE={symbol_type}\n"
            "DESCRIPTION={description}\n"
            "PRICE={price}\n"
            "STOPLIMIT={stoplimit}\n"
            "STRATEGY={strategy}\n"
            "TIMEFRAME={timeframe}\n"
            "BROKER={broker}\n"
            "TIMESTAMP={timestamp}"
        ).format(
            **common_data,
            symbol_type=account.get_symbol_type(symbol).value,
            description=symbol_info.description if symbol_info else "N/A",
            price=price if price else "MARKET",
            stoplimit=stoplimit,
            broker=account.broker.name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        msg_template = "SYMBOL={symbol}, STRATEGY={strategy}, ACCOUNT={account}"
        msg = f"Sending {signal} Order ... " + msg_template.format(**common_data)
        tfmsg = "Time Frame Not completed !!! " + msg_template.format(**common_data)
        riskmsg = "Risk not allowed !!! " + msg_template.format(**common_data)

        return info, sigmsg, msg, tfmsg, riskmsg

    def _check_retcode(self, trade: Trade, position):
        if len(trade.retcodes) > 0:
            for retcode in trade.retcodes:
                if retcode in NON_EXEC_RETCODES[position]:
                    return True
        return False

    def _check_positions_orders(self):
        positions_orders = {}
        try:
            check_mt5_connection(**self.kwargs)
            for order_type in POSITIONS_TYPES + ORDERS_TYPES:
                positions_orders[order_type] = {}
                for symbol in self.symbols:
                    positions_orders[order_type][symbol] = None
                    for id in self.expert_ids:
                        func = getattr(
                            self.trades_instances[symbol], f"get_current_{order_type}"
                        )
                        func_value = func(id=id)
                        if func_value is not None:
                            if positions_orders[order_type][symbol] is None:
                                positions_orders[order_type][symbol] = func_value
                            else:
                                positions_orders[order_type][symbol] += func_value
            return positions_orders
        except Exception as e:
            self._print_exc(
                f"Checking positions and orders, STRATEGY={self.STRATEGY} , ACCOUNT={self.ACCOUNT}",
                e,
            )

    def _long_short_market(self, buys, sells):
        long_market = {
            symbol: buys[symbol] is not None
            and len(buys[symbol]) >= self.max_trades[symbol]
            for symbol in self.symbols
        }
        short_market = {
            symbol: sells[symbol] is not None
            and len(sells[symbol]) >= self.max_trades[symbol]
            for symbol in self.symbols
        }
        return long_market, short_market

    def _display_positions_orders(self, positions_orders):
        for symbol in self.symbols:
            for order_type in POSITIONS_TYPES + ORDERS_TYPES:
                if positions_orders[order_type][symbol] is not None:
                    logger.info(
                        f"Current {order_type.upper()} SYMBOL={symbol}: \
                            {positions_orders[order_type][symbol]}, STRATEGY={self.STRATEGY} , ACCOUNT={self.ACCOUNT}"
                    )

    def _send_notification(self, signal, symbol):
        telegram = self.kwargs.get("telegram", False)
        bot_token = self.kwargs.get("bot_token")
        chat_id = self.kwargs.get("chat_id")
        notify = self.kwargs.get("notify", False)
        if symbol in self.signal_tickers:
            send_message(
                message=signal,
                notify_me=notify,
                telegram=telegram,
                token=bot_token,
                chat_id=chat_id,
            )

    def _logmsg(self, period, symbol):
        logger.info(
            f"End of the {period} !!! SYMBOL={symbol}, STRATEGY={self.STRATEGY} , ACCOUNT={self.ACCOUNT}"
        )

    def _logmsgif(self, period, symbol):
        if len(self.symbols) <= 10:
            self._logmsg(period, symbol)
        elif len(self.symbols) > 10 and symbol == self.symbols[-1]:
            logger.info(
                f"End of the {period} !!! STRATEGY={self.STRATEGY} , ACCOUNT={self.ACCOUNT}"
            )

    def _sleepmsg(self, sleep_time):
        logger.info(f"{self.ACCOUNT} Sleeping for {sleep_time} minutes ...\n")

    def _sleep_over_night(self, sessionmsg):
        sleep_time = self.trades_instances[self.symbols[-1]].sleep_time()
        self._sleepmsg(sleep_time + self.delay)
        time.sleep(60 * sleep_time + self.delay)
        logger.info(sessionmsg)

    def _sleep_over_weekend(self, sessionmsg):
        sleep_time = self.trades_instances[self.symbols[-1]].sleep_time(weekend=True)
        self._sleepmsg(sleep_time + self.delay)
        time.sleep(60 * sleep_time + self.delay)
        logger.info(sessionmsg)

    def _check_is_day_ends(self, trade: Trade, symbol, period_type, today, closing):
        if trade.days_end():
            self._logmsgif("Day", symbol) if today != FRIDAY else self._logmsgif(
                "Week", symbol
            )
            if (
                (
                    period_type == "month"
                    and today == FRIDAY
                    and self._is_month_ends()
                    and closing
                )
                or (period_type == "week" and today == FRIDAY and closing)
                or (period_type == "day" and closing)
                or (period_type == "24/7" and closing)
            ):
                logger.info(
                    f"{self.ACCOUNT} Closing all positions and orders for {symbol} ..."
                )
                for id in self.expert_ids:
                    trade.close_positions(
                        position_type="all", id=id, comment=self.comment
                    )
                trade.statistics(save=True)

    def _is_month_ends(self):
        today = pd.Timestamp(date.today())
        last_business_day = today + pd.tseries.offsets.BMonthEnd(0)
        return today == last_business_day

    def _daily_end_checks(self, today, closing, sessionmsg):
        self.strategy.perform_period_end_checks()
        if self.period_end_action == "break" and closing:
            sys.exit(0)
        elif self.period_end_action == "sleep" and today != FRIDAY or not closing:
            self._sleep_over_night(sessionmsg)
        elif self.period_end_action == "sleep" and today == FRIDAY:
            self._sleep_over_weekend(sessionmsg)

    def _weekly_end_checks(self, today, closing, sessionmsg):
        if today != FRIDAY:
            self._sleep_over_night(sessionmsg)
        elif today == FRIDAY:
            self.strategy.perform_period_end_checks()
            if self.period_end_action == "break" and closing:
                sys.exit(0)
            elif self.period_end_action == "sleep" or not closing:
                self._sleep_over_weekend(sessionmsg)

    def _monthly_end_cheks(self, today, closing, sessionmsg):
        if today != FRIDAY:
            self._sleep_over_night(sessionmsg)
        elif today == FRIDAY and self._is_month_ends() and closing:
            self.strategy.perform_period_end_checks()
            sys.exit(0)
        else:
            self._sleep_over_weekend(sessionmsg)

    def _perform_period_end_actions(
        self,
        today,
        day_end,
        closing,
        sessionmsg,
    ):
        period = self.period.lower()
        for symbol, trade in self.trades_instances.items():
            self._check_is_day_ends(trade, symbol, period, today, closing)

        if day_end:
            self.time_intervals = 0
            match period:
                case "24/7":
                    self.strategy.perform_period_end_checks()
                    self._sleep_over_night(sessionmsg)

                case "day":
                    self._daily_end_checks(today, closing, sessionmsg)

                case "week":
                    self._weekly_end_checks(today, closing, sessionmsg)

                case "month":
                    self._monthly_end_cheks(today, closing, sessionmsg)
                case _:
                    raise ValueError(f"Invalid period {period}")

    def _check(self, buys, sells, symbol):
        if not self.mm:
            return
        if buys is not None or sells is not None:
            self.trades_instances[symbol].break_even(
                mm=self.mm,
                trail=self.trail,
                stop_trail=self.stop_trail,
                trail_after_points=self.trail_after_points,
                be_plus_points=self.be_plus_points,
            )

    def _get_signals_and_weights(self):
        try:
            check_mt5_connection(**self.kwargs)
            signals = self.strategy.calculate_signals()
            weights = (
                self.strategy.apply_risk_management(self.optimizer)
                if hasattr(self.strategy, "apply_risk_management")
                else None
            )
            return signals, weights
        except Exception as e:
            self._print_exc(
                f"Calculating Signals, STRATEGY={self.STRATEGY} , ACCOUNT={self.ACCOUNT}",
                e,
            )
            pass

    def _update_risk(self, weights):
        try:
            check_mt5_connection(**self.kwargs)
            if weights is not None and not all(v == 0 for v in weights.values()):
                assert self.daily_risk is not None
                for symbol in self.symbols:
                    if symbol not in weights:
                        continue
                    trade = self.trades_instances[symbol]
                    dailydd = round(weights[symbol] * self.daily_risk, 5)
                    trade.dailydd = dailydd
        except Exception as e:
            self._print_exc(
                f"Updating Risk, STRATEGY={self.STRATEGY} , ACCOUNT={self.ACCOUNT}",
                e,
            )
            pass

    def _auto_trade(self, sigmsg, symbol) -> bool:
        if self.notify:
            self._send_notification(sigmsg, symbol)
        if self.auto_trade:
            return True
        if not self.auto_trade:
            prompt = (
                f"{sigmsg} \n Enter Y/Yes to accept or N/No to reject this order : "
            )
            if self.prompt_callback is not None:
                auto_trade = self.prompt_callback(prompt)
            else:
                auto_trade = input(prompt)
            if not auto_trade.upper().startswith("Y"):
                info = f"Order Rejected !!! SYMBOL={symbol}, STRATEGY={self.STRATEGY} , ACCOUNT={self.ACCOUNT}"
                logger.info(info)
                if self.notify:
                    self._send_notification(info, symbol)
                return False
        return True

    def _open_buy(
        self, signal, symbol, id, trade: Trade, price, stoplimit, sigmsg, msg, comment
    ):
        if not self._auto_trade(sigmsg, symbol):
            return
        if not self._check_retcode(trade, "BMKT"):
            logger.info(msg)
            trade.open_buy_position(
                action=signal,
                price=price,
                stoplimit=stoplimit,
                id=id,
                mm=self.mm,
                trail=self.trail,
                comment=comment,
            )

    def _open_sell(
        self, signal, symbol, id, trade: Trade, price, stoplimit, sigmsg, msg, comment
    ):
        if not self._auto_trade(sigmsg, symbol):
            return
        if not self._check_retcode(trade, "SMKT"):
            logger.info(msg)
            trade.open_sell_position(
                action=signal,
                price=price,
                stoplimit=stoplimit,
                id=id,
                mm=self.mm,
                trail=self.trail,
                comment=comment,
            )

    def _handle_exit_signals(self, signal, symbol, id, trade: Trade, sigmsg, comment):
        for exit_signal, actions in EXIT_SIGNAL_ACTIONS.items():
            if signal == exit_signal:
                for signal_attr, order_type in actions.items():
                    clos_func = getattr(
                        self.trades_instances[symbol], f"get_current_{signal_attr}"
                    )
                    if clos_func(id=id) is not None:
                        if self.notify:
                            self._send_notification(sigmsg, symbol)
                        close_method = (
                            trade.close_positions
                            if signal_attr in POSITIONS_TYPES
                            else trade.close_orders
                        )
                        close_method(order_type, id=id, comment=comment)

    def _handle_buy_signal(
        self,
        signal,
        symbol,
        id,
        trade,
        price,
        stoplimit,
        buys,
        sells,
        sigmsg,
        msg,
        tfmsg,
        riskmsg,
        comment,
    ):
        if not self.long_market[symbol]:
            if self.use_trade_time:
                if self.time_intervals % self.trade_time == 0 or buys[symbol] is None:
                    self._open_buy(
                        signal,
                        symbol,
                        id,
                        trade,
                        price,
                        stoplimit,
                        sigmsg,
                        msg,
                        comment,
                    )
                else:
                    logger.info(tfmsg)
                    self._check(buys[symbol], sells[symbol], symbol)
            else:
                self._open_buy(
                    signal, symbol, id, trade, price, stoplimit, sigmsg, msg, comment
                )
        else:
            logger.info(riskmsg)

    def _handle_sell_signal(
        self,
        signal,
        symbol,
        id,
        trade,
        price,
        stoplimit,
        buys,
        sells,
        sigmsg,
        msg,
        tfmsg,
        riskmsg,
        comment,
    ):
        if not self.short_market[symbol]:
            if self.use_trade_time:
                if self.time_intervals % self.trade_time == 0 or sells[symbol] is None:
                    self._open_sell(
                        signal,
                        symbol,
                        id,
                        trade,
                        price,
                        stoplimit,
                        sigmsg,
                        msg,
                        comment,
                    )
                else:
                    logger.info(tfmsg)
                    self._check(buys[symbol], sells[symbol], symbol)
            else:
                self._open_sell(
                    signal, symbol, id, trade, price, stoplimit, sigmsg, msg, comment
                )
        else:
            logger.info(riskmsg)

    def _run_trade_algorithm(
        self,
        signal,
        symbol,
        id,
        trade,
        price,
        stoplimit,
        buys,
        sells,
        comment,
    ):
        signal = {"LONG": "BMKT", "BUY": "BMKT", "SHORT": "SMKT", "SELL": "SMKT"}.get(
            signal, signal
        )
        info, sigmsg, msg, tfmsg, riskmsg = self._get_signal_info(
            signal, symbol, price, stoplimit
        )

        if signal not in EXIT_SIGNAL_ACTIONS:
            if signal in NON_EXEC_RETCODES and not self._check_retcode(trade, signal):
                logger.info(info)
            elif signal not in NON_EXEC_RETCODES:
                logger.info(info)

        if signal in EXIT_SIGNAL_ACTIONS:
            self._handle_exit_signals(signal, symbol, id, trade, sigmsg, comment)
        elif signal in BUYS:
            self._handle_buy_signal(
                signal,
                symbol,
                id,
                trade,
                price,
                stoplimit,
                buys,
                sells,
                sigmsg,
                msg,
                tfmsg,
                riskmsg,
                comment,
            )
        elif signal in SELLS:
            self._handle_sell_signal(
                signal,
                symbol,
                id,
                trade,
                price,
                stoplimit,
                buys,
                sells,
                sigmsg,
                msg,
                tfmsg,
                riskmsg,
                comment,
            )

    def _is_closing(self):
        closing = True
        if self.closing_pnl is not None:
            closing = all(
                trade.positive_profit(id=trade.expert_id, th=self.closing_pnl)
                for trade in self.trades_instances.values()
            )
        return closing

    def _sleep(self):
        time.sleep((60 * self.iter_time) - 1.0)
        if self.iter_time == 1:
            self.time_intervals += 1
        elif self.trade_time % self.iter_time == 0:
            self.time_intervals += self.iter_time
        else:
            if self.use_trade_time:
                raise ValueError(
                    f"iter_time must be a multiple of the {self.time_frame} !!!"
                    f"(e.g., if time_frame is 15m, iter_time must be 1.5, 3, 5, 15 etc)"
                )

    def _handle_one_signal(self, signal, today, buys, sells):
        try:
            symbol = signal.symbol
            trade: Trade = self.trades_instances[symbol]
            if trade.trading_time() and today in self.trading_days:
                if signal.action is not None:
                    action = (
                        signal.action.value
                        if isinstance(signal.action, TradeAction)
                        else signal.action
                    )
                    self._run_trade_algorithm(
                        action,
                        symbol,
                        signal.id,
                        trade,
                        signal.price,
                        signal.stoplimit,
                        buys,
                        sells,
                        signal.comment or self.comment,
                    )
            else:
                if len(self.symbols) >= 10:
                    if symbol == self.symbols[-1]:
                        logger.info(
                            f"Not trading Time !!!, STRATEGY={self.STRATEGY} , ACCOUNT={self.ACCOUNT}"
                        )
                else:
                    logger.info(
                        f"Not trading Time !!! SYMBOL={trade.symbol}, STRATEGY={self.STRATEGY} , ACCOUNT={self.ACCOUNT}"
                    )
                self._check(buys[symbol], sells[symbol], symbol)

        except Exception as e:
            msg = (
                f"Error handling signal for SYMBOL={signal.symbol} (SIGNAL: {action}), "
                f"STRATEGY={self.STRATEGY}, ACCOUNT={self.ACCOUNT}"
            )
            self._print_exc(msg, e)

    def _handle_all_signals(self, today, signals, buys, sells, max_workers=50):
        try:
            check_mt5_connection(**self.kwargs)
        except Exception as e:
            msg = "Initial MT5 connection check failed. Aborting signal processing."
            self._print_exc(msg, e)
            return

        if not signals:
            return

        # We want to create a temporary function that
        # already has the 'today', 'buys', and 'sells' arguments filled in.
        # This is necessary because executor.map only iterates over one sequence (signals).
        signal_processor = functools.partial(
            self._handle_one_signal, today=today, buys=buys, sells=sells
        )
        if self.multithread:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # 'map' will apply our worker function to every item in the 'signals' list.
                # It will automatically manage the distribution of tasks to the worker threads.
                # We wrap it in list() to ensure all tasks are complete before moving on.
                list(executor.map(signal_processor, signals))
        else:
            for signal in signals:
                try:
                    signal_processor(signal)
                except Exception as e:
                    self._print_exc(f"Failed to process signal {signal}: ", e)

    def _handle_period_end_actions(self, today):
        try:
            check_mt5_connection(**self.kwargs)
            day_end = all(trade.days_end() for trade in self.trades_instances.values())
            closing = self._is_closing()
            sessionmsg = f"{self.ACCOUNT} STARTING NEW TRADING SESSION ...\n"
            self._perform_period_end_actions(
                today,
                day_end,
                closing,
                sessionmsg,
            )
        except Exception as e:
            msg = f"Handling period end actions, STRATEGY={self.STRATEGY} , ACCOUNT={self.ACCOUNT}"
            self._print_exc(msg, e)
            pass

    def run(self):
        while self._running and not self.shutdown_event.is_set():
            try:
                check_mt5_connection(**self.kwargs)
                positions_orders = self._check_positions_orders()
                if self.show_positions_orders:
                    self._display_positions_orders(positions_orders)
                buys = positions_orders.get("buys")
                sells = positions_orders.get("sells")
                self.long_market, self.short_market = self._long_short_market(
                    buys, sells
                )
                today = datetime.now().strftime("%A").lower()
                signals, weights = self._get_signals_and_weights()
                if len(signals) == 0:
                    for symbol in self.symbols:
                        self._check(buys[symbol], sells[symbol], symbol)
                else:
                    self._update_risk(weights)
                    self._handle_all_signals(today, signals, buys, sells)
                self._sleep()
                self._handle_period_end_actions(today)
            except KeyboardInterrupt:
                self.stop()
                sys.exit(0)
            except Exception as e:
                msg = f"Running Execution Engine, STRATEGY={self.STRATEGY} , ACCOUNT={self.ACCOUNT}"
                self._print_exc(msg, e)
                self._sleep()

    def stop(self):
        """Stops the execution engine."""
        if self._running:
            logger.info(
                f"Stopping Execution Engine for {self.STRATEGY} STRATEGY on {self.ACCOUNT} Account"
            )
            self._running = False
            self.shutdown_event.set()
        logger.info("Execution Engine stopped successfully.")


def RunMt5Engine(account_id: str, **kwargs):
    """Starts an MT5 execution engine for a given account.
    Args:
        account_id: Account ID to run the execution engine on.
        **kwargs: Additional keyword arguments
            _ symbol_list : List of symbols to trade.
            - trades_instances : Dictionary of Trade instances.
            - strategy_cls : Strategy class to use for trading.
    """
    log.info(f"Starting execution engine for {account_id}")

    symbol_list = kwargs.pop("symbol_list")
    trades_instances = kwargs.pop("trades_instances")
    strategy_cls = kwargs.pop("strategy_cls")

    if symbol_list is None or trades_instances is None or strategy_cls is None:
        log.error(f"Missing required arguments for account {account_id}")
        raise ValueError(f"Missing required arguments for account {account_id}")

    try:
        engine = Mt5ExecutionEngine(
            symbol_list, trades_instances, strategy_cls, **kwargs
        )
        engine.run()
    except KeyboardInterrupt:
        log.info(f"Execution engine for {account_id} interrupted by user")
        engine.stop()
        sys.exit(0)
    except Exception as e:
        log.exception(f"Error running execution engine for {account_id}: {e}")
    finally:
        log.info(f"Execution for {account_id} completed")


def RunMt5Engines(accounts: Dict[str, Dict], start_delay: float = 1.0):
    """Runs multiple MT5 execution engines in parallel using multiprocessing.

    Args:
        accounts: Dictionary of accounts to run the execution engines on.
            Keys are the account names or IDs and values are the parameters for the execution engine.
            The parameters are the same as the ones passed to the `Mt5ExecutionEngine` class.
        start_delay: Delay in seconds between starting the processes. Defaults to 1.0.
    """

    processes = {}

    for account_id, params in accounts.items():
        log.info(f"Starting process for {account_id}")
        params["multithread"] = True
        process = mp.Process(target=RunMt5Engine, args=(account_id,), kwargs=params)
        process.start()
        processes[process] = account_id

        if start_delay:
            time.sleep(start_delay)

    for process, account_id in processes.items():
        process.join()
        log.info(f"Process for {account_id} joined")


class TWSExecutionEngine: ...
