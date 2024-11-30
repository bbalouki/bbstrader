import time
from datetime import datetime
from bbstrader.metatrader.trade import Trade
from bbstrader.btengine.strategy import(
    Strategy,
    MT5Strategy
)
from bbstrader.trading.scripts import send_message
from bbstrader.metatrader.account import check_mt5_connection
from typing import (
    Optional, 
    Literal, 
    Tuple, 
    List,  
    Dict
)

__all__ = [
    'MT5ExecutionEngine',
    'TWSExecutionEngine'
]

_TF_MAPPING = {
    '1m':   1,
    '3m':   3,
    '5m':   5,
    '10m':  10,
    '15m':  15,
    '30m':  30,
    '1h':   60,
    '2h':   120,
    '4h':   240,
    '6h':   360,
    '8h':   480,
    '12h':  720,
    'D1':   1440
}

TradingDays = [
    'monday', 
    'tuesday', 
    'wednesday', 
    'thursday', 
    'friday'
]

BUYS    = ['BMKT', 'BLMT', 'BSTP', 'BSTPLMT']
SELLS   = ['SMKT', 'SLMT', 'SSTP', 'SSTPLMT']

ORDERS_TYPES    = ["orders","buy_stops", "sell_stops", "buy_limits", 
                    "sell_limits", "buy_stop_limits", "sell_stop_limits"]
POSITIONS_TYPES = ["positions", "buys", "sells", "profitables", "losings"]

ACTIONS     = ["buys", "sells"]
STOPS       = ["buy_stops", "sell_stops"]
LIMITS      = ["buy_limits", "sell_limits"]
STOP_LIMITS = ["buy_stop_limits", "sell_stop_limits"]

EXIT_SIGNAL_ACTIONS = {
    "EXIT":                 {a: a[:-1] for a in ACTIONS},
    "EXIT_LONG":            {"buys": "buy"},
    "EXIT_SHORT":           {"sells": "sell"},
    "EXIT_STOP":            {stop: stop for stop in STOPS},
    "EXIT_LONG_STOP":       {"buy_stops": "buy_stops"},
    "EXIT_SHORT_STOP":      {"sell_stops": "sell_stops"},
    "EXIT_LIMIT":           {limit: limit for limit in LIMITS},
    "EXIT_LONG_LIMIT":      {"buy_limits": "buy_limits"},
    "EXIT_SHORT_LIMIT":     {"sell_limits": "sell_limits"},
    "EXIT_STOP_LIMIT":      {sl: sl for sl in STOP_LIMITS},
    "EXIT_LONG_STOP_LIMIT": {STOP_LIMITS[0]: STOP_LIMITS[0]},
    "EXIT_SHORT_STOP_LIMIT":{STOP_LIMITS[1]: STOP_LIMITS[1]},
    "EXIT_PROFITABLES":     {"profitables": "profitables"},
    "EXIT_LOSINGS":         {"losings": "losings"},
    "EXIT_ALL_POSITIONS":   {"positions": "all"},
    "EXIT_ALL_ORDERS":      {"orders": "all"}
}


def _mt5_execution(
        symbol_list, trades_instances, strategy_cls, /,
        mm, optimizer, trail, stop_trail, trail_after_points, be_plus_points, show_positions_orders,
        iter_time, use_trade_time, period, period_end_action, closing_pnl, trading_days,
        comment, **kwargs):
    symbols  = symbol_list.copy()
    time_frame = kwargs.get('time_frame', '15m')
    STRATEGY = kwargs.get('strategy_name')
    mtrades  = kwargs.get('max_trades')
    notify   = kwargs.get('notify', False)
    if notify:
        telegram  = kwargs.get('telegram', False)
        bot_token = kwargs.get('bot_token')
        chat_id   = kwargs.get('chat_id')
    
    def update_risk(weights):
        if weights is not None:
            for symbol in symbols:
                if symbol not in weights:
                    continue
                trade = trades_instances[symbol]
                trade.dailydd = round(weights[symbol], 5)

    def _send_notification(signal):
        send_message(message=signal, notify_me=notify, 
                    telegram=telegram, token=bot_token, chat_id=chat_id)
        
    logger   = trades_instances[symbols[0]].logger
    max_trades = {symbol: mtrades[symbol]
                  if mtrades is not None and symbol in mtrades
                  else trades_instances[symbol].max_trade() for symbol in symbols}
    if comment is None:
        trade = trades_instances[symbols[0]]
        comment = f"{trade.expert_name}@{trade.version}"

    def check(buys, sells, symbol):
        if not mm:
            return
        if buys is not None or sells is not None:
            trades_instances[symbol].break_even(
                mm=mm, trail=trail, stop_trail=stop_trail, 
                trail_after_points=trail_after_points, be_plus_points=be_plus_points)
    num_days        = 0
    time_intervals  = 0
    trade_time      = _TF_MAPPING[time_frame]

    long_market   = {symbol: False for symbol in symbols}
    short_market  = {symbol: False for symbol in symbols}
    try:
        check_mt5_connection(**kwargs)
        strategy: MT5Strategy = strategy_cls(symbol_list=symbols, mode='live', **kwargs)
    except Exception as e:
        logger.error(f"Initializing strategy, {e}, STRATEGY={STRATEGY}")
        return
    logger.info(
        f'Running {STRATEGY} Strategy in {time_frame} Interval ...')
    
    while True:
        try:
            check_mt5_connection(**kwargs)
            current_date = datetime.now()
            today = current_date.strftime("%A").lower()
            time.sleep(0.5)
            positions_orders = {}
            for type in POSITIONS_TYPES + ORDERS_TYPES:
                positions_orders[type] = {}
                for symbol in symbols:
                    positions_orders[type][symbol] = None
                    func = getattr(trades_instances[symbol], f"get_current_{type}")
                    positions_orders[type][symbol] = func()
            buys = positions_orders['buys']
            sells = positions_orders['sells']
            for symbol in symbols:
                for type in POSITIONS_TYPES + ORDERS_TYPES:
                    if positions_orders[type][symbol] is not None:
                        if show_positions_orders:
                            logger.info(
                                f"Current {type.upper()} SYMBOL={symbol}: \
                                    {positions_orders[type][symbol]}, STRATEGY={STRATEGY}")
            long_market = {
                symbol: buys[symbol] is not None 
                and len(buys[symbol]) >= max_trades[symbol] for symbol in symbols
            }
            short_market = {
                symbol: sells[symbol] is not None 
                and len(sells[symbol]) >= max_trades[symbol] for symbol in symbols
            }

        except Exception as e:
            logger.error(f"Handling Positions and Orders, {e}, STRATEGY={STRATEGY}")
            continue
        time.sleep(0.5)
        try:
            check_mt5_connection(**kwargs)
            signals = strategy.calculate_signals()
            weights = None
            if hasattr(strategy, 'apply_risk_management'):
                weights = strategy.apply_risk_management(optimizer)
            update_risk(weights)
        except Exception as e:
            logger.error(f"Calculating signal, {e}, STRATEGY={STRATEGY}")
            continue
        for symbol in symbols:
            try:
                check_mt5_connection(**kwargs)
                trade: Trade = trades_instances[symbol]
                tfmsg = f"Time Frame Not completed !!! SYMBOL={trade.symbol}, STRATEGY={STRATEGY}"
                riskmsg = f"Risk not allowed !!! SYMBOL={trade.symbol}, STRATEGY={STRATEGY}"
                signal = signals[symbol]
                if isinstance(signal, dict):
                    signal = signal['action']
                    price = signal['price']
                    stoplimit = signal.get('stoplimit')
                elif isinstance(signal, str):
                    price = None
                    stoplimit = None
                if trade.trading_time() and today in trading_days:
                    if signal is not None:
                        signal = 'BMKT' if signal == 'LONG' else signal
                        signal = 'SMKT' if signal == 'SHORT' else signal
                        info = f"SIGNAL = {signal}, SYMBOL={trade.symbol}, STRATEGY={STRATEGY}, TIMEFRAME={time_frame}"
                        msg = f"Sending {signal} Order ... SYMBOL={trade.symbol}, STRATEGY={STRATEGY}"
                        if signal not in EXIT_SIGNAL_ACTIONS:
                            logger.info(info)
                        if signal in EXIT_SIGNAL_ACTIONS:
                            for exit_signal, actions in EXIT_SIGNAL_ACTIONS.items():
                                for position_type, order_type in actions.items():
                                    if positions_orders[position_type][symbol] is not None:
                                        if notify:
                                            _send_notification(info)
                                        if position_type in POSITIONS_TYPES:
                                            trade.close_positions(position_type=order_type)
                                        else:
                                            trade.close_orders(order_type=order_type)
                        elif signal in BUYS and not long_market[symbol]:
                            if use_trade_time:
                                if time_intervals % trade_time == 0 or buys[symbol] is None:
                                    logger.info(msg)
                                    if notify:
                                        _send_notification(info)
                                    trade.open_buy_position(
                                        action=signal, price=price, stoplimit=stoplimit, mm=mm, comment=comment)
                                else:
                                    logger.info(tfmsg)
                                    check(buys[symbol], sells[symbol], symbol)
                            else:
                                logger.info(msg)
                                if notify:
                                    _send_notification(info)
                                trade.open_buy_position(
                                    action=signal, price=price, stoplimit=stoplimit, mm=mm, comment=comment)
                        elif signal in BUYS and long_market[symbol]:
                            logger.info(riskmsg)
                            check(buys[symbol], sells[symbol], symbol)

                        elif signal in SELLS and not short_market[symbol]:
                            if use_trade_time:
                                if time_intervals % trade_time == 0 or sells[symbol] is None:
                                    logger.info(msg)
                                    if notify:
                                        _send_notification(info)
                                    trade.open_sell_position(
                                        action=signal, price=price, stoplimit=stoplimit, mm=mm, comment=comment)
                                else:
                                    logger.info(tfmsg)
                                    check(buys[symbol], sells[symbol], symbol)
                            else:
                                logger.info(msg)
                                if notify:
                                    _send_notification(info)
                                trade.open_sell_position(
                                    action=signal, price=price, stoplimit=stoplimit, mm=mm, comment=comment)
                        elif signal in SELLS and short_market[symbol]:
                            logger.info(riskmsg)
                            check(buys[symbol], sells[symbol], symbol)
                    else:
                        check(buys[symbol], sells[symbol], symbol)
                else:
                    logger.info(
                        f"Not trading Time !!! SYMBOL={trade.symbol}, STRATEGY={STRATEGY}")
                    check(buys[symbol], sells[symbol], symbol)

            except Exception as e:
                logger.error(f"Handling Signals {e}, SYMBOL={symbol}, STRATEGY={STRATEGY}")
                continue
        time.sleep((60 * iter_time) - 1.0)
        if iter_time == 1:
            time_intervals += 1
        elif trade_time % iter_time == 0:
            time_intervals += iter_time
        else:
            raise ValueError(
                f"iter_time must be a multiple of the {time_frame} !!!"
                f"(e.g; if time_frame is 15m, iter_time must be 1.5, 3, 3, 15 etc)"
            )
        try:
            FRIDAY = 'friday'
            check_mt5_connection(**kwargs)
            day_end = all(trade.days_end() for trade in trades_instances.values())
            if closing_pnl is not None:
                closing = all(trade.positive_profit(id=trade.expert_id, th=closing_pnl) 
                                  for trade in trades_instances.values())
            else: closing = True
            logmsg = lambda period: logger.info(
                f"End of the {period} !!! SYMBOL={trade.symbol}, STRATEGY={STRATEGY}")
            sleepmsg = lambda sleep_time: logger.info(
                f"Sleeping for {sleep_time} minutes ...\n")
            sessionmsg = f"STARTING NEW TRADING SESSION ...\n"
            if period.lower() == 'day':
                for symbol in symbols:
                    trade = trades_instances[symbol]
                    if trade.days_end() and closing:
                        trade.close_positions(position_type='all', comment=comment)
                        logmsg("Day")
                        trade.statistics(save=True)
                if day_end:
                    if period_end_action == 'break' and closing:
                        break
                    elif period_end_action == 'sleep' and today != FRIDAY or not closing:
                        sleep_time = trades_instances[symbols[-1]].sleep_time()
                        sleepmsg(sleep_time)
                        time.sleep(60 * sleep_time)
                        logger.info(sessionmsg)
                    elif period_end_action == 'sleep' and today == FRIDAY:
                        sleep_time = trades_instances[symbols[-1]].sleep_time(weekend=True)
                        sleepmsg(sleep_time)
                        time.sleep(60 * sleep_time)
                        logger.info(sessionmsg)

            elif period.lower() == 'week':
                for symbol in symbols:
                    trade = trades_instances[symbol]
                    if trade.days_end() and today != FRIDAY:
                        logmsg("Day")

                    elif trade.days_end() and today == FRIDAY and closing:
                        trade.close_positions(position_type='all', comment=comment)
                        logmsg("Week")
                        trade.statistics(save=True)
                if day_end and today != FRIDAY:
                    sleep_time = trades_instances[symbols[-1]].sleep_time()
                    sleepmsg(sleep_time)
                    time.sleep(60 * sleep_time)
                    logger.info(sessionmsg)
                elif day_end and today == FRIDAY:
                    if period_end_action == 'break' and closing:
                        break
                    elif period_end_action == 'sleep' or not closing:
                        sleep_time = trades_instances[symbols[-1]].sleep_time(weekend=True)
                        sleepmsg(sleep_time)
                        time.sleep(60 * sleep_time)
                        logger.info(sessionmsg)

            elif period.lower() == 'month':
                for symbol in symbols:
                    trade = trades_instances[symbol]
                    if trade.days_end() and today != FRIDAY:
                        logmsg("Day")
                    elif trade.days_end() and today == FRIDAY:
                        logmsg("Week")
                    elif (
                        trade.days_end()
                        and today == FRIDAY
                        and num_days/len(symbols) >= 20
                    ) and closing:
                        trade.close_positions(position_type='all', comment=comment)
                        logmsg("Month")
                        trade.statistics(save=True)
                if day_end and today != FRIDAY:
                    sleep_time = trades_instances[symbols[-1]].sleep_time()
                    sleepmsg(sleep_time)
                    time.sleep(60 * sleep_time)
                    logger.info(sessionmsg)
                    num_days += 1
                elif day_end and today == FRIDAY:
                    sleep_time = trades_instances[symbols[-1]
                                                ].sleep_time(weekend=True)
                    sleepmsg(sleep_time)
                    time.sleep(60 * sleep_time)
                    logger.info(sessionmsg)
                    num_days += 1
                elif (day_end
                        and today == FRIDAY
                        and num_days/len(symbols) >= 20
                    ):
                    break
        except Exception as e:
            logger.error(f"Handling period end actions, {e}, STRATEGY={STRATEGY}")
            continue


def _tws_execution(*args, **kwargs):
    raise NotImplementedError("TWS Execution is not yet implemented !!!")


class MT5ExecutionEngine():
    """
    The `MT5ExecutionEngine` class serves as the central hub for executing your trading strategies within the `bbstrader` framework. 
    It orchestrates the entire trading process, ensuring seamless interaction between your strategies, market data, and your chosen 
    trading platform.

    Key Features
    ------------

    - **Strategy Execution:** The `MT5ExecutionEngine` is responsible for running your strategy, retrieving signals, and executing trades based on those signals.
    - **Time Management:** You can define a specific time frame for your trades and set the frequency with which the engine checks for signals and manages trades.
    - **Trade Period Control:** Define whether your strategy runs for a day, a week, or a month, allowing for flexible trading durations.
    - **Money Management:** The engine supports optional money management features, allowing you to control risk and optimize your trading performance.
    - **Trading Day Configuration:** You can customize the days of the week your strategy will execute, providing granular control over your trading schedule.
    - **Platform Integration:** The `MT5ExecutionEngine` is currently designed to work with MT5.

    Examples
    --------
    
    >>> from bbstrader.metatrader import create_trade_instance
    >>> from bbstrader.trading.execution import MT5ExecutionEngine
    >>> from bbstrader.trading.strategies import StockIndexCFDTrading
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
    >>>     engine = MT5ExecutionEngine(
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

    def __init__(self,
                 symbol_list:        List[str],
                 trades_instances:   Dict[str, Trade],
                 strategy_cls:       Strategy,
                 /,
                 mm:                 bool = True,
                 optimizer:          str = 'equal',
                 trail:              bool = True,
                 stop_trail:         Optional[int] = None,
                 trail_after_points: Optional[int] = None,
                 be_plus_points:     Optional[int] = None,
                 show_positions_orders: bool = False,
                 iter_time:          int | float = 5,
                 use_trade_time:     bool = True,
                 period:             Literal['day', 'week', 'month'] = 'week',
                 period_end_action:  Literal['break', 'sleep'] = 'break',
                 closing_pnl:        Optional[float] = None,
                 trading_days:       Optional[List[str]] = TradingDays,
                 comment:            Optional[str] = None,
                 **kwargs
        ):
        """
        Args:
            symbol_list : List of symbols to trade
            trades_instances : Dictionary of Trade instances
            strategy_cls : Strategy class to use for trading
            mm : Enable Money Management. Defaults to False.
            optimizer : Risk management optimizer. Defaults to 'equal'.
                See `bbstrader.models.optimization` module for more information.
            show_positions_orders : Print open positions and orders. Defaults to False.
            iter_time : Interval to check for signals and `mm`. Defaults to 5.
            use_trade_time : Open trades after the time is completed. Defaults to True.
            period : Period to trade. Defaults to 'week'.
            period_end_action : Action to take at the end of the period. Defaults to 'break', 
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

        Note:
            1. For `trail` , `stop_trail` , `trail_after_points` , `be_plus_points` see `bbstrader.metatrader.trade.Trade.break_even()` .
            2. All Strategies must inherit from `bbstrader.btengine.strategy.Strategy` or `bbstrader.btengine.strategy.MT5Strategy` class
            and have a `calculate_signals` method that returns a dictionary of signals for each symbol in symbol_list.
            
            3. All strategies must have the following arguments in their `__init__` method:
                - bars (DataHandler): DataHandler instance default to None
                - events (Queue): Queue instance default to None
                - symbol_list (List[str]): List of symbols to trade can be none for backtesting
                - mode (str): Mode of the strategy. Must be either 'live' or 'backtest'
                - **kwargs: Additional keyword arguments
                    The keyword arguments are all the additional arguments passed to the `MT5ExecutionEngine` class,
                    the `Strategy` class, the `DataHandler` class, the `Portfolio` class and the `ExecutionHandler` class.
                - The `bars` and `events` arguments are used for backtesting only.
            
            4. All strategies must generate signals for backtesting and live trading.
            See the `bbstrader.trading.strategies` module for more information on how to create custom strategies.
            See `bbstrader.metatrader.account.check_mt5_connection()` for more details on how to connect to MT5 terminal.
        """
        self.symbol_list = symbol_list
        self.trades_instances = trades_instances
        self.strategy_cls = strategy_cls
        self.mm = mm
        self.optimizer = optimizer
        self.trail = trail
        self.stop_trail = stop_trail
        self.trail_after_points = trail_after_points
        self.be_plus_points = be_plus_points
        self.show_positions_orders = show_positions_orders
        self.iter_time = iter_time
        self.use_trade_time = use_trade_time
        self.period = period
        self.period_end_action = period_end_action
        self.closing_pnl = closing_pnl
        self.trading_days = trading_days
        self.comment = comment
        self.kwargs = kwargs

    def __repr__(self):
        trades = self.trades_instances.keys()
        s = self.strategy_cls.__name__
        return f"MT5ExecutionEngine(Symbols={list(trades)}, Strategy={s})"
    
    def run(self):
        check_mt5_connection(**self.kwargs)
        _mt5_execution(
            self.symbol_list,
            self.trades_instances,
            self.strategy_cls,
            mm=self.mm,
            optimizer=self.optimizer,
            trail=self.trail,
            stop_trail=self.stop_trail,
            trail_after_points=self.trail_after_points,
            be_plus_points=self.be_plus_points,
            show_positions_orders=self.show_positions_orders,
            iter_time=self.iter_time,
            use_trade_time=self.use_trade_time,
            period=self.period,
            period_end_action=self.period_end_action,
            closing_pnl=self.closing_pnl,
            trading_days=self.trading_days,
            comment=self.comment,
            **self.kwargs
        )


class TWSExecutionEngine():...
