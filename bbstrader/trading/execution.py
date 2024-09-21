import time
import MetaTrader5 as mt5
from datetime import datetime
from bbstrader.metatrader.trade import Trade
from bbstrader.trading.strategies import Strategy
from bbstrader.metatrader.account import check_mt5_connection
from typing import Optional, Literal, Tuple, List,  Dict


_TF_MAPPING = {
    '1m':  1,
    '3m':  3,
    '5m':  5,
    '10m': 10,
    '15m': 15,
    '30m': 30,
    '1h':  60,
    '2h':  120,
    '4h':  240,
    'D1':  1440
}

TradingDays = [
    'monday', 
    'tuesday', 
    'wednesday', 
    'thursday', 
    'friday'
]


def _mt5_execution(
        symbol_list, trades_instances, strategy_cls, /,
        mm, trail, stop_trail, trail_after_points, be_plus_points, 
        time_frame, iter_time, period, period_end_action, trading_days,
        comment, **kwargs):
    symbols = symbol_list.copy()
    STRATEGY = kwargs.get('strategy_name')
    _max_trades = kwargs.get('max_trades')
    logger = trades_instances[symbols[0]].logger
    max_trades = {symbol: _max_trades[symbol] for symbol in symbols}
    if comment is None:
        trade = trades_instances[symbols[0]]
        comment = f"{trade.expert_name}@{trade.version}"

    def check(buys: List, sells: List, symbol: str):
        if not mm:
            return
        if buys is not None:
            logger.info(
                f"Checking for Break even, SYMBOL={symbol}...STRATEGY={STRATEGY}")
            trades_instances[symbol].break_even(
                mm=mm, trail=trail, stop_trail=stop_trail, 
                trail_after_points=trail_after_points, be_plus_points=be_plus_points)
        if sells is not None:
            logger.info(
                f"Checking for Break even, SYMBOL={symbol}...STRATEGY={STRATEGY}")
            trades_instances[symbol].break_even(
                mm=mm, trail=trail, stop_trail=stop_trail, 
                trail_after_points=trail_after_points, be_plus_points=be_plus_points)
    num_days = 0
    time_intervals = 0
    trade_time = _TF_MAPPING[time_frame]

    long_market = {symbol: False for symbol in symbols}
    short_market = {symbol: False for symbol in symbols}
    try:
        check_mt5_connection()
        strategy: Strategy = strategy_cls(symbol_list=symbols, mode='live', **kwargs)
    except Exception as e:
        logger.error(f"Error initializing strategy, {e}, STRATEGY={STRATEGY}")
        return
    logger.info(
        f'Running {STRATEGY} Strategy on {symbols} in {time_frame} Interval ...')
    
    while True:
        try:
            check_mt5_connection()
            current_date = datetime.now()
            today = current_date.strftime("%A").lower()
            time.sleep(0.5)
            buys = {
                symbol: trades_instances[symbol].get_current_buys()
                for symbol in symbols
            }
            sells = {
                symbol: trades_instances[symbol].get_current_sells()
                for symbol in symbols
            }
            for symbol in symbols:
                if buys[symbol] is not None:
                    logger.info(
                        f"Current buy positions SYMBOL={symbol}: {buys[symbol]}, STRATEGY={STRATEGY}")
                if sells[symbol] is not None:
                    logger.info(
                        f"Current sell positions SYMBOL={symbol}: {sells[symbol]}, STRATEGY={STRATEGY}")
            long_market = {symbol: buys[symbol] is not None and len(
                buys[symbol]) >= max_trades[symbol] for symbol in symbols}
            short_market = {symbol: sells[symbol] is not None and len(
                sells[symbol]) >= max_trades[symbol] for symbol in symbols}
        except Exception as e:
            logger.error(f"{e}, STRATEGY={STRATEGY}")
            continue
        time.sleep(0.5)
        try:
            check_mt5_connection()
            signals = strategy.calculate_signals()
        except Exception as e:
            logger.error(f"Calculating signal, {e}, STRATEGY={STRATEGY}")
            continue
        for symbol in symbols:
            try:
                check_mt5_connection()
                trade = trades_instances[symbol]
                logger.info(
                    f"Calculating signal... SYMBOL={trade.symbol}, STRATEGY={STRATEGY}")
                signal = signals[symbol]
                if trade.trading_time() and today in trading_days:
                    if signal is not None:
                        logger.info(
                            f"SIGNAL = {signal}, SYMBOL={trade.symbol}, STRATEGY={STRATEGY}")
                        if signal in ("EXIT", "EXIT_LONG") and long_market[symbol]:
                            trade.close_positions(position_type='buy')
                        elif signal in ("EXIT", "EXIT_SHORT") and short_market[symbol]:
                            trade.close_positions(position_type='sell')
                        elif signal == "LONG" and not long_market[symbol]:
                            if time_intervals % trade_time == 0 or buys[symbol] is None:
                                logger.info(
                                    f"Sending buy Order ... SYMBOL={trade.symbol}, STRATEGY={STRATEGY}")
                                trade.open_buy_position(mm=mm, comment=comment)
                            else:
                                check(buys[symbol], sells[symbol], symbol)
                        elif signal == "LONG" and long_market[symbol]:
                            logger.info(
                                f"Sorry Risk not allowed !!! SYMBOL={trade.symbol}, STRATEGY={STRATEGY}")
                            check(buys[symbol], sells[symbol], symbol)

                        elif signal == "SHORT" and not short_market[symbol]:
                            if time_intervals % trade_time == 0 or sells[symbol] is None:
                                logger.info(
                                    f"Sending sell Order ... SYMBOL={trade.symbol}, STRATEGY={STRATEGY}")
                                trade.open_sell_position(
                                    mm=mm, comment=comment)
                            else:
                                check(buys[symbol], sells[symbol], symbol)
                        elif signal == "SHORT" and short_market[symbol]:
                            logger.info(
                                f"Sorry Risk not allowed !!! SYMBOL={trade.symbol}, STRATEGY={STRATEGY}")
                            check(buys[symbol], sells[symbol], symbol)
                    else:
                        logger.info(
                            f"There is no signal !! SYMBOL={trade.symbol}, STRATEGY={STRATEGY}")
                        check(buys[symbol], sells[symbol], symbol)
                else:
                    logger.info(
                        f"Sorry It is Not trading Time !!! SYMBOL={trade.symbol}, STRATEGY={STRATEGY}")
                    check(buys[symbol], sells[symbol], symbol)

            except Exception as e:
                logger.error(f"{e}, SYMBOL={symbol}, STRATEGY={STRATEGY}")
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
        print()
        try:
            check_mt5_connection()
            day_end = all(trade.days_end() for trade in trades_instances.values())
            if period.lower() == 'day':
                for symbol in symbols:
                    trade = trades_instances[symbol]
                    if trade.days_end():
                        trade.close_positions(position_type='all', comment=comment)
                        logger.info(
                            f"End of the Day !!! SYMBOL={trade.symbol}, STRATEGY={STRATEGY}")
                        trade.statistics(save=True)
                if day_end:
                    if period_end_action == 'break':
                        break
                    elif period_end_action == 'sleep':
                        sleep_time = trades_instances[symbols[-1]].sleep_time()
                        logger.info(f"Sleeping for {sleep_time} minutes ...\n")
                        time.sleep(60 * sleep_time)
                        logger.info("STARTING NEW TRADING SESSION ...\n")

            elif period.lower() == 'week':
                for symbol in symbols:
                    trade = trades_instances[symbol]
                    if trade.days_end() and today != 'friday':
                        logger.info(
                            f"End of the Day !!! SYMBOL={trade.symbol}, STRATEGY={STRATEGY}")

                    elif trade.days_end() and today == 'friday':
                        trade.close_positions(position_type='all', comment=comment)
                        logger.info(
                            f"End of the Week !!! SYMBOL={trade.symbol}, STRATEGY={STRATEGY}")
                        trade.statistics(save=True)
                if day_end and today != 'friday':
                    sleep_time = trades_instances[symbols[-1]].sleep_time()
                    logger.info(f"Sleeping for {sleep_time} minutes ...\n")
                    time.sleep(60 * sleep_time)
                    logger.info("STARTING NEW TRADING SESSION ...\n")
                elif day_end and today == 'friday':
                    if period_end_action == 'break':
                        break
                    elif period_end_action == 'sleep':
                        sleep_time = trades_instances[symbols[-1]].sleep_time(weekend=True)
                        logger.info(f"Sleeping for {sleep_time} minutes ...\n")
                        time.sleep(60 * sleep_time)
                        logger.info("STARTING NEW TRADING SESSION ...\n")

            elif period.lower() == 'month':
                for symbol in symbols:
                    trade = trades_instances[symbol]
                    if trade.days_end() and today != 'friday':
                        logger.info(
                            f"End of the Day !!! SYMBOL={trade.symbol}, STRATEGY={STRATEGY}")

                    elif trade.days_end() and today == 'friday':
                        logger.info(
                            f"End of the Week !!! SYMBOL={trade.symbol}, STRATEGY={STRATEGY}")
                    elif (
                        trade.days_end()
                        and today == 'friday'
                        and num_days/len(symbols) >= 20
                    ):
                        trade.close_positions(position_type='all', comment=comment)
                        logger.info(
                            f"End of the Month !!! SYMBOL={trade.symbol}, STRATEGY={STRATEGY}")
                        trade.statistics(save=True)
                if day_end and today != 'friday':
                    sleep_time = trades_instances[symbols[-1]].sleep_time()
                    logger.info(f"Sleeping for {sleep_time} minutes ...\n")
                    time.sleep(60 * sleep_time)
                    logger.info("STARTING NEW TRADING SESSION ...\n")
                    num_days += 1
                elif day_end and today == 'friday':
                    sleep_time = trades_instances[symbols[-1]
                                                ].sleep_time(weekend=True)
                    logger.info(f"Sleeping for {sleep_time} minutes ...\n")
                    time.sleep(60 * sleep_time)
                    logger.info("STARTING NEW TRADING SESSION ...\n")
                    num_days += 1
                elif (day_end
                        and today == 'friday'
                        and num_days/len(symbols) >= 20
                    ):
                    break
        except Exception as e:
            logger.error(f"Handling period end actions, {e}, STRATEGY={STRATEGY}")
            continue


def _tws_execution(*args, **kwargs):
    raise NotImplementedError("TWS Execution is not yet implemented !!!")

_TERMINALS = {
    'MT5': _mt5_execution,
    'TWS': _tws_execution
}
class ExecutionEngine():
    """
    The `ExecutionEngine` class serves as the central hub for executing your trading strategies within the `bbstrader` framework. 
    It orchestrates the entire trading process, ensuring seamless interaction between your strategies, market data, and your chosen 
    trading platform (currently MetaTrader 5 (MT5) and Interactive Brokers TWS).

    Key Features
    ------------

    - **Strategy Execution:** The `ExecutionEngine` is responsible for running your strategy, retrieving signals, and executing trades based on those signals.
    - **Time Management:** You can define a specific time frame for your trades and set the frequency with which the engine checks for signals and manages trades.
    - **Trade Period Control:** Define whether your strategy runs for a day, a week, or a month, allowing for flexible trading durations.
    - **Money Management:** The engine supports optional money management features, allowing you to control risk and optimize your trading performance.
    - **Trading Day Configuration:** You can customize the days of the week your strategy will execute, providing granular control over your trading schedule.
    - **Platform Integration:** The `ExecutionEngine` is currently designed to work with both MT5 and TWS platforms, ensuring compatibility and flexibility in your trading environment.

    Examples
    --------
    
    >>> from bbstrader.metatrader import create_trade_instance
    >>> from bbstrader.trading.execution import ExecutionEngine
    >>> from bbstrader.trading.strategies import StockIndexCFDTrading
    >>> from bbstrader.metatrader.utils import config_logger
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
    >>>     engine = ExecutionEngine(
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
    >>>     engine.run(terminal='MT5')
    """

    def __init__(self,
                 symbol_list:        List[str],
                 trades_instances:   Dict[str, Trade],
                 strategy_cls:       Strategy,
                 /,
                 mm:                 Optional[bool] = True,
                 trail:              Optional[bool] = True,
                 stop_trail:         Optional[int] = None,
                 trail_after_points: Optional[int] = None,
                 be_plus_points:     Optional[int] = None,
                 time_frame:         Optional[str] = '15m',
                 iter_time:          Optional[int | float] = 5,
                 period:             Literal['day', 'week', 'month'] = 'week',
                 period_end_action:  Literal['break', 'sleep'] = 'break',
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
            time_frame : Time frame to trade. Defaults to '15m'.
            iter_time : Interval to check for signals and `mm`. Defaults to 5.
            period : Period to trade. Defaults to 'week'.
            period_end_action : Action to take at the end of the period. Defaults to 'break', 
                this only applies when period is 'day', 'week'.
            trading_days : Trading days in a week. Defaults to monday to friday.
            comment: Comment for trades. Defaults to None.
            **kwargs: Additional keyword arguments
                - strategy_name (Optional[str]): Strategy name. Defaults to None.
                - max_trades (Dict[str, int]): Maximum trades per symbol. Defaults to None.

        Note:
            1. For `trail` , `stop_trail` , `trail_after_points` , `be_plus_points` see `bbstrader.metatrader.trade.Trade.break_even()` .
            2. All Strategies must inherit from `bbstrader.btengine.strategy.Strategy` class
            and have a `calculate_signals` method that returns a dictionary of signals for each symbol in symbol_list.
            
            3. All strategies must have the following arguments in their `__init__` method:
                - bars (DataHandler): DataHandler instance default to None
                - events (Queue): Queue instance default to None
                - symbol_list (List[str]): List of symbols to trade can be none for backtesting
                - mode (str): Mode of the strategy. Must be either 'live' or 'backtest'
                - **kwargs: Additional keyword arguments
                    The keyword arguments are all the additional arguments passed to the `ExecutionEngine` class,
                    the `Strategy` class, the `DataHandler` class, the `Portfolio` class and the `ExecutionHandler` class.
                - The `bars` and `events` arguments are used for backtesting only.
            
            4. All strategies must generate signals for backtesting and live trading.
            See the `bbstrader.trading.strategies` module for more information on how to create custom strategies.
        """
        self.symbol_list = symbol_list
        self.trades_instances = trades_instances
        self.strategy_cls = strategy_cls
        self.mm = mm
        self.trail = trail
        self.stop_trail = stop_trail
        self.trail_after_points = trail_after_points
        self.be_plus_points = be_plus_points
        self.time_frame = time_frame
        self.iter_time = iter_time
        self.period = period
        self.period_end_action = period_end_action
        self.trading_days = trading_days
        self.comment = comment
        self.kwargs = kwargs

    def run(self, terminal: Literal['MT5', 'TWS']):
        if terminal not in _TERMINALS:
            raise ValueError(
                f"Invalid terminal: {terminal}. Must be either 'MT5' or 'TWS'")
        elif terminal == 'MT5':
            check_mt5_connection()
        _TERMINALS[terminal](
                self.symbol_list,
                self.trades_instances,
                self.strategy_cls,
                mm=self.mm,
                trail=self.trail,
                stop_trail=self.stop_trail,
                trail_after_points=self.trail_after_points,
                be_plus_points=self.be_plus_points,
                time_frame=self.time_frame,
                iter_time=self.iter_time,
                period=self.period,
                period_end_action=self.period_end_action,
                trading_days=self.trading_days,
                comment=self.comment,
                **self.kwargs
            )
