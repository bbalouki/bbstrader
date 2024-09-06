from math import log
import time
from datetime import datetime
from bbstrader.metatrader.trade import Trade
from bbstrader.trading.strategies import Strategy
from typing import Optional, Literal, List, Tuple, Dict
import MetaTrader5 as mt5
from bbstrader.metatrader.account import INIT_MSG
from bbstrader.metatrader.utils import raise_mt5_error


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

TRADING_DAYS = [
    'monday', 
    'tuesday', 
    'wednesday', 
    'thursday', 
    'friday'
]

def _check_mt5_connection():
    if not mt5.initialize():
        raise_mt5_error(INIT_MSG)

def _mt5_execution(
    symbol_list, trades_instances, strategy_cls, /,
        mm, time_frame, iter_time, period, trading_days,
        comment, **kwargs
):
    symbols = symbol_list.copy()
    STRATEGY = kwargs.get('strategy_name')
    _max_trades = kwargs.get('max_trades')
    logger = kwargs.get('logger')
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
            trades_instances[symbol].break_even(mm=mm)
        if sells is not None:
            logger.info(
                f"Checking for Break even, SYMBOL={symbol}...STRATEGY={STRATEGY}")
            trades_instances[symbol].break_even(mm=mm)
    num_days = 0
    time_intervals = 0
    trade_time = _TF_MAPPING[time_frame]

    long_market = {symbol: False for symbol in symbols}
    short_market = {symbol: False for symbol in symbols}

    logger.info(
        f'Running {STRATEGY} Strategy on {symbols} in {time_frame} Interval ...')
    strategy: Strategy = strategy_cls(
        symbol_list=symbols, mode='live', **kwargs)

    while True:
        try:
            _check_mt5_connection()
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
        time.sleep(0.5)
        for symbol in symbols:
            try:
                trade = trades_instances[symbol]
                logger.info(
                    f"Calculating signal... SYMBOL={trade.symbol}, STRATEGY={STRATEGY}")
                signal = strategy.calculate_signals()[symbol]
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
        if period.lower() == 'day':
            for symbol in symbols:
                trade = trades_instances[symbol]
                if trade.days_end():
                    trade.close_positions(position_type='all', comment=comment)
                    logger.info(
                        f"End of the Day !!! SYMBOL={trade.symbol}, STRATEGY={STRATEGY}")
                    trade.statistics(save=True)
            if trades_instances[symbols[-1]].days_end():
                break

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
            if trades_instances[symbols[-1]].days_end() and today != 'friday':
                sleep_time = trades_instances[symbols[-1]].sleep_time()
                logger.info(f"Sleeping for {sleep_time} minutes ...")
                time.sleep(60 * sleep_time)
                logger.info("\nSTARTING NEW TRADING SESSION ...")
            elif trades_instances[symbols[-1]].days_end() and today == 'friday':
                break

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
            if trades_instances[symbols[-1]].days_end() and today != 'friday':
                sleep_time = trades_instances[symbols[-1]].sleep_time()
                logger.info(f"Sleeping for {sleep_time} minutes ...")
                time.sleep(60 * sleep_time)
                logger.info("\nSTARTING NEW TRADING SESSION ...")
                num_days += 1
            elif trades_instances[symbols[-1]].days_end() and today == 'friday':
                sleep_time = trades_instances[symbols[-1]
                                              ].sleep_time(weekend=True)
                logger.info(f"Sleeping for {sleep_time} minutes ...")
                time.sleep(60 * sleep_time)
                logger.info("\nSTARTING NEW TRADING SESSION ...")
                num_days += 1
            elif (trades_instances[symbols[-1]].days_end()
                  and today == 'friday'
                  and num_days/len(symbols) >= 20
                  ):
                break


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
                 time_frame:         Optional[str] = '15m',
                 iter_time:          Optional[int | float] = 5,
                 period:             Literal['day', 'week', 'month'] = 'week',
                 trading_days:       Optional[List[str]] = TRADING_DAYS,
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
            trading_days : Trading days in a week. Defaults to monday to friday.
            comment: Comment for trades. Defaults to None.
            **kwargs: Additional keyword arguments
                - strategy_name (Optional[str]): Strategy name. Defaults to None.
                - max_trades (Dict[str, int]): Maximum trades per symbol. Defaults to None.
                - logger (Optional[logging.Logger]): Logger instance. Defaults to None.

        Note:
            1. All Strategies must inherit from `bbstrader.btengine.strategy.Strategy` class
            and have a `calculate_signals` method that returns a dictionary of signals for each symbol in symbol_list.
            
            2. All strategies must have the following arguments in their `__init__` method:
                - bars (DataHandler): DataHandler instance default to None
                - events (Queue): Queue instance default to None
                - symbol_list (List[str]): List of symbols to trade can be none for backtesting
                - mode (str): Mode of the strategy. Must be either 'live' or 'backtest'
                - **kwargs: Additional keyword arguments
                    The keyword arguments are all the additional arguments passed to the `ExecutionEngine` class,
                    the `Strategy` class, the `DataHandler` class, the `Portfolio` class and the `ExecutionHandler` class.
                - The `bars` and `events` arguments are used for backtesting only.
            
            3. All strategies must generate signals for backtesting and live trading.
            See the `bbstrader.trading.strategies` module for more information on how to create custom strategies.
        """
        self.symbol_list = symbol_list
        self.trades_instances = trades_instances
        self.strategy_cls = strategy_cls
        self.mm = mm
        self.time_frame = time_frame
        self.iter_time = iter_time
        self.period = period
        self.trading_days = trading_days
        self.comment = comment
        self.kwargs = kwargs

    def run(self, terminal: Literal['MT5', 'TWS']):
        if terminal not in _TERMINALS:
            raise ValueError(
                f"Invalid terminal: {terminal}. Must be either 'MT5' or 'TWS'")
        _TERMINALS[terminal](
                self.symbol_list,
                self.trades_instances,
                self.strategy_cls,
                mm=self.mm,
                time_frame=self.time_frame,
                iter_time=self.iter_time,
                period=self.period,
                trading_days=self.trading_days,
                comment=self.comment,
                **self.kwargs
            )
