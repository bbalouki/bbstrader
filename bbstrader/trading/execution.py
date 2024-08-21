import time
import pandas as pd
import numpy as np
from datetime import datetime
from metatrader.rates import Rates
from metatrader.trade import Trade
from .utils import tf_mapping
from strategies import (
    ArimaGarchStrategy, SMAStrategy, KLFStrategy, OrnsteinUhlenbeck,
)
from models import HMMRiskManager
from metatrader.utils import config_logger
from typing import Optional, Literal, List, Tuple

logger = config_logger(log_file='trade.log', console_log=False)

TRADING_DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
# ========  SMA TRADING ======================
def sma_trading(
    trade: Trade,
    tf: Optional[str] = '1h',
    sma: Optional[int] = 35,
    lma: Optional[int] = 80,
    mm: Optional[bool] = True,
    max_t: Optional[int] = 1,
    iter_time: Optional[int |float] = 30,
    risk_manager: str = 'hmm',
    period: Literal['day', 'week', 'month'] = 'week',
    **kwargs
):
    """
    Executes a Simple Moving Average (SMA) trading strategy 
    with optional risk management using Hidden Markov Models (HMM).

    Parameters
    ==========
    trade : Trade
        The Trade object that encapsulates trading operations like 
        opening, closing positions, etc.
    tf : str, optional
        Time frame for the trading strategy, defaults to '1h' (1 hour).
    sma : int, optional
        Short Moving Average period, defaults to 35.
    lma : int, optional
        Long Moving Average period, defaults to 80.
    mm : bool, optional
        Money management flag to enable/disable money management, defaults to True.
    max_t : int, optional
        Maximum number of trades allowed, defaults to 1.
    iter_time : Union[int, float], optional
        Iteration time for the trading loop, defaults to 30 seconds.
    risk_manager : str
        Specifies the risk management strategy to use, 
        'hmm' for Hidden Markov Model.
    period : Literal['day', 'week', 'month'], optional
        Trading period to reset statistics and close positions, 
        can be 'day', 'week', or 'month', defaults to 'week'.
    **kwargs
        Additional keyword arguments for the HMM risk manager.

    Returns
    =======
    None

    Notes
    =====
    This function integrates a trading strategy based on simple moving averages 
    with an optional risk management layer using HMM.
    It periodically checks for trading signals and executes buy or sell orders 
    based on the strategy signals and risk management conditions.
    The trading period (day, week, month) determines when to reset statistics 
    and close all positions.

    This function includes an infinite loop with time delays designed 
    to run continuously during market hours. Ensure proper exception handling 
    and resource management when integrating into a live trading environment.
    """

    def check(buys: list, sells: list):
        if buys is not None or sells is not None:
            logger.info(f"Checking for Break even SYMBOL={trade.symbol}...")
            trade.break_even()

    time_frame_mapping = tf_mapping()
    if tf == 'D1':
        trade_time = trade.get_minutes()
    else:
        trade_time = time_frame_mapping[tf]

    rate = Rates(trade.symbol, tf, 0)
    data = rate.get_rates_from_pos()
    strategy = SMAStrategy(short_window=sma, long_window=lma)
    hmm = HMMRiskManager(data=data, verbose=True,
                         iterations=1000, **kwargs)
    time_intervals = 0
    long_market = False
    short_market = False
    num_days = 0
    logger.info(
        f'Running SMA Strategy on {trade.symbol} in {tf} Interval ...\n')
    while True:
        current_date = datetime.now()
        today = current_date.strftime("%A")
        try:
            buys = trade.get_current_buys()
            if buys is not None:
                logger.info(
                    f"Current buy positions SYMBOL={trade.symbol}: {buys}, STRATEGY=SMA")
            sells = trade.get_current_sells()
            if sells is not None:
                logger.info(
                    f"Current sell positions SYMBOL={trade.symbol}: {sells}, STRATEGY=SMA")
            long_market = buys is not None and len(buys) >= max_t
            short_market = sells is not None and len(sells) >= max_t

            time.sleep(0.5)
            sig_rate = Rates(trade.symbol, tf, 0, lma)
            hmm_data = sig_rate.get_returns.values
            current_regime = hmm.which_trade_allowed(hmm_data)
            logger.info(f'CURRENT REGIME = {current_regime}, SYMBOL={trade.symbol}, STRATEGY=SMA')
            ma_data = sig_rate.get_close.values
            signal = strategy.calculate_signals(ma_data)
            logger.info(f"Calculating signal...SYMBOL={trade.symbol}, STRATEGY=SMA")
            comment = f"{trade.expert_name}@{trade.version}"
            if trade.trading_time() and today in TRADING_DAYS:
                if signal is not None:
                    logger.info(f"SIGNAL = {signal}, SYMBOL={trade.symbol}, STRATEGY=SMA")
                    if signal == "EXIT" and short_market:
                        trade.close_positions(position_type='sell')
                        short_market = False
                    elif signal == "EXIT" and long_market:
                        trade.close_positions(position_type='buy')
                        long_market = False

                    if current_regime == 'LONG':
                        if signal == "LONG" and not long_market:
                            if time_intervals % trade_time == 0 or buys is None:
                                logger.info(f"Sending buy Order .... SYMBOL={trade.symbol}, STRATEGY=SMA")
                                trade.open_buy_position(mm=mm, comment=comment)
                            else:
                                check(buys, sells)
                        elif signal == "LONG" and long_market:
                            logger.info(f"Sorry Risk not allowed !!! SYMBOL={trade.symbol}, STRATEGY=SMA")
                            check(buys, sells)

                    elif current_regime == 'SHORT':
                        if signal == "SHORT" and not short_market:
                            if time_intervals % trade_time == 0 or sells is None:
                                logger.info(f"Sending Sell Order .... SYMBOL={trade.symbol}, STRATEGY=SMA")
                                trade.open_sell_position(
                                    mm=mm, comment=comment)
                            else:
                                check(buys, sells)
                        elif signal == "SHORT" and short_market:
                            logger.info(f"Sorry Risk not Allowed !!! SYMBOL={trade.symbol}, STRATEGY=SMA")
                            check(buys, sells)
                else:
                    logger.info(f"There is no signal !! SYMBOL={trade.symbol}, STRATEGY=SMA")
                    check(buys, sells)
            else:
                logger.info(f"Sorry It is Not trading Time !!! SYMBOL={trade.symbol}, STRATEGY=SMA")
                check(buys, sells)
        except Exception as e:
            logger.error(f"{e}, SYMBOL={trade.symbol}, STRATEGY=SMA")
        time.sleep((60 * iter_time) - 1.5)
        if iter_time == 1:
            time_intervals += 1
        elif iter_time == trade_time:
            time_intervals += trade_time
        else:
            time_intervals += (trade_time/iter_time)
        if period.lower() == 'month':
            if trade.days_end() and today != 'Friday':
                sleep_time = trade.sleep_time()
                logger.info(f"End of the Day !!! SYMBOL={trade.symbol}, STRATEGY=SMA")
                time.sleep(60 * sleep_time)
                num_days += 1

            elif trade.days_end() and today == 'Friday':
                logger.info(f"End of the Week !!! SYMBOL={trade.symbol}, STRATEGY=SMA")
                sleep_time = trade.sleep_time(weekend=True)
                time.sleep(60 * sleep_time)
                num_days += 1

            elif (
                trade.days_end()
                and today == 'Friday'
                and num_days >= 20
            ):
                trade.close_positions(position_type='all', comment=comment)
                logger.info(f"End of the Month !!! SYMBOL={trade.symbol}, STRATEGY=SMA")
                trade.statistics(save=True)
                break

        elif period.lower() == 'week':
            if trade.days_end() and today != 'Friday':
                sleep_time = trade.sleep_time()
                time.sleep(60 * sleep_time)

            elif trade.days_end() and today == 'Friday':
                trade.close_positions(position_type='all', comment=comment)
                logger.info(f"End of the Week !!! SYMBOL={trade.symbol}, STRATEGY=SMA")
                trade.statistics(save=True)
                break

        elif period.lower() == 'day':
            if trade.days_end():
                trade.close_positions(position_type='all', comment=comment)
                logger.info(f"End of the Day !!! SYMBOL={trade.symbol}, STRATEGY=SMA")
                trade.statistics(save=True)
                break


# ========= PAIR TRADING =====================
def pair_trading(
    pair: List[str] | Tuple[str],
    p0: Trade,
    p1: Trade,
    tf: str,
    /,
    max_t: Optional[int] = 1,
    mm: Optional[bool] = True,
    iter_time: Optional[int | float] = 30,
    risk_manager: Optional[str] = None,
    rm_ticker: Optional[str] = None,
    rm_window: Optional[int] = None,
    period: Literal['day', 'week', 'month'] = 'month',
    **kwargs
):
    """
    Implements a pair trading strategy with optional risk management 
    using Hidden Markov Models (HMM). This strategy trades pairs of assets 
    based on their historical price relationship, seeking to capitalize on converging prices.

    :param pair (list[str] | tuple[str]): The trading pair represented as a list or tuple of symbols (e.g., ['AAPL', 'GOOG']).
    :param p0 (Trade): Trade object for the first asset in the pair.
    :param p1 (Trade): Trade object for the second asset in the pair.
    :param tf (str): Time frame for the trading strategy (e.g., '1h' for 1 hour).
    :param max_t (int, optional): Maximum number of trades allowed at any time for each asset in the pair, defaults to 1.
    
    :param mm (bool, optional): Money management flag to enable/disable money management, defaults to True.
    :param iter_time (int | float ,optional): Iteration time (in minutes) for the trading loop, defaults to 30.
    :param risk_manager: Specifies the risk management model to use default is None , Hidden Markov Model ('hmm) Can be use.
    :param rm_window: Window size for the risk model use for the prediction, defaults to None. Must be specified if `risk_manager` is not None.
    
    :param period (str, optional): Trading period to reset statistics and close positions, can be 'day', 'week', or 'month'. 
    :param kwargs: Additional keyword arguments for HMM risk manager.

    This function continuously evaluates the defined pair for trading opportunities 
    based on the strategy logic, taking into account the specified risk management 
    approach if applicable. It aims to profit from the mean reversion behavior typically
    observed in closely related financial instruments.

    Note: 
    This function includes an infinite loop with time delays designed to run continuously during market hours.
    Proper exception handling and resource management are crucial for live trading environments.
    """
    regime = False
    if risk_manager is not None:
        assert rm_ticker is not None
        assert rm_window is not None
        regime = True

    def p0_check(p0_positions):
        if p0_positions is not None:
            logger.info(f"Checking for breakeven on {pair[0]} positions...STRATEGY=KLF")
            p0.break_even()

    def p1_check(p1_positions):
        if p1_positions is not None:
            logger.info(f"Checking for breakeven on {pair[1]} positions...STRATEGY=KLF")
            p1.break_even()

    time_frame_mapping = tf_mapping()
    if tf == 'D1':
        trade_time = p0.get_minutes()
    else:
        trade_time = time_frame_mapping[tf]

    if regime:
        if risk_manager == 'hmm':
            rate = Rates(rm_ticker, tf, 0)
            data = rate.get_rates_from_pos()
            hmm = HMMRiskManager(data=data, verbose=True, iterations=5000, **kwargs)

    time_intervals = 0
    long_market = False
    short_market = False
    num_days = 0
    logger.info(
        f'Running KLF Strategy on {pair[0]} and {pair[1]} in {tf} Interval ...\n')
    while True:
        current_date = datetime.now()
        today = current_date.strftime("%A")
        try:
            # Data Retrieval
            p0_ = Rates(pair[0], tf, 0, 10)
            p1_ = Rates(pair[1], tf, 0, 10)

            p0_data = p0_.get_close
            p1_data = p1_.get_close
            prices = np.array(
                [p0_data.values[-1], p1_data.values[-1]]
            )
            strategy = KLFStrategy(pair)
            if regime:
                if risk_manager == 'hmm':
                    hmm_data = Rates(rm_ticker, tf, 0, rm_window)
                    returns = hmm_data.get_returns.values
                    current_regime = hmm.which_trade_allowed(returns)
                    logger.info(f'CURRENT REGIME ={current_regime}, STRATEGY=KLF')
            else:
                current_regime = None

            p0_positions = p0.get_current_open_positions()
            time.sleep(0.5)
            p1_positions = p1.get_current_open_positions()
            time.sleep(0.5)
            p1_buys = p1.get_current_buys()
            p0_buys = p0.get_current_buys()
            time.sleep(0.5)
            if p1_buys is not None:
                logger.info(f"Current buy positions on {pair[1]}: {p1_buys}, STRATEGY=KLF")
            if p0_buys is not None:
                logger.info(f"Current buy positions on {pair[0]}: {p0_buys}, STRATEGY=KLF")
            time.sleep(0.5)
            p1_sells = p1.get_current_sells()
            p0_sells = p0.get_current_sells()
            time.sleep(0.5)
            if p1_sells is not None:
                logger.info(f"Current sell positions on {pair[1]}: {p1_sells}, STRATEGY=KLF")
            if p0_sells is not None:
                logger.info(f"Current sell positions on {pair[0]}: {p0_sells}, STRATEGY=KLF")

            p1_long_market = p1_buys is not None and len(p1_buys) >= max_t
            p0_long_market = p0_buys is not None and len(p0_buys) >= max_t
            p1_short_market = p1_sells is not None and len(p1_sells) >= max_t
            p0_short_market = p0_sells is not None and len(p0_sells) >= max_t

            logger.info(f"Calculating Signals SYMBOL={pair}...STRATEGY=KLF")
            signals = strategy.calculate_signals(prices)
            comment = f"{p0.expert_name}@{p0.version}"

            if signals is not None:
                logger.info(f'SIGNALS = {signals}, STRATEGY=KLF')
                if p0.trading_time() and today in TRADING_DAYS:
                    p1_signal = signals[pair[1]]
                    p0_signal = signals[pair[0]]
                    if p1_signal == "EXIT" and p0_signal == "EXIT":
                        if p1_positions is not None:
                            logger.info(f"Exiting Positions On [{pair[1]}], STRATEGY=KLF")
                            p1.close_positions(position_type='all', comment=comment)
                            p1_long_market = False
                            p1_short_market = False
                        if p0_positions is not None:
                            logger.info(f"Exiting Positions On [{pair[0]}], STRATEGY=KLF")
                            p0.close_positions(position_type='all', comment=comment)
                            p1_long_market = False
                            p1_short_market = False
                    if current_regime is not None:
                        if (
                            p1_signal == "LONG"
                            and p0_signal == "SHORT"
                            and current_regime == 'LONG'
                        ):
                            if not p1_long_market:
                                if time_intervals % trade_time == 0 or p1_buys is None:
                                    logger.info(f"Going LONG on [{pair[1]}], STRATEGY=KLF")
                                    p1.open_buy_position(
                                        mm=mm, comment=comment)
                                else:
                                    p1_check(p1_positions)
                            else:
                                logger.info(f"Sorry Risk Not allowed on [{pair[1]}], STRATEGY=KLF")
                                p1_check(p1_positions)

                            if not p0_short_market:
                                if time_intervals % trade_time == 0 or p0_sells is None:
                                    logger.info(f"Going SHORT on [{pair[0]}]")
                                    p0.open_sell_position(
                                        mm=mm, comment=comment)
                                else:
                                    p0_check(p0_positions)
                            else:
                                logger.info(
                                    f"Sorry Risk Not allowed on [{pair[0]}], STRATEGY=KLF")
                                p0_check(p0_positions)
                        elif (
                            p1_signal == "SHORT"
                            and p0_signal == "LONG"
                            and current_regime == 'SHORT'
                        ):
                            if not p1_short_market:
                                if time_intervals % trade_time == 0 or p1_sells is None:
                                    logger.info(f"Going SHORT on [{pair[1]}], STRATEGY=KLF")
                                    p1.open_sell_position(
                                        mm=mm, comment=comment)
                                else:
                                    p1_check(p1_positions)
                            else:
                                logger.info(f"Sorry Risk Not allowed on [{pair[1]}], STRATEGY=KLF")
                                p1_check(p1_positions)

                            if not p0_long_market:
                                if time_intervals % trade_time == 0 or p0_buys is None:
                                    logger.info(f"Going LONG on [{pair[0]}], STRATEGY=KLF")
                                    p0.open_buy_position(
                                        mm=mm, comment=comment)
                                else:
                                    p0_check(p0_positions)
                            else:
                                logger.info(
                                    f"Sorry Risk Not allowed on [{pair[0]}], STRATEGY=KLF")
                                p0_check(p0_positions)
                    else:
                        if (
                            p1_signal == "LONG"
                            and p0_signal == "SHORT"
                        ):
                            if not p1_long_market:
                                if time_intervals % trade_time == 0 or p1_buys is None:
                                    logger.info(f"Going LONG on [{pair[1]}], STRATEGY=KLF")
                                    p1.open_buy_position(
                                        mm=mm, comment=comment)
                                else:
                                    p1_check(p1_positions)
                            else:
                                logger.info(f"Sorry Risk Not allowed on [{pair[1]}], STRATEGY=KLF")
                                p1_check(p1_positions)

                            if not p0_short_market:
                                if time_intervals % trade_time == 0 or p0_sells is None:
                                    logger.info(f"Going SHORT on [{pair[0]}], STRATEGY=KLF")
                                    p0.open_sell_position(
                                        mm=mm, comment=comment)
                                else:
                                    p0_check(p0_positions)
                            else:
                                logger.info(
                                    f"Sorry Risk Not allowed on [{pair[0]}], STRATEGY=KLF")
                                p0_check(p0_positions)
                        elif (
                            p1_signal == "SHORT"
                            and p0_signal == "LONG"
                        ):
                            if not p1_short_market:
                                if time_intervals % trade_time == 0 or p1_sells is None:
                                    logger.info(f"Going SHORT on [{pair[1]}], STRATEGY=KLF")
                                    p1.open_sell_position(
                                        mm=mm, comment=comment)
                                else:
                                    p1_check(p1_positions)
                            else:
                                logger.info(f"Sorry Risk Not allowed on [{pair[1]}], STRATEGY=KLF")
                                p1_check(p1_positions)

                            if not p0_long_market:
                                if time_intervals % trade_time == 0 or p0_buys is None:
                                    logger.info(f"Going LONG on [{pair[0]}], STRATEGY=KLF")
                                    p0.open_buy_position(
                                        mm=mm, comment=comment)
                                else:
                                    p0_check(p0_positions)
                            else:
                                logger.info(
                                    f"Sorry Risk Not allowed on [{pair[0]}], STRATEGY=KLF")
                                p0_check(p0_positions)
                else:
                    logger.info(
                        f"It is Not trading time !!! STRATEGY=KLF, SYMBOLS={pair}")
                    p0_check(p0_positions)
                    p1_check(p1_positions)
            else:
                logger.info(
                    f"There is no signal !!! STRATEGY=KLF, SYMBOLS={pair}")
                    
                p0_check(p0_positions)
                p1_check(p1_positions)

        except Exception as e:
            logger.error(f"{e}, STRATEGY=KLF, SYMBOLS={pair}")

        time.sleep((60 * iter_time) - 2.5)

        if iter_time == 1:
            time_intervals += 1
        elif iter_time == trade_time:
            time_intervals += trade_time
        else:
            time_intervals += (trade_time/iter_time)

        if period.lower() == 'month':
            if p0.days_end() and today != 'Friday':
                logger.info(
                    f"End of the Day !!! STRATEGY=KLF, SYMBOLS={pair}")
                    
                sleep_time = p0.sleep_time()
                time.sleep(60 * sleep_time)
                num_days += 1

            elif p0.days_end() and today == 'Friday':
                logger.info(
                    f"End of the Week !!! STRATEGY=KLF, SYMBOLS={pair}")
                sleep_time = p0.sleep_time(weekend=True)
                time.sleep(60 * sleep_time)
                num_days += 1

            elif (
                    p0.days_end()
                and today == 'Friday'
                and num_days >= 20
            ):
                p0.close_positions(position_type='all', comment=comment)
                p1.close_positions(position_type='all', comment=comment)
                logger.info(
                    f"End of the Month !!! STRATEGY=KLF, SYMBOLS={pair}")                    
                p0.statistics(save=True)
                p1.statistics(save=True)
                break

        elif period.lower() == 'week':
            if p0.days_end() and today != 'Friday':
                sleep_time = p0.sleep_time()
                time.sleep(60 * sleep_time)

            elif p0.days_end() and today == 'Friday':
                p0.close_positions(position_type='all', comment=comment)
                p1.close_positions(position_type='all', comment=comment)
                logger.info(
                    f"End of the Week !!! STRATEGY=KLF, SYMBOLS={pair}")
                p0.statistics(save=True)
                p1.statistics(save=True)
                break

        elif period.lower() == 'day':
            if p0.days_end():
                p0.close_positions(position_type='all', comment=comment)
                p1.close_positions(position_type='all', comment=comment)
                logger.info(
                    f"End of the Day !!! STRATEGY=KLF, SYMBOLS={pair}")                   
                p0.statistics(save=True)
                p1.statistics(save=True)
                break


# ========= ORNSTEIN UHLENBECK TRADING ========
def ou_trading(
    trade: Trade,
    tf: Optional[str] = '1h',
    p: Optional[int] = 20,
    n: Optional[int] = 20,
    ou_window: Optional[int] = 2000,
    max_t: Optional[int] = 1,
    mm: Optional[bool] = True,
    iter_time: Optional[int | float] = 30,
    risk_manager: Optional[str] = None,
    rm_window: Optional[int] = None,
    period: Literal['day', 'week', 'month'] = 'month',
    **kwargs
):
    """
    Executes the Ornstein-Uhlenbeck (OU) trading strategy, 
    incorporating various risk management and trading frequency adjustments.

    :param trade: A `Trade` instance, containing methods and attributes for executing trades.
    :param tf: Time frame for the trading strategy, default is '1h'.
    :param mm: Boolean indicating if money management is enabled, default is True.
    :param max_t: Maximum number of trades allowed at any given time, default is 1.
    :param p: Period length for calculating returns, default is 20.
    :param n: Window size for the Ornstein-Uhlenbeck strategy calculation, default is 20.
    :param iter_time: Iteration time for the trading loop, can be an integer or float. 
    :param ou_window: Lookback period for the OU strategy, defaults to 2000.
    :param risk_manager: Specifies the risk management model to use
        default is None , Hidden Markov Model ('hmm) Can be use.
    :param rm_window: Window size for the risk model use for the prediction, defaults to None. 
        Must be specified if `risk_manager` is not None.
    :param period: Defines the trading period as 'month', 'week', or 'day'
        affecting how and when positions are closed.
    :param kwargs: Additional keyword arguments for risk management models or other customizations.

    This function manages trading based on the OU strategy, adjusting for risk and time-based criteria. 
    It includes handling of trading sessions, buy/sell signal generation, risk management through the HMM model, and period-based
    trading evaluation.
    """
    regime = False
    if risk_manager is not None:
        if risk_manager.lower() == 'hmm':
            assert rm_window is not None
            regime = True

    rate = Rates(trade.symbol, tf, 0)
    data = rate.get_rates_from_pos()
    def check(buys: list, sells: list):
        if buys is not None or sells is not None:
            logger.info(f"Checking for Break even on {trade.symbol}... STRATEGY=OU")
            trade.break_even()

    time_frame_mapping = tf_mapping()
    if tf == 'D1':
        trade_time = trade.get_minutes()
    else:
        trade_time = time_frame_mapping[tf]

    if regime:
        if risk_manager == 'hmm':
            hmm = HMMRiskManager(data=data, verbose=True, **kwargs)
    strategy = OrnsteinUhlenbeck(data['Close'].values[-ou_window:], timeframe=tf)

    time_intervals = 0
    long_market = False
    short_market = False
    num_days = 0
    logger.info(f'Running OU Strategy on {trade.symbol} in {tf} Interval ...\n')
    while True:
        current_date = datetime.now()
        today = current_date.strftime("%A")
        try:
            buys = trade.get_current_buys()
            if buys is not None:
                logger.info(f"Current buy positions on {trade.symbol}: {buys}, STRATEGY=OU")
            sells = trade.get_current_sells()
            if sells is not None:
                logger.info(f"Current sell positions on {trade.symbol}: {sells}, STRATEGY=OU")
            long_market = buys is not None and len(buys) >= max_t
            short_market = sells is not None and len(sells) >= max_t

            time.sleep(0.5)
            if regime:
                if risk_manager == 'hmm':
                    hmm_returns = Rates(trade.symbol, tf, 0, rm_window)
                    hmm_returns_val = hmm_returns.get_returns.values
                    current_regime = hmm.which_trade_allowed(hmm_returns_val)
                    logger.info(
                        f'CURRENT REGIME = {current_regime}, SYMBOL={trade.symbol}, STRATEGY=OU')
            else:
                current_regime = None
            logger.info(f"Calculating signal... SYMBOL={trade.symbol}, STRATEGY=OU")
            ou_returns = Rates(trade.symbol, tf, 0, p)
            returns_val = ou_returns.get_returns.values
            signal = strategy.calculate_signals(returns_val, p=p, n=n)
            comment = f"{trade.expert_name}@{trade.version}"
            if trade.trading_time() and today in TRADING_DAYS:
                if signal is not None:
                    logger.info(f"SIGNAL = {signal}, SYMBOL={trade.symbol}, STRATEGY=OU")
                    if signal == "LONG" and short_market:
                        trade.close_positions(position_type='sell')
                        short_market = False
                    elif signal == "SHORT" and long_market:
                        trade.close_positions(position_type='buy')
                        long_market = False
                    if current_regime is not None:
                        if current_regime == 'LONG':
                            if signal == "LONG" and not long_market:
                                if time_intervals % trade_time == 0 or buys is None:
                                    logger.info(f"Sending buy Order .... SYMBOL={trade.symbol}, STRATEGY=OU")
                                    trade.open_buy_position(
                                        mm=mm, comment=comment)
                                else:
                                    check(buys, sells)

                            elif signal == "LONG" and long_market:
                                logger.info(f"Sorry Risk not allowed !!! SYMBOL={trade.symbol}, STRATEGY=OU")
                                check(buys, sells)

                        elif current_regime == 'SHORT':
                            if signal == "SHORT" and not short_market:
                                if time_intervals % trade_time == 0 or sells is None:
                                    logger.info(f"Sending Sell Order .... SYMBOL={trade.symbol}, STRATEGY=OU")
                                    trade.open_sell_position(
                                        mm=mm, comment=comment)
                                else:
                                    check(buys, sells)
                            elif signal == "SHORT" and short_market:
                                logger.info(f"Sorry Risk not Allowed !!! SYMBOL={trade.symbol}, STRATEGY=OU")
                                check(buys, sells)
                    else:
                        if signal == "LONG" and not long_market:
                            if time_intervals % trade_time == 0 or buys is None:
                                logger.info(f"Sending buy Order .... SYMBOL={trade.symbol}, STRATEGY=OU")
                                trade.open_buy_position(mm=mm, comment=comment)
                            else:
                                check(buys, sells)

                        elif signal == "LONG" and long_market:
                            logger.info(f"Sorry Risk not allowed !!! SYMBOL={trade.symbol}, STRATEGY=OU")
                            check(buys, sells)

                        if signal == "SHORT" and not short_market:
                            if time_intervals % trade_time == 0 or sells is None:
                                logger.info(f"Sending Sell Order .... SYMBOL={trade.symbol}, STRATEGY=OU")
                                trade.open_sell_position(
                                    mm=mm, comment=comment)
                            else:
                                check(buys, sells)
                        elif signal == "SHORT" and short_market:
                            logger.info(f"Sorry Risk not Allowed !!! SYMBOL={trade.symbol}, STRATEGY=OU")
                            check(buys, sells)
                else:
                    logger.info(f"There is no signal !!! SYMBOL={trade.symbol}, STRATEGY=OU")
                    check(buys, sells)
            else:
                logger.info(f"Sorry It is Not trading Time !!! SYMBOL={trade.symbol}, STRATEGY=OU")
                check(buys, sells)
        except Exception as e:
            print(f"{e}, SYMBOL={trade.symbol}, STRATEGY=OU")
        time.sleep((60 * iter_time) - 1.5)
        if iter_time == 1:
            time_intervals += 1
        elif iter_time == trade_time:
            time_intervals += trade_time
        else:
            time_intervals += (trade_time/iter_time)

        if period.lower() == 'month':
            if trade.days_end() and today != 'Friday':
                logger.info(f"End of the Day !!! SYMBOL={trade.symbol}, STRATEGY=OU")
                sleep_time = trade.sleep_time()
                time.sleep(60 * sleep_time)
                num_days += 1

            elif trade.days_end() and today == 'Friday':
                logger.info(f"End of the Week !!! SYMBOL={trade.symbol}, STRATEGY=OU")
                sleep_time = trade.sleep_time(weekend=True)
                time.sleep(60 * sleep_time)
                num_days += 1

            elif (
                    trade.days_end()
                and today == 'Friday'
                and num_days >= 20
            ):
                trade.close_positions(position_type='all', comment=comment)
                logger.info(f"End of the Month !!! SYMBOL={trade.symbol}, STRATEGY=OU")
                trade.statistics(save=True)
                break

        elif period.lower() == 'week':
            if trade.days_end() and today != 'Friday':
                sleep_time = trade.sleep_time()
                time.sleep(60 * sleep_time)

            elif trade.days_end() and today == 'Friday':
                trade.close_positions(position_type='all', comment=comment)
                logger.info(f"End of the Week !!! SYMBOL={trade.symbol}, STRATEGY=OU")
                trade.statistics(save=True)
                break

        elif period.lower() == 'day':
            if trade.days_end():
                trade.close_positions(position_type='all', comment=comment)
                logger.info(f"End of the Day !!! SYMBOL={trade.symbol}, STRATEGY=OU")
                trade.statistics(save=True)
                break


# ========= ARIMA + GARCH TRADING =======================
def arch_trading(
    trade: Trade,
    tf: str = 'D1',
    k: int = 500, 
    max_t: Optional[int] = 1,
    mm: Optional[bool] = True,
    iter_time: Optional[int | float] = 30,
    risk_manager: Optional[str] = None,
    rm_window: Optional[int] = None,
    period: Literal['day', 'week', 'month'] = 'month',
    **kwargs
):
    """
    Executes trading based on the ARCH (Autoregressive Conditional Heteroskedasticity) model, with 
    the capability to incorporate a risk management strategy, specifically a Hidden Markov Model (HMM), 
    to adjust trading decisions based on the market regime.

    :param trade: A `Trade` instance, necessary for executing trades and managing positions.
    :param tf: Time frame for the trading data, default is 'D1' (daily).
    :param k: Number of past points to consider for the ARCH model analysis, default is 500.
    :param mm: Boolean flag indicating if money management strategies should be applied, default is True.
    :param max_t: Maximum number of trades allowed at any given time, default is 1.
    :param iter_time: Time in minutes between each iteration of the trading loop. Can be an integer or float.
    :param risk_manager: Specifies the risk management model to use. Default is None. 
    :param rm_window: Window size for the risk model use for the prediction, defaults to None. 
        Must be specified if `risk_manager` is not None.
    :param period: Trading period to consider for closing positions, options are 'month', 'week', or 'day'. 
                   This affects the frequency at which statistics are calculated and positions are closed.
    :param kwargs: Additional keyword arguments for the risk management models or other strategy-specific settings.

    This function is designed to perform trading based on ARCH model predictions, managing risk using an HMM where 
    applicable, and handling trade executions and position management based on the specified parameters. It includes 
    considerations for trading times, money management, and periodic evaluation of trading performance.
    """
    regime = False
    if risk_manager is not None:
        if risk_manager.lower() == 'hmm':
            assert rm_window is not None
            regime = True

    def check(buys: list, sells: list):
        if buys is not None or sells is not None:
            logger.info(f"Checking for Break even on {trade.symbol}...")
            trade.break_even()

    time_frame_mapping = tf_mapping()
    if tf == 'D1':
        trade_time = trade.get_minutes()
    else:
        trade_time = time_frame_mapping[tf]

    rate = Rates(trade.symbol, tf, 0)
    data = rate.get_rates_from_pos()
    strategy = ArimaGarchStrategy(trade.symbol, data, k=k)
    if regime:
        if risk_manager == 'hmm':
            hmm = HMMRiskManager(data=data, verbose=True, iterations=5000, **kwargs)

    time_intervals = 0
    long_market = False
    short_market = False
    num_days = 0
    logger.info(
        f'Running ARIMA + GARCH Strategy on {trade.symbol} in {tf} Interval ...\n')
    while True:
        current_date = datetime.now()
        today = current_date.strftime("%A")
        try:
            buys = trade.get_current_buys()
            if buys is not None:
                logger.info(f"Current buy positions on {trade.symbol}: {buys}, STRATEGY=ARCH")
            sells = trade.get_current_sells()
            if sells is not None:
                logger.info(f"Current sell positions on {trade.symbol}: {sells}, STRATEGY=ARCH")
            long_market = buys is not None and len(buys) >= max_t
            short_market = sells is not None and len(sells) >= max_t

            time.sleep(0.5)
            if regime:
                if risk_manager == 'hmm':
                    hmm_returns = Rates(trade.symbol, tf, 0, rm_window)
                    hmm_returns_val = hmm_returns.get_returns.values
                    current_regime = hmm.which_trade_allowed(hmm_returns_val)
                    logger.info(f'CURRENT REGIME = {current_regime}, SYMBOL={trade.symbol}, STRATEGY=ARCH')
            else:
                current_regime = None
            logger.info(f"Calculating Signal ... SYMBOL={trade.symbol}, STRATEGY=ARCH")
            arch_data = Rates(trade.symbol, tf, 0, k)
            rates = arch_data.get_rates_from_pos()
            arch_returns = strategy.load_and_prepare_data(rates)
            window_data = arch_returns['diff_log_return'].iloc[-k:]
            signal = strategy.calculate_signals(window_data)

            comment = f"{trade.expert_name}@{trade.version}"
            if trade.trading_time() and today in TRADING_DAYS:
                if signal is not None:
                    logger.info(f"SIGNAL = {signal}, SYMBOL={trade.symbol}, STRATEGY=ARCH")
                    if signal == "LONG" and short_market:
                        trade.close_positions(position_type='sell')
                        short_market = False
                    elif signal == "SHORT" and long_market:
                        trade.close_positions(position_type='buy')
                        long_market = False
                    if current_regime is not None:
                        if current_regime == 'LONG':
                            if signal == "LONG" and not long_market:
                                if time_intervals % trade_time == 0 or buys is None:
                                    logger.info(f"Sending buy Order .... SYMBOL={trade.symbol}, STRATEGY=ARCH")
                                    trade.open_buy_position(
                                        mm=mm, comment=comment)
                                else:
                                    check(buys, sells)

                            elif signal == "LONG" and long_market:
                                logger.info(f"Sorry Risk not allowed !!! SYMBOL={trade.symbol}, STRATEGY=ARCH")
                                check(buys, sells)

                        elif current_regime == 'SHORT':
                            if signal == "SHORT" and not short_market:
                                if time_intervals % trade_time == 0 or sells is None:
                                    logger.info(f"Sending Sell Order .... SYMBOL={trade.symbol}, STRATEGY=ARCH")
                                    trade.open_sell_position(
                                        mm=mm, comment=comment)
                                else:
                                    check(buys, sells)
                            elif signal == "SHORT" and short_market:
                                logger.info(f"Sorry Risk not Allowed !!! SYMBOL={trade.symbol}, STRATEGY=ARCH")
                                check(buys, sells)
                    else:
                        if signal == "LONG" and not long_market:
                            if time_intervals % trade_time == 0 or buys is None:
                                logger.info(f"Sending buy Order .... SYMBOL={trade.symbol}, STRATEGY=ARCH")
                                trade.open_buy_position(mm=mm, comment=comment)
                            else:
                                check(buys, sells)

                        elif signal == "LONG" and long_market:
                            logger.info(f"Sorry Risk not allowed !!! SYMBOL={trade.symbol}, STRATEGY=ARCH")
                            check(buys, sells)

                        if signal == "SHORT" and not short_market:
                            if time_intervals % trade_time == 0 or sells is None:
                                logger.info(f"Sending Sell Order .... SYMBOL={trade.symbol}, STRATEGY=ARCH")
                                trade.open_sell_position(
                                    mm=mm, comment=comment)
                            else:
                                check(buys, sells)

                        elif signal == "SHORT" and short_market:
                            logger.info(f"Sorry Risk not Allowed !!! SYMBOL={trade.symbol}, STRATEGY=ARCH")
                            check(buys, sells)
                else:
                    logger.info("There is no signal !!")
                    check(buys, sells)
            else:
                logger.info(f"Sorry It is Not trading Time !!! SYMBOL={trade.symbol}, STRATEGY=ARCH")
                check(buys, sells)

        except Exception as e:
            print(f"{e}, SYMBOL={trade.symbol}, STRATEGY=ARCH")

        time.sleep((60 * iter_time) - 1.5)
        if iter_time == 1:
            time_intervals += 1
        elif iter_time == trade_time:
            time_intervals += trade_time
        else:
            time_intervals += (trade_time/iter_time)

        if period.lower() == 'month':
            if trade.days_end() and today != 'Friday':
                logger.info(f"End of the Day !!! SYMBOL={trade.symbol}, STRATEGY=ARCH")
                sleep_time = trade.sleep_time()
                time.sleep(60 * sleep_time)
                num_days += 1

            elif trade.days_end() and today == 'Friday':
                logger.info(f"End of the Week !!! SYMBOL={trade.symbol}, STRATEGY=ARCH")
                sleep_time = trade.sleep_time(weekend=True)
                time.sleep(60 * sleep_time)
                num_days += 1

            elif (
                    trade.days_end()
                and today == 'Friday'
                and num_days >= 20
            ):
                trade.close_positions(position_type='all', comment=comment)
                logger.info(f"End of the Month !!! SYMBOL={trade.symbol}, STRATEGY=ARCH")
                trade.statistics(save=True)
                break

        elif period.lower() == 'week':
            if trade.days_end() and today != 'Friday':
                sleep_time = trade.sleep_time()
                time.sleep(60 * sleep_time)

            elif trade.days_end() and today == 'Friday':
                trade.close_positions(position_type='all', comment=comment)
                logger.info(f"End of the Week !!! SYMBOL={trade.symbol}, STRATEGY=ARCH")
                trade.statistics(save=True)
                break

        elif period.lower() == 'day':
            if trade.days_end():
                trade.close_positions(position_type='all', comment=comment)
                logger.info(f"End of the Day !!! SYMBOL={trade.symbol}, STRATEGY=ARCH")
                trade.statistics(save=True)
                break
