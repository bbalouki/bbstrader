import time
import pandas as pd
import numpy as np
from datetime import datetime
from mtrader5.rates import Rates
from mtrader5.trade import Trade
from trading.mt5.utils import tf_mapping
from strategies.sma import SMAStrategy
from strategies.arch import ArimaGarchStrategy
from strategies.klf import KLFStrategy
from strategies.ou import OrnsteinUhlenbeck
from risk_models.hmm import HMMRiskManager


MAX_BARS = 10_000_000
TRADING_DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


# ========  SMA TRADING ======================
def sma_trading(
    trade: Trade,
    tf: str = '1h',
    sma: int = 35,
    lma: int = 80,
    mm: bool = True,
    max_t: int = 1,
    iter_time: int | float = 30,
    risk_manager: str = 'hmm',
    period: str = 'week',
    **kwargs
):
    """
    Executes a Simple Moving Average (SMA) trading strategy 
    with optional risk management using Hidden Markov Models (HMM).

    Parameters
    ==========
    :param trade (Trade): The Trade object that encapsulates 
        trading operations like opening, closing positions, etc.
    :param tf (str, optional): Time frame for the trading strategy
        defaults to '1h' (1 hour).
    :param sma (int, optional): Short Moving Average period, defaults to 35.
    :param lma (int, optional): Long Moving Average period, defaults to 80.
    :param mm (bool, optional): Money management flag 
        to enable/disable money management, defaults to True.
    :param max_t (int, optional): Maximum number of trades allowed, defaults to 1.
    :param iter_time (int | float, optional): Iteration time for the trading loop.
    :param risk_manager (str ): Specifies the risk management strategy to use
      'hmm' for Hidden Markov Model. Defaults to 'hmm'.
    :param period (str, optional): Trading period to reset statistics and close positions
        can be 'day', 'week', or 'month'. Defaults to 'week'.
    :param **kwargs: Additional keyword arguments for HMM risk manager.

    The function integrates a trading strategy based on simple moving averages, 
    with an optional risk management layer using HMM.
    It periodically checks for trading signals and executes buy or sell orders 
    based on the strategy signals and risk management conditions.
    The trading period (day, week, month) determines when to reset statistics and close all positions.

    Note: This function includes an infinite loop with time delays
      designed to run continuously during market hours.
          Make sure to handle exceptions and ensure proper resource management 
          when integrating into a live trading environment.
    """
    if risk_manager is None:
        raise ValueError (
            "For SMAStrategy , the Risk Manger is required",
            "Please privde a valid risk manager or set it to 'None'"   
        )

    def check(buys: list, sells: list):
        if buys is not None or sells is not None:
            print(f"\nChecking for Break even on {trade.symbol}...")
            trade.break_even()

    time_frame_mapping = tf_mapping()
    if tf == 'D1':
        trade_time = trade.get_minutes()
    else: trade_time = time_frame_mapping[tf]

    rate = Rates(trade.symbol, tf, 0, MAX_BARS)
    data = rate.get_rate_frame()
    data['Date'] = pd.to_datetime(data['Date'], unit='s')
    data.set_index('Date', inplace=True)
    strategy = SMAStrategy(short_window=sma, long_window=lma)
    hmm = HMMRiskManager(data=data, verbose=True,
                             iterations=1000, **kwargs)
    time_intervals = 0
    long_market = False
    short_market = False
    num_days = 0
    print(f'\nRunning SMA Strategy on {trade.symbol} in {tf} Interval ...\n')
    while True:
        current_date = datetime.now()
        today = current_date.strftime("%A")
        try:
            buys = trade.get_current_buys()
            if buys is not None:
                print(f"\nCurrent buy positions on {trade.symbol}: \n{buys}")
            sells = trade.get_current_sells()
            if sells is not None:
                print(f"\nCurrent sell positions on {trade.symbol}: \n{sells}")
            long_market = buys is not None and len(buys) >= max_t
            short_market = sells is not None and len(sells) >= max_t

            time.sleep(0.5)
            sig_rate = Rates(trade.symbol, tf, 0, lma)
            hmm_data = sig_rate.get_returns.values
            current_regime = hmm.which_trade_allowed(hmm_data)
            print(f'CURRENT REGIME : {current_regime}')
            ma_data = sig_rate.get_close.values
            signal = strategy.calculate_signals(ma_data)
            print("Calculating signal ...")
            print(f"\nSIGNAL : {signal}")
            comment = f"{trade.expert_name}@{trade.version}"
            now = datetime.now().strftime("%H:%M:%S")
            if trade.trading_time() and today in TRADING_DAYS:
                if signal is not None:
                    if signal == "EXIT" and short_market:
                        print(f'\nTime: {now}')
                        trade.close_all_sells()
                        short_market = False
                    elif signal == "EXIT" and long_market:
                        print(f'\nTime: {now}')
                        trade.close_all_buys()
                        long_market = False

                    if current_regime == 'LONG':
                        if signal == "LONG" and not long_market:
                            if time_intervals % trade_time == 0 or buys is None:
                                print("Sending buy Order ....")
                                trade.open_buy_position(mm=mm, comment=comment)
                            else:
                                print(f'\nTime: {now}')
                                check(buys, sells)
                        elif signal == "LONG" and long_market:
                            print(f'\nTime: {now}')
                            print("Sorry Risk not allowed !!!")
                            check(buys, sells)

                    elif current_regime == 'SHORT':
                        if signal == "SHORT" and not short_market:
                            if time_intervals % trade_time == 0 or sells is None:
                                print("Sending Sell Order ....")
                                trade.open_sell_position(mm=mm, comment=comment)
                            else:
                                print(f'\nTime: {now}')
                                check(buys, sells)
                        elif signal == "SHORT" and short_market:
                            print(f'\nTime: {now}')
                            print("Sorry Risk not Allowed !!!")
                            check(buys, sells)
                else:
                    print(f'\nTime: {now}')
                    print("There is no signal !!")
                    check(buys, sells)
            else:
                print(f'\nTime: {now}')
                print("Sorry It is Not trading Time !!!")
                check(buys, sells)
        except Exception as e:
            print(f'Time: {now}')
            print(f"Error: {e}")
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
                time.sleep(60 * sleep_time)
                num_days += 1

            elif trade.days_end() and today == 'Friday':
                sleep_time = trade.sleep_time(weekend=True)
                time.sleep(60 * sleep_time)
                num_days += 1

            elif (
                trade.days_end()
                and today == 'Friday'
                and num_days >= 20
            ):
                trade.close_all_positions(comment=comment)
                print("\nEnd of the Month !!!")
                trade.statistics(save=True)
                break

        elif period.lower() == 'week':
            if trade.days_end() and today != 'Friday':
                sleep_time = trade.sleep_time()
                time.sleep(60 * sleep_time)

            elif trade.days_end() and today == 'Friday':
                trade.close_all_positions(comment=comment)
                print("\nEnd of the Week !!!")
                trade.statistics(save=True)
                break

        elif period.lower() == 'day':
            if trade.days_end():
                trade.close_all_positions(comment=comment)
                print("\nEnd of the Day !!!")
                trade.statistics(save=True)
                break


# ========= PAIR TRADING =====================
def pair_trading(
    pair: list[str] | tuple[str],
    p0: Trade,
    p1: Trade,
    tf: str,
    /,
    max_t: int = 1,
    mm: bool = True,
    iter_time: int | float = 30,
    risk_manager: str | None = None, # 'hmm',
    rm_ticker: str = None,
    rm_window: int = None,
    period: str = 'month',  # day , week, month
    **kwargs
):
    """
    Implements a pair trading strategy with optional risk management 
    using Hidden Markov Models (HMM). This strategy trades pairs of assets 
    based on their historical price relationship, seeking to capitalize on converging prices.

    Parameters
    ==========
    :param pair (list[str] | tuple[str]): The trading pair
        represented as a list or tuple of symbols (e.g., ['AAPL', 'GOOG']).
    :param p0 (Trade): Trade object for the first asset in the pair.
    :param p1 (Trade): Trade object for the second asset in the pair.
    :param tf (str): Time frame for the trading strategy (e.g., '1h' for 1 hour).
    :param max_t (int, optional): Maximum number of trades allowed at any time 
        for each asset in the pair, defaults to 1.
    :param mm (bool, optional): Money management flag to enable/disable 
        money management, defaults to True.
    :param iter_time (int | float ,optional): Iteration time (in minutes) 
        for the trading loop, defaults to 30.
    :param risk_manager: Specifies the risk management model to use
        default is None , Hidden Markov Model ('hmm) Can be use.
    :param rm_window: Window size for the risk model use for the prediction, defaults to None. 
        Must be specified if `risk_manager` is not None.
    :param period (str, optional): Trading period to reset statistics 
        and close positions, can be 'day', 'week', or 'month'. Defaults to 'month'.
    :param **kwargs: Additional keyword arguments for HMM risk manager.

    This function continuously evaluates the defined pair for trading opportunities 
    based on the strategy logic, taking into account the specified risk management 
        approach if applicable. It aims to profit from the mean reversion behavior typically
    observed in closely related financial instruments.

    Note: This function includes an infinite loop with time delays
    designed to run continuously during market hours.
    Proper exception handling and resource management are crucial for live trading environments.
    """
    regime = False
    if risk_manager is not None:
        assert rm_ticker is not None
        assert rm_window is not None
        regime = True

    def p0_check(p0_positions):
        if p0_positions is not None:
            print(f"Checking for breakeven on {pair[0]} positions...\n")
            p0.break_even()

    def p1_check(p1_positions):
        if p1_positions is not None:
            print(f"Checking for breakeven on {pair[1]} positions...\n")
            p1.break_even()

    time_frame_mapping = tf_mapping()
    if tf == 'D1':
        trade_time = p0.get_minutes()
    else: trade_time = time_frame_mapping[tf]

    if regime:
        if risk_manager == 'hmm':
            rate = Rates(rm_ticker, tf, 0, MAX_BARS)
            data = rate.get_rate_frame()
            data['Date'] = pd.to_datetime(data['Date'], unit='s')
            data.set_index('Date', inplace=True)
            hmm = HMMRiskManager(
                data=data, verbose=True, iterations=5000, **kwargs)

    time_intervals = 0
    long_market = False
    short_market = False
    num_days = 0
    print(
        f'\nRunning KLF Strategy on {pair[0]} and {pair[1]} in {tf} Interval ...\n')
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
                    print(f'\nCURRENT REGIME : {current_regime}')
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
                print(f"\nCurrent buy positions on {pair[1]}: {p1_buys}")
            if p0_buys is not None:
                print(f"\nCurrent buy positions on {pair[0]}: {p0_buys}")
            time.sleep(0.5)
            p1_sells = p1.get_current_sells()
            p0_sells = p0.get_current_sells()
            time.sleep(0.5)
            if p1_sells is not None:
                print(f"Current sell positions on {pair[1]}: {p1_sells}")
            if p0_sells is not None:
                print(f"Current sell positions on {pair[0]}: {p0_sells}")

            p1_long_market = p1_buys is not None and len(p1_buys) >= max_t
            p0_long_market = p0_buys is not None and len(p0_buys) >= max_t
            p1_short_market = p1_sells is not None and len(p1_sells) >= max_t
            p0_short_market = p0_sells is not None and len(p0_sells) >= max_t

            print("\nCalculating Signals ...")
            signals = strategy.calculate_signals(prices)
            print(f'SIGNALS : {signals}')
            comment = f"{p0.expert_name}@{p0.version}"

            if signals is not None:
                now = datetime.now().strftime("%H:%M:%S")
                if p0.trading_time() and today in TRADING_DAYS:
                    p1_signal = signals[pair[1]]
                    p0_signal = signals[pair[0]]
                    if p1_signal == "EXIT" and p0_signal == "EXIT":
                        if p1_positions is not None:
                            print(f'\nTime: {now}')
                            print(f"Exiting Positions On [{pair[1]}]")
                            p1.close_all_positions(comment=comment)
                            p1_long_market = False
                            p1_short_market = False
                        if p0_positions is not None:
                            print(f'\nTime: {now}')
                            print(f"Exiting Positions On [{pair[0]}]")
                            p0.close_all_positions(comment=comment)
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
                                    print(f"\nGoing LONG on [{pair[1]}]")
                                    p1.open_buy_position(mm=mm, comment=comment)
                                else:
                                    print(f'\nTime: {now}')
                                    p1_check(p1_positions)
                            else:
                                print(f'\nTime: {now}')
                                print(f"Sorry Risk Not allowed on [{pair[1]}]")
                                p1_check(p1_positions)

                            if not p0_short_market:
                                if time_intervals % trade_time == 0 or p0_sells is None:
                                    print(f"\nGoing SHORT on [{pair[0]}]")
                                    p0.open_sell_position(mm=mm, comment=comment)
                                else:
                                    print(f'\nTime: {now}')
                                    p0_check(p0_positions)
                            else:
                                print(f'\nTime: {now}')
                                print(f"Sorry Risk Not allowed on [{pair[0]}] ")
                                p0_check(p0_positions)
                        elif (
                            p1_signal == "SHORT"
                            and p0_signal == "LONG"
                            and current_regime == 'SHORT'
                        ):
                            if not p1_short_market:
                                if time_intervals % trade_time == 0 or p1_sells is None:
                                    print(f"\nGoing SHORT on [{pair[1]}]")
                                    p1.open_sell_position(mm=mm, comment=comment)
                                else:
                                    print(f'\nTime: {now}')
                                    p1_check(p1_positions)
                            else:
                                print(f'\nTime: {now}')
                                print(f"Sorry Risk Not allowed on [{pair[1]}]")
                                p1_check(p1_positions)

                            if not p0_long_market:
                                if time_intervals % trade_time == 0 or p0_buys is None:
                                    print(f"\nGoing LONG on [{pair[0]}]")
                                    p0.open_buy_position(mm=mm, comment=comment)
                                else:
                                    print(f'\nTime: {now}')
                                    p0_check(p0_positions)
                            else:
                                print(f'\nTime: {now}')
                                print(f"Sorry Risk Not allowed on [{pair[0]}] ")
                                p0_check(p0_positions)
                    else:
                        if (
                            p1_signal == "LONG"
                            and p0_signal == "SHORT"
                        ):
                            if not p1_long_market:
                                if time_intervals % trade_time == 0 or p1_buys is None:
                                    print(f"\nGoing LONG on [{pair[1]}]")
                                    p1.open_buy_position(mm=mm, comment=comment)
                                else:
                                    p1_check(p1_positions)
                            else:
                                print(f'\nTime: {now}')
                                print(f"Sorry Risk Not allowed on [{pair[1]}]")
                                p1_check(p1_positions)

                            if not p0_short_market:
                                if time_intervals % trade_time == 0 or p0_sells is None:
                                    print(f"\nGoing SHORT on [{pair[0]}]")
                                    p0.open_sell_position(mm=mm, comment=comment)
                                else:
                                    print(f'\nTime: {now}')
                                    p0_check(p0_positions)
                            else:
                                print(f'\nTime: {now}')
                                print(f"Sorry Risk Not allowed on [{pair[0]}] ")
                                p0_check(p0_positions)
                        elif (
                            p1_signal == "SHORT"
                            and p0_signal == "LONG"
                        ):
                            if not p1_short_market:
                                if time_intervals % trade_time == 0 or p1_sells is None:
                                    print(f"\nGoing SHORT on [{pair[1]}]")
                                    p1.open_sell_position(mm=mm, comment=comment)
                                else:
                                    p1_check(p1_positions)
                            else:
                                print(f'\nTime: {now}')
                                print(f"Sorry Risk Not allowed on [{pair[1]}]")
                                p1_check(p1_positions)

                            if not p0_long_market:
                                if time_intervals % trade_time == 0 or p0_buys is None:
                                    print(f"\nGoing LONG on [{pair[0]}]")
                                    p0.open_buy_position(mm=mm, comment=comment)
                                else:
                                    print(f'\nTime: {now}')
                                    p0_check(p0_positions)
                            else:
                                print(f'\nTime: {now}')
                                print(f"Sorry Risk Not allowed on [{pair[0]}] ")
                                p0_check(p0_positions)
                else:
                    print(f'\nTime: {now}')
                    print("It is Not trading time !!")
                    p0_check(p0_positions)
                    p1_check(p1_positions)
            else:
                print(f'\nTime: {now}')
                print("There is no signal !!")
                p0_check(p0_positions)
                p1_check(p1_positions)

        except Exception as e:
            print(f'Time: {now}')
            print(f"Error: {e}")

        time.sleep((60 * iter_time) - 2.5)

        if iter_time == 1:
            time_intervals += 1
        elif iter_time == trade_time:
            time_intervals += trade_time
        else:
            time_intervals += (trade_time/iter_time)

        if period.lower() == 'month':
            if p0.days_end() and today != 'Friday':
                sleep_time = p0.sleep_time()
                time.sleep(60 * sleep_time)
                num_days += 1

            elif p0.days_end() and today == 'Friday':
                sleep_time = p0.sleep_time(weekend=True)
                time.sleep(60 * sleep_time)
                num_days += 1

            elif (
                    p0.days_end()
                and today == 'Friday'
                and num_days >= 20
            ):
                p0.close_all_positions(comment=comment)
                p1.close_all_positions(comment=comment)
                print("\nEnd of the Month !!!")
                p0.statistics(save=True)
                p1.statistics(save=True)
                break

        elif period.lower() == 'week':
            if p0.days_end() and today != 'Friday':
                sleep_time = p0.sleep_time()
                time.sleep(60 * sleep_time)

            elif p0.days_end() and today == 'Friday':
                p0.close_all_positions(comment=comment)
                p1.close_all_positions(comment=comment)
                print("\nEnd of the Week !!!")
                p0.statistics(save=True)
                p1.statistics(save=True)
                break

        elif period.lower() == 'day':
            if p0.days_end():
                p0.close_all_positions(comment=comment)
                p1.close_all_positions(comment=comment)
                print("\nEnd of the Day !!!")
                p0.statistics(save=True)
                p1.statistics(save=True)
                break


# ========= ORNSTEIN UHLENBECK TRADING ========
def ou_trading(
    trade: Trade,
    tf: str = '1h',
    mm: bool = True,
    max_t: int = 1,
    p: int = 20,
    n: int = 20,
    iter_time: int | float = 30,
    ou_window: int = 2000,
    risk_manager: str | None = None, # 'hmm' is currently supported
    rm_window: int | None = None,
    period: str = 'week',  # (month , week, day)
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

    rate = Rates(trade.symbol, tf, 0, MAX_BARS)
    data = rate.get_rate_frame()
    data['Date'] = pd.to_datetime(data['Date'], unit='s')
    data.set_index('Date', inplace=True)

    def check(buys: list, sells: list):
        if buys is not None or sells is not None:
            print(f"\nChecking for Break even on {trade.symbol}...")
            trade.break_even()

    time_frame_mapping = tf_mapping()
    if tf == 'D1':
        trade_time = trade.get_minutes()
    else: trade_time = time_frame_mapping[tf]

    if regime:
        if risk_manager == 'hmm':
            hmm = HMMRiskManager(data=data, verbose=True, **kwargs)
    strategy = OrnsteinUhlenbeck(
        data['Close'].values[-ou_window:], timeframe=tf)

    time_intervals = 0
    long_market = False
    short_market = False
    num_days = 0
    print(f'\nRunning OU Strategy on {trade.symbol} in {tf} Interval ...\n')
    while True:   
        current_date = datetime.now()
        today = current_date.strftime("%A")
        try:
            buys = trade.get_current_buys()
            if buys is not None:
                print(f"\nCurrent buy positions on {trade.symbol}: \n{buys}")
            sells = trade.get_current_sells()
            if sells is not None:
                print(f"\nCurrent sell positions on {trade.symbol}: \n{sells}")
            long_market = buys is not None and len(buys) >= max_t
            short_market = sells is not None and len(sells) >= max_t

            time.sleep(0.5)
            if regime:
                if risk_manager ==  'hmm':
                    hmm_returns = Rates(trade.symbol, tf, 0, rm_window)
                    hmm_returns_val = hmm_returns.get_returns.values
                    current_regime = hmm.which_trade_allowed(hmm_returns_val)
                    print(f'CURRENT REGIME : {current_regime}')
            else:
                current_regime = None
            print("Calculating signal ..")
            ou_returns = Rates(trade.symbol, tf, 0, p)
            returns_val = ou_returns.get_returns.values
            signal = strategy.calculate_signals(returns_val, p=p, n=n)
            print(f"SIGNAL : {signal}")
            comment = f"{trade.expert_name}@{trade.version}"
            now = datetime.now().strftime("%H:%M:%S")
            if trade.trading_time() and today in TRADING_DAYS:
                if signal is not None:
                    if signal == "LONG" and short_market:
                        print(f'Time: {now}')
                        trade.close_all_sells()
                        short_market  = False
                    elif signal == "SHORT" and long_market:
                        print(f'Time: {now}')
                        trade.close_all_buys()
                        long_market =  False
                    if current_regime is not None:
                        if current_regime == 'LONG':
                            if signal == "LONG" and not long_market:
                                if time_intervals % trade_time == 0 or buys is None:
                                    print("Sending buy Order ....")
                                    trade.open_buy_position(mm=mm, comment=comment)
                                else:
                                    print(f'\nTime: {now}')
                                    check(buys, sells)

                            elif signal == "LONG" and long_market:
                                print(f'\nTime: {now}')
                                print("Sorry Risk not allowed !!!")
                                check(buys, sells)

                        elif current_regime == 'SHORT':
                            if signal == "SHORT" and not short_market:
                                if time_intervals % trade_time == 0 or sells is None:
                                    print("Sending Sell Order ....")
                                    trade.open_sell_position(mm=mm, comment=comment)
                                else:
                                    print(f'\nTime: {now}')
                                    check(buys, sells)
                            elif signal == "SHORT" and short_market:
                                print(f'\nTime: {now}')
                                print("Sorry Risk not Allowed !!!")
                                check(buys, sells)
                    else:
                        if signal == "LONG" and not long_market:
                            if time_intervals % trade_time == 0 or buys is None:
                                print("Sending buy Order ....")
                                trade.open_buy_position(mm=mm, comment=comment)
                            else:
                                print(f'\nTime: {now}')
                                check(buys, sells)

                        elif signal == "LONG" and long_market:
                            print(f'\nTime: {now}')
                            print("\nSorry Risk not allowed !!!")
                            check(buys, sells)

                        if signal == "SHORT" and not short_market:
                            if time_intervals % trade_time == 0 or sells is None:
                                print("Sending Sell Order ....")
                                trade.open_sell_position(mm=mm, comment=comment)
                            else:
                                print(f'\nTime: {now}')
                                check(buys, sells)
                        elif signal == "SHORT" and short_market:
                            print(f'\nTime: {now}')
                            print("Sorry Risk not Allowed !!!")
                            check(buys, sells)
                else:
                    print(f'\nTime: {now}')
                    print("There is no signal !!")
                    check(buys, sells)
            else:
                print(f'\nTime: {now}')
                print("Sorry It is Not trading Time !!!")
                check(buys, sells)
        except Exception as e:
            print(f'Time: {now}')
            print(f"Error: {e}")
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
                time.sleep(60 * sleep_time)
                num_days += 1

            elif trade.days_end() and today == 'Friday':
                sleep_time = trade.sleep_time(weekend=True)
                time.sleep(60 * sleep_time)
                num_days += 1

            elif (
                    trade.days_end()
                and today == 'Friday'
                and num_days >= 20
            ):
                trade.close_all_positions(comment=comment)
                print("\nEnd of the Month !!!")
                trade.statistics(save=True)
                break

        elif period.lower() == 'week':
            if trade.days_end() and today != 'Friday':
                sleep_time = trade.sleep_time()
                time.sleep(60 * sleep_time)

            elif trade.days_end() and today == 'Friday':
                trade.close_all_positions(comment=comment)
                print("\nEnd of the Week !!!")
                trade.statistics(save=True)
                break

        elif period.lower() == 'day':
            if trade.days_end():
                trade.close_all_positions(comment=comment)
                print("\nEnd of the Day !!!")
                trade.statistics(save=True)
                break


# ========= ARIMA + GARCH TRADING =======================
def arch_trading(
    trade: Trade,
    tf: str = 'D1',
    k: int = 500,
    mm: bool = True,
    max_t: int = 1,
    iter_time: int | float = 30,
    risk_manager: str | None = None, # 'hmm',
    rm_window: int | None = None,
    period: str = 'month',  # (month , week, day)
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
            print(f"\nChecking for Break even on {trade.symbol}...")
            trade.break_even()

    time_frame_mapping = tf_mapping()
    if tf == 'D1':
        trade_time = trade.get_minutes()
    else: trade_time = time_frame_mapping[tf]

    rate = Rates(trade.symbol, tf, 0, MAX_BARS)
    data = rate.get_rate_frame()
    data['Date'] = pd.to_datetime(data['Date'], unit='s')
    data.set_index('Date', inplace=True)
    strategy = ArimaGarchStrategy(trade.symbol, data, k=k)
    if regime:
        if risk_manager == 'hmm':
            hmm = HMMRiskManager(
                data=data, verbose=True, iterations=5000, **kwargs)

    time_intervals = 0
    long_market = False
    short_market = False
    num_days = 0
    print(f'\nRunning ARIMA + GARCH Strategy on {trade.symbol} in {tf} Interval ...\n')
    while True:  
        current_date = datetime.now()
        today = current_date.strftime("%A")
        try:
            buys = trade.get_current_buys()
            if buys is not None:
                print(f"\nCurrent buy positions on {trade.symbol}: \n{buys}")
            sells = trade.get_current_sells()
            if sells is not None:
                print(f"\nCurrent sell positions on {trade.symbol}: \n{sells}")
            long_market = buys is not None and len(buys) >= max_t
            short_market = sells is not None and len(sells) >= max_t

            time.sleep(0.5)
            if regime:
                if risk_manager == 'hmm':
                    hmm_returns = Rates(trade.symbol, tf, 0, rm_window)
                    hmm_returns_val = hmm_returns.get_returns.values
                    current_regime = hmm.which_trade_allowed(hmm_returns_val)
                    print(f'CURRENT REGIME : {current_regime}')
            else:
                current_regime = None
            print("Calculating Signal ...")
            arch_data = Rates(trade.symbol, tf, 0, k)
            rates = arch_data.get_rate_frame()
            arch_returns = strategy.load_and_prepare_data(rates)
            window_data = arch_returns['diff_log_return'].iloc[-k:]
            signal = strategy.calculate_signals(window_data)
            print(f"SIGNAL : {signal}")

            comment = f"{trade.expert_name}@{trade.version}"
            now = datetime.now().strftime("%H:%M:%S")
            if trade.trading_time() and today in TRADING_DAYS:
                if signal is not None:
                    if signal == "LONG" and short_market:
                        print(f'\nTime: {now}')
                        trade.close_all_sells()
                        short_market = False
                    elif signal == "SHORT" and long_market:
                        print(f'\nTime: {now}')
                        trade.close_all_buys()
                        long_market = False
                    if current_regime is not None:
                        if current_regime == 'LONG':
                            if signal == "LONG" and not long_market:
                                if time_intervals % trade_time == 0 or buys is None:
                                    print("Sending buy Order ....")
                                    trade.open_buy_position(mm=mm, comment=comment)
                                else:
                                    print(f'Time: {now}')
                                    check(buys, sells)

                            elif signal == "LONG" and long_market:
                                print(f'\nTime: {now}')
                                print("Sorry Risk not allowed !!!")
                                check(buys, sells)

                        elif current_regime == 'SHORT':
                            if signal == "SHORT" and not short_market:
                                if time_intervals % trade_time == 0 or sells is None:
                                    print("Sending Sell Order ....")
                                    trade.open_sell_position(mm=mm, comment=comment)
                                else:
                                    print(f'\nTime: {now}')
                                    check(buys, sells)
                            elif signal == "SHORT" and short_market:
                                print(f'\nTime: {now}')
                                print("Sorry Risk not Allowed !!!")
                                check(buys, sells)
                    else:
                        if signal == "LONG" and not long_market:
                            if time_intervals % trade_time == 0 or buys is None:
                                print("Sending buy Order ....")
                                trade.open_buy_position(mm=mm, comment=comment)
                            else:
                                print(f'\nTime: {now}')
                                check(buys, sells)

                        elif signal == "LONG" and long_market:
                            print(f'\nTime: {now}')
                            print("\nSorry Risk not allowed !!!")
                            check(buys, sells)

                        if signal == "SHORT" and not short_market:
                            if time_intervals % trade_time == 0 or sells is None:
                                print("Sending Sell Order ....")
                                trade.open_sell_position(mm=mm, comment=comment)
                            else:
                                print(f'\nTime: {now}')
                                check(buys, sells)

                        elif signal == "SHORT" and short_market:
                            print(f'\nTime: {now}')
                            print("Sorry Risk not Allowed !!!")
                            check(buys, sells)
                else:
                    print(f'\nTime: {now}')
                    print("There is no signal !!")
                    check(buys, sells)
            else:
                print(f'\nTime: {now}')
                print("Sorry It is Not trading Time !!!")
                check(buys, sells)

        except Exception as e:
            print(f'Time: {now}')
            print(f"Error: {e}")

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
                time.sleep(60 * sleep_time)
                num_days += 1

            elif trade.days_end() and today == 'Friday':
                sleep_time = trade.sleep_time(weekend=True)
                time.sleep(60 * sleep_time)
                num_days += 1

            elif (
                    trade.days_end()
                and today == 'Friday'
                and num_days >= 20
            ):
                trade.close_all_positions(comment=comment)
                print("\nEnd of the Month !!!")
                trade.statistics(save=True)
                break

        elif period.lower() == 'week':
            if trade.days_end() and today != 'Friday':
                sleep_time = trade.sleep_time()
                time.sleep(60 * sleep_time)

            elif trade.days_end() and today == 'Friday':
                trade.close_all_positions(comment=comment)
                print("\nEnd of the Week !!!")
                trade.statistics(save=True)
                break

        elif period.lower() == 'day':
            if trade.days_end():
                trade.close_all_positions(comment=comment)
                print("\nEnd of the Day !!!")
                trade.statistics(save=True)
                break
