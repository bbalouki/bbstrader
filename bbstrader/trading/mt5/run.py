import time
import argparse
from mtrader5.trade import Trade
from trading.mt5.execution import (
    sma_trading, pair_trading, ou_trading, arch_trading)
from trading.mt5.utils import (
    add_sma_trading_arguments, add_pair_trading_arguments,
    add_ou_trading_arguments, add_arch_trading_arguments,
    init_trade
    )

def run_sma_trading():
    # Create parser
    parser = argparse.ArgumentParser(
        description='Run SMA trading strategy.')

    # Add arguments for sma_trading parameters
    parser = add_sma_trading_arguments(parser)

    # Parse arguments
    args = parser.parse_args()

    # Initialize Trade with command-line arguments
    trade = init_trade(args)

    # Call sma_trading with command-line arguments
    sma_trading(
        trade, 
        tf=args.tf, 
        sma=args.sma, 
        lma=args.lma, 
        mm=args.mm, 
        max_t=args.max_t, 
        iter_time=args.iter_time, 
        risk_manager=args.risk_manager, 
        period=args.period
    )

def run_pair_trading(pair=True, pchange_sl=3.0):
    # Create parser
    parser = argparse.ArgumentParser(
        description='Run Pair trading strategy.')

    # Add arguments for pair_trading parameters
    parser = add_pair_trading_arguments(
        parser, pair=pair, pchange_sl=pchange_sl)

    # Parse arguments
    args = parser.parse_args()
    tickers = tuple(args.pair)

    # Initialize Trade with command-line arguments
    p0 = init_trade(args, symbol=tickers[0])
    time.sleep(5)
    p1 = init_trade(args, symbol=tickers[1])
    # Call pair_trading with command-line arguments
    pair_trading(
        tickers,
        p0,
        p1,
        args.tf,
        ols=args.ols,
        max_t=args.max_t,
        mm=args.mm,
        iter_time=args.iter_time,
        risk_manager=args.risk_manager,
        hmm_ticker=args.hmm_ticker,
        period=args.period
    )

def run_ou_trading():
    # Create parser
    parser = argparse.ArgumentParser(
        description='Run Ornstein-Uhlenbeck trading strategy.')

    # Add arguments for ou_trading parameters
    parser = add_ou_trading_arguments(parser)

    # Parse arguments
    args = parser.parse_args()

    # Initialize Trade with command-line arguments
    trade = init_trade(args)

    # Call ou_trading with command-line arguments
    ou_trading(
        trade,
        tf=args.tf,
        n=args.n,
        p=args.p,
        max_t=args.max_t,
        mm=args.mm,
        iter_time=args.iter_time,
        ou_window=args.ou_window,
        hmm_window=args.hmm_window,
        risk_manager=args.risk_manager,
        period=args.period
    )


def run_arch_trading():
    # Create parser
    parser = argparse.ArgumentParser(
        description='Run ARIMA + GARCH trading strategy.')

    # Add arguments for arch_trading parameters
    parser = add_arch_trading_arguments(parser)

    # Parse arguments
    args = parser.parse_args()

    # Initialize Trade with command-line arguments
    trade = init_trade(args)

    # Call arch_trading with command-line arguments
    arch_trading(
        trade,
        tf=args.tf,
        max_t=args.max_t,
        mm=args.mm,
        k=args.k,
        hmm_window=args.hmm_window,
        iter_time=args.iter_time,
        risk_manager=args.risk_manager,
        period=args.period
    )