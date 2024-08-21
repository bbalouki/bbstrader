import time
import argparse
from bbstrader.metatrader.trade import Trade
from bbstrader.trading.execution import (
    sma_trading, pair_trading, ou_trading, arch_trading)
from bbstrader.trading.utils import (
    add_sma_trading_arguments, add_pair_trading_arguments,
    add_ou_trading_arguments, add_arch_trading_arguments,
    init_trade)

__all__ = [
    "run_sma_trading",
    "run_pair_trading",
    "run_ou_trading",
    "run_arch_trading",
]
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
        max_t=args.mxt, 
        iter_time=args.it, 
        risk_manager=args.rm, 
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
        ols=args.rmw,
        max_t=args.mxt,
        mm=args.mm,
        iter_time=args.it,
        risk_manager=args.rm,
        rm_ticker=args.rmt,
        rm_window=args.rmw,
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
        max_t=args.mxt,
        mm=args.mm,
        iter_time=args.it,
        ou_window=args.ouw,
        rm_window=args.rmw,
        risk_manager=args.rm,
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
        max_t=args.mxt,
        mm=args.mm,
        k=args.k,
        iter_time=args.it,
        risk_manager=args.rm,
        rm_window=args.rmw,
        period=args.period
    )