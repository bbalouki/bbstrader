import argparse
import sys

import pyfiglet
from colorama import Fore

from bbstrader.btengine.scripts import backtest
from bbstrader.metatrader.scripts import copy_trades
from bbstrader.trading.scripts import execute_strategy

DESCRIPTION = "BBSTRADER"
USAGE_TEXT = """
    Usage:
        python -m bbstrader --run <module> [options]

    Modules:
        copier: Copy trades from one MetaTrader account to another or multiple accounts
        backtest: Backtest a strategy, see bbstrader.btengine.backtest.run_backtest
        execution: Execute a strategy, see bbstrader.trading.execution.Mt5ExecutionEngine
    
    python -m bbstrader --run <module> --help for more information on the module
"""

FONT = pyfiglet.figlet_format("BBSTRADER", font="big")


def main():
    print(Fore.BLUE + FONT)
    print(Fore.WHITE + "")
    parser = argparse.ArgumentParser(
        prog="BBSTRADER",
        usage=USAGE_TEXT,
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )
    parser.add_argument("--run", type=str, nargs="?", default=None, help="Run a module")
    args, unknown = parser.parse_known_args()
    if ("-h" in sys.argv or "--help" in sys.argv) and args.run is None:
        print(Fore.WHITE + USAGE_TEXT)
        sys.exit(0)
    if args.run == "copier":
        copy_trades(unknown)
    elif args.run == "backtest":
        backtest(unknown)
    elif args.run == "execution":
        execute_strategy(unknown)


if __name__ == "__main__":
    main()
