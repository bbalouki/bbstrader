import argparse
import multiprocessing
import sys
from enum import Enum

import pyfiglet
from colorama import Fore

from bbstrader.btengine.scripts import backtest
from bbstrader.core.scripts import send_news_feed
from bbstrader.metatrader.scripts import copy_trades
from bbstrader.trading.scripts import execute_strategy


class _Module(Enum):
    COPIER = "copier"
    BACKTEST = "backtest"
    EXECUTION = "execution"
    NEWS_FEED = "news_feed"


FONT = pyfiglet.figlet_format("BBSTRADER", font="big")


def main():
    DESCRIPTION = "BBSTRADER"
    USAGE_TEXT = """
    Usage:
        python -m bbstrader --run <module> [options]

    Modules:
        copier: Copy trades from one MetaTrader account to another or multiple accounts
        backtest: Backtest a strategy, see bbstrader.btengine.backtest.run_backtest
        execution: Execute a strategy, see bbstrader.trading.execution.Mt5ExecutionEngine
        news_feed: Send news feed from Coindesk to Telegram channel
    
    python -m bbstrader --run <module> --help for more information on the module
    """
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
    try:
        match args.run:
            case _Module.COPIER.value:
                copy_trades(unknown)
            case _Module.BACKTEST.value:
                backtest(unknown)
            case _Module.EXECUTION.value:
                execute_strategy(unknown)
            case _Module.NEWS_FEED.value:
                send_news_feed(unknown)
            case _:
                print(Fore.RED + f"Unknown module: {args.run}")
                sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(Fore.RED + f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.RED + "\nExecution interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(Fore.RED + f"Error: {e}")
        sys.exit(1)
