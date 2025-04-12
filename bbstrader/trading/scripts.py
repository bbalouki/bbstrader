import argparse
import json
import multiprocessing as mp
import os
import sys

from bbstrader.btengine import MT5Strategy, Strategy
from bbstrader.core.utils import load_class, load_module
from bbstrader.metatrader.trade import create_trade_instance
from bbstrader.trading.execution import RunMt5Engine

EXECUTION_PATH = os.path.expanduser("~/.bbstrader/execution/execution.py")
CONFIG_PATH = os.path.expanduser("~/.bbstrader/execution/execution.json")


def load_config(config_path, strategy_name, account=None):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    try:
        config = config[strategy_name]
    except KeyError:
        raise ValueError(
            f"Strategy {strategy_name} not found in the configuration file."
        )
    if account is not None:
        try:
            config = config[account]
        except KeyError:
            raise ValueError(f"Account {account} not found in the configuration file.")
    if config.get("symbol_list") is None:
        raise ValueError("symbol_list is required in the configuration file.")
    if config.get("trades_kwargs") is None:
        raise ValueError("trades_kwargs is required in the configuration file.")
    return config


def worker_function(account, args):
    strategy_module = load_module(args.path)
    strategy_class = load_class(strategy_module, args.strategy, (MT5Strategy, Strategy))

    config = load_config(args.config, args.strategy, account)
    symbol_list = config.pop("symbol_list")
    trades_kwargs = config.pop("trades_kwargs")
    trades = create_trade_instance(symbol_list, trades_kwargs)

    kwargs = {
        "symbol_list": symbol_list,
        "trades_instances": trades,
        "strategy_cls": strategy_class,
        "account": account,
        **config,
    }
    RunMt5Engine(account, **kwargs)


def RunMt5Terminal(args):
    if args.parallel:
        if len(args.account) == 0:
            raise ValueError(
                "account or accounts are required when running in parallel"
            )

        processes = []
        try:
            for account in args.account:
                p = mp.Process(target=worker_function, args=(account, args))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        except Exception as e:
            print(f"Error in parallel execution: {e}")
            raise e
        except KeyboardInterrupt:
            print("\nTerminating Execution...")
            for p in processes:
                p.terminate()
            for p in processes:
                p.join()
            print("Execution terminated")
    else:
        worker_function(args.account[0], args)


def RunTWSTerminal(args):
    raise NotImplementedError("RunTWSTerminal is not implemented yet")

def execute_strategy(unknown):
    HELP_MSG = """
    Execute a strategy on one or multiple MT5 accounts.

    Usage:
        python -m bbstrader --run execution [options]

    Options:
        -s, --strategy: Strategy class name to run
        -a, --account: Account(s) name(s) or ID(s) to run the strategy on (must be the same as in the configuration file)
        -p, --path: Path to the execution file (default: ~/.bbstrader/execution/execution.py)
        -c, --config: Path to the configuration file (default: ~/.bbstrader/execution/execution.json)
        -l, --parallel: Run the strategy in parallel (default: False)
        -t, --terminal: Terminal to use (default: MT5)
        -h, --help: Show this help message and exit
    
    Note:
        The configuration file must contain all the required parameters 
        to create trade instances for each account and strategy.
        The configuration file must be a dictionary with the following structure:
        If parallel is True:
        {
            "strategy_name": {
                "account_name": {
                    "symbol_list": ["symbol1", "symbol2"],
                    "trades_kwargs": {"param1": "value1", "param2": "value2"}
                    **other_parameters (for the strategy and the execution engine)
                }
            }
        }
        If parallel is False:
        {
            "strategy_name": {
                "symbol_list": ["symbol1", "symbol2"],
                "trades_kwargs": {"param1": "value1", "param2": "value2"}
                **other_parameters (for the strategy and the execution engine)
            }
        }
        See bbstrader.metatrader.trade.create_trade_instance for more details on the trades_kwargs.
        See bbstrader.trading.execution.Mt5ExecutionEngine for more details on the other parameters.
        
        All other paramaters must be python built-in types. 
        If you have custom type you must set them in your strategy class 
        or run the Mt5ExecutionEngine directly, don't run on CLI.
    """
    if "-h" in unknown or "--help" in unknown:
        print(HELP_MSG)
        sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--strategy", type=str, required=True)
    parser.add_argument("-a", "--account", type=str, nargs="*", default=[])
    parser.add_argument("-p", "--path", type=str, default=EXECUTION_PATH)
    parser.add_argument("-c", "--config", type=str, default=CONFIG_PATH)
    parser.add_argument("-l", "--parallel", action="store_true")
    parser.add_argument(
        "-t", "--terminal", type=str, default="MT5", choices=["MT5", "TWS"]
    )
    args = parser.parse_args(unknown)

    if args.terminal == "MT5":
        RunMt5Terminal(args)
    elif args.terminal == "TWS":
        RunTWSTerminal(args)
