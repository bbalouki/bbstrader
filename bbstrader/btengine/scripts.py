import argparse
import json
import os
import sys
from datetime import datetime

from bbstrader.btengine.backtest import run_backtest
from bbstrader.btengine.data import (
    CSVDataHandler,
    DataHandler,
    EODHDataHandler,
    FMPDataHandler,
    MT5DataHandler,
    YFDataHandler,
)
from bbstrader.btengine.execution import (
    ExecutionHandler,
    MT5ExecutionHandler,
    SimExecutionHandler,
)
from bbstrader.core.utils import load_class, load_module

BACKTEST_PATH = os.path.expanduser("~/.bbstrader/backtest/backtest.py")
CONFIG_PATH = os.path.expanduser("~/.bbstrader/backtest/backtest.json")

DATA_HANDLER_MAP = {
    "csv": CSVDataHandler,
    "mt5": MT5DataHandler,
    "yf": YFDataHandler,
    "eodh": EODHDataHandler,
    "fmp": FMPDataHandler,
}

EXECUTION_HANDLER_MAP = {
    "sim": SimExecutionHandler,
    "mt5": MT5ExecutionHandler,
}


def load_exc_handler(module, handler_name):
    return load_class(module, handler_name, ExecutionHandler)


def load_data_handler(module, handler_name):
    return load_class(module, handler_name, DataHandler)


def load_strategy(module, strategy_name):
    from bbstrader.btengine.strategy import MT5Strategy, Strategy

    return load_class(module, strategy_name, (Strategy, MT5Strategy))


def load_config(config_path, strategy_name):
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file {config_path} not found. Please create it."
        )

    with open(config_path, "r") as f:
        config = json.load(f)
    try:
        config = config[strategy_name]
    except KeyError:
        raise ValueError(
            f"Strategy {strategy_name} not found in the configuration file."
        )

    required_fields = ["symbol_list", "start_date", "data_handler", "execution_handler"]
    for field in required_fields:
        if not config.get(field):
            raise ValueError(f"{field} is required in the configuration file.")

    config["start_date"] = datetime.strptime(config["start_date"], "%Y-%m-%d")

    if config.get("execution_handler") not in EXECUTION_HANDLER_MAP:
        try:
            backtest_module = load_module(BACKTEST_PATH)
            exc_handler_class = load_exc_handler(
                backtest_module, config["execution_handler"]
            )
        except Exception as e:
            raise ValueError(f"Invalid execution handler: {e}")
    else:
        exc_handler_class = EXECUTION_HANDLER_MAP[config["execution_handler"]]

    if config.get("data_handler") not in DATA_HANDLER_MAP:
        try:
            backtest_module = load_module(BACKTEST_PATH)
            data_handler_class = load_data_handler(
                backtest_module, config["data_handler"]
            )
        except Exception as e:
            raise ValueError(f"Invalid data handler: {e}")
    else:
        data_handler_class = DATA_HANDLER_MAP[config["data_handler"]]

    config["execution_handler"] = exc_handler_class
    config["data_handler"] = data_handler_class

    return config


def backtest(unknown):
    HELP_MSG = """
    Usage:
        python -m bbstrader --run backtest [options]

    Options:
        -s, --strategy: Strategy class name to run
        -c, --config: Configuration file path (default: ~/.bbstrader/backtest/backtest.json)
        -p, --path: Path to the backtest file (default: ~/.bbstrader/backtest/backtest.py)
    
    Note:
        The configuration file must contain all the required parameters 
        for the data handler and execution handler and strategy.
        See bbstrader.btengine.BacktestEngine for more details on the parameters.
    """
    if "-h" in unknown or "--help" in unknown:
        print(HELP_MSG)
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Backtesting Engine CLI")
    parser.add_argument(
        "-s", "--strategy", type=str, required=True, help="Strategy class name to run"
    )
    parser.add_argument(
        "-c", "--config", type=str, default=CONFIG_PATH, help="Configuration file path"
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=BACKTEST_PATH,
        help="Path to the backtest file",
    )
    args = parser.parse_args(unknown)
    config = load_config(args.config, args.strategy)
    strategy_module = load_module(args.path)
    strategy_class = load_strategy(strategy_module, args.strategy)

    symbol_list = config.pop("symbol_list")
    start_date = config.pop("start_date")
    data_handler = config.pop("data_handler")
    execution_handler = config.pop("execution_handler")

    try:
        run_backtest(
            symbol_list,
            start_date,
            data_handler,
            strategy_class,
            exc_handler=execution_handler,
            **config,
        )
    except Exception as e:
        print(f"Error: {e}")
