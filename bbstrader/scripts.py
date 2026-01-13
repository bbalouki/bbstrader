import argparse
import asyncio
import importlib
import importlib.util
import json
import multiprocessing
import multiprocessing as mp
import os
import sys
import textwrap
import time
from datetime import datetime, timedelta
from types import ModuleType
from typing import Any, Dict, List, Literal, Type

import nltk
from loguru import logger
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer

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
from bbstrader.btengine.strategy import BaseStrategy
from bbstrader.core.data import FinancialNews
from bbstrader.core.strategy import Strategy
from bbstrader.metatrader._copier import main as RunCopyApp
from bbstrader.metatrader.copier import RunCopier, config_copier, copier_worker_process
from bbstrader.metatrader.trade import create_trade_instance
from bbstrader.trading.execution import RunMt5Engine
from bbstrader.trading.strategy import LiveStrategy
from bbstrader.trading.utils import send_telegram_message

EXECUTION_PATH = os.path.expanduser("~/.bbstrader/execution/execution.py")
CONFIG_PATH = os.path.expanduser("~/.bbstrader/execution/execution.json")
BACKTEST_PATH = os.path.expanduser("~/.bbstrader/backtest/backtest.py")
CONFIG_PATH = os.path.expanduser("~/.bbstrader/backtest/backtest.json")


DATA_HANDLER_MAP: Dict[str, Type[DataHandler]] = {
    "csv": CSVDataHandler,
    "mt5": MT5DataHandler,
    "yf": YFDataHandler,
    "eodh": EODHDataHandler,
    "fmp": FMPDataHandler,
}

EXECUTION_HANDLER_MAP: Dict[str, Type[ExecutionHandler]] = {
    "sim": SimExecutionHandler,
    "mt5": MT5ExecutionHandler,
}


__all__ = ["load_module", "load_class"]


def load_module(file_path: str) -> ModuleType:
    """Load a module from a file path.
    Args:
        file_path: Path to the file to load.
    Returns:
        The loaded module.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Strategy file {file_path} not found. Please create it."
        )
    spec = importlib.util.spec_from_file_location("bbstrader.cli", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module at {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_class(module: ModuleType, class_name: str, base_class: Type) -> Type:
    """Load a class from a module.
    Args:
        module: The module to load the class from.
        class_name: The name of the class to load.
        base_class: The base class that the class must inherit from.
    """
    if not hasattr(module, class_name):
        raise AttributeError(f"{class_name} not found in {module}")
    class_ = getattr(module, class_name)
    if not issubclass(class_, base_class):
        raise TypeError(f"{class_name} must inherit from {base_class}.")
    return class_


##############################################################
###################### BACKTESTING ###########################
##############################################################


def load_exc_handler(module: ModuleType, handler_name: str) -> Type[ExecutionHandler]:
    return load_class(module, handler_name, ExecutionHandler)  # type: ignore


def load_data_handler(module: ModuleType, handler_name: str) -> Type[DataHandler]:
    return load_class(module, handler_name, DataHandler)  # type: ignore


def load_strategy(module: ModuleType, strategy_name: str) -> Type[Strategy]:
    return load_class(module, strategy_name, (Strategy, BaseStrategy))  # type: ignore


def load_backtest_config(config_path: str, strategy_name: str) -> Dict[str, Any]:
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


def backtest(unknown: List[str]) -> None:
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
    config = load_backtest_config(args.config, args.strategy)
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


##############################################################
###################### LIVE EXECUTION ########################
##############################################################


def load_live_config(config_path, strategy_name, account=None):
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
    strategy_class = load_class(
        strategy_module, args.strategy, (LiveStrategy, Strategy)
    )

    config = load_live_config(args.config, args.strategy, account)
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


##########################################################
##################### TRADE COPIER #######################
##########################################################


def copier_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="CLI",
        choices=("CLI", "GUI"),
        help="Run the copier in the terminal or using the GUI",
    )
    parser.add_argument(
        "-s", "--source", type=str, nargs="?", default=None, help="Source section name"
    )
    parser.add_argument(
        "-I", "--id", type=int, default=0, help="Source Account unique ID"
    )
    parser.add_argument(
        "-U",
        "--unique",
        action="store_true",
        help="Specify if the source account is only master",
    )
    parser.add_argument(
        "-d",
        "--destinations",
        type=str,
        nargs="*",
        default=None,
        help="Destination section names",
    )
    parser.add_argument(
        "-i", "--interval", type=float, default=0.1, help="Update interval in seconds"
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        default=None,
        type=str,
        help="Config file name or path",
    )
    parser.add_argument(
        "-t",
        "--start",
        type=str,
        nargs="?",
        default=None,
        help="Start time in HH:MM format",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=str,
        nargs="?",
        default=None,
        help="End time in HH:MM format",
    )
    parser.add_argument(
        "-M",
        "--multiprocess",
        action="store_true",
        help="Run each destination account in a separate process.",
    )
    return parser


def copy_trades(unknown):
    HELP_MSG = """
    Usage:
        python -m bbstrader --run copier [options]

    Options:
        -m, --mode: CLI for terminal app and GUI for Desktop app
        -s, --source: Source Account section name
        -I, --id: Source Account unique ID
        -U, --unique: Specify if the source account is only master 
        -d, --destinations: Destination Account section names (multiple allowed)
        -i, --interval: Update interval in seconds
        -M, --multiprocess: When set to True, each destination account runs in a separate process.
        -c, --config: .ini file or path (default: ~/.bbstrader/copier/copier.ini)
        -t, --start: Start time in HH:MM format
        -e, --end: End time in HH:MM format
    """
    if "-h" in unknown or "--help" in unknown:
        print(HELP_MSG)
        sys.exit(0)

    copy_parser = argparse.ArgumentParser("Trades Copier", add_help=False)
    copy_parser = copier_args(copy_parser)
    copy_args = copy_parser.parse_args(unknown)

    if copy_args.mode == "GUI":
        RunCopyApp()

    elif copy_args.mode == "CLI":
        source, destinations = config_copier(
            source_section=copy_args.source,
            dest_sections=copy_args.destinations,
            inifile=copy_args.config,
        )
        source["id"] = copy_args.id
        source["unique"] = copy_args.unique
        if copy_args.multiprocess:
            copier_processes = []
            for dest_config in destinations:
                process = multiprocessing.Process(
                    target=copier_worker_process,
                    args=(
                        source,
                        dest_config,
                        copy_args.interval,
                        copy_args.start,
                        copy_args.end,
                    ),
                )
                process.start()
                copier_processes.append(process)
            for process in copier_processes:
                process.join()
        else:
            RunCopier(
                source,
                destinations,
                copy_args.interval,
                copy_args.start,
                copy_args.end,
            )


############################################################
##################### NEWS FEED ############################
############################################################


def summarize_text(text: str, sentences_count: int = 5) -> str:
    """
    Generate a summary using TextRank algorithm.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)


def format_coindesk_article(article: Dict[str, Any]) -> str:
    if not all(
        k in article
        for k in (
            "body",
            "title",
            "published_on",
            "sentiment",
            "keywords",
            "status",
            "url",
        )
    ):
        return ""
    summary = summarize_text(article["body"], sentences_count=3)
    text = (
        f"📰 {article['title']}\n"
        f"Published Date: {article['published_on']}\n"
        f"Sentiment: {article['sentiment']}\n"
        f"Status: {article['status']}\n"
        f"Keywords: {article['keywords']}\n\n"
        f"🔍 Summary\n"
        f"{textwrap.fill(summary, width=80)}"
        f"\n\n👉 Visit {article['url']} for full article."
    )
    return text


def format_fmp_article(article: Dict[str, Any]) -> str:
    if not all(k in article for k in ("title", "date", "content", "tickers")):
        return ""
    summary = summarize_text(article["content"], sentences_count=3)
    text = (
        f"📰 {article['title']}\n"
        f"Published Date: {article['date']}\n"
        f"Keywords: {article['tickers']}\n\n"
        f"🔍 Summary\n"
        f"{textwrap.fill(summary, width=80)}"
    )
    return text


async def send_articles(
    articles: List[Dict[str, Any]],
    token: str,
    id: str,
    source: Literal["coindesk", "fmp"],
    interval: int = 15,
) -> None:
    for article in articles:
        message = ""
        if source == "coindesk":
            published_on = article.get("published_on")
            if isinstance(
                published_on, datetime
            ) and published_on >= datetime.now() - timedelta(minutes=interval):
                article["published_on"] = published_on.strftime("%Y-%m-%d %H:%M:%S")
                message = format_coindesk_article(article)
        else:
            message = format_fmp_article(article)
        if message == "":
            continue
        await asyncio.sleep(2)  # To avoid hitting Telegram rate limits
        await send_telegram_message(token, id, text=message)


def send_news_feed(unknown: List[str]) -> None:
    HELP_MSG = """
    Send news feed from Coindesk to Telegram channel.
    This script fetches the latest news articles from Coindesk, summarizes them,
    and sends them to a specified Telegram channel at regular intervals.

    Usage:
        python -m bbstrader --run news_feed [options]

    Options:
        -q, --query: The news to look for (default: "")
        -t, --token: Telegram bot token
        -I, --id: Telegram Chat id
            --fmp: Financial Modeling Prop Api Key
        -i, --interval: Interval in minutes to fetch news (default: 15)

    Note:
        The script will run indefinitely, fetching news every 15 minutes.
        Use Ctrl+C to stop the script.
    """

    if "-h" in unknown or "--help" in unknown:
        print(HELP_MSG)
        sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q", "--query", type=str, default="", help="The news to look for"
    )
    parser.add_argument(
        "-t",
        "--token",
        type=str,
        required=True,
        help="Telegram bot token",
    )
    parser.add_argument("-I", "--id", type=str, required=True, help="Telegram Chat id")
    parser.add_argument(
        "--fmp", type=str, default="", help="Financial Modeling Prop Api Key"
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=15,
        help="Interval in minutes to fetch news (default: 15)",
    )
    args = parser.parse_args(unknown)

    nltk.download("punkt", quiet=True)
    news = FinancialNews()
    fmp_news = news.get_fmp_news(api=args.fmp) if args.fmp else None
    logger.info(f"Starting the News Feed on {args.interval} minutes")
    while True:
        try:
            fmp_articles: List[Dict[str, Any]] = []
            if fmp_news is not None:
                fmp_articles = fmp_news.get_latest_articles(limit=5)
            coindesk_articles = news.get_coindesk_news(query=args.query)
            if coindesk_articles and isinstance(coindesk_articles, list):
                asyncio.run(
                    send_articles(
                        coindesk_articles,  # type: ignore
                        args.token,
                        args.id,
                        "coindesk",
                        interval=args.interval,
                    )
                )
            if len(fmp_articles) != 0:
                asyncio.run(send_articles(fmp_articles, args.token, args.id, "fmp"))
            time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            logger.info("Stopping the News Feed ...")
            sys.exit(0)
        except Exception as e:
            logger.error(e)
