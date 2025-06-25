import multiprocessing
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Tuple

from loguru import logger as log

from bbstrader.config import BBSTRADER_DIR
from bbstrader.metatrader.account import Account, check_mt5_connection
from bbstrader.metatrader.trade import Trade
from bbstrader.metatrader.utils import TradeOrder, TradePosition, trade_retcode_message

try:
    import MetaTrader5 as Mt5
except ImportError:
    import bbstrader.compat  # noqa: F401


__all__ = ["TradeCopier", "RunCopier", "RunMultipleCopier", "config_copier"]


log.add(
    f"{BBSTRADER_DIR}/logs/copier.log",
    enqueue=True,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
)
global logger
logger = log


def fix_lot(fixed):
    if fixed == 0 or fixed is None:
        raise ValueError("Fixed lot must be a number")
    return fixed


def multiply_lot(lot, multiplier):
    if multiplier == 0 or multiplier is None:
        raise ValueError("Multiplier lot must be a number")
    return lot * multiplier


def percentage_lot(lot, percentage):
    if percentage == 0 or percentage is None:
        raise ValueError("Percentage lot must be a number")
    return round(lot * percentage / 100, 2)


def dynamic_lot(source_lot, source_eqty: float, dest_eqty: float):
    if source_eqty == 0 or dest_eqty == 0:
        raise ValueError("Source or destination account equity is zero")
    try:
        ratio = dest_eqty / source_eqty
        return round(source_lot * ratio, 2)
    except ZeroDivisionError:
        raise ValueError("Source or destination account equity is zero")


def fixed_lot(lot, symbol, destination) -> float:
    def _volume_step(value):
        value_str = str(value)
        if "." in value_str and value_str != "1.0":
            decimal_index = value_str.index(".")
            num_digits = len(value_str) - decimal_index - 1
            return num_digits
        elif value_str == "1.0":
            return 0
        else:
            return 0

    def _check_lot(lot: float, symbol_info) -> float:
        if lot > symbol_info.volume_max:
            return symbol_info.volume_max / 2
        elif lot < symbol_info.volume_min:
            return symbol_info.volume_min
        return lot

    s_info = Account(**destination).get_symbol_info(symbol)
    volume_step = s_info.volume_step
    steps = _volume_step(volume_step)
    if float(steps) >= float(1):
        return _check_lot(round(lot, steps), s_info)
    else:
        return _check_lot(round(lot), s_info)


Mode = Literal["fix", "multiply", "percentage", "dynamic", "replicate"]


def calculate_copy_lot(
    source_lot,
    symbol: str,
    destination: dict,
    mode: Mode = "dynamic",
    value=None,
    source_eqty: float = None,
    dest_eqty: float = None,
):
    if mode == "replicate":
        return fixed_lot(source_lot, symbol, destination)
    elif mode == "fix":
        return fixed_lot(fix_lot(value), symbol, destination)
    elif mode == "multiply":
        lot = multiply_lot(source_lot, value)
        return fixed_lot(lot, symbol, destination)
    elif mode == "percentage":
        lot = percentage_lot(source_lot, value)
        return fixed_lot(lot, symbol, destination)
    elif mode == "dynamic":
        lot = dynamic_lot(source_lot, source_eqty, dest_eqty)
        return fixed_lot(lot, symbol, destination)
    else:
        raise ValueError("Invalid mode selected")


def get_copy_symbols(destination: dict, source: dict):
    symbols = destination.get("symbols", "all")
    src_account = Account(**source)
    dest_account = Account(**destination)
    if symbols == "all" or symbols == "*":
        src_symbols = src_account.get_symbols()
        dest_symbols = dest_account.get_symbols()
        for s in src_symbols:
            if s not in dest_symbols:
                err_msg = (
                    f"To use 'all' or '*', Source account@{src_account.number} "
                    f"and destination account@{dest_account.number} "
                    f"must be the same type and have the same symbols"
                )
                raise ValueError(err_msg)
    elif isinstance(symbols, (list, dict)):
        return symbols
    elif isinstance(symbols, str):
        if "," in symbols:
            return symbols.split(",")
        if " " in symbols:
            return symbols.split()


class TradeCopier(object):
    """
    ``TradeCopier`` responsible for copying trading orders and positions from a source account to multiple destination accounts.

    This class facilitates the synchronization of trades between a source account and multiple destination accounts.
    It handles copying new orders, modifying existing orders, updating and closing positions based on updates from the source account.

    """

    __slots__ = (
        "source",
        "destinations",
        "errors",
        "sleeptime",
        "start_time",
        "end_time",
        "shutdown_event",
        "custom_logger",
    )
    shutdown_event: threading.Event

    def __init__(
        self,
        source: Dict,
        destinations: List[dict],
        sleeptime: float = 0.1,
        start_time: str = None,
        end_time: str = None,
        shutdown_event=None,
        custom_logger=None,
    ):
        """
        Initializes the ``TradeCopier`` instance, setting up the source and destination trading accounts for trade copying.

        Args:
            source (dict):
                A dictionary containing the connection details for the source trading account. This dictionary
                **must** include all parameters required to successfully connect to the source account.
                Refer to the ``bbstrader.metatrader.check_mt5_connection`` function for a comprehensive list
                of required keys and their expected values.  Common parameters include, but are not limited to

                    - `login`:  The account login ID (integer).
                    - `password`: The account password (string).
                    - `server`:  The server address (string), e.g., "Broker-Demo".
                    - `path`:  The path to the MetaTrader 5 installation directory (string).
                    - ``portable``:  A boolean indicating whether to open MetaTrader 5 installation in portable mode.

            destinations (List[dict]):
                A list of dictionaries, where each dictionary represents a destination trading account to which
                trades will be copied.  Each destination dictionary **must** contain the following keys

                    - Authentication details (e.g., `login`, `password`, `server`)
                    Identical in structure and requirements to the `source` dictionary,
                    ensuring a connection can be established to the destination account.
                    Refer to ``bbstrader.metatrader.check_mt5_connection``.

                    - `symbols` (Union[List[str], Dict[str, str], str])
                    Specifies which symbols should be copied from the source
                    account to this destination account.  Possible values include
                    `List[str]` A list of strings, where each string is a symbol to be copied.
                        The same symbol will be traded on the destination account.  Example `["EURUSD", "GBPUSD"]`
                    `Dict[str, str]` A dictionary mapping source symbols to destination symbols.
                        This allows for trading a different symbol on the destination account than the one traded on the source.
                        Example `{"EURUSD": "EURUSD_i", "GBPUSD": "GBPUSD_i"}`.
                    `"all"` or `"*"`  Indicates that all symbols traded on the source account should be
                        copied to this destination account, using the same symbol name.

                    - `mode` (str) The risk management mode to use.  Valid options are
                    `"fix"` Use a fixed lot size.  The `value` key must specify the fixed lot size.
                    `"multiply"` Multiply the source account's lot size by a factor.
                        The `value` key must specify the multiplier.
                    `"percentage"`  Trade a percentage of the source account's lot size.
                        The `value` key must specify the percentage (as a decimal, e.g., 50 for 50%).
                    `"dynamic"` Calculate the lot size dynamically based on account equity and risk parameters.
                        The `value` key is ignored.
                    `"replicate"` Copy the exact lot size from the source account. The `value` key is ignored.

                    - `value` (float, optional)  A numerical value used in conjunction with the selected `mode`.
                        Its meaning depends on the chosen `mode` (see above). Required for "fix", "multiply",
                        and "percentage" modes; optional for "dynamic".

                    - `slippage` (float, optional) The maximum allowed slippage in percentage when opening trades on the destination account,
                    defaults to 0.1% (0.1), if the slippage exceeds this value, the trade will not be copied.

                    - `comment` (str, optional) An optional comment to be added to trades opened on the destination account,
                    defaults to an empty string.

                    - ``copy_what`` (str, optional)
                    Specifies what to copy from the source account to the destination accounts.  Valid options are
                    `"orders"` Copy only orders from the source account to the destination accounts.
                    `"positions"` Copy only positions from the source account to the destination accounts.
                    `"all"` Copy both orders and positions from the source account to the destination accounts.
                    Defaults to `"all"`.

            sleeptime (float, optional):
                The time interval in seconds between each iteration of the trade copying process.
                Defaults to 0.1 seconds. It can be useful if you know the frequency of new trades on the source account.
        Note:
            The source account and the destination accounts must be connected to different MetaTrader 5 platforms.
            you can copy the initial installation of MetaTrader 5 to a different directory and rename it to create a new instance
            Then you can connect destination accounts to the new instance while the source account is connected to the original instance.
        """
        self.source = source
        self.destinations = destinations
        self.sleeptime = sleeptime
        self.start_time = start_time
        self.end_time = end_time
        self.shutdown_event = shutdown_event
        self._add_logger(custom_logger)
        self._add_copy()
        self.errors = set()

    def _add_logger(self, custom_logger):
        if custom_logger:
            global logger
            logger = custom_logger

    def _add_copy(self):
        self.source["copy"] = True
        for destination in self.destinations:
            destination["copy"] = True

    def source_orders(self, symbol=None):
        check_mt5_connection(**self.source)
        return Account(**self.source).get_orders(symbol=symbol)

    def source_positions(self, symbol=None):
        check_mt5_connection(**self.source)
        return Account(**self.source).get_positions(symbol=symbol)

    def destination_orders(self, destination: dict, symbol=None):
        check_mt5_connection(**destination)
        return Account(**destination).get_orders(symbol=symbol)

    def destination_positions(self, destination: dict, symbol=None):
        check_mt5_connection(**destination)
        return Account(**destination).get_positions(symbol=symbol)

    def get_copy_symbol(self, symbol, destination: dict = None, type="destination"):
        symbols = get_copy_symbols(destination, self.source)
        if isinstance(symbols, list):
            if symbol in symbols:
                return symbol
        if isinstance(symbols, dict):
            if type == "destination":
                if symbol in symbols.keys():
                    return symbols[symbol]
            if type == "source":
                for k, v in symbols.items():
                    if v == symbol:
                        return k
        raise ValueError(f"Symbol {symbol} not found in {type} account")

    def isorder_modified(self, source: TradeOrder, dest: TradeOrder):
        if source.type == dest.type and source.ticket == dest.magic:
            return (
                source.sl != dest.sl
                or source.tp != dest.tp
                or source.price_open != dest.price_open
                or source.price_stoplimit != dest.price_stoplimit
            )
        return False

    def isposition_modified(self, source: TradePosition, dest: TradePosition):
        if source.type == dest.type and source.ticket == dest.magic:
            return source.sl != dest.sl or source.tp != dest.tp
        return False

    def slippage(self, source: TradeOrder | TradePosition, destination: dict) -> bool:
        slippage = destination.get("slippage", 0.1)
        if slippage is None:
            return False
        if hasattr(source, "profit"):
            if source.type in [0, 1] and source.profit < 0:
                return False
        delta = ((source.price_current - source.price_open) / source.price_open) * 100
        if source.type in [0, 3, 4, 6] and delta > slippage:
            return True
        if source.type in [1, 2, 5, 7] and delta < -slippage:
            return True
        return False

    def iscopy_time(self):
        if self.start_time is None or self.end_time is None:
            return True
        else:
            start_time = datetime.strptime(self.start_time, "%H:%M").time()
            end_time = datetime.strptime(self.end_time, "%H:%M").time()
            if start_time <= datetime.now().time() <= end_time:
                return True
            return False

    def copy_new_trade(
        self, trade: TradeOrder | TradePosition, action_type: dict, destination: dict
    ):
        if not self.iscopy_time():
            return
        check_mt5_connection(**destination)
        volume = trade.volume if hasattr(trade, "volume") else trade.volume_initial
        symbol = self.get_copy_symbol(trade.symbol, destination)
        lot = calculate_copy_lot(
            volume,
            symbol,
            destination,
            mode=destination.get("mode", "fix"),
            value=destination.get("value", 0.01),
            source_eqty=Account(**self.source).get_account_info().margin_free,
            dest_eqty=Account(**destination).get_account_info().margin_free,
        )

        trade_instance = Trade(
            symbol=symbol, **destination, max_risk=100.0, logger=None
        )
        try:
            action = action_type[trade.type]
        except KeyError:
            return
        try:
            if trade_instance.open_position(
                action,
                volume=lot,
                sl=trade.sl,
                tp=trade.tp,
                id=trade.ticket,
                symbol=symbol,
                mm=trade.sl != 0 and trade.tp != 0,
                price=trade.price_open if trade.type not in [0, 1] else None,
                stoplimit=trade.price_stoplimit if trade.type in [6, 7] else None,
                comment=destination.get("comment", trade.comment + "#Copied"),
            ):
                logger.info(
                    f"Copy {action} Order #{trade.ticket} from @{self.source.get('login')}::{trade.symbol} "
                    f"to @{destination.get('login')}::{symbol}"
                )
            else:
                logger.error(
                    f"Error copying {action} Order #{trade.ticket} from @{self.source.get('login')}::{trade.symbol} "
                    f"to @{destination.get('login')}::{symbol}"
                )
        except Exception as e:
            self.log_error(e, symbol=symbol)

    def copy_new_order(self, order: TradeOrder, destination: dict):
        action_type = {
            2: "BLMT",
            3: "SLMT",
            4: "BSTP",
            5: "SSTP",
            6: "BSTPLMT",
            7: "SSTPLMT",
        }
        self.copy_new_trade(order, action_type, destination)

    def modify_order(self, ticket, symbol, source_order: TradeOrder, destination: dict):
        check_mt5_connection(**destination)
        account = Account(**destination)
        request = {
            "action": Mt5.TRADE_ACTION_MODIFY,
            "order": ticket,
            "symbol": symbol,
            "price": source_order.price_open,
            "sl": source_order.sl,
            "tp": source_order.tp,
            "stoplimit": source_order.price_stoplimit,
        }
        result = account.send_order(request)
        if result.retcode != Mt5.TRADE_RETCODE_DONE:
            msg = trade_retcode_message(result.retcode)
            logger.error(
                f"Error modifying Order #{ticket} on @{destination.get('login')}::{symbol}, {msg}, "
                f"SOURCE=@{self.source.get('login')}::{source_order.symbol}"
            )
        if result.retcode == Mt5.TRADE_RETCODE_DONE:
            logger.info(
                f"Modify Order #{ticket} on @{destination.get('login')}::{symbol}, "
                f"SOURCE=@{self.source.get('login')}::{source_order.symbol}"
            )

    def remove_order(self, src_symbol, order: TradeOrder, destination: dict):
        check_mt5_connection(**destination)
        trade = Trade(symbol=order.symbol, **destination, logger=None)
        if trade.close_order(order.ticket, id=order.magic):
            logger.info(
                f"Close Order #{order.ticket} on @{destination.get('login')}::{order.symbol}, "
                f"SOURCE=@{self.source.get('login')}::{src_symbol}"
            )
        else:
            logger.error(
                f"Error closing Order #{order.ticket} on @{destination.get('login')}::{order.symbol}, "
                f"SOURCE=@{self.source.get('login')}::{src_symbol}"
            )

    def copy_new_position(self, position: TradePosition, destination: dict):
        action_type = {0: "BMKT", 1: "SMKT"}
        self.copy_new_trade(position, action_type, destination)

    def modify_position(
        self, ticket, symbol, source_pos: TradePosition, destination: dict
    ):
        check_mt5_connection(**destination)
        account = Account(**destination)
        request = {
            "action": Mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": source_pos.sl,
            "tp": source_pos.tp,
        }
        result = account.send_order(request)
        if result.retcode != Mt5.TRADE_RETCODE_DONE:
            msg = trade_retcode_message(result.retcode)
            logger.error(
                f"Error modifying Position #{ticket} on @{destination.get('login')}::{symbol},  {msg}, "
                f"SOURCE=@{self.source.get('login')}::{source_pos.symbol}"
            )
        if result.retcode == Mt5.TRADE_RETCODE_DONE:
            logger.info(
                f"Modify Position #{ticket} on @{destination.get('login')}::{symbol}, "
                f"SOURCE=@{self.source.get('login')}::{source_pos.symbol}"
            )

    def remove_position(self, src_symbol, position: TradePosition, destination: dict):
        check_mt5_connection(**destination)
        trade = Trade(symbol=position.symbol, **destination, logger=None)
        if trade.close_position(position.ticket, id=position.magic):
            logger.info(
                f"Close Position #{position.ticket} on @{destination.get('login')}::{position.symbol}, "
                f"SOURCE=@{self.source.get('login')}::{src_symbol}"
            )
        else:
            logger.error(
                f"Error closing Position #{position.ticket} on @{destination.get('login')}::{position.symbol}, "
                f"SOURCE=@{self.source.get('login')}::{src_symbol}"
            )

    def filter_positions_and_orders(self, pos_or_orders, symbols=None):
        if symbols is None:
            return pos_or_orders
        elif isinstance(symbols, list):
            return [pos for pos in pos_or_orders if pos.symbol in symbols]
        elif isinstance(symbols, dict):
            return [
                pos
                for pos in pos_or_orders
                if pos.symbol in symbols.keys() or pos.symbol in symbols.values()
            ]

    def get_positions(self, destination: dict):
        source_positions = self.source_positions() or []
        dest_symbols = get_copy_symbols(destination, self.source)
        dest_positions = self.destination_positions(destination) or []
        source_positions = self.filter_positions_and_orders(
            source_positions, symbols=dest_symbols
        )
        dest_positions = self.filter_positions_and_orders(
            dest_positions, symbols=dest_symbols
        )
        return source_positions, dest_positions

    def get_orders(self, destination: dict):
        source_orders = self.source_orders() or []
        dest_symbols = get_copy_symbols(destination, self.source)
        dest_orders = self.destination_orders(destination) or []
        source_orders = self.filter_positions_and_orders(
            source_orders, symbols=dest_symbols
        )
        dest_orders = self.filter_positions_and_orders(
            dest_orders, symbols=dest_symbols
        )
        return source_orders, dest_orders

    def _copy_new_orders(self, destination):
        source_orders, destination_orders = self.get_orders(destination)
        # Check for new orders
        dest_ids = [order.magic for order in destination_orders]
        for source_order in source_orders:
            if source_order.ticket not in dest_ids:
                if not self.slippage(source_order, destination):
                    self.copy_new_order(source_order, destination)

    def _copy_modified_orders(self, destination):
        # Check for modified orders
        source_orders, destination_orders = self.get_orders(destination)
        for source_order in source_orders:
            for destination_order in destination_orders:
                if source_order.ticket == destination_order.magic:
                    if self.isorder_modified(source_order, destination_order):
                        ticket = destination_order.ticket
                        symbol = destination_order.symbol
                        self.modify_order(ticket, symbol, source_order, destination)

    def _copy_closed_orders(self, destination):
        # Check for closed orders
        source_orders, destination_orders = self.get_orders(destination)
        source_ids = [order.ticket for order in source_orders]
        for destination_order in destination_orders:
            if destination_order.magic not in source_ids:
                src_symbol = self.get_copy_symbol(
                    destination_order.symbol, destination, type="source"
                )
                self.remove_order(src_symbol, destination_order, destination)

    def _sync_positions(self, what, destination):
        # Update postions
        source_positions, _ = self.get_positions(destination)
        _, destination_orders = self.get_orders(destination)
        for source_position in source_positions:
            for destination_order in destination_orders:
                if source_position.ticket == destination_order.magic:
                    self.remove_order(
                        source_position.symbol, destination_order, destination
                    )
                    if what in ["all", "positions"]:
                        if not self.slippage(source_position, destination):
                            self.copy_new_position(source_position, destination)

    def _sync_orders(self, destination):
        # Update orders
        _, destination_positions = self.get_positions(destination)
        source_orders, _ = self.get_orders(destination)
        for destination_position in destination_positions:
            for source_order in source_orders:
                if destination_position.magic == source_order.ticket:
                    self.remove_position(
                        source_order.symbol, destination_position, destination
                    )
                    if not self.slippage(source_order, destination):
                        self.copy_new_order(source_order, destination)

    def _copy_what(self, destination):
        if not destination.get("copy", False):
            raise ValueError("Destination account not set to copy mode")
        return destination.get("copy_what", "all")

    def copy_orders(self, destination: dict):
        what = self._copy_what(destination)
        if what not in ["all", "orders"]:
            return
        check_mt5_connection(**destination)
        self._copy_new_orders(destination)
        self._copy_modified_orders(destination)
        self._copy_closed_orders(destination)
        self._sync_positions(what, destination)
        self._sync_orders(destination)

    def _copy_new_positions(self, destination):
        source_positions, destination_positions = self.get_positions(destination)
        # Check for new positions
        dest_ids = [pos.magic for pos in destination_positions]
        for source_position in source_positions:
            if source_position.ticket not in dest_ids:
                if not self.slippage(source_position, destination):
                    self.copy_new_position(source_position, destination)

    def _copy_modified_positions(self, destination):
        # Check for modified positions
        source_positions, destination_positions = self.get_positions(destination)
        for source_position in source_positions:
            for destination_position in destination_positions:
                if source_position.ticket == destination_position.magic:
                    if self.isposition_modified(source_position, destination_position):
                        ticket = destination_position.ticket
                        symbol = destination_position.symbol
                        self.modify_position(
                            ticket, symbol, source_position, destination
                        )

    def _copy_closed_position(self, destination):
        # Check for closed positions
        source_positions, destination_positions = self.get_positions(destination)
        source_ids = [pos.ticket for pos in source_positions]
        for destination_position in destination_positions:
            if destination_position.magic not in source_ids:
                src_symbol = self.get_copy_symbol(
                    destination_position.symbol, destination, type="source"
                )
                self.remove_position(src_symbol, destination_position, destination)

    def copy_positions(self, destination: dict):
        what = self._copy_what(destination)
        if what not in ["all", "positions"]:
            return
        check_mt5_connection(**destination)
        self._copy_new_positions(destination)
        self._copy_modified_positions(destination)
        self._copy_closed_position(destination)

    def log_error(self, e, symbol=None):
        error_msg = repr(e)
        if error_msg not in self.errors:
            self.errors.add(error_msg)
            add_msg = f"SYMBOL={symbol}" if symbol else ""
            logger.error(f"Error encountered: {error_msg}, {add_msg}")

    def run(self):
        logger.info("Trade Copier Running ...")
        logger.info(f"Source Account: {self.source.get('login')}")
        while True:
            if self.shutdown_event and self.shutdown_event.is_set():
                logger.info(
                    "Shutdown event received, stopping Trade Copier gracefully."
                )
                break
            try:
                for destination in self.destinations:
                    if self.shutdown_event and self.shutdown_event.is_set():
                        break
                    if destination.get("path") == self.source.get("path"):
                        err_msg = "Source and destination accounts are on the same \
                            MetaTrader 5 installation which is not allowed."
                        logger.error(err_msg)
                        continue
                    self.copy_orders(destination)
                    self.copy_positions(destination)
                    Mt5.shutdown()
                    time.sleep(0.1)

                if self.shutdown_event and self.shutdown_event.is_set():
                    logger.info(
                        "Shutdown event received during destination processing, exiting."
                    )
                    break

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received, stopping the Trade Copier ...")
                if self.shutdown_event:
                    self.shutdown_event.set()
                break
            except Exception as e:
                self.log_error(e)
                if self.shutdown_event and self.shutdown_event.is_set():
                    logger.error(
                        "Error occurred after shutdown signaled, exiting loop."
                    )
                    break

            # Check shutdown event before sleeping
            if self.shutdown_event and self.shutdown_event.is_set():
                logger.info("Shutdown event checked before sleep, exiting.")
                break
            time.sleep(self.sleeptime)
        logger.info("Trade Copier has shut down.")


def RunCopier(
    source: dict,
    destinations: list,
    sleeptime: float,
    start_time: str,
    end_time: str,
    shutdown_event=None,
    custom_logger=None,
):
    copier = TradeCopier(
        source,
        destinations,
        sleeptime,
        start_time,
        end_time,
        shutdown_event,
        custom_logger,
    )
    copier.run()


def RunMultipleCopier(
    accounts: List[dict],
    sleeptime: float = 0.1,
    start_delay: float = 1.0,
    start_time: str = None,
    end_time: str = None,
    shutdown_event=None,
    custom_logger=None,
):
    processes = []

    for account in accounts:
        source = account.get("source")
        destinations = account.get("destinations")

        if not source or not destinations:
            logger.warning("Skipping account due to missing source or destinations.")
            continue
        paths = set([source.get("path")] + [dest.get("path") for dest in destinations])
        if len(paths) == 1:
            logger.warning(
                "Skipping account due to same MetaTrader 5 installation path."
            )
            continue
        logger.info(f"Starting process for source account @{source.get('login')}")
        process = multiprocessing.Process(
            target=RunCopier,
            args=(
                source,
                destinations,
                sleeptime,
                start_time,
                end_time,
            ),
            kwargs=dict(shutdown_event=shutdown_event, custom_logger=custom_logger),
        )
        processes.append(process)
        process.start()

        if start_delay:
            time.sleep(start_delay)

    for process in processes:
        process.join()


def _strtodict(string: str) -> dict:
    string = string.strip().replace("\n", "").replace(" ", "").replace('"""', "")
    if string.endswith(","):
        string = string[:-1]
    return dict(item.split(":") for item in string.split(","))


def _parse_symbols(section):
    symbols: str = section.get("symbols")
    symbols = symbols.strip().replace("\n", " ").replace('"""', "")
    if symbols in ["all", "*"]:
        section["symbols"] = symbols
    elif ":" in symbols:
        symbols = _strtodict(symbols)
        section["symbols"] = symbols
    elif " " in symbols and "," not in symbols:
        symbols = symbols.split()
        section["symbols"] = symbols
    elif "," in symbols:
        symbols = symbols.replace(" ", "").split(",")
        section["symbols"] = symbols
    else:
        raise ValueError("""
        Invalid symbols format.
        You can use space or comma separated symbols in one line or multiple lines using triple quotes.
        You can also use a dictionary to map source symbols to destination symbols as shown below.
        Or if you want to copy all symbols, use "all" or "*".

        symbols = EURUSD, GBPUSD, USDJPY (space separated)
        symbols = EURUSD,GBPUSD,USDJPY (comma separated)
        symbols = EURUSD.s:EURUSD_i, GBPUSD.s:GBPUSD_i, USDJPY.s:USDJPY_i (dictionary) 
        symbols = all (copy all symbols)
        symbols = * (copy all symbols) """)


def config_copier(
    source_section: str = None,
    dest_sections: str | List[str] = None,
    inifile: str | Path = None,
) -> Tuple[dict, List[dict]]:
    """
    Read the configuration file and return the source and destination account details.

    Args:
        inifile (str | Path): The path to the INI configuration file.
        source_section (str): The section name of the source account, defaults to "SOURCE".
        dest_sections (str | List[str]): The section name(s) of the destination account(s).

    Returns:
        Tuple[dict, List[dict]]: A tuple containing the source account and a list of destination accounts.

    Example:
        ```python
        from pathlib import Path
        config_file = ~/.bbstrader/copier/copier.ini
        source, destinations = config_copier(config_file, "SOURCE", ["DEST1", "DEST2"])
        ```
    """
    from bbstrader.core.utils import dict_from_ini

    if not inifile:
        inifile = Path().home() / ".bbstrader" / "copier" / "copier.ini"
        if not inifile.exists() or not inifile.is_file():
            raise FileNotFoundError(f"{inifile} not found")

    if not source_section:
        source_section = "SOURCE"

    config = dict_from_ini(inifile)
    try:
        source = config.pop(source_section)
    except KeyError:
        raise ValueError(f"Source section {source_section} not found in {inifile}")
    dest_sections = dest_sections or config.keys()
    if not dest_sections:
        raise ValueError("No destination sections found in the configuration file")

    destinations = []

    if isinstance(dest_sections, str):
        dest_sections = [dest_sections]

    for dest_section in dest_sections:
        try:
            section = config[dest_section]
        except KeyError:
            raise ValueError(
                f"Destination section {dest_section} not found in {inifile}"
            )
        _parse_symbols(section)
        destinations.append(section)

    return source, destinations
