import multiprocessing as mp
import time
from datetime import datetime
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Dict, List, Literal, Tuple

from loguru import logger as log

from bbstrader.config import BBSTRADER_DIR
from bbstrader.metatrader.account import Account, check_mt5_connection
from bbstrader.metatrader.trade import FILLING_TYPE
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


ORDER_TYPE = {
    0: (Mt5.ORDER_TYPE_BUY, "BUY"),
    1: (Mt5.ORDER_TYPE_SELL, "SELL"),
    2: (Mt5.ORDER_TYPE_BUY_LIMIT, "BUY LIMIT"),
    3: (Mt5.ORDER_TYPE_SELL_LIMIT, "SELL LIMIT"),
    4: (Mt5.ORDER_TYPE_BUY_STOP, "BUY STOP"),
    5: (Mt5.ORDER_TYPE_SELL_STOP, "SELL STOP"),
    6: (Mt5.ORDER_TYPE_BUY_STOP_LIMIT, "BUY STOP LIMIT"),
    7: (Mt5.ORDER_TYPE_SELL_STOP_LIMIT, "SELL STOP LIMIT"),
}


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


def get_symbols_from_string(symbols_string: str):
    if not symbols_string:
        raise ValueError("Input Error", "Tickers string cannot be empty.")
    string = (
        symbols_string.strip().replace("\n", "").replace(" ", "").replace('"""', "")
    )
    if ":" in string and "," in string:
        if string.endswith(","):
            string = string[:-1]
        return dict(item.split(":") for item in string.split(","))
    elif ":" in string and "," not in string:
        raise ValueError("Each key pairs value must be separeted by ','")
    elif "," in string and ":" not in string:
        return string.split(",")
    else:
        raise ValueError("""
        Invalid symbols format.
        You can use comma separated symbols in one line or multiple lines using triple quotes.
        You can also use a dictionary to map source symbols to destination symbols as shown below.
        Or if you want to copy all symbols, use "all" or "*".

        symbols = EURUSD,GBPUSD,USDJPY (comma separated)
        symbols = EURUSD.s:EURUSD_i, GBPUSD.s:GBPUSD_i, USDJPY.s:USDJPY_i (dictionary) 
        symbols = all (copy all symbols)
        symbols = * (copy all symbols) """)


def get_copy_symbols(destination: dict, source: dict) -> List[str] | Dict[str, str]:
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
                    f"If not Use a dictionary to map source symbols to destination symbols "
                    f"(e.g., EURUSD.s:EURUSD_i, GBPUSD.s:GBPUSD_i, USDJPY.s:USDJPY_i"
                    f"Where EURUSD.s is the source symbols and EURUSD_i is the corresponding symbol"
                )
                raise ValueError(err_msg)
        return dest_symbols
    elif isinstance(symbols, (list, dict)):
        return symbols
    elif isinstance(symbols, str):
        return get_symbols_from_string(symbols)
    else:
        raise ValueError("Invalide symbols provided")


class TradeCopier(object):
    """
    ``TradeCopier`` responsible for copying trading orders and positions from a source account to multiple destination accounts.

    This class facilitates the synchronization of trades between a source account and multiple destination accounts.
    It handles copying new orders, modifying existing orders, updating and closing positions based on updates from the source account.

    """

    __slots__ = (
        "source",
        "source_id",
        "source_isunique",
        "destinations",
        "errors",
        "sleeptime",
        "start_time",
        "end_time",
        "shutdown_event",
        "custom_logger",
        "log_queue",
        "_last_session",
        "_running",
    )

    source: Dict
    source_id: int
    source_isunique: bool
    destinations: List[dict]
    shutdown_event: Event
    log_queue: mp.Queue

    def __init__(
        self,
        source: Dict,
        destinations: List[dict],
        /,
        sleeptime: float = 0.1,
        start_time: str = None,
        end_time: str = None,
        *,
        custom_logger=None,
        shutdown_event=None,
        log_queue=None,
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
                    - `portable`:  A boolean indicating whether to open MetaTrader 5 installation in portable mode.
                    - `id`: A unique identifier for all trades opened buy the source source account.
                        This Must be a positive number greater than 0 and less than 2^32 / 2.
                    - `unique`: A boolean indication whehter to allow destination accounts to copy from other sources.
                        If Set to True, all destination accounts won't be allow to accept trades from other accounts even
                        manually opened positions or orders will be removed.

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

            start_time (str, optional): The time (HH:MM) from which the copier start copying from the source.
            end_time (str, optional): The time (HH:MM) from which the copier stop copying from the source.
            sleeptime (float, optional): The delay between each check from the source account.
            custom_logger (Any, Optional): Used to set a cutum logger (default is ``loguru.logger``)
            shutdown_event (Any, Otional): Use to terminal the copy process when runs in a custum environment like web App or GUI.
            log_queue (multiprocessing.Queue, Optional): Use to send log to an external program, usefule in GUI apps

        Note:
            The source account and the destination accounts must be connected to different MetaTrader 5 platforms.
            you can copy the initial installation of MetaTrader 5 to a different directory and rename it to create a new instance
            Then you can connect destination accounts to the new instance while the source account is connected to the original instance.
        """
        self.source = source
        self.source_id = source.get("id", 0)
        self.source_isunique = source.get("unique", True)
        self.destinations = destinations
        self.sleeptime = sleeptime
        self.start_time = start_time
        self.end_time = end_time
        self.errors = set()
        self.log_queue = log_queue
        self._add_logger(custom_logger)
        self._validate_source()
        self._add_copy()
        self.shutdown_event = (
            shutdown_event if shutdown_event is not None else mp.Event()
        )
        self._last_session = datetime.now().date()
        self._running = True

    @property
    def running(self):
        """Check if the Trade Copier is running."""
        return self._running

    def _add_logger(self, custom_logger):
        if custom_logger:
            global logger
            logger = custom_logger

    def _add_copy(self):
        self.source["copy"] = self.source.get("copy", True)
        for destination in self.destinations:
            destination["copy"] = destination.get("copy", True)

    def _log_message(self, message, type="info"):
        if self.log_queue:
            try:
                now = datetime.now()
                formatted = (
                    now.strftime("%Y-%m-%d %H:%M:%S.")
                    + f"{int(now.microsecond / 1000):03d}"
                )
                space = len("warning")
                self.log_queue.put(
                    f"{formatted} |{type.upper()} {' '*(space - len(type))}|  - {message}"
                )
            except Exception:
                pass
        else:
            logmethod = logger.info if type == "info" else logger.error
            logmethod(message)

    def log_error(self, e, symbol=None):
        if datetime.now().date() > self._last_session:
            self._last_session = datetime.now().date()
            self.errors.clear()
        error_msg = repr(e)
        if error_msg not in self.errors:
            self.errors.add(error_msg)
            add_msg = f"SYMBOL={symbol}" if symbol else ""
            message = f"Error encountered: {error_msg}, {add_msg}"
            self._log_message(message, type="error")

    def _validate_source(self):
        if not self.source_isunique:
            try:
                assert self.source_id >= 1
            except AssertionError:
                raise ValueError(
                    "Non Unique source account must have a valide ID , (e.g., source['id'] = 1234)"
                )

    def add_destinations(self, destination_accounts: List[dict]):
        self.stop()
        destinations = destination_accounts.copy()
        for destination in destinations:
            destination["copy"] = True
            self.destinations.append(destination)
        self.run()

    def _get_magic(self, ticket: int) -> int:
        return int(str(self.source_id) + str(ticket)) if self.source_id >= 1 else ticket

    def _select_symbol(self, symbol: str, destination: dict):
        selected = Mt5.symbol_select(symbol, True)
        if not selected:
            self._log_message(
                f"Failed to select {destination.get('login')}::{symbol}, error code =",
                Mt5.last_error(),
                type="error",
            )

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
        if source.type == dest.type and self._get_magic(source.ticket) == dest.magic:
            return (
                source.sl != dest.sl
                or source.tp != dest.tp
                or source.price_open != dest.price_open
                or source.price_stoplimit != dest.price_stoplimit
            )
        return False

    def isposition_modified(self, source: TradePosition, dest: TradePosition):
        if source.type == dest.type and self._get_magic(source.ticket) == dest.magic:
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

    def _update_filling_type(self, request, result):
        new_result = result
        if result.retcode == Mt5.TRADE_RETCODE_INVALID_FILL:
            for fill in FILLING_TYPE:
                request["type_filling"] = fill
                new_result = Mt5.order_send(request)
                if new_result.retcode == Mt5.TRADE_RETCODE_DONE:
                    break
        return new_result

    def copy_new_trade(self, trade: TradeOrder | TradePosition, destination: dict):
        if not self.iscopy_time():
            return
        check_mt5_connection(**destination)
        symbol = self.get_copy_symbol(trade.symbol, destination)
        self._select_symbol(symbol, destination)

        volume = trade.volume if hasattr(trade, "volume") else trade.volume_initial
        lot = calculate_copy_lot(
            volume,
            symbol,
            destination,
            mode=destination.get("mode", "fix"),
            value=destination.get("value", 0.01),
            source_eqty=Account(**self.source).get_account_info().margin_free,
            dest_eqty=Account(**destination).get_account_info().margin_free,
        )
        trade_action = (
            Mt5.TRADE_ACTION_DEAL if trade.type in [0, 1] else Mt5.TRADE_ACTION_PENDING
        )
        action = ORDER_TYPE[trade.type][1]
        tick = Mt5.symbol_info_tick(symbol)
        price = tick.ask if trade.type == 0 else tick.bid
        try:
            request = dict(
                symbol=symbol,
                action=trade_action,
                volume=lot,
                price=price,
                sl=trade.sl,
                tp=trade.tp,
                type=ORDER_TYPE[trade.type][0],
                magic=self._get_magic(trade.ticket),
                deviation=Mt5.symbol_info(symbol).spread,
                comment=destination.get("comment", trade.comment + "#Copied"),
                type_time=Mt5.ORDER_TIME_GTC,
                type_filling=Mt5.ORDER_FILLING_FOK,
            )
            if trade.type not in [0, 1]:
                request["price"] = trade.price_open

            if trade.type in [6, 7]:
                request["stoplimit"] = trade.price_stoplimit

            result = Mt5.order_send(request)
            if result.retcode != Mt5.TRADE_RETCODE_DONE:
                result = self._update_filling_type(request, result)
            if result.retcode == Mt5.TRADE_RETCODE_DONE:
                self._log_message(
                    f"Copy {action} Order #{trade.ticket} from @{self.source.get('login')}::{trade.symbol} "
                    f"to @{destination.get('login')}::{symbol}",
                )
            if result.retcode != Mt5.TRADE_RETCODE_DONE:
                self._log_message(
                    f"Error copying {action} Order #{trade.ticket} from @{self.source.get('login')}::{trade.symbol} "
                    f"to @{destination.get('login')}::{symbol}, {trade_retcode_message(result.retcode)}",
                    type="error",
                )
        except Exception as e:
            self.log_error(e, symbol=symbol)

    def copy_new_order(self, order: TradeOrder, destination: dict):
        self.copy_new_trade(order, destination)

    def modify_order(self, ticket, symbol, source_order: TradeOrder, destination: dict):
        check_mt5_connection(**destination)
        self._select_symbol(symbol, destination)
        request = {
            "action": Mt5.TRADE_ACTION_MODIFY,
            "order": ticket,
            "symbol": symbol,
            "price": source_order.price_open,
            "sl": source_order.sl,
            "tp": source_order.tp,
            "stoplimit": source_order.price_stoplimit,
        }
        result = Mt5.order_send(request)
        if result.retcode != Mt5.TRADE_RETCODE_DONE:
            result = self._update_filling_type(request, result)
        if result.retcode == Mt5.TRADE_RETCODE_DONE:
            self._log_message(
                f"Modify {ORDER_TYPE[source_order.type][1]} Order #{ticket} on @{destination.get('login')}::{symbol}, "
                f"SOURCE=@{self.source.get('login')}::{source_order.symbol}"
            )
        if result.retcode != Mt5.TRADE_RETCODE_DONE:
            self._log_message(
                f"Error modifying {ORDER_TYPE[source_order.type][1]} Order #{ticket} on @{destination.get('login')}::{symbol},"
                f"SOURCE=@{self.source.get('login')}::{source_order.symbol},  {trade_retcode_message(result.retcode)}",
                type="error",
            )

    def remove_order(self, src_symbol, order: TradeOrder, destination: dict):
        check_mt5_connection(**destination)
        self._select_symbol(order.symbol, destination)
        request = {
            "action": Mt5.TRADE_ACTION_REMOVE,
            "order": order.ticket,
        }
        result = Mt5.order_send(request)
        if result.retcode != Mt5.TRADE_RETCODE_DONE:
            result = self._update_filling_type(request, result)
        if result.retcode == Mt5.TRADE_RETCODE_DONE:
            self._log_message(
                f"Close {ORDER_TYPE[order.type][1]} Order #{order.ticket}  on @{destination.get('login')}::{order.symbol}, "
                f"SOURCE=@{self.source.get('login')}::{src_symbol}"
            )
        if result.retcode != Mt5.TRADE_RETCODE_DONE:
            self._log_message(
                f"Error closing {ORDER_TYPE[order.type][1]} Order #{order.ticket} on @{destination.get('login')}::{order.symbol}, "
                f"SOURCE=@{self.source.get('login')}::{src_symbol}, {trade_retcode_message(result.retcode)}",
                type="error",
            )

    def copy_new_position(self, position: TradePosition, destination: dict):
        self.copy_new_trade(position, destination)

    def modify_position(
        self, ticket, symbol, source_pos: TradePosition, destination: dict
    ):
        check_mt5_connection(**destination)
        self._select_symbol(symbol, destination)
        request = {
            "action": Mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": source_pos.sl,
            "tp": source_pos.tp,
        }
        result = Mt5.order_send(request)
        if result.retcode != Mt5.TRADE_RETCODE_DONE:
            result = self._update_filling_type(request, result)
        if result.retcode == Mt5.TRADE_RETCODE_DONE:
            self._log_message(
                f"Modify {ORDER_TYPE[source_pos.type][1]} Position #{ticket} on @{destination.get('login')}::{symbol}, "
                f"SOURCE=@{self.source.get('login')}::{source_pos.symbol}"
            )
        if result.retcode != Mt5.TRADE_RETCODE_DONE:
            self._log_message(
                f"Error modifying {ORDER_TYPE[source_pos.type][1]} Position #{ticket} on @{destination.get('login')}::{symbol}, "
                f"SOURCE=@{self.source.get('login')}::{source_pos.symbol}, {trade_retcode_message(result.retcode)}",
                type="error",
            )

    def remove_position(self, src_symbol, position: TradePosition, destination: dict):
        check_mt5_connection(**destination)
        self._select_symbol(position.symbol, destination)
        position_type = (
            Mt5.ORDER_TYPE_SELL if position.type == 0 else Mt5.ORDER_TYPE_BUY
        )
        request = {
            "action": Mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": position_type,
            "position": position.ticket,
            "price": position.price_current,
            "deviation": int(Mt5.symbol_info(position.symbol).spread),
            "type_time": Mt5.ORDER_TIME_GTC,
            "type_filling": Mt5.ORDER_FILLING_FOK,
        }
        result = Mt5.order_send(request)
        if result.retcode != Mt5.TRADE_RETCODE_DONE:
            result = self._update_filling_type(request, result)

        if result.retcode == Mt5.TRADE_RETCODE_DONE:
            self._log_message(
                f"Close {ORDER_TYPE[position.type][1]} Position #{position.ticket} "
                f"on @{destination.get('login')}::{position.symbol}, "
                f"SOURCE=@{self.source.get('login')}::{src_symbol}"
            )
        if result.retcode != Mt5.TRADE_RETCODE_DONE:
            self._log_message(
                f"Error closing {ORDER_TYPE[position.type][1]} Position #{position.ticket} "
                f"on @{destination.get('login')}::{position.symbol}, "
                f"SOURCE=@{self.source.get('login')}::{src_symbol},  {trade_retcode_message(result.retcode)}",
                type="error",
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

    def get_positions(
        self, destination: dict
    ) -> Tuple[List[TradePosition], List[TradePosition]]:
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

    def get_orders(
        self, destination: dict
    ) -> Tuple[List[TradeOrder], List[TradeOrder]]:
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

    def _copy_what(self, destination):
        if not destination.get("copy", False):
            raise ValueError("Destination account not set to copy mode")
        return destination.get("copy_what", "all")

    def _isvalide_magic(self, magic):
        ticket = str(magic)
        id = str(self.source_id)
        return (
            ticket != id
            and ticket.startswith(id)
            and ticket[: len(id)] == id
            and int(ticket[: len(id)]) == self.source_id
        )

    def _copy_new_orders(self, destination):
        source_orders, destination_orders = self.get_orders(destination)
        # Check for new orders
        dest_ids = [order.magic for order in destination_orders]
        for source_order in source_orders:
            if self._get_magic(source_order.ticket) not in dest_ids:
                if not self.slippage(source_order, destination):
                    self.copy_new_order(source_order, destination)

    def _copy_modified_orders(self, destination):
        # Check for modified orders
        source_orders, destination_orders = self.get_orders(destination)
        for source_order in source_orders:
            for destination_order in destination_orders:
                if self._get_magic(source_order.ticket) == destination_order.magic:
                    if self.isorder_modified(source_order, destination_order):
                        ticket = destination_order.ticket
                        symbol = destination_order.symbol
                        self.modify_order(ticket, symbol, source_order, destination)

    def _copy_closed_orders(self, destination):
        # Check for closed orders
        source_orders, destination_orders = self.get_orders(destination)
        source_ids = [self._get_magic(order.ticket) for order in source_orders]
        for destination_order in destination_orders:
            if destination_order.magic not in source_ids:
                if self.source_isunique or self._isvalide_magic(
                    destination_order.magic
                ):
                    src_symbol = self.get_copy_symbol(
                        destination_order.symbol, destination, type="source"
                    )
                    self.remove_order(src_symbol, destination_order, destination)

    def _sync_orders(self, destination):
        # Update orders
        _, destination_positions = self.get_positions(destination)
        source_orders, _ = self.get_orders(destination)
        for destination_position in destination_positions:
            for source_order in source_orders:
                if destination_position.magic == self._get_magic(source_order.ticket):
                    self.remove_position(
                        source_order.symbol, destination_position, destination
                    )
                    if not self.slippage(source_order, destination):
                        self.copy_new_order(source_order, destination)

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
            if self._get_magic(source_position.ticket) not in dest_ids:
                if not self.slippage(source_position, destination):
                    self.copy_new_position(source_position, destination)

    def _copy_modified_positions(self, destination):
        # Check for modified positions
        source_positions, destination_positions = self.get_positions(destination)
        for source_position in source_positions:
            for destination_position in destination_positions:
                if (
                    self._get_magic(source_position.ticket)
                    == destination_position.magic
                ):
                    if self.isposition_modified(source_position, destination_position):
                        ticket = destination_position.ticket
                        symbol = destination_position.symbol
                        self.modify_position(
                            ticket, symbol, source_position, destination
                        )

    def _copy_closed_position(self, destination):
        # Check for closed positions
        source_positions, destination_positions = self.get_positions(destination)
        source_ids = [self._get_magic(pos.ticket) for pos in source_positions]
        for destination_position in destination_positions:
            if destination_position.magic not in source_ids:
                if self.source_isunique or self._isvalide_magic(
                    destination_position.magic
                ):
                    src_symbol = self.get_copy_symbol(
                        destination_position.symbol, destination, type="source"
                    )
                    self.remove_position(src_symbol, destination_position, destination)

    def _sync_positions(self, what, destination):
        # Update postions
        source_positions, _ = self.get_positions(destination)
        _, destination_orders = self.get_orders(destination)
        for source_position in source_positions:
            for destination_order in destination_orders:
                if self._get_magic(source_position.ticket) == destination_order.magic:
                    self.remove_order(
                        source_position.symbol, destination_order, destination
                    )
                    if what in ["all", "positions"]:
                        if not self.slippage(source_position, destination):
                            self.copy_new_position(source_position, destination)

    def copy_positions(self, destination: dict):
        what = self._copy_what(destination)
        if what not in ["all", "positions"]:
            return
        check_mt5_connection(**destination)
        self._copy_new_positions(destination)
        self._copy_modified_positions(destination)
        self._copy_closed_position(destination)

    def start_copy_process(self, destination: dict):
        """
        Worker process: copy orders and positions for a single destination account.
        """
        if destination.get("path") == self.source.get("path"):
            self._log_message(
                f"Source and destination accounts are on the same  "
                f"MetaTrader 5 installation {self.source.get('path')} which is not allowed."
            )
            return

        self._log_message(
            f"Copy started for source @{self.source.get('login')} "
            f" and destination @{destination.get('login')}"
        )
        while not self.shutdown_event.is_set():
            try:
                self.copy_positions(destination)
                self.copy_orders(destination)
            except KeyboardInterrupt:
                self._log_message(
                    "KeyboardInterrupt received, stopping the Trade Copier ..."
                )
                self.stop()
            except Exception as e:
                self.log_error(e)

        self._log_message(
            f"Process exiting for destination @{destination.get('login')} due to shutdown event."
        )

    def run(self):
        """
        Entry point to start the copier.
        This will loop through the destinations it was given and process them.
        """
        self._log_message(
            f"Copier instance started for source @{self.source.get('login')}"
        )
        try:
            while not self.shutdown_event.is_set():
                for destination in self.destinations:
                    if self.shutdown_event.is_set():
                        break

                    if destination.get("path") == self.source.get("path"):
                        self._log_message(
                            f"Source and destination accounts are on the same "
                            f"MetaTrader 5 installation {self.source.get('path')} which is not allowed."
                        )
                        continue
                    try:
                        self.copy_positions(destination)
                        self.copy_orders(destination)
                    except Exception as e:
                        self.log_error(e)
                time.sleep(self.sleeptime)

        except KeyboardInterrupt:
            self._log_message(
                "KeyboardInterrupt received, stopping the copier instance..."
            )
            self.shutdown_event.set()

        self._log_message(
            f"Copier instance for source @{self.source.get('login')} is shutting down."
        )

    def stop(self):
        """
        Stop the Trade Copier gracefully by setting the shutdown event.
        """
        if self._running:
            self._log_message(
                f"Signaling stop for Trade Copier on source account @{self.source.get('login')}..."
            )
            self._running = False
            self.shutdown_event.set()
        self._log_message("Trade Copier stopped successfully.")


def copier_worker_process(
    source_config: dict,
    destination_config: dict,
    sleeptime: float,
    start_time: str,
    end_time: str,
    /,
    custom_logger=None,
    shutdown_event=None,
    log_queue=None,
):
    """A top-level worker function for handling a single source-to-destination copy task.

    This function is the cornerstone of the robust, multi-process architecture. It is
    designed to be the `target` of a `multiprocessing.Process`. By being a top-level
    function, it avoids pickling issues on Windows and ensures that each copy task
    runs in a completely isolated process.

    A controller (like a GUI or a master script) should spawn one process with this
    target for each destination account it needs to manage.

    Args:
        source_config (dict): Configuration dictionary for the source account.
            Must contain 'login', 'password', 'server', and 'path'.
        destination_config (dict): Configuration dictionary for a *single*
            destination account.
        sleeptime (float): The time in seconds to wait between copy cycles.
        start_time (str): The time of day to start copying (e.g., "08:00").
        end_time (str): The time of day to stop copying (e.g., "22:00").
        custom_logger: An optional custom logger instance.
        shutdown_event (multiprocessing.Event): An event object that, when set,
            will signal this process to terminate gracefully.
        log_queue (multiprocessing.Queue): A queue for sending log messages back
            to the parent process in a thread-safe manner.
    """
    copier = TradeCopier(
        source_config,
        [destination_config],
        sleeptime=sleeptime,
        start_time=start_time,
        end_time=end_time,
        custom_logger=custom_logger,
        shutdown_event=shutdown_event,
        log_queue=log_queue,
    )
    copier.start_copy_process(destination_config)


def RunCopier(
    source: dict,
    destinations: list,
    sleeptime: float,
    start_time: str,
    end_time: str,
    /,
    custom_logger=None,
    shutdown_event=None,
    log_queue=None,
):
    """Initializes and runs a TradeCopier instance in a single process.

    This function serves as a straightforward wrapper to start a copying session
    that handles one source account and one or more destination accounts
    *sequentially* within the same thread. It does not create any new processes itself.

    This is useful for:
    - Simpler, command-line based use cases.
    - Scenarios where parallelism is not required.
    - As the target for `RunMultipleCopier`, where each process handles a
      full source-to-destinations session.

    Args:
        source (dict): Configuration dictionary for the source account.
        destinations (list): A list of configuration dictionaries, one for each
            destination account to be processed sequentially.
        sleeptime (float): The time in seconds to wait after completing a full
            cycle through all destinations.
        start_time (str): The time of day to start copying (e.g., "08:00").
        end_time (str): The time of day to stop copying (e.g., "22:00").
        custom_logger: An optional custom logger instance.
        shutdown_event (multiprocessing.Event): An event to signal shutdown.
        log_queue (multiprocessing.Queue): A queue for log messages.
    """
    copier = TradeCopier(
        source,
        destinations,
        sleeptime=sleeptime,
        start_time=start_time,
        end_time=end_time,
        custom_logger=custom_logger,
        shutdown_event=shutdown_event,
        log_queue=log_queue,
    )
    copier.run()


def RunMultipleCopier(
    accounts: List[dict],
    sleeptime: float = 0.01,
    start_delay: float = 1.0,
    start_time: str = None,
    end_time: str = None,
    shutdown_event=None,
    custom_logger=None,
    log_queue=None,
):
    """Manages multiple, independent trade copying sessions in parallel.

    This function acts as a high-level manager that takes a list of account
    setups and creates a separate, dedicated process for each one. Each process
    is responsible for copying from one source account to its associated list of
    destination accounts.

    The parallelism occurs at the **source account level**. Within each spawned
    process, the destinations for that source are handled sequentially by `RunCopier`.

    Example `accounts` structure:
    [
        { "source": {...}, "destinations": [{...}, {...}] },  # -> Process 1
        { "source": {...}, "destinations": [{...}] }          # -> Process 2
    ]

    Args:
        accounts (List[dict]): A list of account configurations. Each item in the
            list must be a dictionary with a 'source' key and a 'destinations' key.
        sleeptime (float): The sleep time passed down to each `RunCopier` process.
        start_delay (float): A delay in seconds between starting each new process.
            This helps prevent resource contention by staggering the initialization
            of multiple MetaTrader 5 terminals.
        start_time (str): The start time passed down to each `RunCopier` process.
        end_time (str): The end time passed down to each `RunCopier` process.
        shutdown_event (multiprocessing.Event): An event to signal shutdown to all
            child processes.
        custom_logger: An optional custom logger instance.
        log_queue (multiprocessing.Queue): A queue for aggregating log messages
            from all child processes.
    """
    processes = []

    for account in accounts:
        source = account.get("source")
        destinations = account.get("destinations")

        if not source or not destinations:
            logger.warning("Skipping account due to missing source or destinations.")
            continue
        paths = set([source.get("path")] + [dest.get("path") for dest in destinations])
        if len(paths) == 1 and len(destinations) >= 1:
            logger.warning(
                "Skipping account: source and destination cannot share the same MetaTrader 5 terminal path."
            )
            continue
        logger.info(f"Starting process for source account @{source.get('login')}")
        process = mp.Process(
            target=RunCopier,
            args=(
                source,
                destinations,
                sleeptime,
                start_time,
                end_time,
            ),
            kwargs=dict(
                custom_logger=custom_logger,
                shutdown_event=shutdown_event,
                log_queue=log_queue,
            ),
        )
        processes.append(process)
        process.start()

        if start_delay:
            time.sleep(start_delay)

    for process in processes:
        process.join()


def _parse_symbols(section):
    symbols: str = section.get("symbols")
    symbols = symbols.strip().replace("\n", " ").replace('"""', "")
    if symbols in ["all", "*"]:
        section["symbols"] = symbols
    else:
        symbols = get_symbols_from_string(symbols)
        section["symbols"] = symbols


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
