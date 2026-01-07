from datetime import datetime

import MetaTrader5 as mt5
import pytz

from bbstrader.api import (
    AccountInfo,
    BookInfo,
    Handlers,
    OrderCheckResult,
    OrderSentResult,
    RateInfo,
    SymbolInfo,
    TerminalInfo,
    TickInfo,
    TradeDeal,
    TradeOrder,
    TradePosition,
    TradeRequest,
)


def _convert_obj(obj, nt_type):
    """Converts a single MT5 object to the target NamedTuple type."""
    if obj is None:
        return None

    # Create a dictionary of the fields from the source object
    if hasattr(obj, "_asdict"):
        field_data: dict = obj._asdict()
        object_type = nt_type()
        for k, v in field_data.items():
            try:
                setattr(object_type, k, v)
            except Exception as e:
                print(f"Error on {type(obj)} k: {k}, v: {v}")
                print(e)

        # Handle nested object conversion for specific types
        if nt_type is OrderCheckResult or nt_type is OrderSentResult:
            if "request" in field_data and field_data["request"] is not None:
                field_data["request"] = _convert_obj(
                    field_data["request"], TradeRequest
                )

        return object_type
    else:
        raise TypeError(f"Can not handle object of type {type(obj)}")


def _convert_list(obj_list, nt_type):
    if obj_list is None:
        return None
    return [_convert_obj(obj, nt_type) for obj in obj_list]


def _convert_nparray(arr, nt_type):
    if arr is None:
        return None
    return [nt_type(**{name: row[name] for name in arr.dtype.names}) for row in arr]


ENUM_TYPES = {"action", "type", "type_time", "type_filling"}


def _build_request(req: TradeRequest) -> dict:
    request = {}
    for attr in dir(mt5.TradeRequest):
        if attr.startswith("_") or attr.startswith("n_"):
            continue
        if attr not in ENUM_TYPES and attr == 0 or "":
            continue
        request[attr] = getattr(req, attr)


def check_order(request: TradeRequest) -> OrderCheckResult:
    if isinstance(request, dict):
        return mt5.order_check(request)
    
    return mt5.order_check(_build_request(request))


def send_order(request: TradeRequest) -> OrderSentResult:
    if isinstance(request, dict):
        return mt5.order_send(request)
    return mt5.order_send(_build_request(request))


def get_time(ts):
    timezone = pytz.timezone("UTC")
    if isinstance(ts, datetime):
        return ts
    elif isinstance(ts, int):
        return datetime.fromtimestamp(ts, tz=timezone)
    else:
        raise ValueError(f"Invalide Time format {type(ts)} must be and int or datetime")


def get_mt5_handlers():
    """
    Exhaustively maps all MetaTrader 5 Python functions to the C++ Handlers struct,
    converting return values to the appropriate NamedTuple types.
    """
    h = Handlers()

    # 1. System & Session Management (Functions returning structs are wrapped)
    h.init_auto = lambda: mt5.initialize()
    h.init_path = lambda path: mt5.initialize(path)
    h.init_full = (
        lambda path, login, password, server, timeout, portable: mt5.initialize(
            path=path,
            login=login,
            password=password,
            server=server,
            timeout=timeout,
            portable=portable,
        )
    )
    h.login = lambda login, password, server, timeout: mt5.login(
        login=login, password=password, server=server, timeout=timeout
    )
    h.shutdown = mt5.shutdown
    h.get_version = mt5.version
    h.get_last_error = mt5.last_error
    h.get_terminal_info = lambda: _convert_obj(mt5.terminal_info(), TerminalInfo)
    h.get_account_info = lambda: _convert_obj(mt5.account_info(), AccountInfo)

    # 2. Symbols & Market Depth (Level 2)
    h.get_total_symbols = mt5.symbols_total
    h.get_symbols_all = lambda: _convert_list(mt5.symbols_get(), SymbolInfo)
    h.get_symbol_info = lambda symbol: _convert_obj(mt5.symbol_info(symbol), SymbolInfo)
    h.select_symbol = mt5.symbol_select
    h.get_symbols_by_group = lambda group: _convert_list(
        mt5.symbols_get(group), SymbolInfo
    )
    h.subscribe_book = mt5.market_book_add
    h.unsubscribe_book = mt5.market_book_release
    h.get_book_info = lambda symbol: _convert_list(
        mt5.market_book_get(symbol), BookInfo
    )

    # 3. Market Data (Rates & Ticks)
    h.get_rates_by_date = lambda symbol, timeframe, date_from, count: _convert_nparray(
        mt5.copy_rates_from(symbol, timeframe, get_time(get_time(date_from)), count),
        RateInfo,
    )
    h.get_rates_by_pos = lambda symbol, timeframe, start_pos, count: _convert_nparray(
        mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count), RateInfo
    )
    h.get_rates_by_range = (
        lambda symbol, timeframe, date_from, date_to: _convert_nparray(
            mt5.copy_rates_range(
                symbol, timeframe, get_time(date_from), get_time(date_to)
            ),
            RateInfo,
        )
    )
    h.get_ticks_by_date = lambda symbol, date_from, count, flags: _convert_nparray(
        mt5.copy_ticks_from(symbol, get_time(date_from), count, flags), TickInfo
    )
    h.get_ticks_by_range = lambda symbol, date_from, date_to, flags: _convert_nparray(
        mt5.copy_ticks_range(symbol, get_time(date_from), get_time(date_to), flags),
        TickInfo,
    )
    h.get_tick_info = lambda symbol: _convert_obj(
        mt5.symbol_info_tick(symbol), TickInfo
    )

    # 4. Trading Operations
    h.check_order = lambda request: _convert_obj(check_order(request), OrderCheckResult)
    h.send_order = lambda request: _convert_obj(send_order(request), OrderSentResult)
    h.calc_margin = mt5.order_calc_margin
    h.calc_profit = mt5.order_calc_profit

    # 5. Active Orders & Positions
    h.get_orders_all = lambda: _convert_list(mt5.orders_get(), TradeOrder)
    h.get_orders_by_symbol = lambda symbol: _convert_list(
        mt5.orders_get(symbol=symbol), TradeOrder
    )
    h.get_orders_by_group = lambda group: _convert_list(
        mt5.orders_get(group=group), TradeOrder
    )
    h.get_order_by_ticket = lambda ticket: _convert_obj(
        mt5.orders_get(ticket=ticket), TradeOrder
    )
    h.get_total_orders = mt5.orders_total
    h.get_positions_all = lambda: _convert_list(mt5.positions_get(), TradePosition)
    h.get_positions_symbol = lambda symbol: _convert_list(
        mt5.positions_get(symbol=symbol), TradePosition
    )
    h.get_positions_group = lambda group: _convert_list(
        mt5.positions_get(group=group), TradePosition
    )
    h.get_position_ticket = lambda ticket: _convert_obj(
        mt5.positions_get(ticket=ticket), TradePosition
    )
    h.get_total_positions = mt5.positions_total

    # 6. Trade History (Orders & Deals)
    h.get_hist_orders_range = lambda date_from, date_to, group: _convert_list(
        mt5.history_orders_get(get_time(date_from), get_time(date_to), group),
        TradeOrder,
    )
    h.get_hist_order_ticket = lambda ticket: _convert_obj(
        mt5.history_orders_get(ticket=ticket), TradeOrder
    )
    h.get_hist_orders_pos = lambda position, count: _convert_list(
        mt5.history_orders_get(position=position), TradeOrder
    )
    h.get_hist_orders_total = mt5.history_orders_total
    h.get_hist_deals_range = lambda date_from, date_to, group: _convert_list(
        mt5.history_deals_get(get_time(date_from), get_time(date_to), group), TradeDeal
    )
    h.get_hist_deals_ticket = lambda ticket: _convert_obj(
        mt5.history_deals_get(ticket=ticket), TradeDeal
    )
    h.get_hist_deals_pos = lambda position, count: _convert_list(
        mt5.history_deals_get(position=position), TradeDeal
    )
    h.get_hist_deals_total = mt5.history_deals_total

    return h
