from datetime import datetime

import pytz

from bbstrader.api.metatrader_client import (  # type: ignore
    AccountInfo,
    BookInfo,
    MetaTraderHandlers,
    OrderCheckResult,
    OrderSentResult,
    SymbolInfo,
    TerminalInfo,
    TickInfo,
    TradeDeal,
    TradeOrder,
    TradePosition,
    TradeRequest,
)

try:
    import MetaTrader5 as mt5
except ImportError:
    import bbstrader.compat  # noqa: F401


def _convert_obj(obj, obj_type):
    """Converts a single MT5 object to the target C++ class instance."""
    if obj is None:
        return None

    if hasattr(obj, "_asdict"):
        field_data: dict = obj._asdict()
        instance = obj_type()

        for k, v in field_data.items():
            if not hasattr(instance, k):
                continue
            if (
                k == "request"
                and hasattr(v, "_asdict")
                and obj_type in (OrderCheckResult, OrderSentResult)
            ):
                setattr(instance, k, v._asdict())
            else:
                setattr(instance, k, v)

        return instance
    else:
        raise TypeError(f"Expected MT5 namedtuple, got {type(obj)}")


def _convert_list(obj_list, obj_type):
    if obj_list is None:
        return None
    return [_convert_obj(obj, obj_type) for obj in obj_list]


def _build_request(req: TradeRequest | dict) -> dict:
    if isinstance(req, dict):
        return req

    request = {}
    attrs = [
        "action",
        "magic",
        "order",
        "symbol",
        "volume",
        "price",
        "stoplimit",
        "sl",
        "tp",
        "deviation",
        "type",
        "type_filling",
        "type_time",
        "expiration",
        "comment",
        "position",
        "position_by",
    ]
    enum_types = {"action", "type", "type_time", "type_filling"}
    for attr in attrs:
        val = getattr(req, attr)
        if attr in enum_types or (val != 0 and val != ""):
            request[attr] = val
    return request


def check_order(request: TradeRequest) -> OrderCheckResult:
    return mt5.order_check(_build_request(request))


def send_order(request: TradeRequest) -> OrderSentResult:
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
    converting return values to the appropriate types.
    """
    h = MetaTraderHandlers()

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
    h.get_rates_by_date = (
        lambda symbol, timeframe, date_from, count: mt5.copy_rates_from(
            symbol, timeframe, get_time(date_from), count
        )
    )
    h.get_rates_by_pos = (
        lambda symbol, timeframe, start_pos, count: mt5.copy_rates_from_pos(
            symbol, timeframe, start_pos, count
        )
    )
    h.get_rates_by_range = (
        lambda symbol, timeframe, date_from, date_to: mt5.copy_rates_range(
            symbol, timeframe, get_time(date_from), get_time(date_to)
        )
    )
    h.get_ticks_by_date = lambda symbol, date_from, count, flags: mt5.copy_ticks_from(
        symbol, get_time(date_from), count, flags
    )
    h.get_ticks_by_range = (
        lambda symbol, date_from, date_to, flags: mt5.copy_ticks_range(
            symbol, get_time(date_from), get_time(date_to), flags
        )
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
        (mt5.orders_get(ticket=ticket) or [None])[0], TradeOrder
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
        (mt5.positions_get(ticket=ticket) or [None])[0], TradePosition
    )
    h.get_total_positions = mt5.positions_total

    # 6. Trade History (Orders & Deals)
    h.get_hist_orders_group = lambda date_from, date_to, group: _convert_list(
        mt5.history_orders_get(get_time(date_from), get_time(date_to), group=group),
        TradeOrder,
    )
    h.get_hist_orders_range = lambda date_from, date_to: _convert_list(
        mt5.history_orders_get(get_time(date_from), get_time(date_to)),
        TradeOrder,
    )
    h.get_hist_order_ticket = lambda ticket: _convert_obj(
        mt5.history_orders_get(ticket=ticket), TradeOrder
    )
    h.get_hist_orders_pos = lambda position: _convert_list(
        mt5.history_orders_get(position=position), TradeOrder
    )
    h.get_hist_orders_total = mt5.history_orders_total
    h.get_hist_deals_group = lambda date_from, date_to, group: _convert_list(
        mt5.history_deals_get(get_time(date_from), get_time(date_to), group=group),
        TradeDeal,
    )
    h.get_hist_deals_range = lambda date_from, date_to: _convert_list(
        mt5.history_deals_get(get_time(date_from), get_time(date_to)), TradeDeal
    )

    h.get_hist_deals_ticket = lambda ticket: _convert_obj(
        mt5.history_deals_get(ticket=ticket), TradeDeal
    )
    h.get_hist_deals_pos = lambda position: _convert_list(
        mt5.history_deals_get(position=position), TradeDeal
    )
    h.get_hist_deals_total = mt5.history_deals_total

    return h


Mt5Handlers = get_mt5_handlers()
