import MetaTrader5 as mt5
from bbstrader.api import (
    Handlers,
    AccountInfo,
    BookInfo,
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
    field_data = {field: getattr(obj, field) for field in nt_type._fields}

    # Handle nested object conversion for specific types
    if nt_type is OrderCheckResult or nt_type is OrderSentResult:
        if 'request' in field_data and field_data['request'] is not None:
            field_data['request'] = _convert_obj(field_data['request'], TradeRequest)

    return nt_type(**field_data)


def _convert_list(obj_list, nt_type):
    """Converts a list of MT5 objects to a list of NamedTuples."""
    if obj_list is None:
        return None
    return [_convert_obj(obj, nt_type) for obj in obj_list]


def _convert_nparray(arr, nt_type):
    """Converts a numpy structured array to a list of NamedTuples."""
    if arr is None:
        return None
    return [nt_type(**{name: row[name] for name in arr.dtype.names}) for row in arr]


def get_mt5_handlers():
    """
    Exhaustively maps all MetaTrader 5 Python functions to the C++ Handlers struct,
    converting return values to the appropriate NamedTuple types.
    """
    h = Handlers()

    # 1. System & Session Management (Functions returning structs are wrapped)
    h.init_auto = lambda: mt5.initialize()
    h.init_path = lambda path: mt5.initialize(path)
    h.init_full = lambda path, login, password, server, timeout, portable: mt5.initialize(
        path=path, login=login, password=password, server=server, timeout=timeout, portable=portable
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
    h.get_symbols_by_group = lambda group: _convert_list(mt5.symbols_get(group), SymbolInfo)
    h.subscribe_book = mt5.market_book_add
    h.unsubscribe_book = mt5.market_book_release
    h.get_book_info = lambda symbol: _convert_list(mt5.market_book_get(symbol), BookInfo)

    # 3. Market Data (Rates & Ticks)
    h.get_rates_by_date = lambda *a, **kw: _convert_nparray(mt5.copy_rates_from(*a, **kw), RateInfo)
    h.get_rates_by_pos = lambda *a, **kw: _convert_nparray(mt5.copy_rates_from_pos(*a, **kw), RateInfo)
    h.get_rates_by_range = lambda *a, **kw: _convert_nparray(mt5.copy_rates_range(*a, **kw), RateInfo)
    h.get_ticks_by_date = lambda *a, **kw: _convert_nparray(mt5.copy_ticks_from(*a, **kw), TickInfo)
    h.get_ticks_by_range = lambda *a, **kw: _convert_nparray(mt5.copy_ticks_range(*a, **kw), TickInfo)
    h.get_tick_info = lambda symbol: _convert_obj(mt5.symbol_info_tick(symbol), TickInfo)

    # 4. Trading Operations
    h.check_order = lambda request: _convert_obj(mt5.order_check(request), OrderCheckResult)
    h.send_order = lambda request: _convert_obj(mt5.order_send(request), OrderSentResult)
    h.calc_margin = mt5.order_calc_margin
    h.calc_profit = mt5.order_calc_profit

    # 5. Active Orders & Positions
    h.get_orders_all = lambda: _convert_list(mt5.orders_get(), TradeOrder)
    h.get_orders_by_symbol = lambda symbol: _convert_list(mt5.orders_get(symbol=symbol), TradeOrder)
    h.get_orders_by_group = lambda group: _convert_list(mt5.orders_get(group=group), TradeOrder)
    h.get_order_by_ticket = lambda ticket: _convert_obj(mt5.orders_get(ticket=ticket), TradeOrder)
    h.get_total_orders = mt5.orders_total
    h.get_positions_all = lambda: _convert_list(mt5.positions_get(), TradePosition)
    h.get_positions_symbol = lambda symbol: _convert_list(mt5.positions_get(symbol=symbol), TradePosition)
    h.get_positions_group = lambda group: _convert_list(mt5.positions_get(group=group), TradePosition)
    h.get_position_ticket = lambda ticket: _convert_obj(mt5.positions_get(ticket=ticket), TradePosition)
    h.get_total_positions = mt5.positions_total

    # 6. Trade History (Orders & Deals)
    h.get_hist_orders_range = lambda *a, **kw: _convert_list(mt5.history_orders_get(*a, **kw), TradeOrder)
    h.get_hist_order_ticket = lambda ticket: _convert_obj(mt5.history_orders_get(ticket=ticket), TradeOrder)
    h.get_hist_orders_pos = lambda position, count: _convert_list(mt5.history_orders_get(position=position, count=count), TradeOrder)
    h.get_hist_orders_total = mt5.history_orders_total
    h.get_hist_deals_range = lambda *a, **kw: _convert_list(mt5.history_deals_get(*a, **kw), TradeDeal)
    h.get_hist_deals_ticket = lambda ticket: _convert_obj(mt5.history_deals_get(ticket=ticket), TradeDeal)
    h.get_hist_deals_pos = lambda position, count: _convert_list(mt5.history_deals_get(position=position, count=count), TradeDeal)
    h.get_hist_deals_total = mt5.history_deals_total

    return h
