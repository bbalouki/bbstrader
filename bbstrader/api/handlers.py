import MetaTrader5 as mt5
from bbstrader.api import Handlers


def get_mt5_handlers():
    """
    Exhaustively maps all MetaTrader 5 Python functions to the C++ Handlers struct.
    """
    h = Handlers()

    # 1. System & Session Management
    h.init_auto = lambda: mt5.initialize()
    h.init_path = lambda path: mt5.initialize(path)
    h.init_full = lambda path, login, password, server, timeout, portable: mt5.initialize(
        path=path,
        login=login,
        password=password,
        server=server,
        timeout=timeout,
        portable=portable,
    )
    h.login = lambda login, password, server, timeout: mt5.login(
        login=login, password=password, server=server, timeout=timeout
    )
    h.shutdown = mt5.shutdown
    h.get_version = mt5.version
    h.get_last_error = mt5.last_error
    h.get_terminal_info = mt5.terminal_info
    h.get_account_info = mt5.account_info

    # 2. Symbols & Market Depth (Level 2)
    h.get_total_symbols = mt5.symbols_total
    h.get_symbols_all = lambda: mt5.symbols_get()
    h.get_symbol_info = mt5.symbol_info
    h.select_symbol = mt5.symbol_select
    h.get_symbols_by_group = lambda group: mt5.symbols_get(group)
    h.subscribe_book = mt5.market_book_add
    h.unsubscribe_book = mt5.market_book_release
    h.get_book_info = mt5.market_book_get

    # 3. Market Data (Rates & Ticks)
    h.get_rates_by_date = mt5.copy_rates_from
    h.get_rates_by_pos = mt5.copy_rates_from_pos
    h.get_rates_by_range = mt5.copy_rates_range
    h.get_ticks_by_date = mt5.copy_ticks_from
    h.get_ticks_by_range = mt5.copy_ticks_range
    h.get_tick_info = mt5.symbol_info_tick

    # 4. Trading Operations
    h.check_order = mt5.order_check
    h.send_order = mt5.order_send
    h.calc_margin = mt5.order_calc_margin
    h.calc_profit = mt5.order_calc_profit

    # 5. Active Orders & Positions
    h.get_orders_all = lambda: mt5.orders_get()
    h.get_orders_by_symbol = lambda symbol: mt5.orders_get(symbol=symbol)
    h.get_orders_by_group = lambda group: mt5.orders_get(group=group)
    h.get_order_by_ticket = lambda ticket: mt5.orders_get(ticket=ticket)
    h.get_total_orders = mt5.orders_total
    h.get_positions_all = lambda: mt5.positions_get()
    h.get_positions_symbol = lambda symbol: mt5.positions_get(symbol=symbol)
    h.get_positions_group = lambda group: mt5.positions_get(group=group)
    h.get_position_ticket = lambda ticket: mt5.positions_get(ticket=ticket)
    h.get_total_positions = mt5.positions_total

    # 6. Trade History (Orders & Deals)
    h.get_hist_orders_range = mt5.history_orders_get
    h.get_hist_order_ticket = lambda ticket: mt5.history_orders_get(ticket=ticket)
    h.get_hist_orders_pos = lambda position: mt5.history_orders_get(position=position)
    h.get_hist_orders_total = mt5.history_orders_total
    h.get_hist_deals_range = mt5.history_deals_get
    h.get_hist_deals_ticket = lambda ticket: mt5.history_deals_get(ticket=ticket)
    h.get_hist_deals_pos = lambda position: mt5.history_deals_get(position=position)
    h.get_hist_deals_total = mt5.history_deals_total

    return h
