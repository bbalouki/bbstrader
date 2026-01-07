import collections.abc
import typing
from typing import overload

from _typeshed import Incomplete

class AccountInfo:
    assets: float
    balance: float
    commission_blocked: float
    company: str
    credit: float
    currency: str
    currency_digits: int
    equity: float
    fifo_close: bool
    leverage: int
    liabilities: float
    limit_orders: int
    login: int
    margin: float
    margin_free: float
    margin_initial: float
    margin_level: float
    margin_maintenance: float
    margin_mode: int
    margin_so_call: float
    margin_so_mode: int
    margin_so_so: float
    name: str
    profit: float
    server: str
    trade_allowed: bool
    trade_expert: bool
    trade_mode: int
    def __init__(self) -> None: ...

class BookInfo:
    price: float
    type: int
    volume: int
    volume_real: float
    def __init__(self) -> None: ...

class Handlers:
    calc_margin: Incomplete
    calc_profit: Incomplete
    check_order: Incomplete
    get_account_info: Incomplete
    get_book_info: Incomplete
    get_hist_deals_pos: Incomplete
    get_hist_deals_range: Incomplete
    get_hist_deals_ticket: Incomplete
    get_hist_deals_total: collections.abc.Callable[
        [typing.SupportsInt, typing.SupportsInt], int
    ]
    get_hist_order_ticket: Incomplete
    get_hist_orders_pos: Incomplete
    get_hist_orders_range: Incomplete
    get_hist_orders_total: collections.abc.Callable[
        [typing.SupportsInt, typing.SupportsInt], int
    ]
    get_last_error: collections.abc.Callable[[], int]
    get_order_by_ticket: Incomplete
    get_orders_all: Incomplete
    get_orders_by_group: Incomplete
    get_orders_by_symbol: Incomplete
    get_position_ticket: Incomplete
    get_positions_all: Incomplete
    get_positions_group: Incomplete
    get_positions_symbol: Incomplete
    get_rates_by_date: Incomplete
    get_rates_by_pos: Incomplete
    get_rates_by_range: Incomplete
    get_symbol_info: Incomplete
    get_symbols_all: Incomplete
    get_symbols_by_group: Incomplete
    get_terminal_info: Incomplete
    get_tick_info: Incomplete
    get_ticks_by_date: Incomplete
    get_ticks_by_range: Incomplete
    get_total_orders: collections.abc.Callable[[], int]
    get_total_positions: collections.abc.Callable[[], int]
    get_total_symbols: collections.abc.Callable[[], int]
    get_version: collections.abc.Callable[[], str | None]
    init_auto: collections.abc.Callable[[], bool]
    init_full: collections.abc.Callable[
        [str, typing.SupportsInt, str, str, typing.SupportsInt, bool], bool
    ]
    init_path: collections.abc.Callable[[str], bool]
    login: collections.abc.Callable[
        [typing.SupportsInt, str, str, typing.SupportsInt], bool
    ]
    select_symbol: collections.abc.Callable[[str, bool], bool]
    send_order: Incomplete
    shutdown: collections.abc.Callable[[], None]
    subscribe_book: collections.abc.Callable[[str], bool]
    unsubscribe_book: collections.abc.Callable[[str], bool]
    def __init__(self) -> None: ...

class MetaTraderClient:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, arg0: Handlers) -> None: ...
    def account_info(self, *args, **kwargs): ...
    def copy_rates_from(self, *args, **kwargs): ...
    def copy_rates_from_pos(self, *args, **kwargs): ...
    def copy_rates_range(self, *args, **kwargs): ...
    def copy_ticks_from(self, *args, **kwargs): ...
    def copy_ticks_range(self, *args, **kwargs): ...
    def history_deals_get(self, *args, **kwargs): ...
    def history_deals_get_by_pos(self, *args, **kwargs): ...
    def history_deals_total(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> int: ...
    def history_orders_get(self, *args, **kwargs): ...
    def history_orders_get_by_pos(self, *args, **kwargs): ...
    def history_orders_total(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsInt
    ) -> int: ...
    @overload
    def initialize(self) -> bool: ...
    @overload
    def initialize(self, arg0: str) -> bool: ...
    @overload
    def initialize(
        self,
        arg0: str,
        arg1: typing.SupportsInt,
        arg2: str,
        arg3: str,
        arg4: typing.SupportsInt,
        arg5: bool,
    ) -> bool: ...
    def last_error(self) -> int: ...
    def login(
        self, arg0: typing.SupportsInt, arg1: str, arg2: str, arg3: typing.SupportsInt
    ) -> bool: ...
    def market_book_add(self, arg0: str) -> bool: ...
    def market_book_get(self, *args, **kwargs): ...
    def market_book_release(self, arg0: str) -> bool: ...
    def order_calc_margin(
        self, arg0, arg1: str, arg2: typing.SupportsFloat, arg3: typing.SupportsFloat
    ) -> float | None: ...
    def order_calc_profit(
        self,
        arg0,
        arg1: str,
        arg2: typing.SupportsFloat,
        arg3: typing.SupportsFloat,
        arg4: typing.SupportsFloat,
    ) -> float | None: ...
    def order_check(self, *args, **kwargs): ...
    def order_get_by_ticket(self, *args, **kwargs): ...
    def order_send(self, *args, **kwargs): ...
    def orders_get(self, *args, **kwargs): ...
    def orders_get_by_group(self, *args, **kwargs): ...
    def orders_total(self) -> int: ...
    def position_get_by_ticket(self, *args, **kwargs): ...
    def positions_get(self, *args, **kwargs): ...
    def positions_get_by_group(self, *args, **kwargs): ...
    def positions_total(self) -> int: ...
    def shutdown(self) -> None: ...
    def symbol_info(self, *args, **kwargs): ...
    def symbol_info_tick(self, *args, **kwargs): ...
    def symbol_select(self, arg0: str, arg1: bool) -> bool: ...
    def symbols_get(self, *args, **kwargs): ...
    def symbols_total(self) -> int: ...
    def terminal_info(self, *args, **kwargs): ...
    def version(self) -> str | None: ...

class OrderCheckResult:
    balance: float
    comment: str
    equity: float
    margin: float
    margin_free: float
    margin_level: float
    profit: float
    request: TradeRequest
    retcode: int
    def __init__(self) -> None: ...

class OrderSentResult:
    ask: float
    bid: float
    comment: str
    deal: int
    order: int
    price: float
    request: TradeRequest
    request_id: int
    retcode: int
    retcode_external: int
    volume: float
    def __init__(self) -> None: ...

class RateInfo:
    close: float
    high: float
    low: float
    open: float
    real_volume: int
    spread: int
    tick_volume: int
    time: int
    def __init__(self) -> None: ...

class SymbolInfo:
    ask: float
    askhigh: float
    asklow: float
    bank: str
    basis: str
    bid: float
    bidhigh: float
    bidlow: float
    category: str
    chart_mode: int
    currency_base: str
    currency_margin: str
    currency_profit: str
    custom: bool
    description: str
    digits: int
    exchange: str
    expiration_mode: int
    expiration_time: int
    filling_mode: int
    formula: str
    isin: str
    last: float
    lasthigh: float
    lastlow: float
    margin_hedged: float
    margin_hedged_use_leg: bool
    margin_initial: float
    margin_maintenance: float
    name: str
    option_mode: int
    option_right: int
    option_strike: float
    order_gtc_mode: int
    order_mode: int
    page: str
    path: str
    point: float
    price_change: float
    price_greeks_delta: float
    price_greeks_gamma: float
    price_greeks_omega: float
    price_greeks_rho: float
    price_greeks_theta: float
    price_greeks_vega: float
    price_sensitivity: float
    price_theoretical: float
    price_volatility: float
    select: bool
    session_aw: float
    session_buy_orders: int
    session_buy_orders_volume: float
    session_close: float
    session_deals: int
    session_interest: float
    session_open: float
    session_price_limit_max: float
    session_price_limit_min: float
    session_price_settlement: float
    session_sell_orders: int
    session_sell_orders_volume: float
    session_turnover: float
    session_volume: float
    spread: int
    spread_float: bool
    start_time: int
    swap_long: float
    swap_mode: int
    swap_rollover3days: int
    swap_short: float
    ticks_bookdepth: int
    time: int
    trade_accrued_interest: float
    trade_calc_mode: int
    trade_contract_size: float
    trade_exemode: int
    trade_face_value: float
    trade_freeze_level: int
    trade_liquidity_rate: float
    trade_mode: int
    trade_stops_level: int
    trade_tick_size: float
    trade_tick_value: float
    trade_tick_value_loss: float
    trade_tick_value_profit: float
    visible: bool
    volume: int
    volume_limit: float
    volume_max: float
    volume_min: float
    volume_real: float
    volume_step: float
    volumehigh: int
    volumehigh_real: float
    volumelow: int
    volumelow_real: float
    def __init__(self) -> None: ...

class TerminalInfo:
    build: int
    codepage: int
    commondata_path: str
    community_account: bool
    community_balance: float
    community_connection: bool
    company: str
    connected: bool
    data_path: str
    dlls_allowed: bool
    email_enabled: bool
    ftp_enabled: bool
    language: str
    maxbars: int
    mqid: bool
    name: str
    notifications_enabled: bool
    path: str
    ping_last: int
    retransmission: float
    trade_allowed: bool
    tradeapi_disabled: bool
    def __init__(self) -> None: ...

class TickInfo:
    ask: float
    bid: float
    flags: int
    last: float
    time: int
    time_msc: int
    volume: int
    volume_real: float
    def __init__(self) -> None: ...

class TradeDeal:
    comment: str
    commission: float
    entry: int
    external_id: str
    fee: float
    magic: int
    order: int
    position_id: int
    price: float
    profit: float
    reason: int
    swap: float
    symbol: str
    ticket: int
    time: int
    time_msc: int
    type: int
    volume: float
    def __init__(self) -> None: ...

class TradeOrder:
    comment: str
    external_id: str
    magic: int
    position_by_id: int
    position_id: int
    price_current: float
    price_open: float
    price_stoplimit: float
    reason: int
    sl: float
    state: int
    symbol: str
    ticket: int
    time_done: int
    time_done_msc: int
    time_expiration: int
    time_setup: int
    time_setup_msc: int
    tp: float
    type: int
    type_filling: int
    type_time: int
    volume_current: float
    volume_initial: float
    def __init__(self) -> None: ...

class TradePosition:
    comment: str
    external_id: str
    identifier: int
    magic: int
    price_current: float
    price_open: float
    profit: float
    reason: int
    sl: float
    swap: float
    symbol: str
    ticket: int
    time: int
    time_msc: int
    time_update: int
    time_update_msc: int
    tp: float
    type: int
    volume: float
    def __init__(self) -> None: ...

class TradeRequest:
    action: int
    comment: str
    deviation: int
    expiration: int
    magic: int
    order: int
    position: int
    position_by: int
    price: float
    sl: float
    stoplimit: float
    symbol: str
    tp: float
    type: int
    type_filling: int
    type_time: int
    volume: float
    def __init__(self) -> None: ...
