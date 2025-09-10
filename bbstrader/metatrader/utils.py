from datetime import datetime
from enum import Enum
from typing import NamedTuple, Optional

import numpy as np

try:
    import MetaTrader5 as MT5
except ImportError:
    import bbstrader.compat  # noqa: F401


__all__ = [
    "TIMEFRAMES",
    "TimeFrame",
    "TerminalInfo",
    "AccountInfo",
    "SymbolInfo",
    "SymbolType",
    "TickInfo",
    "TradeRequest",
    "OrderCheckResult",
    "OrderSentResult",
    "TradeOrder",
    "TradePosition",
    "TradeDeal",
    "InvalidBroker",
    "GenericFail",
    "InvalidParams",
    "HistoryNotFound",
    "InvalidVersion",
    "AuthFailed",
    "RateInfo",
    "RateDtype",
    "TickDtype",
    "TickFlag",
    "UnsupportedMethod",
    "AutoTradingDisabled",
    "InternalFailSend",
    "InternalFailReceive",
    "InternalFailInit",
    "InternalFailConnect",
    "InternalFailTimeout",
    "trade_retcode_message",
    "raise_mt5_error",
]

# TIMEFRAME is an enumeration with possible chart period values
# See https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrom_py#timeframe
TIMEFRAMES = {
    "1m": MT5.TIMEFRAME_M1,
    "2m": MT5.TIMEFRAME_M2,
    "3m": MT5.TIMEFRAME_M3,
    "4m": MT5.TIMEFRAME_M4,
    "5m": MT5.TIMEFRAME_M5,
    "6m": MT5.TIMEFRAME_M6,
    "10m": MT5.TIMEFRAME_M10,
    "12m": MT5.TIMEFRAME_M12,
    "15m": MT5.TIMEFRAME_M15,
    "20m": MT5.TIMEFRAME_M20,
    "30m": MT5.TIMEFRAME_M30,
    "1h": MT5.TIMEFRAME_H1,
    "2h": MT5.TIMEFRAME_H2,
    "3h": MT5.TIMEFRAME_H3,
    "4h": MT5.TIMEFRAME_H4,
    "6h": MT5.TIMEFRAME_H6,
    "8h": MT5.TIMEFRAME_H8,
    "12h": MT5.TIMEFRAME_H12,
    "D1": MT5.TIMEFRAME_D1,
    "W1": MT5.TIMEFRAME_W1,
    "MN1": MT5.TIMEFRAME_MN1,
}


class TimeFrame(Enum):
    """
    Rrepresent a time frame object
    """

    M1 = TIMEFRAMES["1m"]
    M2 = TIMEFRAMES["2m"]
    M3 = TIMEFRAMES["3m"]
    M4 = TIMEFRAMES["4m"]
    M5 = TIMEFRAMES["5m"]
    M6 = TIMEFRAMES["6m"]
    M10 = TIMEFRAMES["10m"]
    M12 = TIMEFRAMES["12m"]
    M15 = TIMEFRAMES["15m"]
    M20 = TIMEFRAMES["20m"]
    M30 = TIMEFRAMES["30m"]
    H1 = TIMEFRAMES["1h"]
    H2 = TIMEFRAMES["2h"]
    H3 = TIMEFRAMES["3h"]
    H4 = TIMEFRAMES["4h"]
    H6 = TIMEFRAMES["6h"]
    H8 = TIMEFRAMES["8h"]
    H12 = TIMEFRAMES["12h"]
    D1 = TIMEFRAMES["D1"]
    W1 = TIMEFRAMES["W1"]
    MN1 = TIMEFRAMES["MN1"]

    def __str__(self):
        """Return the string representation of the time frame."""
        return self.name


class TerminalInfo(NamedTuple):
    """
    Represents general information about the trading terminal.
    See https://www.mql5.com/en/docs/constants/environment_state/terminalstatus
    """

    community_account: bool
    community_connection: bool
    connected: bool
    dlls_allowed: bool
    trade_allowed: bool
    tradeapi_disabled: bool
    email_enabled: bool
    ftp_enabled: bool
    notifications_enabled: bool
    mqid: bool
    build: int
    maxbars: int
    codepage: int
    ping_last: int
    community_balance: float
    retransmission: float
    company: str
    name: str
    language: str
    path: str
    data_path: str
    commondata_path: str


class AccountInfo(NamedTuple):
    """
    Represents information about a trading account.
    See https://www.mql5.com/en/docs/constants/environment_state/accountinformation
    """

    login: int
    trade_mode: int
    leverage: int
    limit_orders: int
    margin_so_mode: int
    trade_allowed: bool
    trade_expert: bool
    margin_mode: int
    currency_digits: int
    fifo_close: bool
    balance: float
    credit: float
    profit: float
    equity: float
    margin: float
    margin_free: float
    margin_level: float
    margin_so_call: float
    margin_so_so: float
    margin_initial: float
    margin_maintenance: float
    assets: float
    liabilities: float
    commission_blocked: float
    name: str
    server: str
    currency: str
    company: str


class SymbolInfo(NamedTuple):
    """
    Represents detailed information about a financial instrument.
    See https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants
    """

    custom: bool
    chart_mode: int
    select: bool
    visible: bool
    session_deals: int
    session_buy_orders: int
    session_sell_orders: int
    volume: int
    volumehigh: int
    volumelow: int
    time: datetime
    digits: int
    spread: int
    spread_float: bool
    ticks_bookdepth: int
    trade_calc_mode: int
    trade_mode: int
    start_time: int
    expiration_time: int
    trade_stops_level: int
    trade_freeze_level: int
    trade_exemode: int
    swap_mode: int
    swap_rollover3days: int
    margin_hedged_use_leg: bool
    expiration_mode: int
    filling_mode: int
    order_mode: int
    order_gtc_mode: int
    option_mode: int
    option_right: int
    bid: float
    bidhigh: float
    bidlow: float
    ask: float
    askhigh: float
    asklow: float
    last: float
    lasthigh: float
    lastlow: float
    volume_real: float
    volumehigh_real: float
    volumelow_real: float
    option_strike: float
    point: float
    trade_tick_value: float
    trade_tick_value_profit: float
    trade_tick_value_loss: float
    trade_tick_size: float
    trade_contract_size: float
    trade_accrued_interest: float
    trade_face_value: float
    trade_liquidity_rate: float
    volume_min: float
    volume_max: float
    volume_step: float
    volume_limit: float
    swap_long: float
    swap_short: float
    margin_initial: float
    margin_maintenance: float
    session_volume: float
    session_turnover: float
    session_interest: float
    session_buy_orders_volume: float
    session_sell_orders_volume: float
    session_open: float
    session_close: float
    session_aw: float
    session_price_settlement: float
    session_price_limit_min: float
    session_price_limit_max: float
    margin_hedged: float
    price_change: float
    price_volatility: float
    price_theoretical: float
    price_greeks_delta: float
    price_greeks_theta: float
    price_greeks_gamma: float
    price_greeks_vega: float
    price_greeks_rho: float
    price_greeks_omega: float
    price_sensitivity: float
    basis: str
    category: str
    currency_base: str
    currency_profit: str
    currency_margin: str
    bank: str
    description: str
    exchange: str
    formula: str
    isin: str
    name: str
    page: str
    path: str


class SymbolType(Enum):
    """
    Represents the type of a symbol.
    """

    FOREX = "FOREX"  # Forex currency pairs
    FUTURES = "FUTURES"  # Futures contracts
    STOCKS = "STOCKS"  # Stocks and shares
    BONDS = "BONDS"  # Bonds
    CRYPTO = "CRYPTO"  # Cryptocurrencies
    ETFs = "ETFs"  # Exchange-Traded Funds
    INDICES = "INDICES"  # Market indices
    COMMODITIES = "COMMODITIES"  # Commodities
    OPTIONS = "OPTIONS"  # Options contracts
    unknown = "UNKNOWN"  # Unknown or unsupported type


TickDtype = np.dtype(
    [
        ("time", "<i8"),
        ("bid", "<f8"),
        ("ask", "<f8"),
        ("last", "<f8"),
        ("volume", "<u8"),
        ("time_msc", "<i8"),
        ("flags", "<u4"),
        ("volume_real", "<f8"),
    ]
)

TickFlag = {
    "all": MT5.COPY_TICKS_ALL,
    "info": MT5.COPY_TICKS_INFO,
    "trade": MT5.COPY_TICKS_TRADE,
}


class TickInfo(NamedTuple):
    """
    Represents the last tick for the specified financial instrument.
    * time:     Time of the last prices update
    * bid:      Current Bid price
    * ask:      Current Ask price
    * last:     Price of the last deal (Last)
    * volume:   Volume for the current Last price
    * time_msc: Time of a price last update in milliseconds
    * flags:    Tick flags
    * volume_real:  Volume for the current Last price with greater accuracy
    """

    time: datetime
    bid: float
    ask: float
    last: float
    volume: int
    time_msc: int
    flags: int
    volume_real: float


RateDtype = np.dtype(
    [
        ("time", "<i8"),
        ("open", "<f8"),
        ("high", "<f8"),
        ("low", "<f8"),
        ("close", "<f8"),
        ("tick_volume", "<u8"),
        ("spread", "<i4"),
        ("real_volume", "<u8"),
    ]
)


class RateInfo(NamedTuple):
    """
    Reprents a candle (bar) for a specified period.
    * time: Time in seconds since 1970.01.01 00:00
    * open: Open price
    * high: High price
    * low: Low price
    * close: Close price
    * tick_volume: Tick volume
    * spread: Spread value
    * real_volume: Real volume

    """

    time: int
    open: float
    high: float
    low: float
    close: float
    tick_volume: float
    spread: int
    real_volume: float


class BookInfo(NamedTuple):
    """
    Represents the structure of a book.
    * type: Type of the order (buy/sell)
    * price: Price of the order
    * volume: Volume of the order in lots
    * volume_dbl: Volume with greater accuracy

    """

    type: int
    price: float
    volume: float
    volume_dbl: float


class TradeRequest(NamedTuple):
    """
    Represents a Trade Request Structure
    See https://www.mql5.com/en/docs/constants/structures/mqltraderequest
    """

    action: int
    magic: int
    order: int
    symbol: str
    volume: float
    price: float
    stoplimit: float
    sl: float
    tp: float
    deviation: int
    type: int
    type_filling: int
    type_time: int
    expiration: int
    comment: str
    position: int
    position_by: int


class OrderCheckResult(NamedTuple):
    """
    The Structure of Results of a Trade Request Check
    See https://www.mql5.com/en/docs/constants/structures/mqltradecheckresult
    """

    retcode: int
    balance: float
    equity: float
    profit: float
    margin: float
    margin_free: float
    margin_level: float
    comment: str
    request: TradeRequest


class OrderSentResult(NamedTuple):
    """
    The Structure of a Trade Request Result
    See https://www.mql5.com/en/docs/constants/structures/mqltraderesult
    """

    retcode: int
    deal: int
    order: int
    volume: float
    price: float
    bid: float
    ask: float
    comment: str
    request_id: int
    retcode_external: int
    request: TradeRequest


class TradeOrder(NamedTuple):
    """
    Represents a trade order.
    See https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties
    """

    ticket: int
    time_setup: int
    time_setup_msc: int
    time_done: int
    time_done_msc: int
    time_expiration: int
    type: int
    type_time: int
    type_filling: int
    state: int
    magic: int
    position_id: int
    position_by_id: int
    reason: int
    volume_initial: float
    volume_current: float
    price_open: float
    sl: float  # Stop Loss
    tp: float  # Take Profit
    price_current: float
    price_stoplimit: float
    symbol: str
    comment: str
    external_id: str


class TradePosition(NamedTuple):
    """
    Represents a trade position with attributes like ticket, open/close prices,
    volume, profit, and other trading details.
    See https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties
    """

    ticket: int
    time: int
    time_msc: int
    time_update: int
    time_update_msc: int
    type: int
    magic: int
    identifier: int
    reason: int
    volume: float
    price_open: float
    sl: float  # Stop Loss
    tp: float  # Take Profit
    price_current: float
    swap: float
    profit: float
    symbol: str
    comment: str
    external_id: str


class TradeDeal(NamedTuple):
    """
    Represents a trade deal execution.
    See https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties
    """

    ticket: int
    order: int
    time: int
    time_msc: int
    type: int
    entry: int
    magic: int
    position_id: int
    reason: int
    volume: float
    price: float
    commission: float
    swap: float
    profit: float
    fee: float
    symbol: str
    comment: str
    external_id: str


class InvalidBroker(Exception):
    """Exception raised for invalid broker errors."""

    def __init__(self, message="Invalid broker."):
        super().__init__(message)


class MT5TerminalError(Exception):
    """Base exception class for trading-related errors."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message

    def __repr__(self) -> str:
        msg_str = str(self.message) if self.message is not None else ""
        return f"{self.code} - {self.__class__.__name__}: {msg_str}"


class GenericFail(MT5TerminalError):
    """Exception raised for generic failure."""

    def __init__(self, message="Generic fail"):
        super().__init__(MT5.RES_E_FAIL, message)


class InvalidParams(MT5TerminalError):
    """Exception raised for invalid arguments or parameters."""

    def __init__(self, message="Invalid arguments or parameters."):
        super().__init__(MT5.RES_E_INVALID_PARAMS, message)


class HistoryNotFound(MT5TerminalError):
    """Exception raised when no history is found."""

    def __init__(self, message="No history found."):
        super().__init__(MT5.RES_E_NOT_FOUND, message)


class InvalidVersion(MT5TerminalError):
    """Exception raised for an invalid version."""

    def __init__(self, message="Invalid version."):
        super().__init__(MT5.RES_E_INVALID_VERSION, message)


class AuthFailed(MT5TerminalError):
    """Exception raised for authorization failure."""

    def __init__(self, message="Authorization failed."):
        super().__init__(MT5.RES_E_AUTH_FAILED, message)


class UnsupportedMethod(MT5TerminalError):
    """Exception raised for an unsupported method."""

    def __init__(self, message="Unsupported method."):
        super().__init__(MT5.RES_E_UNSUPPORTED, message)


class AutoTradingDisabled(MT5TerminalError):
    """Exception raised when auto-trading is disabled."""

    def __init__(self, message="Auto-trading is disabled."):
        super().__init__(MT5.RES_E_AUTO_TRADING_DISABLED, message)


class InternalFailError(MT5TerminalError):
    """Base exception class for internal IPC errors."""

    pass


class InternalFailSend(InternalFailError):
    """Exception raised for internal IPC send failure."""

    def __init__(self, message="Internal IPC send failed."):
        super().__init__(MT5.RES_E_INTERNAL_FAIL_SEND, message)


class InternalFailReceive(InternalFailError):
    """Exception raised for internal IPC receive failure."""

    def __init__(self, message="Internal IPC receive failed."):
        super().__init__(MT5.RES_E_INTERNAL_FAIL_RECEIVE, message)


class InternalFailInit(InternalFailError):
    """Exception raised for internal IPC initialization failure."""

    def __init__(self, message="Internal IPC initialization failed."):
        super().__init__(MT5.RES_E_INTERNAL_FAIL_INIT, message)


class InternalFailConnect(InternalFailError):
    """Exception raised for no IPC connection."""

    def __init__(self, message="No IPC connection."):
        super().__init__(MT5.RES_E_INTERNAL_FAIL_CONNECT, message)


class InternalFailTimeout(InternalFailError):
    """Exception raised for an internal timeout."""

    def __init__(self, message="Internal timeout."):
        super().__init__(MT5.RES_E_INTERNAL_FAIL_TIMEOUT, message)


RES_E_FAIL = 1  # Generic error
RES_E_INVALID_PARAMS = 2  # Invalid parameters
RES_E_NOT_FOUND = 3  # Not found
RES_E_INVALID_VERSION = 4  # Invalid version
RES_E_AUTH_FAILED = 5  # Authorization failed
RES_E_UNSUPPORTED = 6  # Unsupported method
RES_E_AUTO_TRADING_DISABLED = 7  # Autotrading disabled

# Actual internal error codes from MetaTrader5
RES_E_INTERNAL_FAIL_CONNECT = -10000
RES_E_INTERNAL_FAIL_INIT = -10001
RES_E_INTERNAL_FAIL_SEND = -10006
RES_E_INTERNAL_FAIL_RECEIVE = -10007
RES_E_INTERNAL_FAIL_TIMEOUT = -10008

# Dictionary to map error codes to exception classes
_ERROR_CODE_TO_EXCEPTION_ = {
    MT5.RES_E_FAIL: GenericFail,
    MT5.RES_E_INVALID_PARAMS: InvalidParams,
    MT5.RES_E_NOT_FOUND: HistoryNotFound,
    MT5.RES_E_INVALID_VERSION: InvalidVersion,
    MT5.RES_E_AUTH_FAILED: AuthFailed,
    MT5.RES_E_UNSUPPORTED: UnsupportedMethod,
    MT5.RES_E_AUTO_TRADING_DISABLED: AutoTradingDisabled,
    MT5.RES_E_INTERNAL_FAIL_SEND: InternalFailSend,
    MT5.RES_E_INTERNAL_FAIL_RECEIVE: InternalFailReceive,
    MT5.RES_E_INTERNAL_FAIL_INIT: InternalFailInit,
    MT5.RES_E_INTERNAL_FAIL_CONNECT: InternalFailConnect,
    MT5.RES_E_INTERNAL_FAIL_TIMEOUT: InternalFailTimeout,
    RES_E_FAIL: GenericFail,
    RES_E_INVALID_PARAMS: InvalidParams,
    RES_E_NOT_FOUND: HistoryNotFound,
    RES_E_INVALID_VERSION: InvalidVersion,
    RES_E_AUTH_FAILED: AuthFailed,
    RES_E_UNSUPPORTED: UnsupportedMethod,
    RES_E_AUTO_TRADING_DISABLED: AutoTradingDisabled,
    RES_E_INTERNAL_FAIL_SEND: InternalFailSend,
    RES_E_INTERNAL_FAIL_RECEIVE: InternalFailReceive,
    RES_E_INTERNAL_FAIL_INIT: InternalFailInit,
    RES_E_INTERNAL_FAIL_CONNECT: InternalFailConnect,
    RES_E_INTERNAL_FAIL_TIMEOUT: InternalFailTimeout,
}


def raise_mt5_error(message: Optional[str] = None):
    """Raises an exception based on the given error code.

    Args:
        message: An optional custom error message.

    Raises:
        MT5TerminalError: A specific exception based on the error code.
    """
    if message and isinstance(message, Exception):
        message = str(message)
    error = _ERROR_CODE_TO_EXCEPTION_.get(MT5.last_error()[0])
    if error is not None:
        raise Exception(f"{error(None)} {message or MT5.last_error()[1]}")
    else:
        raise Exception(f"{message or MT5.last_error()[1]}")


_ORDER_FILLING_TYPE_ = "https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type_filling"
_ORDER_TYPE_ = "https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type"
_POSITION_IDENTIFIER_ = "https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_integer"
_FIFO_RULE_ = "https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_info_integer"

_TRADE_RETCODE_MESSAGES_ = {
    10004: "Requote: The price has changed, please try again",
    10006: "Request rejected",
    10007: "Request canceled by trader",
    10008: "Order placed",
    10009: "Request completed",
    10010: "Only part of the request was completed",
    10011: "Request processing error",
    10012: "Request canceled by timeout",
    10013: "Invalid request",
    10014: "Invalid volume in the request",
    10015: "Invalid price in the request",
    10016: "Invalid stops in the request",
    10017: "Trade is disabled",
    10018: "Market is closed",
    10019: "Insufficient funds to complete the request",
    10020: "Prices changed",
    10021: "No quotes to process the request",
    10022: "Invalid order expiration date in the request",
    10023: "Order state changed",
    10024: "Too many requests, please try again later",
    10025: "No changes in request",
    10026: "Autotrading disabled by server",
    10027: "Autotrading disabled by client terminal",
    10028: "Request locked for processing",
    10029: "Order or position frozen",
    10030: "Invalid order filling type: see" + " " + _ORDER_FILLING_TYPE_,
    10031: "No connection with the trade server",
    10032: "Operation allowed only for live accounts",
    10033: "The number of pending orders has reached the limit",
    10034: "Order/position volume limit for the symbol reached",
    10035: "Incorrect or prohibited order type: see" + " " + _ORDER_TYPE_,
    10036: "Position with the specified ID has already been closed: see"
    + " "
    + _POSITION_IDENTIFIER_,
    10038: "Close volume exceeds the current position volume",
    10039: "A close order already exists for this position",
    10040: "Maximum number of open positions reached",
    10041: "Pending order activation rejected, order canceled",
    10042: "Only long positions are allowed",
    10043: "Only short positions are allowed",
    10044: "Only position closing is allowed",
    10045: "Position closing allowed only by FIFO rule: see" + " " + _FIFO_RULE_,
    10046: "Opposite positions on this symbol are disabled",
}


def trade_retcode_message(code, display=False, add_msg=""):
    """
    Retrieves a user-friendly message corresponding to a given trade return code.

    Args:
        code (int): The trade return code to look up.
        display (bool, optional): Whether to print the message to the console. Defaults to False.

    Returns:
        str: The message associated with the provided trade return code. If the code is not found,
             it returns "Unknown trade error.".
    """
    message = _TRADE_RETCODE_MESSAGES_.get(code, "Unknown trade error")
    if display:
        print(message + add_msg)
    return message
