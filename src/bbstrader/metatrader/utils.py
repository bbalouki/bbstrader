from enum import Enum
from typing import NamedTuple, Optional

import numpy as np

try:
    import MetaTrader5 as MT5
except ImportError:
    import bbstrader.compat  # noqa: F401


__all__ = [
    "TIMEFRAMES",
    "RateInfo",
    "RateDtype",
    "TimeFrame",
    "SymbolType",
    "InvalidBroker",
    "GenericFail",
    "InvalidParams",
    "HistoryNotFound",
    "InvalidVersion",
    "AuthFailed",
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

    def __init__(self, code, message):
        super().__init__(code, message)


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
    exception = _ERROR_CODE_TO_EXCEPTION_.get(MT5.last_error()[0])
    if exception is not None:
        raise exception(f"{message or MT5.last_error()[1]}")
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


_ADMIRAL_MARKETS_URL_ = "https://one.justmarkets.link/a/tufvj0xugm/registration/trader"
_JUST_MARKETS_URL_ = "https://one.justmarkets.link/a/tufvj0xugm/registration/trader"
_FTMO_URL_ = "https://trader.ftmo.com/?affiliates=JGmeuQqepAZLMcdOEQRp"

INIT_MSG = (
    f"\n* Check your internet connection\n"
    f"* Make sure MT5 is installed and active\n"
    f"* Looking for a boker? See [{_ADMIRAL_MARKETS_URL_}] "
    f"or [{_JUST_MARKETS_URL_}]\n"
    f"* Looking for a prop firm? See [{_FTMO_URL_}]\n"
)
