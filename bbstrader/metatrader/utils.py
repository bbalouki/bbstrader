import MetaTrader5 as MT5

__all__ = [
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
    "TIMEFRAMES",
    "ADMIRAL_MARKETS_URL",
    "JUST_MARKETS_URL",
    "INIT_MSG"
]
ADMIRAL_MARKETS_URL = "https://cabinet.a-partnership.com/visit/?bta=35537&brand=admiralmarkets"
ADMIRAL_MARKETS_PRODUCTS = "Stock, ETFs, Indices, Commodity, Futures and Forex"
JUST_MARKETS_URL = "https://one.justmarkets.link/a/tufvj0xugm/registration/trader"
INIT_MSG = (
    f"\n* Ensure you have MT5 terminal install on you machine \n"
    f"* Ensure you have an active MT5 demo Account \n"
    f"* Ensure you have a good internet connexion \n"
    f"* If you want to trade {ADMIRAL_MARKETS_PRODUCTS} see {ADMIRAL_MARKETS_URL}\n"
    f"* If you want to trade Crypto and or Forex see {JUST_MARKETS_URL}"
)

TIMEFRAMES = {
    '1m':  MT5.TIMEFRAME_M1,
    '2m':  MT5.TIMEFRAME_M2,
    '3m':  MT5.TIMEFRAME_M3,
    '4m':  MT5.TIMEFRAME_M4,
    '5m':  MT5.TIMEFRAME_M5,
    '6m':  MT5.TIMEFRAME_M6,
    '10m': MT5.TIMEFRAME_M10,
    '12m': MT5.TIMEFRAME_M12,
    '15m': MT5.TIMEFRAME_M15,
    '20m': MT5.TIMEFRAME_M20,
    '30m': MT5.TIMEFRAME_M30,
    '1h':  MT5.TIMEFRAME_H1,
    '2h':  MT5.TIMEFRAME_H2,
    '3h':  MT5.TIMEFRAME_H3,
    '4h':  MT5.TIMEFRAME_H4,
    '6h':  MT5.TIMEFRAME_H6,
    '8h':  MT5.TIMEFRAME_H8,
    '12h': MT5.TIMEFRAME_H12,
    'D1':  MT5.TIMEFRAME_D1,
    'W1':  MT5.TIMEFRAME_W1,
    'MN1': MT5.TIMEFRAME_MN1
}


class MT5TerminalError(Exception):
    """Base exception class for trading-related errors."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message

    def __str__(self) -> str:
        if self.message is None:
            return f"{self.__class__.__name__}"
        else:
            return f"{self.__class__.__name__}, {self.message}"

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


# Dictionary to map error codes to exception classes
_ERROR_CODE_TO_EXCEPTION = {
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
}


def raise_mt5_error(message=None):
    """Raises an exception based on the given error code.

    Args:
        message: An optional custom error message.

    Raises:
        MT5TerminalError: A specific exception based on the error code.
    """
    error = _ERROR_CODE_TO_EXCEPTION.get(MT5.last_error()[0])
    raise Exception(f"{error(None)} {message or MT5.last_error()[1]}")


_ORDER_FILLING_TYPE = "https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type_filling"
_ORDER_TYPE = "https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type"
_POSITION_IDENTIFIER = "https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_property_integer"
_FIFO_RULE = "https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_info_integer"
_TRADE_RETCODE_MESSAGES = {
    10004: "Requote: The price has changed, please try again.",
    10006: "Request rejected.",
    10007: "Request canceled by trader.",
    10008: "Order placed.",
    10009: "Request completed.",
    10010: "Only part of the request was completed.",
    10011: "Request processing error.",
    10012: "Request canceled by timeout.",
    10013: "Invalid request.",
    10014: "Invalid volume in the request.",
    10015: "Invalid price in the request.",
    10016: "Invalid stops in the request.",
    10017: "Trade is disabled.",
    10018: "Market is closed.",
    10019: "Insufficient funds to complete the request.",
    10020: "Prices changed.",
    10021: "No quotes to process the request.",
    10022: "Invalid order expiration date in the request.",
    10023: "Order state changed.",
    10024: "Too many requests, please try again later.",
    10025: "No changes in request.",
    10026: "Autotrading disabled by server.",
    10027: "Autotrading disabled by client terminal.",
    10028: "Request locked for processing.",
    10029: "Order or position frozen.",
    10030: "Invalid order filling type: see" + " "+_ORDER_FILLING_TYPE,
    10031: "No connection with the trade server.",
    10032: "Operation allowed only for live accounts.",
    10033: "The number of pending orders has reached the limit.",
    10034: "Order/position volume limit for the symbol reached.",
    10035: "Incorrect or prohibited order type: see" + " " + _ORDER_TYPE,
    10036: "Position with the specified ID has already been closed: see"+" "+_POSITION_IDENTIFIER,
    10038: "Close volume exceeds the current position volume.",
    10039: "A close order already exists for this position.",
    10040: "Maximum number of open positions reached.",
    10041: "Pending order activation rejected, order canceled.",
    10042: "Only long positions are allowed for this symbol.",
    10043: "Only short positions are allowed for this symbol.",
    10044: "Only position closing is allowed for this symbol.",
    10045: "Position closing allowed only by FIFO rule: see" + " " + _FIFO_RULE,
    10046: "Opposite positions on this symbol are disabled."
}


def trade_retcode_message(code, display=False):
    """
    Retrieves a user-friendly message corresponding to a given trade return code.

    Args:
        code (int): The trade return code to look up.
        display (bool, optional): Whether to print the message to the console. Defaults to False.

    Returns:
        str: The message associated with the provided trade return code. If the code is not found,
             it returns "Unknown trade error.".
    """
    message = _TRADE_RETCODE_MESSAGES.get(code, "Unknown trade error.")
    if display:
        print(message)
    return message
