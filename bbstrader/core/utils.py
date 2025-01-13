from enum import Enum
from dataclasses import dataclass


class TradeAction(Enum):
    """
    An enumeration class for trade actions.
    """
    BUY = "LONG"
    LONG = "LONG"
    SELL = "SHORT"
    EXIT = "EXIT"
    BMKT = "BMKT"
    SMKT = "SMKT"
    BLMT = "BLMT"
    SLMT = "SLMT"
    BSTP = "BSTP"
    SSTP = "SSTP"
    SHORT = "SHORT"
    BSTPLMT = "BSTPLMT"
    SSTPLMT = "SSTPLMT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    EXIT_STOP = "EXIT_STOP"
    EXIT_LIMIT = "EXIT_LIMIT"
    EXIT_LONG_STOP = "EXIT_LONG_STOP"
    EXIT_LONG_LIMIT = "EXIT_LONG_LIMIT"
    EXIT_SHORT_STOP = "EXIT_SHORT_STOP"
    EXIT_SHORT_LIMIT = "EXIT_SHORT_LIMIT"
    EXIT_LONG_STOP_LIMIT = "EXIT_LONG_STOP_LIMIT"
    EXIT_SHORT_STOP_LIMIT = "EXIT_SHORT_STOP_LIMIT"
    EXIT_PROFITABLES = "EXIT_PROFITABLES"
    EXIT_LOSINGS = "EXIT_LOSINGS"
    EXIT_ALL_POSITIONS = "EXIT_ALL_POSITIONS"
    EXIT_ALL_ORDERS = "EXIT_ALL_ORDERS"

    def __str__(self):
        return self.value


@dataclass()
class TradeSignal:
    """
    A dataclass for storing trading signal.
    """

    id: int
    symbol: str
    action: TradeAction
    price: float = None
    stoplimit: float = None

    def __repr__(self):
        return (
            f"TradeSignal(id={self.id}, symbol='{self.symbol}', "
            f"action='{self.action.value}', price={self.price}, stoplimit={self.stoplimit})"
        )
