"""Broker-neutral execution abstraction.

A thin ``Broker`` interface decouples strategy/execution logic from any specific
venue, so the same strategy can target MT5 today and IBKR / a crypto exchange
later by swapping the adapter. ``PaperBroker`` is a fully in-memory simulated
adapter useful for paper trading, tests, and as the reference implementation
of the contract. Live adapters (e.g. an MT5 adapter over
:mod:`bbstrader.metatrader`) implement the same methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

__all__ = [
    "OrderSide",
    "OrderType",
    "BrokerOrder",
    "BrokerPosition",
    "AccountInfo",
    "Broker",
    "PaperBroker",
]


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


@dataclass
class BrokerOrder:
    """A venue-neutral order request and its assigned id once submitted."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    id: Optional[int] = None


@dataclass
class BrokerPosition:
    """An open position: signed quantity and volume-weighted average price."""

    symbol: str
    quantity: float
    avg_price: float


@dataclass
class AccountInfo:
    """A snapshot of account cash, mark-to-market equity and currency."""

    cash: float
    equity: float
    currency: str = "USD"


class Broker(ABC):
    """The execution contract every venue adapter implements."""

    @abstractmethod
    def connect(self) -> bool:
        """Open the connection to the venue; return True on success."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection to the venue."""
        ...

    @abstractmethod
    def account(self) -> AccountInfo:
        """Return the current account snapshot (cash, equity, currency)."""
        ...

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """Return the latest market price for ``symbol``.

        Args:
            symbol (str): The instrument to price.

        Returns:
            float: The latest price.
        """
        ...

    @abstractmethod
    def submit_order(self, order: BrokerOrder) -> BrokerOrder:
        """Submit ``order`` to the venue and return it with venue fields set.

        Args:
            order (BrokerOrder): The order to submit.

        Returns:
            BrokerOrder: The submitted order, populated with its assigned id.
        """
        ...

    @abstractmethod
    def positions(self) -> List[BrokerPosition]:
        """Return the currently open positions."""
        ...

    @abstractmethod
    def orders(self) -> List[BrokerOrder]:
        """Return the currently open (resting) orders."""
        ...


class PaperBroker(Broker):
    """An in-memory simulated broker with immediate market fills.

    Maintains cash, positions (volume-weighted average price) and an order log.
    Prices are set with :meth:`set_price`; market orders fill at the current
    price, limit/stop orders rest until :meth:`set_price` crosses their level.
    """

    def __init__(self, cash: float = 100000.0, currency: str = "USD") -> None:
        """Initialise the paper broker with starting cash.

        Args:
            cash (float): The opening cash balance.
            currency (str): The account currency code.
        """
        self._cash = float(cash)
        self.currency = currency
        self._positions: Dict[str, BrokerPosition] = {}
        self._orders: List[BrokerOrder] = []
        self._open_orders: List[BrokerOrder] = []
        self._prices: Dict[str, float] = {}
        self._next_id = 1
        self._connected = False
        self.realized_pnl = 0.0

    # connection - #
    def connect(self) -> bool:
        """Mark the broker connected; always succeeds for the paper broker."""
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Mark the broker disconnected."""
        self._connected = False

    # market data  #
    def set_price(self, symbol: str, price: float) -> None:
        """Update the market price and trigger any resting orders it crosses."""
        self._prices[symbol] = float(price)
        self._check_open_orders(symbol)

    def get_price(self, symbol: str) -> float:
        """Return the last price set for ``symbol``.

        Args:
            symbol (str): The instrument to price.

        Returns:
            float: The most recently set price.

        Raises:
            KeyError: If no price has been set for ``symbol``.
        """
        if symbol not in self._prices:
            raise KeyError(f"No price set for {symbol}.")
        return self._prices[symbol]

    # accounting
    def account(self) -> AccountInfo:
        """Return the account snapshot (cash, mark-to-market equity, currency)."""
        return AccountInfo(
            cash=self._cash, equity=self.equity(), currency=self.currency
        )

    def equity(self) -> float:
        """Return cash plus the mark-to-market value of all open positions."""
        market_value = sum(
            pos.quantity * self._prices.get(sym, pos.avg_price)
            for sym, pos in self._positions.items()
        )
        return self._cash + market_value

    def positions(self) -> List[BrokerPosition]:
        """Return the open (non-zero quantity) positions."""
        return [p for p in self._positions.values() if p.quantity != 0]

    def orders(self) -> List[BrokerOrder]:
        """Return the resting (not yet filled) orders."""
        return list(self._open_orders)

    # order handling
    def submit_order(self, order: BrokerOrder) -> BrokerOrder:
        """Submit an order, filling market orders immediately at the set price.

        Market orders fill at ``order.price`` or the current market price;
        limit/stop orders rest and fill when :meth:`set_price` crosses them.

        Args:
            order (BrokerOrder): The order to submit. Its ``id`` is assigned here.

        Returns:
            BrokerOrder: The same order with its assigned ``id``.

        Raises:
            ValueError: If ``order.quantity`` is not positive.
        """
        if order.quantity <= 0:
            raise ValueError("order quantity must be positive.")
        order.id = self._next_id
        self._next_id += 1
        self._orders.append(order)
        if order.order_type is OrderType.MARKET:
            price = order.price or self.get_price(order.symbol)
            self._fill(order, price)
        else:
            self._open_orders.append(order)
            # A resting order may fill immediately if already crossed.
            if order.symbol in self._prices:
                self._check_open_orders(order.symbol)
        return order

    def _check_open_orders(self, symbol: str) -> None:
        """Fill any resting orders on ``symbol`` whose trigger the price crosses.

        Args:
            symbol (str): The instrument whose latest price just changed.
        """
        price = self._prices[symbol]
        for order in list(self._open_orders):
            if order.symbol != symbol or order.price is None:
                continue
            crossed = (
                (
                    order.order_type is OrderType.LIMIT
                    and order.side is OrderSide.BUY
                    and price <= order.price
                )
                or (
                    order.order_type is OrderType.LIMIT
                    and order.side is OrderSide.SELL
                    and price >= order.price
                )
                or (
                    order.order_type is OrderType.STOP
                    and order.side is OrderSide.BUY
                    and price >= order.price
                )
                or (
                    order.order_type is OrderType.STOP
                    and order.side is OrderSide.SELL
                    and price <= order.price
                )
            )
            if crossed:
                self._fill(order, order.price)
                self._open_orders.remove(order)

    def _fill(self, order: BrokerOrder, price: float) -> None:
        """Apply a fill: update cash, position basis and realized P&L.

        Adds to or reduces the symbol's position, recomputing the weighted
        average entry price and booking realized P&L when a position is reduced
        or flipped through zero.

        Args:
            order (BrokerOrder): The order being filled.
            price (float): The fill price.
        """
        signed = order.quantity if order.side is OrderSide.BUY else -order.quantity
        pos = self._positions.get(order.symbol)
        if pos is None or pos.quantity == 0:
            self._positions[order.symbol] = BrokerPosition(order.symbol, signed, price)
        else:
            new_qty = pos.quantity + signed
            if pos.quantity * signed > 0:
                # Adding to the position: update the weighted average price.
                pos.avg_price = (
                    pos.avg_price * pos.quantity + price * signed
                ) / new_qty
            elif new_qty != 0 and pos.quantity * new_qty < 0:
                # Flipped through zero: realize the closed leg, reset basis.
                self.realized_pnl += (price - pos.avg_price) * pos.quantity
                pos.avg_price = price
            else:
                # Reducing/closing: realize P&L on the closed quantity.
                self.realized_pnl += (price - pos.avg_price) * (-signed)
            pos.quantity = new_qty
        self._cash -= signed * price
