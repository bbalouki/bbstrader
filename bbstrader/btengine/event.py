from datetime import datetime
from typing import Literal

__all__ = [
    "Event",
    "MarketEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent"
]


class Event(object):
    """
    Event is base class providing an interface for all subsequent
    (inherited) events, that will trigger further events in the
    trading infrastructure.
    Since in many implementations the Event objects will likely develop greater 
    complexity, it is thus being "future-proofed" by creating a class hierarchy. 
    The Event class is simply a way to ensure that all events have a common interface
    and can be handled in a consistent manner.
    """
    ...


class MarketEvent(Event):
    """
    Market Events are triggered when the outer while loop of the backtesting 
    system begins a new `"heartbeat"`. It occurs when the `DataHandler` object 
    receives a new update of market data for any symbols which are currently 
    being tracked. It is used to `trigger the Strategy object` generating 
    new `trading signals`. The event object simply contains an identification 
    that it is a market event, with no other structure.
    """

    def __init__(self):
        """
        Initialises the MarketEvent.
        """
        self.type = 'MARKET'


class SignalEvent(Event):
    """
    The `Strategy object` utilises market data to create new `SignalEvents`. 
    The SignalEvent contains a `strategy ID`, a `ticker symbol`, a `timestamp` 
    for when it was generated, a `direction` (long or short) and a `"strength"` 
    indicator (this is useful for mean reversion strategies) and the `quantiy` 
    to buy or sell. The `SignalEvents` are utilised by the `Portfolio object` 
    as advice for how to trade.
    """

    def __init__(self,
                 strategy_id: int,
                 symbol: str,
                 datetime: datetime,
                 signal_type: Literal['LONG', 'SHORT', 'EXIT'],
                 quantity: int | float = 100,
                 strength: int | float = 1.0,
                 price: int | float = None,
                 stoplimit: int | float = None
                 ):
        """
        Initialises the SignalEvent.

        Args:
            strategy_id (int): The unique identifier for the strategy that
                generated the signal.

            symbol (str): The ticker symbol, e.g. 'GOOG'.
            datetime (datetime): The timestamp at which the signal was generated.
            signal_type (str): 'LONG' or 'SHORT' or 'EXIT'.
            quantity (int | float): An optional integer (or float) representing the order size.
            strength (int | float): An adjustment factor "suggestion" used to scale
                quantity at the portfolio level. Useful for pairs strategies.
            price (int | float): An optional price to be used when the signal is generated.
            stoplimit (int | float): An optional stop-limit price for the signal
        """
        self.type = 'SIGNAL'
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.quantity = quantity
        self.strength = strength
        self.price = price
        self.stoplimit = stoplimit


class OrderEvent(Event):
    """
    When a Portfolio object receives `SignalEvents` it assesses them 
    in the wider context of the portfolio, in terms of risk and position sizing. 
    This ultimately leads to `OrderEvents` that will be sent to an `ExecutionHandler`.

    The `OrderEvents` is slightly more complex than a `SignalEvents` since 
    it contains a quantity field in addition to the aforementioned properties 
    of SignalEvent. The quantity is determined by the Portfolio constraints. 
    In addition the OrderEvent has a `print_order()` method, used to output the 
    information to the console if necessary.
    """

    def __init__(self,
                 symbol: str,
                 order_type: Literal['MKT', 'LMT', 'STP', 'STPLMT'],
                 quantity: int | float,
                 direction: Literal['BUY', 'SELL'],
                 price: int | float = None,
                 signal: str = None
                 ):
        """
        Initialises the order type, setting whether it is
        a Market order ('MKT') or Limit order ('LMT'), or Stop order ('STP').
        a quantity (integral or float) and its direction ('BUY' or 'SELL').

        Args:
            symbol (str): The instrument to trade.
            order_type (str): 'MKT' or 'LMT' for Market or Limit. 
            quantity (int | float): Non-negative number for quantity.            
            direction (str): 'BUY' or 'SELL' for long or short.
            price (int | float): The price at which to order.
            signal (str): The signal that generated the order.
        """
        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction
        self.price = price
        self.signal = signal

        def print_order(self):
            """
            Outputs the values within the Order.
            """
            print(
                "Order: Symbol=%s, Type=%s, Quantity=%s, Direction=%s, Price=%s" %
                (self.symbol, self.order_type, self.quantity, self.direction, self.price)
            )


class FillEvent(Event):
    """
    When an `ExecutionHandler` receives an `OrderEvent` it must transact the order.
    Once an order has been transacted it generates a `FillEvent`, which describes 
    the cost of purchase or sale as well as the transaction costs, such as fees 
    or slippage.

    The `FillEvent` is the Event with the greatest complexity. 
    It contains a `timestamp` for when an order was filled, the `symbol` 
    of the order and the `exchange` it was executed on, the `quantity` 
    of shares transacted, the `actual price of the purchase` and the `commission 
    incurred`.

    The commission is calculated using the Interactive Brokers commissions. 
    For US API orders this commission is `1.30 USD` minimum per order, with a flat 
    rate of either 0.013 USD or 0.08 USD per share depending upon whether 
    the trade size is below or above `500 units` of stock.
    """

    def __init__(self,
                 timeindex: datetime,
                 symbol: str,
                 exchange: str,
                 quantity: int | float,
                 direction: Literal['BUY', 'SELL'],
                 fill_cost: int | float | None,
                 commission: float | None = None,
                 order: str = None
                 ):
        """
        Initialises the FillEvent object. Sets the symbol, exchange,
        quantity, direction, cost of fill and an optional
        commission.

        If commission is not provided, the Fill object will
        calculate it based on the trade size and Interactive
        Brokers fees.

        Args:
            timeindex (datetime): The bar-resolution when the order was filled.
            symbol (str): The instrument which was filled.
            exchange (str): The exchange where the order was filled.
            quantity (int | float): The filled quantity.
            direction (str): The direction of fill `('LONG', 'SHORT', 'EXIT')`
            fill_cost (int | float): Price of the shares when filled.
            commission (float | None): An optional commission sent from IB.
            order (str): The order that this fill is related
        """
        self.type = 'FILL'
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        # Calculate commission
        if commission is None:
            self.commission = self.calculate_ib_commission()
        else:
            self.commission = commission
        self.order = order

    def calculate_ib_commission(self):
        """
        Calculates the fees of trading based on an Interactive
        Brokers fee structure for API, in USD.
        This does not include exchange or ECN fees.
        Based on "US API Directed Orders":
        https://www.interactivebrokers.com/en/index.php?f=commission&p=stocks2
        """
        full_cost = 1.3
        if self.quantity <= 500:
            full_cost = max(1.3, 0.013 * self.quantity)
        else:
            full_cost = max(1.3, 0.008 * self.quantity)
        return full_cost
