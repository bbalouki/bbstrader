from datetime import datetime


class Event(object):
    """
    Event is base class providing an interface for all subsequent
    (inherited) events, that will trigger further events in the
    trading infrastructure.
    """
    ...


class MarketEvent(Event):
    """
    Handles the event of receiving a new market update with
    corresponding bars.
    """

    def __init__(self):
        """
        Initialises the MarketEvent.
        """
        self.type = 'MARKET'


class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.
    This is received by a Portfolio object and acted upon.
    """

    def __init__(self,
                 strategy_id: int,
                 symbol: str,
                 datetime: datetime,
                 signal_type: str,
                 quantity: int | float = 100,
                 strength: int | float = 1.0
                 ):
        """
        Initialises the SignalEvent.

        Args:
            strategy_id (int): The unique identifier for the strategy that
                generated the signal.

            symbol (str): The ticker symbol, e.g. 'GOOG'.
            datetime (datetime): The timestamp at which the signal was generated.
            signal_type (str): 'LONG' or 'SHORT'.
            quantity (int | float): An optional integer (or float) representing the order size.
            strength (int | float): An adjustment factor "suggestion" used to scale
                quantity at the portfolio level. Useful for pairs strategies.
        """
        self.type = 'SIGNAL'
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.quantity = quantity
        self.strength = strength


class OrderEvent(Event):
    """
    Handles the event of sending an Order to an execution system.
    The order contains a symbol (e.g. GOOG), a type (market or limit),
    quantity and a direction.
    """

    def __init__(self,
                 symbol: str,
                 order_type: str,
                 quantity: int | float,
                 direction: str
                 ):
        """
        Initialises the order type, setting whether it is
        a Market order ('MKT') or Limit order ('LMT'), has
        a quantity (integral or float) and its direction ('BUY' or 'SELL').

        Args:
            symbol (str): The instrument to trade.
            order_type (str): 'MKT' or 'LMT' for Market or Limit. 
            quantity (int | float): Non-negative number for quantity.            
            direction (str): 'BUY' or 'SELL' for long or short.
        """
        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction

        def print_order(self):
            """
            Outputs the values within the Order.
            """
            print(
                "Order: Symbol=%s, Type=%s, Quantity=%s, Direction=%s" %
                (self.symbol, self.order_type, self.quantity, self.direction)
            )


class FillEvent(Event):
    """
    Encapsulates the notion of a Filled Order, as returned
    from a brokerage. Stores the quantity of an instrument
    actually filled and at what price. In addition, stores
    the commission of the trade from the brokerage.
    """

    def __init__(self,
                 timeindex: datetime,
                 symbol: str,
                 exchange: str,
                 quantity: int | float,
                 direction: str,
                 fill_cost: int | float,
                 commission: float | None = None
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
            direction (str): The direction of fill ('BUY' or 'SELL')
            fill_cost (int | float): The holdings value in dollars.
            commission (float | None): An optional commission sent from IB.
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
