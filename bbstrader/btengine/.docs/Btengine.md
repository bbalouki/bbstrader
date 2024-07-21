# Events
The first component to be discussed is the Event class hierarchy. In this infrastructure there are four types of events which allow communication between all components via an event queue. They are a `MarketEvent`, `SignalEvent`, `OrderEvent` and `FillEvent`.

## `Event()`
The parent class in the hierarchy is called Event. It is a base class and does not provide any functionality or specific interface. Since in many implementations the Event objects will likely develop greater complexity, it is thus being "future-proofed" by creating a class hierarchy. The Event class is simply a way to ensure that all events have a common interface and can be handled in a consistent manner.

## `MarketEvent()`
Market Events are triggered when the outer while loop of the backtesting system begins a new `"heartbeat"`. It occurs when the `DataHandler` object receives a new update of market data for any symbols which are currently being tracked. It is used to `trigger the Strategy object` generating new `trading signals`. The event object simply contains an identification that it is a market event, with no other structure.

## `SignalEvent()`
The `Strategy object` utilises market data to create new `SignalEvents`. The SignalEvent contains a `strategy ID`, a `ticker symbol`, a `timestamp` for when it was generated, a `direction` (long or short) and a `"strength"` indicator (this is useful for mean reversion strategies) and the `quantiy` to buy or sell. The `SignalEvents` are utilised by the `Portfolio object` as advice for how to trade.

## `OrderEvent()`
When a Portfolio object receives `SignalEvents` it assesses them in the wider context of the portfolio, in terms of risk and position sizing. This ultimately leads to `OrderEvents` that will be sent to an `ExecutionHandler`.

The `OrderEvents` is slightly more complex than a `SignalEvents` since it contains a quantity field in addition to the aforementioned properties of SignalEvent. The quantity is determined by the Portfolio constraints. In addition the OrderEvent has a `print_order()` method, used to output the information to the console if necessary.

## `FillEvent()`
When an `ExecutionHandler` receives an `OrderEvent` it must transact the order. Once an order has been transacted it generates a `FillEvent`, which describes the cost of purchase or sale as well as the transaction costs, such as fees or slippage.

The `FillEvent` is the Event with the greatest complexity. It contains a `timestamp` for when an order was filled, the `symbol` of the order and the `exchange` it was executed on, the `quantity` of shares transacted, the `actual price of the purchase` and the `commission incurred`.

The commission is calculated using the Interactive Brokers commissions. For US API orders this commission is `1.30 USD` minimum per order, with a flat rate of either 0.013 USD or 0.08 USD per share depending upon whether the trade size is below or above `500 units` of stock.


# Data Handler
One of the goals of an event-driven trading system is to minimise duplication of code between the backtesting element and the live execution element. Ideally it would be optimal to utilise the same signal generation methodology and portfolio management components for both historical testing and live trading. In order for this to work the Strategy object which generates the Signals, and the Portfolio object which provides Orders based on them, must utilise an identical interface to a market feed for both historic and live running.

This motivates the concept of a class hierarchy based on a `DataHandler object`, which givesall subclasses an interface for providing market data to the remaining components within thesystem. In this way any subclass data handler can be "swapped out", without affecting strategy or portfolio calculation.

Specific example subclasses could include `HistoricCSVDataHandler`, `EODHDDataHandler`, `SecuritiesMasterDataHandler`, `IBMarketFeedDataHandler` etc. We are only going to consider the creation of a historic CSV data handler, which will load intraday CSV data for equities in an `Open-Low-High-Close-Adj Close-Volume-Returns` set of bars. This can then be used to `"drip feed"` on a bar-by-bar basis the data into the Strategy and Portfolio classes on every heartbeat of the system, thus avoiding lookahead bias.

# Strategy
A `Strategy()` object encapsulates all calculation on market data that generate advisory signals to a Portfolio object. Thus all of the "strategy logic" resides within this class. We opted to separate out the Strategy and Portfolio objects for this backtester, since we believe this is more amenable to the situation of multiple strategies feeding "ideas" to a larger Portfolio, which then can handle its own risk (such as sector allocation, leverage). In higher frequency trading, the strategy and portfolio concepts will be tightly coupled and extremely hardware dependent.

At this stage in the event-driven backtester development there is no concept of an indicator or filter, such as those found in technical trading. These are also good candidates for creating a class hierarchy.

The strategy hierarchy is relatively simple as it consists of an abstract base class with a single pure virtual method for generating SignalEvent objects. In order to create the Strategy hierarchy it is necessary to import NumPy, pandas, the Queue object (which has become queue in Python 3), abstract base class tools and the SignalEvent

# Portfolio
This  describes a `Portfolio()` object that keeps track of the positions within a portfolio
and generates orders of a fixed quantity of stock based on signals. More sophisticated portfolio objects could include risk management and position sizing tools (such as the `Kelly Criterion`).

The portfolio order management system is possibly the most complex component of an eventdriven backtester.  Its role is to keep track of all current market positions as well as the market value of the positions (known as the "holdings"). This is simply an estimate of the liquidation value of the position and is derived in part from the data handling facility of the backtester.

In addition to the positions and holdings management the portfolio must also be aware of risk factors and position sizing techniques in order to optimise orders that are sent to a brokerage or other form of market access.

Unfortunately, Portfolio and `Order Management Systems (OMS)` can become rather complex!
So let's keep the Portfolio object relatively straightforward  anf improve it foward.

Continuing in the vein of the Event class hierarchy a Portfolio object must be able to handle `SignalEvent` objects, generate `OrderEvent` objects and interpret `FillEvent` objects to update positions. Thus it is no surprise that the Portfolio objects are often the largest component of event-driven systems, in terms of lines of code (LOC).

The initialisation of the Portfolio object requires access to the bars `DataHandler`, the  `Event Queue`, a start datetime stamp and an initial capital value (defaulting to `100,000 USD`) and others parameter based on the `Strategy` requirement.

The `Portfolio` is designed to handle position sizing and current holdings, but will carry out trading orders in a "dumb" manner by simply sending them directly to the brokerage with a predetermined fixed quantity size, irrespective of cash held. These are all unrealistic assumptions, but they help to outline how a portfolio order management system (OMS) functions in an eventdriven fashion.

The portfolio contains the `all_positions` and `current_positions` members. The former stores a list of all previous positions recorded at the timestamp of a market data event. A position is simply the quantity of the asset held. Negative positions mean the asset has been shorted.

The latter current_positions dictionary stores contains the current positions for the last market bar update, for each symbol.

In addition to the positions data the portfolio stores `holdings`, which describe the current market value of the positions held. "Current market value" in this instance means the closing price obtained from the current market bar, which is clearly an approximation, but is reasonable enough for the time being. `all_holdings` stores the historical list of all symbol holdings, while current_holdings stores the most up to date dictionary of all symbol holdings values.

# Execution Handler

`ExecutionHandler()` is a class hierarchy that will represent a simulated order handling mechanism and ultimately tie into a brokerage or other means of market connectivity.

The ExecutionHandler described here is exceedingly simple, since it fills all orders at the
current market price. This is highly unrealistic, but serves as a good baseline for improvement.

# Backtest

The `Backtest()` object encapsulates the event-handling logic and essentially ties together all of the other classes.

The Backtest object is designed to carry out a nested while-loop event-driven system in order to handle the events placed on the Event Queue object. The outer while-loop is known as the "heartbeat loop" and decides the temporal resolution of the backtesting system. In a live environment this value will be a positive number, such as 600 seconds (every ten minutes). Thus the market data and positions will only be updated on this timeframe.

For the backtester described here the "heartbeat" can be set to zero, irrespective of the strategy frequency, since the data is already available by virtue of the fact it is historical! We can run the backtest at whatever speed we like, since the event-driven system is agnostic to when the data became available, so long as it has an associated timestamp. 

The inner while-loop actually processes the signals and sends them to the correct component depending upon the event type. Thus the Event Queue is continually being populated and depopulated with events. This is what it means for a system to be event-driven.

The initialisation of the Backtest object requires the CSV directory, the full symbol list of traded symbols, the initial capital, the heartbeat time in milliseconds, the start datetime stamp of the backtest as well as the DataHandler, ExecutionHandler, Portfolio and Strategy objects.

A Queue is used to hold the events. The signals, orders and fills are counted.
For a MarketEvent, the Strategy object is told to recalculate new signals, while the Portfolio object is told to reindex the time. If a SignalEvent object is received the Portfolio is told to handle the new signal and convert it into a set of OrderEvents, if appropriate. If an OrderEvent is received the ExecutionHandler is sent the order to be transmitted to the broker (if in a real trading setting). Finally, if a FillEvent is received, the Portfolio will update itself to be aware of the new positions.
