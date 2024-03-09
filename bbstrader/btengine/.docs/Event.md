# Events
The first component to be discussed is the Event class hierarchy. In this infrastructure there are four types of events which allow communication between all components via an event queue. They are a `MarketEvent`, `SignalEvent`, `OrderEvent` and `FillEvent`.

## Event()
The parent class in the hierarchy is called Event. It is a base class and does not provide any functionality or specific interface. Since in many implementations the Event objects will likely develop greater complexity, it is thus being "future-proofed" by creating a class hierarchy. The Event class is simply a way to ensure that all events have a common interface and can be handled in a consistent manner.

## MarketEvent()
Market Events are triggered when the outer while loop of the backtesting system begins a new `"heartbeat"`. It occurs when the `DataHandler` object receives a new update of market data for any symbols which are currently being tracked. It is used to `trigger the Strategy object` generating new `trading signals`. The event object simply contains an identification that it is a market event, with no other structure.

## SignalEvent()
The `Strategy object` utilises market data to create new `SignalEvents`. The SignalEvent contains a `strategy ID`, a `ticker symbol`, a `timestamp` for when it was generated, a `direction` (long or short) and a `"strength"` indicator (this is useful for mean reversion strategies) and the `quantiy` to buy or sell. The `SignalEvents` are utilised by the `Portfolio object` as advice for how to trade.

## OrderEvent()
When a Portfolio object receives `SignalEvents` it assesses them in the wider context of the portfolio, in terms of risk and position sizing. This ultimately leads to `OrderEvents` that will be sent to an `ExecutionHandler`.

The `OrderEvents` is slightly more complex than a `SignalEvents` since it contains a quantity field in addition to the aforementioned properties of SignalEvent. The quantity is determined by the Portfolio constraints. In addition the OrderEvent has a `print_order()` method, used to output the information to the console if necessary.

## FillEvent
When an `ExecutionHandler` receives an `OrderEvent` it must transact the order. Once an order has been transacted it generates a `FillEvent`, which describes the cost of purchase or sale as well as the transaction costs, such as fees or slippage.

The `FillEvent` is the Event with the greatest complexity. It contains a `timestamp` for when an order was filled, the `symbol` of the order and the `exchange` it was executed on, the `quantity` of shares transacted, the `actual price of the purchase` and the `commission incurred`.

The commission is calculated using the Interactive Brokers commissions. For US API orders this commission is `1.30 USD` minimum per order, with a flat rate of either 0.013 USD or 0.08 USD per share depending upon whether the trade size is below or above `500 units` of stock.
