# Portfolio
This  describes a Portfolio object that keeps track of the positions within a portfolio
and generates orders of a fixed quantity of stock based on signals. More sophisticated portfolio objects could include risk management and position sizing tools (such as the `Kelly Criterion`).

The portfolio order management system is possibly the most complex component of an eventdriven backtester.  Its role is to keep track of all current market positions as well as the market value of the positions (known as the "holdings"). This is simply an estimate of the liquidation value of the position and is derived in part from the data handling facility of the backtester.

In addition to the positions and holdings management the portfolio must also be aware of risk factors and position sizing techniques in order to optimise orders that are sent to a brokerage or other form of market access.

Unfortunately, Portfolio and `Order Management Systems (OMS)` can become rather complex!
So let's keep the Portfolio object relatively straightforward  anf improve it foward.

Continuing in the vein of the Event class hierarchy a Portfolio object must be able to
handle `SignalEvent` objects, generate `OrderEvent` objects and interpret `FillEvent` objects to update positions. Thus it is no surprise that the Portfolio objects are often the largest component of event-driven systems, in terms of lines of code (LOC).

The initialisation of the Portfolio object requires access to the bars `DataHandler`, the  `Event Queue`, a start datetime stamp and an initial capital value (defaulting to `100,000 USD`) and others parameter based on the `Strategy` requirement.

The `Portfolio` is designed to handle position sizing and current holdings, but will carry out trading orders in a "dumb" manner by simply sending them directly to the brokerage with a predetermined fixed quantity size, irrespective of cash held. These are all unrealistic assumptions, but they help to outline how a portfolio order management system (OMS) functions in an eventdriven fashion.

The portfolio contains the `all_positions` and `current_positions` members. The former
stores a list of all previous positions recorded at the timestamp of a market data event. A position is simply the quantity of the asset held. Negative positions mean the asset has been shorted.

The latter current_positions dictionary stores contains the current positions for the last market bar update, for each symbol.

In addition to the positions data the portfolio stores `holdings`, which describe the current market value of the positions held. "Current market value" in this instance means the closing price obtained from the current market bar, which is clearly an approximation, but is reasonable enough for the time being. `all_holdings` stores the historical list of all symbol holdings, while current_holdings stores the most up to date dictionary of all symbol holdings values.
