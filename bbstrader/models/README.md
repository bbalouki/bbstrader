# Models
The `models` module serves as a framework for implementing various types of financial models. Below are the two currently implemented models:

## `RiskModel`
The `RiskModel` class is an abstract base class for creating risk management strategies in financial markets. It assists in decision-making regarding permissible trades and optimal asset allocation to optimize the risk-reward ratio.

### Methods
`which_trade_allowed(self, returns_val)`: Determines the types of trades permissible under current market conditions based on financial metrics.
`which_quantity_allowed(self)`: Defines the strategy for asset allocation within the portfolio to maintain an optimal risk-reward balance.