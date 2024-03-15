# Trade Class Documentation

The `Trade` class extends the functionalities of the `RiskManagement` class, integrating specific trading operations with risk management strategies for executing trades on the MetaTrader 5 (MT5) platform. It enables the automated execution of trades based on defined parameters, ensuring that risk management protocols are adhered to.

## Dependencies
- Inherits from `RiskManagement` class.
- `os`, `csv`: For file operations, including saving statistics to CSV files.
- `datetime`: For handling dates and times.
- `MetaTrader5 (Mt5)`: The MT5 Python package for integration with the MetaTrader 5 terminal.
- `numpy`: For numerical operations.
- `time`: For introducing delays in execution.

## Class: Trade

### Description
`Trade` provides a structured approach to executing trades while managing risk. It includes functionalities for opening, managing, and closing positions, along with detailed reporting on trading activity and risk parameters.

### Initialization
The constructor initializes the `Trade` class with trading and expert advisor-specific parameters, along with inherited risk management settings.

#### Parameters
- `symbol (str)`: Trading symbol.
- `expert_name (str)`: Name of the expert advisor.
- `expert_id (int)`: Unique identifier for the expert advisor or the strategy.
- `version (str)`: Version of the expert advisor.
- `target (float)`: Daily profit target as a percentage.
- `start_time (str)`: Time when the expert advisor starts trading.
- `finishing_time (str)`: Time after which no new positions are opened.
- `ending_time (str)`: Time after which any open positions are closed.

### Methods

#### `initialize`
Attempts to initialize a connection with the MT5 terminal. If initialization fails, it retries after a delay or shuts down the terminal.

#### `select_symbol`
Selects and ensures the trading symbol is visible in the MT5 terminal.

#### `prepare_symbol`
Checks and prepares the selected symbol for trading.

#### `summary`
Displays a brief summary of the trading program and its parameters.

#### `risk_management`
Shows the current risk management settings and account status.

#### `statistics`
Calculates and displays statistics for the trading session. Optionally saves the statistics to a CSV file.

#### `open_buy_position`
Executes a buy order based on predefined parameters.

#### `open_sell_position`
Executes a sell order based on predefined parameters.

#### `request_result`
Processes the result of a trade request and handles retries for failed attempts.

#### `open_position`
Convenience method to open either a buy or sell position.

#### `get_opened_orders`
Returns a list of all opened orders tickets.

#### `get_opened_positions`
Returns a list of all opened position tickets.

#### `get_buy_positions`, `get_sell_positions`, `get_be_positions`
Retrieve lists of buy, sell, and break-even position tickets, respectively.

#### `get_current_open_positions`
Gets all current open positions for the specified symbol.

#### `get_current_open_orders`
Gets all current open orders for the specified symbol.

#### `get_current_win_trades`
Identifies profitable trades that have reached or exceeded the break-even point.

#### `positive_profit`
Checks if the total profit from open positions exceeds a minimum threshold.

#### `get_current_buys`, `get_current_sells`
Retrieves current open buy and sell positions.

#### `_risk_free`
Determines whether it is safe to take additional trades based on the maximum allowed trades and losses.

#### `check`
Verifies if all conditions are met for taking a new position.

#### `_check`
Performs checks and actions at the end of the trading session or when conditions no longer permit trading.

#### `break_even`
Evaluates positions for setting break-even levels.

#### `set_break_even`
Sets a break-even level for a given position.

#### `_break_even_request`
Sends a request to set a position to break even.

#### `win_trade`
Determines whether a position is profitable based on a threshold.

#### `profit_target`
Checks if the profit target for the session has been reached.

#### `close_position`
Closes a specific position by its ticket number.

#### `close_all_positions`, `close_all_buys`, `close_all_sells`
Convenience methods for closing all positions, all buy positions, or all sell positions.

#### `get_stats`
Generates trading statistics for the session.

#### `sharpe`
Calculates the Sharpe ratio for the trading activity.

#### `days_end`, `trading_time`, `sleep_time`
Utility methods for determining the end of the trading day, whether it is currently within the trading window, and calculating the sleep time until the next trading session.

### Usage
To use the `Trade` class, instantiate it with the required parameters, including the trading symbol, expert advisor settings, and risk management parameters. Then, utilize its methods to execute trades, manage positions, and review trading performance.
```python
import time
from mtrader5.trade import Trade

# Initialize the Trade class with parameters
trade = Trade(
    symbol="EURUSD",              # Symbol to trade
    expert_name="MyExpertAdvisor",# Name of the expert advisor
    expert_id=12345,              # Unique ID for the expert advisor
    version="1.0",                # Version of the expert advisor
    target=5.0,                   # Daily profit target in percentage
    start_time="09:00",           # Start time for trading
    finishing_time="17:00",       # Time to stop opening new positions
    ending_time="17:30",          # Time to close any open positions
    max_risk=2.0,                 # Maximum risk allowed on the account in percentage
    daily_risk=1.0,               # Daily risk allowed in percentage
    max_trades=5,                 # Maximum number of trades per session
    rr=2.0,                       # Risk-reward ratio
    account_leverage=True,        # Use account leverage in calculations
    std_stop=True,                # Use standard deviation for stop loss calculation
    sl=20,                        # Stop loss in points (optional)
    tp=30,                        # Take profit in points (optional)
    be=10                         # Break-even in points (optional)
)

# Example to open a buy position
trade.open_buy_position(mm=True, comment="Opening Buy Position")

# Example to open a sell position
trade.open_sell_position(mm=True, comment="Opening Sell Position")

# Check current open positions
opened_positions = trade.get_opened_positions
if opened_positions is not None:
    print(f"Current open positions: {opened_positions}")

# Close all open positions at the end of the trading session
if trade.days_end():
    trade.close_all_positions(comment="Closing all positions at day's end")

# Print trading session statistics
trade.statistics(save=True, dir="my_trading_stats")

# Sleep until the next trading session if needed (example usage)
sleep_time = trade.sleep_time()
print(f"Sleeping for {sleep_time} minutes until the next trading session.")
time.sleep(sleep_time * 60)
```
Note: This is a simplified example. In a real-world scenario, you would need to handle timing and execution logic based on real-time market data and possibly incorporate additional checks and balances as per your trading strategy requirements.

This class enables the automation of trading strategies on the MT5 platform, incorporating sophisticated risk management and reporting functionalities.