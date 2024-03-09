
# MetaTrader 5 Account Management Class Documentation

This documentation covers the `Account` class designed for interacting with the MetaTrader 5 (MT5) trading platform. The `Account` class provides functionalities to access and manage trading account information, terminal status, financial instrument details, active orders, open positions, and trading history.

## Dependencies
- `datetime`: For handling dates and times.
- `MetaTrader5 (mt5)`: The MT5 Python package for integration with the MetaTrader 5 terminal.
- `pandas (pd)`: For data manipulation and analysis, particularly with structured data.

## Class: Account

### Description
The `Account` class is utilized to retrieve information about the current trading account or a specific account. It enables interaction with the MT5 terminal to manage account details, execute trades, and analyze financial data.

### Initialization
Upon instantiation, the class attempts to initialize a connection with the MT5 terminal. If the initialization fails, it prints the error code and exits the application.

#### Syntax
```python
def __init__(self):
```

### Methods

#### `get_account_info`
Retrieves information about the current trading account or a specified account.

##### Parameters
- `account (int)`: The MT5 account number.
- `password (str)`: The MT5 account password.
- `server (str)`: The MT5 account server (e.g., broker or terminal server).

##### Returns
- A namedtuple structure containing account information, or `None` in case of an error.

#### `print_account_info`
Prints the current account information in a structured format using pandas DataFrame.

##### Syntax
```python
def print_account_info(self):
```

#### `get_terminal_info`
Fetches information about the connected MetaTrader 5 client terminal's status and settings.

##### Returns
- A pandas DataFrame containing terminal information, or `None` in case of an error.

#### `get_symbols`
Retrieves all financial instruments available in the MetaTrader 5 terminal.

##### Returns
- A list of symbols, or `None` in case of an error.

#### `show_symbol_info`
Prints detailed information about a specific symbol.

##### Parameters
- `symbol (str)`: The name of the symbol.

##### Returns
- A pandas DataFrame containing symbol properties, or `None` in case of an error.

#### `get_symbol_info`
Gets detailed properties of a specified symbol.

##### Parameters
- `symbol (str)`: The name of the symbol.

##### Returns
- A namedtuple containing symbol properties, or `None` in case of an error.

#### `get_orders`
Retrieves active orders from the trading account.

##### Returns
- A pandas DataFrame containing details of active orders, or `None` in case of an error.

#### `get_positions`
Fetches open positions from the trading account.

##### Returns
- A pandas DataFrame containing open positions details, or `None` in case of an error.

#### `get_trade_history`
Gets deals from the trading history within a specified time interval.

##### Parameters
- `date_from (datetime)`: Start date for the trading history.
- `date_to (datetime)`: End date for the trading history.
- `group (str)`: Specifies the group of deals (optional).
- `save (bool)`: If `True`, saves the trading history to a CSV file.

##### Returns
- A pandas DataFrame containing the trading history, or `None` in case of an error.

### Examples
```python
# Instantiating the Account class
account = Account()

# Getting account information
account_info = account.get_account_info()

# Printing account information
account.print_account_info()

# Getting terminal information
terminal_info = account.get_terminal_info()

# Retrieving and printing symbol information
symbol_info = account.show_symbol_info('EURUSD')

# Getting active orders
orders = account.get_orders()

# Fetching open positions
positions = account.get_positions()

# Accessing trade history
from_date = datetime(2020, 1, 1)
to_date = datetime.now()
trade_history = account.get_trade_history(from_date, to_date)
```

This class provides a comprehensive interface for managing and analyzing a trading account within the MetaTrader 5 platform, offering a variety of methods to interact with and retrieve information from the terminal.
