# Rates Class Documentation

The `Rates` class facilitates the retrieval of financial instrument data from the MetaTrader 5 (MT5) terminal. It provides access to historical data for a specified symbol and time frame, including open, high, low, close prices,  volume, and Returns calculated as percentage change.

## Dependencies
- `datetime`: For handling dates and times.
- `MetaTrader5 (Mt5)`: The MT5 package for integration with the MetaTrader 5 terminal.
- `pandas`: For data manipulation and analysis, particularly with structured data.
- `time`: For handling delays during initialization retries.

## Class: Rates

### Description
`Rates` is designed to fetch historical data for financial instruments available in the MT5 terminal. It supports various time frames and allows for data retrieval within a specified range.

### Initialization
The constructor initializes the `Rates` class with parameters for the financial instrument and the desired data range.

#### Parameters
- `symbol (str)`: Name of the financial instrument (e.g., "EURUSD").
- `time_frame (str)`: Time frame for the data retrieval (e.g., "1m", "1h", "D1").
- `start_pos (int)`: Starting index from which data is requested.
- `count (int)`: Number of data points (bars) to retrieve.

### Methods

#### `get_rate_frame`
Returns the retrieved data as a pandas DataFrame.

#### `get_rate`
Fetches the data based on the initialized parameters and returns it as a pandas DataFrame.

#### `get_history`
Retrieves historical data for the specified time range. It can also save the data to a CSV file if required.

#### Properties
Properties provide convenient access to specific data columns:
- `get_time`: Returns the datetime of each bar.
- `get_open`: Returns the opening prices.
- `get_high`: Returns the highest prices.
- `get_low`: Returns the lowest prices.
- `get_close`: Returns the closing prices.
- `get_adj_close`: Returns the adjusted closing prices (same as close prices for this class).
- `get_volume`: Returns the trading volume.

### Usage Example

```python
from datetime import datetime, timedelta
import pytz

# Example usage of the Rates class
symbol = "EURUSD"
time_frame = "1h"
start_pos = 0
count = 100

# Initialize the Rates class
rates = Rates(symbol=symbol, time_frame=time_frame, start_pos=start_pos, count=count)

# Retrieve and display the data frame
rates_data_frame = rates.get_rate_frame()
print(rates_data_frame.head())

# Example to fetch historical data within a specific date range
timezone = pytz.timezone("Etc/UTC")
date_from = datetime(2020, 1, 1, tzinfo=timezone)
date_to = datetime.now(tz=timezone)

# Fetch historical data and save to CSV
rates.get_history(date_from=date_from, date_to=date_to, save=True)
```

This example demonstrates how to initialize the `Rates` class, retrieve historical price data for a specified symbol and time frame, and optionally save the data to a CSV file.
