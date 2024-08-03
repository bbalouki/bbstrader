from datetime import datetime
import MetaTrader5 as Mt5
import pandas as pd
import time


class Rates(object):
    """

    The `Rates` class facilitates the retrieval of financial instrument data 
    from the MetaTrader 5 (MT5) terminal. It provides access to historical data 
    for a specified symbol and time frame, including open, high, low, close prices,  
    volume, and Returns calculated as percentage change.

    Exemple:
    >>> from datetime import datetime, timedelta
    >>> import pytz

    >>> # Example usage of the Rates class
    >>> symbol = "EURUSD"
    >>> time_frame = "1h"
    >>> start_pos = 0
    >>> count = 100

    >>> # Initialize the Rates class
    >>> rates = Rates(symbol=symbol, time_frame=time_frame, start_pos=start_pos, count=count)

    >>> # Retrieve and display the data frame
    >>> rates_data_frame = rates.get_rates()
    >>> print(rates_data_frame.head())

    >>> # Example to fetch historical data within a specific date range
    >>> timezone = pytz.timezone("Etc/UTC")
    >>> date_from = datetime(2020, 1, 1, tzinfo=timezone)
    >>> date_to = datetime.now(tz=timezone)

    >>> # Fetch historical data and save to CSV
    >>> rates.get_history(date_from=date_from, date_to=date_to, save=True)
    """
    def __init__(
        self,
        symbol: str,
        time_frame: str = "D1",
        start_pos: int = 0,
        count: int = 100
    ):
        """
        Initialize the Rates class to retrieve financial instrument data.

        Args:
            symbol (str): Financial instrument name (e.g., `"EURUSD"`).
            time_frame (str): The time frame on which the program is working
                `(1m, 3m, 5m, 10m, 15m, 30m, 1h, 2h, 4h, D1)`.
            start_pos (int): Initial index of the bar the data are requested from.
            count (int): Number of bars to receive.
        """
        self.symbol = symbol
        time_frame_mapping = {
            '1m':  Mt5.TIMEFRAME_M1,
            '3m':  Mt5.TIMEFRAME_M3,
            '5m':  Mt5.TIMEFRAME_M5,
            '10m': Mt5.TIMEFRAME_M10,
            '15m': Mt5.TIMEFRAME_M15,
            '30m': Mt5.TIMEFRAME_M30,
            '1h':  Mt5.TIMEFRAME_H1,
            '2h':  Mt5.TIMEFRAME_H2,
            '4h':  Mt5.TIMEFRAME_H4,
            'D1':  Mt5.TIMEFRAME_D1,
        }
        if time_frame not in time_frame_mapping:
            raise ValueError("Unsupported time frame")
        self.time_frame = time_frame_mapping[time_frame]
        self.start_pos = start_pos
        self.count = count
        self.initialize()
    
    def get_rates(self) -> pd.DataFrame:
        """
        Get bars from the MetaTrader 5 terminal.

        Returns:
        -   Bars as the pd.DataFrame
        """
        data = Mt5.copy_rates_from_pos(
            self.symbol, self.time_frame, self.start_pos, self.count)
        if data is None:
            if Mt5.last_error() == Mt5.RES_E_INTERNAL_FAIL_CONNECT:
                print(f"Error retrieving data: {Mt5.last_error()}")
        else:
            # create DataFrame out of the obtained data
            data = pd.DataFrame(data)
            data = data[['time', 'open', 'high',
                         'low', 'close', 'tick_volume']]
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            data['Adj Close'] = data['Close']
            # Reordering the columns
            data = data[['Date', 'Open', 'High',
                         'Low', 'Close', 'Adj Close', 'Volume']]
            data['Date'] = pd.to_datetime(data['Date'], unit='s')
            data['Returns'] = data['Adj Close'].pct_change().dropna()
        return data

    def get_history(
        self,
        date_from:  datetime,
        date_to:  datetime = datetime.now(),
        save:  bool = False
    ) -> pd.DataFrame | None:
        """
        Get bars in the specified date range from the MetaTrader 5 terminal.

        Args:
            date_from (datetime): Date the bars are requested from. 
                Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01. 
                Bars with the open time >= date_from are returned. Required unnamed parameter.
            
            date_to (datetime): Same as `date_from`
            save (bool): If set to True, a csv file will be create a to save data loaded

        Returns:
        -   Bars as the dataframe with `'Date', 'Open', 'High', 'Low', 'Close','Adj Close', 'Volume'`. 
        -   None in case of an error.

        Example:
        >>> # set time zone to UTC
        >>> timezone = pytz.timezone("Etc/UTC")
        >>> # create 'datetime' objects in UTC time zone to avoid 
        >>> # the implementation of a local time zone offset
        >>> utc_from = datetime(2020, 1, 10, tzinfo=timezone)
        >>> utc_to = datetime(2020, 1, 11, hour = 13, tzinfo=timezone)
        >>> # get bars from USDJPY M5 within the interval 
        >>> # of 2020.01.10 00:00 - 2020.01.11 13:00 in UTC time zone
        >>> rates = Rates("USDJPY", time_frame="m5")
        >>> history = rates.get_history(utc_from, utc_to, save=True)
        """
        rates = Mt5.copy_rates_range(
            self.symbol, self.time_frame, date_from, date_to)
        if len(rates) != None:
            # create DataFrame out of the obtained data
            data = pd.DataFrame(rates)
            data = data[['time', 'open', 'high',
                         'low', 'close', 'tick_volume']]
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            data['Adj Close'] = data['Close']
            # Reordering the columns
            data = data[['Date', 'Open', 'High',
                         'Low', 'Close', 'Adj Close', 'Volume']]
            data['Date'] = pd.to_datetime(data['Date'], unit='s')
            data['Returns'] = data['Adj Close'].pct_change()
            data.dropna(inplace=True)
            data.set_index('Date', inplace=True)
            if save:
                file = f"{self.symbol}.csv"
                data.to_csv(file)
            return data
        else:
            raise Exception(
                f"Sorry we can't get the history, error={Mt5.last_error()}")
            return None


    def initialize(self):
        try:
            Mt5.initialize()
        except Exception as e:
            if Mt5.last_error() == Mt5.RES_E_INTERNAL_FAIL_TIMEOUT:
                print("initialize() failed, error code =", Mt5.last_error())
                print("Trying again ....")
                time.sleep(60*3)
                if not Mt5.initialize():
                    print("initialize() failed, error code =", Mt5.last_error())
                    Mt5.shutdown()
