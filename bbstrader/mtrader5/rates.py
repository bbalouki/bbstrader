from datetime import datetime
import MetaTrader5 as Mt5
import pandas as pd
import time


class Rates:
    """
    The Rates class is used to retrieve financial instrument data.
    """
    def __init__(
        self,
        symbol: str,
        time_frame: str,
        start_pos: int,
        count: int
    ):
        """
        Initialize the Rates class to retrieve financial instrument data.

        Parameters
        ==========
        :param symbol (str) : Financial instrument name (e.g., "EURUSD").
        :param time_frame (str) : The time frame on which the program is working
            (1m, 3m, 5m, 10m, 15m, 30m, 1h, 2h, 4h, D1)
        :param start_pos (int): Initial index of the bar the data are requested from.
        :count (int): Number of bars to receive.
        :returns : Bars as the numpy array with the named 
            time, open, high, low, close, tick_volume, spread, and real_volume columns. 

        Returns
        :return: None in case of an error.
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
        self.data = self.get_rate()

    def get_rate_frame(self):
        return self.data
    
    @property
    def get_time(self):
        """
        Returns the date associated with the data.
        """
        return self.data['Date']
    
    @property
    def get_open(self):
        """
        Returns the open price of the financial instrument.
        """
        return self.data['Open']
    
    @property
    def get_high(self):
        """
        Returns the high price of the financial instrument.
        """
        return self.data['High']
    
    @property
    def get_low(self):
        """
        Returns the low price of the financial instrument.
        """
        return self.data['Low']
    
    @property
    def get_close(self):
        """
        Returns the close price of the financial instrument.
        """
        return self.data['Close']

    @property
    def get_adj_close(self):
        """
        Returns the adjusted close price of the financial instrument.
        """
        return self.data['Adj Close']
    
    @property
    def get_returns(self):
        """
        Returns the Returns of the current symbol
        """
        return self.data['Returns']
    
    @property
    def get_volume(self):
        """
        Returns the volume of the financial instrument.
        """
        return self.data['Volume']
    
    def get_rate(self):
        """
        Get bars from the MetaTrader 5 terminal.

        Returns
        :return : Bars as the pd.DataFrame
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
    ) -> None:
        """
        Get bars in the specified date range from the MetaTrader 5 terminal.

        Parameters
        ==========
        :param date_from (datetime) : Date the bars are requested from. 
            Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01. 
            Bars with the open time >= date_from are returned. Required unnamed parameter.
        :param : date_to (datetime): Same as  date_from
        :param save (bool) : Boolean value , if set to True ;
            a csv file will be create a to save data loaded

        Returns
        - Returns bars as the dataframe with the 
            'Date', 'Open', 'High', 'Low', 'Close','Adj Close', 'Volume'. 
        - Returns None in case of an error.

        Example:
        ```
        # set time zone to UTC
        timezone = pytz.timezone("Etc/UTC")
        # create 'datetime' objects in UTC time zone to avoid 
        # the implementation of a local time zone offset
        utc_from = datetime(2020, 1, 10, tzinfo=timezone)
        utc_to = datetime(2020, 1, 11, hour = 13, tzinfo=timezone)
        # get bars from USDJPY M5 within the interval of 2020.01.10 00:00 - 2020.01.11 13:00 in UTC time zone
        history = get_history(utc_from, utc_to, save=True)
        ```
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
        if not Mt5.initialize():
            if Mt5.last_error() == Mt5.RES_E_INTERNAL_FAIL_TIMEOUT:
                print("initialize() failed, error code =", Mt5.last_error())
                print("Trying again ....")
                time.sleep(60*3)
                if not Mt5.initialize():
                    print("initialize() failed, error code =", Mt5.last_error())
                    Mt5.shutdown()
