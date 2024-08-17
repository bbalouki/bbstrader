import pandas as pd
import MetaTrader5 as Mt5
from datetime import datetime
from typing import Union, Optional
from bbstrader.metatrader.utils import (
    raise_mt5_error, TimeFrame, TIMEFRAMES, INIT_MSG
)
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar


class Rates(object):
    """
    Provides methods to retrieve historical financial data from MetaTrader 5.

    This class encapsulates interactions with the MetaTrader 5 (MT5) terminal
    to fetch historical price data for a given symbol and timeframe. It offers
    flexibility in retrieving data either by specifying a starting position
    and count of bars or by providing a specific date range.

    Example:
        >>> rates = Rates("EURUSD", "1h")
        >>> df = rates.get_historical_data(
        ...     date_from=datetime(2023, 1, 1),
        ...     date_to=datetime(2023, 1, 10),
        ... )
        >>> print(df.head())
    """

    def __init__(
        self,
        symbol: str,
        time_frame: TimeFrame = 'D1',
        start_pos: Optional[int] = None,
        count: Optional[int] = None,
    ):
        """
        Initializes a new Rates instance.

        Args:
            symbol (str): Financial instrument symbol (e.g., "EURUSD").
            time_frame (str): Timeframe string (e.g., "D1", "1h", "5m").
            start_pos (int, optional): Starting index for data retrieval.
            count (int, optional): Number of bars to retrieve.

        Raises:
            ValueError: If the provided timeframe is invalid.
        """
        self.symbol = symbol
        self.time_frame = self._validate_time_frame(time_frame)
        self.start_pos = start_pos
        self.count = count
        self._mt5_initialized()

    def _validate_time_frame(self, time_frame: str) -> int:
        """Validates and returns the MT5 timeframe code."""
        if time_frame not in TIMEFRAMES:
            raise ValueError(
                f"Unsupported time frame '{time_frame}'. "
                f"Possible values are: {list(TIMEFRAMES.keys())}"
            )
        return TIMEFRAMES[time_frame]

    def _mt5_initialized(self):
        """Ensures the MetaTrader 5 Terminal is initialized."""
        if not Mt5.initialize():
            raise_mt5_error(message=INIT_MSG)

    def _fetch_data(
        self, start: Union[int, datetime],
        count: Union[int, datetime]
    ) -> Union[pd.DataFrame, None]:
        """Fetches data from MT5 and returns a DataFrame or None."""
        try:
            if isinstance(start, int) and isinstance(count, int):
                rates = Mt5.copy_rates_from_pos(
                    self.symbol, self.time_frame, start, count
                )
            elif isinstance(start, datetime) and isinstance(count, datetime):
                rates = Mt5.copy_rates_range(
                    self.symbol, self.time_frame, start, count
                )
            if rates is None:
                return None

            df = pd.DataFrame(rates)
            return self._format_dataframe(df)
        except Exception as e:
            raise_mt5_error(e)

    def _format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Formats the raw MT5 data into a standardized DataFrame."""
        df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Adj Close'] = df['Close']
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        df['Date'] = pd.to_datetime(df['Date'], unit='s')
        df.set_index('Date', inplace=True)
        return df

    def get_rates_from_pos(self) -> Union[pd.DataFrame, None]:
        """
        Retrieves historical data starting from a specific position.

        Uses the `start_pos` and `count` attributes specified during
        initialization to fetch data.

        Returns:
            Union[pd.DataFrame, None]: A DataFrame containing historical
            data if successful, otherwise None.
        """
        if self.start_pos is None or self.count is None:
            raise ValueError(
                "Both 'start_pos' and 'count' must be provided "
                "when calling 'get_rates_from_pos'."
            )
        df = self._fetch_data(self.start_pos, self.count)
        return df

    def get_historical_data(
        self,
        date_from: datetime,
        date_to: datetime = datetime.now(),
        save_csv: Optional[str] = None,
    ) -> Union[pd.DataFrame, None]:
        """
        Retrieves historical data within a specified date range.

        Args:
            date_from (datetime): Starting date for data retrieval.
            date_to (datetime, optional): Ending date for data retrieval.
                Defaults to the current time.
            save_csv (str, optional): File path to save the data as a CSV.
                If None, the data won't be saved.

        Returns:
            Union[pd.DataFrame, None]: A DataFrame containing historical data
                if successful, otherwise None.
        """
        df = self._fetch_data(date_from, date_to)
        if save_csv and df is not None:
            df.to_csv(save_csv)
        return df


def get_pos_index(
        start_date: str,
        time_frame: TimeFrame = 'D1',
        session_duration: float = 6.5):
    """
    Calculate the starting index for a given time frame and session duration.
    This is use in the `Rates()` if you want to get data starting from specific
    time in the past.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        time_frame (str): TimeFrame (e.g., 'D1', '1h', '15m').
        session_duration (float): Number of trading hours per day.

    Returns:
        int: Starting index.

    Note:
        For `session_duration` check your broker symbols details

    Exemple:
    >>> index = get_pos_index(
    ... start_date="2024-07-15", time_frame='1h', session_duration=6.5)
    >>> print('start_pos:', index)
    start_pos: 129
    """
    # Create a custom business day calendar
    us_business_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(datetime.now())

    # Generate a range of business days
    trading_days = pd.date_range(
        start=start_date, end=end_date, freq=us_business_day)

    # Calculate the number of trading days
    trading_days = len(trading_days)
    td = trading_days
    time_frame_mapping = {}
    for minutes in [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60, 120, 180, 240, 360, 480, 720]:
        key = f"{minutes//60}h" if minutes >= 60 else f"{minutes}m"
    time_frame_mapping[key] = int(td * (60 / minutes) * session_duration)

    time_frame_mapping['D1'] = int(td)
    if time_frame not in time_frame_mapping:
        pv = list(time_frame_mapping.keys())
        raise ValueError(f"Unsupported time frame, Possible Values are {pv}")

    index = time_frame_mapping.get(time_frame, 0)-1
    return max(index, 0)
