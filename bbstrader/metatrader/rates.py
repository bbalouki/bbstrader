import pandas as pd
import MetaTrader5 as Mt5
from datetime import datetime
from typing import Union, Optional
from bbstrader.metatrader.utils import (
    raise_mt5_error, TimeFrame, TIMEFRAMES
)
from bbstrader.metatrader.account import INIT_MSG
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

MAX_BARS = 10_000_000


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
        start_pos: Union[int | str] = 0,
        count: Optional[int] = MAX_BARS,
        session_duration: Optional[float] = None
    ):
        """
        Initializes a new Rates instance.

        Args:
            symbol (str): Financial instrument symbol (e.g., "EURUSD").
            time_frame (str): Timeframe string (e.g., "D1", "1h", "5m").
            start_pos (int, | str): Starting index (int)  or date (str) for data retrieval.
            count (int, optional): Number of bars to retrieve default is
                the maximum bars availble in the MT5 terminal.
            session_duration (float): Number of trading hours per day.

        Raises:
            ValueError: If the provided timeframe is invalid.

        Notes:
            If `start_pos` is an str, it must be in 'YYYY-MM-DD' format.
            For `session_duration` check your broker symbols details
        """
        self.symbol = symbol
        self.time_frame = self._validate_time_frame(time_frame)
        self.sd = session_duration
        self.start_pos = self._get_start_pos(start_pos, time_frame)
        self.count = count
        self._mt5_initialized()

    def _get_start_pos(self, index, time_frame):
        if isinstance(index, int):
            start_pos = index
        elif isinstance(index, str):
            assert self.sd is not None, \
                ValueError("Please provide the session_duration in hour")
            start_pos = self._get_pos_index(index, time_frame, self.sd)
        return start_pos

    def _get_pos_index(self, start_date, time_frame, sd):
        # Create a custom business day calendar
        us_business_day = CustomBusinessDay(
            calendar=USFederalHolidayCalendar())

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(datetime.now())

        # Generate a range of business days
        trading_days = pd.date_range(
            start=start_date, end=end_date, freq=us_business_day)

        # Calculate the number of trading days
        trading_days = len(trading_days)
        td = trading_days
        time_frame_mapping = {}
        for minutes in [1, 2, 3, 4, 5, 6, 10, 12, 15, 20,
                        30, 60, 120, 180, 240, 360, 480, 720]:
            key = f"{minutes//60}h" if minutes >= 60 else f"{minutes}m"
            time_frame_mapping[key] = int(td * (60 / minutes) * sd)
        time_frame_mapping['D1'] = int(td)

        if time_frame not in time_frame_mapping:
            pv = list(time_frame_mapping.keys())
            raise ValueError(
                f"Unsupported time frame, Possible Values are {pv}")

        index = time_frame_mapping.get(time_frame, 0)-1
        return max(index, 0)

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
        df = df.copy()
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
        save_csv: Optional[bool] = False,
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
            df.to_csv(f"{self.symbol}.csv")
        return df
