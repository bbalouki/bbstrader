from datetime import datetime
from typing import Optional, Union

import pandas as pd
from exchange_calendars import get_calendar, get_calendar_names
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from bbstrader.metatrader.account import AMG_EXCHANGES, Account, check_mt5_connection
from bbstrader.metatrader.utils import TIMEFRAMES, TimeFrame, raise_mt5_error, SymbolType

try:
    import MetaTrader5 as Mt5
except ImportError:
    import bbstrader.compat  # noqa: F401


__all__ = [
    "Rates",
    "download_historical_data",
    "get_data_from_pos",
    "get_data_from_date",
]

MAX_BARS = 10_000_000

IDX_CALENDARS = {
    "CAD": "XTSE",
    "AUD": "XASX",
    "GBP": "XLON",
    "HKD": "XSHG",
    "ZAR": "XJSE",
    "CHF": "XSWX",
    "NOK": "XOSL",
    "EUR": "XETR",
    "SGD": "XSES",
    "USD": "us_futures",
    "JPY": "us_futures",
}

COMD_CALENDARS = {
    "Energies": "us_futures",
    "Metals": "us_futures",
    "Agricultures": "CBOT",
    "Bonds": {"USD": "CBOT", "EUR": "EUREX"},
}

CALENDARS = {
    SymbolType.FOREX: "us_futures",
    SymbolType.STOCKS: AMG_EXCHANGES,
    SymbolType.ETFs: AMG_EXCHANGES,
    SymbolType.INDICES: IDX_CALENDARS,
    SymbolType.COMMODITIES: COMD_CALENDARS,
    SymbolType.CRYPTO: "24/7",
    SymbolType.FUTURES: None,
}

SESSION_TIMEFRAMES = [
    Mt5.TIMEFRAME_D1,
    Mt5.TIMEFRAME_W1,
    Mt5.TIMEFRAME_H12,
    Mt5.TIMEFRAME_MN1,
]


class Rates(object):
    """
    Provides methods to retrieve historical financial data from MetaTrader 5.

    This class encapsulates interactions with the MetaTrader 5 (MT5) terminal
    to fetch historical price data for a given symbol and timeframe. It offers
    flexibility in retrieving data either by specifying a starting position
    and count of bars or by providing a specific date range.

    Notes:
        1. Befor using this class, ensure that the `Max bars in chart` in you terminal
        is set to a value that is greater than the number of bars you want to retrieve
        or just set it to Unlimited.
        In your MT5 terminal, go to `Tools` -> `Options` -> `Charts` -> `Max bars in chart`.

        2. The `open, high, low, close, adjclose, returns,
        volume` properties returns data in  Broker's timezone by default.

        See `bbstrader.metatrader.account.check_mt5_connection()` for more details on how to connect to MT5 terminal.

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
        timeframe: TimeFrame = "D1",
        start_pos: Union[int, str] = 0,
        count: Optional[int] = MAX_BARS,
        session_duration: Optional[float] = None,
        **kwargs,
    ):
        """
        Initializes a new Rates instance.

        Args:
            symbol (str): Financial instrument symbol (e.g., "EURUSD").
            timeframe (str): Timeframe string (e.g., "D1", "1h", "5m").
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
        self.time_frame = self._validate_time_frame(timeframe)
        self.sd = session_duration
        self.start_pos = self._get_start_pos(start_pos, timeframe)
        self.count = count
        self.__initializ_mt5(**kwargs)
        self.__account = Account(**kwargs)
        self.__data = self.get_rates_from_pos()

    def __initializ_mt5(self, **kwargs):
        check_mt5_connection(**kwargs)

    def _get_start_pos(self, index, time_frame):
        if isinstance(index, int):
            start_pos = index
        elif isinstance(index, str):
            assert self.sd is not None, ValueError(
                "Please provide the session_duration in hour"
            )
            start_pos = self._get_pos_index(index, time_frame, self.sd)
        return start_pos

    def _get_pos_index(self, start_date, time_frame, sd):
        # Create a custom business day calendar
        us_business_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(datetime.now())

        # Generate a range of business days
        trading_days = pd.date_range(
            start=start_date, end=end_date, freq=us_business_day
        )

        # Calculate the number of trading days
        trading_days = len(trading_days)
        td = trading_days
        time_frame_mapping = {}
        for minutes in [
            1,
            2,
            3,
            4,
            5,
            6,
            10,
            12,
            15,
            20,
            30,
            60,
            120,
            180,
            240,
            360,
            480,
            720,
        ]:
            key = f"{minutes // 60}h" if minutes >= 60 else f"{minutes}m"
            time_frame_mapping[key] = int(td * (60 / minutes) * sd)
        time_frame_mapping["D1"] = int(td)

        if time_frame not in time_frame_mapping:
            pv = list(time_frame_mapping.keys())
            raise ValueError(f"Unsupported time frame, Possible Values are {pv}")

        index = time_frame_mapping.get(time_frame, 0) - 1
        return max(index, 0)

    def _validate_time_frame(self, time_frame: str) -> int:
        """Validates and returns the MT5 timeframe code."""
        if time_frame not in TIMEFRAMES:
            raise ValueError(
                f"Unsupported time frame '{time_frame}'. "
                f"Possible values are: {list(TIMEFRAMES.keys())}"
            )
        return TIMEFRAMES[time_frame]

    def _fetch_data(
        self,
        start: Union[int, datetime, pd.Timestamp],
        count: Union[int, datetime, pd.Timestamp],
        lower_colnames=False,
        utc=False,
    ) -> Union[pd.DataFrame, None]:
        """Fetches data from MT5 and returns a DataFrame or None."""
        try:
            rates = None
            if isinstance(start, int) and isinstance(count, int):
                rates = Mt5.copy_rates_from_pos(
                    self.symbol, self.time_frame, start, count
                )
            elif isinstance(start, (datetime, pd.Timestamp)) and isinstance(count, int):
                rates = Mt5.copy_rates_from(self.symbol, self.time_frame, start, count)
            elif isinstance(start, (datetime, pd.Timestamp)) and isinstance(
                count, (datetime, pd.Timestamp)
            ):
                rates = Mt5.copy_rates_range(self.symbol, self.time_frame, start, count)
            if rates is None:
                return None

            df = pd.DataFrame(rates)
            return self._format_dataframe(df, lower_colnames=lower_colnames, utc=utc)
        except Exception as e:
            raise_mt5_error(e)

    def _format_dataframe(
        self, df: pd.DataFrame, lower_colnames=False, utc=False
    ) -> pd.DataFrame:
        """Formats the raw MT5 data into a standardized DataFrame."""
        df = df.copy()
        df = df[["time", "open", "high", "low", "close", "tick_volume"]]
        df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df["Adj Close"] = df["Close"]
        df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        df["Date"] = pd.to_datetime(df["Date"], unit="s", utc=utc)
        df.set_index("Date", inplace=True)
        if lower_colnames:
            df.columns = df.columns.str.lower().str.replace(" ", "_")
            df.index.name = df.index.name.lower().replace(" ", "_")
        return df

    def _filter_data(
        self, df: pd.DataFrame, date_from=None, date_to=None, fill_na=False
    ) -> pd.DataFrame:
        df = df.copy()
        symbol_type = self.__account.get_symbol_type(self.symbol)
        currencies = self.__account.get_currency_rates(self.symbol)
        s_info = self.__account.get_symbol_info(self.symbol)
        if symbol_type in CALENDARS:
            if symbol_type == SymbolType.STOCKS or symbol_type == SymbolType.ETFs:
                for exchange in CALENDARS[symbol_type]:
                    if exchange in get_calendar_names():
                        symbols = self.__account.get_stocks_from_exchange(
                            exchange_code=exchange
                        )
                        if self.symbol in symbols:
                            calendar = get_calendar(exchange, side="right")
                            break
            elif symbol_type == SymbolType.INDICES:
                calendar = get_calendar(
                    CALENDARS[symbol_type][currencies["mc"]], side="right"
                )
            elif symbol_type == SymbolType.COMMODITIES:
                for commodity in CALENDARS[symbol_type]:
                    if commodity in s_info.path:
                        calendar = get_calendar(
                            CALENDARS[symbol_type][commodity], side="right"
                        )
            elif symbol_type == SymbolType.FUTURES:
                if "Index" in s_info.path:
                    calendar = get_calendar(
                        CALENDARS[SymbolType.INDICES][currencies["mc"]], side="right"
                    )
                else:
                    for commodity, cal in COMD_CALENDARS.items():
                        if self.symbol in self.__account.get_future_symbols(
                            category=commodity
                        ):
                            if commodity == "Bonds":
                                calendar = get_calendar(
                                    cal[currencies["mc"]], side="right"
                                )
                            else:
                                calendar = get_calendar(cal, side="right")
            else:
                calendar = get_calendar(CALENDARS[symbol_type], side="right")
            date_from = date_from or df.index[0]
            date_to = date_to or df.index[-1]
            if self.time_frame in SESSION_TIMEFRAMES:
                valid_sessions = calendar.sessions_in_range(date_from, date_to)
            else:
                valid_sessions = calendar.minutes_in_range(date_from, date_to)
            if self.time_frame in [Mt5.TIMEFRAME_M1, Mt5.TIMEFRAME_D1]:
                # save the index name of the dataframe
                index_name = df.index.name
                if fill_na:
                    if isinstance(fill_na, bool):
                        method = "nearest"
                    if isinstance(fill_na, str):
                        method = fill_na
                    df = df.reindex(valid_sessions, method=method)
                else:
                    df.reindex(valid_sessions, method=None)
                df.index = df.index.rename(index_name)
            else:
                df = df[df.index.isin(valid_sessions)]
        return df

    def _check_filter(self, filter, utc):
        if filter and self.time_frame not in SESSION_TIMEFRAMES and not utc:
            utc = True
        elif filter and self.time_frame in SESSION_TIMEFRAMES and utc:
            utc = False
        return utc

    def get_rates_from_pos(
        self, filter=False, fill_na=False, lower_colnames=False, utc=False
    ) -> Union[pd.DataFrame, None]:
        """
        Retrieves historical data starting from a specific position.

        Uses the `start_pos` and `count` attributes specified during
        initialization to fetch data.

        Args:
            filter : See `Rates.get_historical_data` for more details.
            fill_na : See `Rates.get_historical_data` for more details.
            lower_colnames : If True, the column names will be converted to lowercase.
            utc (bool, optional): If True, the data will be in UTC timezone.
                Defaults to False.

        Returns:
            Union[pd.DataFrame, None]: A DataFrame containing historical
            data if successful, otherwise None.

        Raises:
            ValueError: If `start_pos` or `count` is not provided during
                initialization.

        Notes:
            The Datetime for this method is in Broker's timezone.
        """
        if self.start_pos is None or self.count is None:
            raise ValueError(
                "Both 'start_pos' and 'count' must be provided "
                "when calling 'get_rates_from_pos'."
            )
        utc = self._check_filter(filter, utc)
        df = self._fetch_data(
            self.start_pos, self.count, lower_colnames=lower_colnames, utc=utc
        )
        if df is None:
            return None
        if filter:
            return self._filter_data(df, fill_na=fill_na)
        return df

    def get_rates_from(
        self,
        date_from: datetime | pd.Timestamp,
        count: int = MAX_BARS,
        filter=False,
        fill_na=False,
        lower_colnames=False,
        utc=False,
    ) -> Union[pd.DataFrame, None]:
        """
        Retrieves historical data within a specified date range.

        Args:
            date_from : Starting date for data retrieval.
                The data will be retrieved from this date going to the past.

            count : Number of bars to retrieve.

            filter : See `Rates.get_historical_data` for more details.
            fill_na : See `Rates.get_historical_data` for more details.
            lower_colnames : If True, the column names will be converted to lowercase.
            utc (bool, optional): If True, the data will be in UTC timezone.
                Defaults to False.

        Returns:
            Union[pd.DataFrame, None]: A DataFrame containing historical
            data if successful, otherwise None.
        """
        utc = self._check_filter(filter, utc)
        df = self._fetch_data(date_from, count, lower_colnames=lower_colnames, utc=utc)
        if df is None:
            return None
        if filter:
            return self._filter_data(df, fill_na=fill_na)
        return df

    @property
    def open(self):
        return self.__data["Open"]

    @property
    def high(self):
        return self.__data["High"]

    @property
    def low(self):
        return self.__data["Low"]

    @property
    def close(self):
        return self.__data["Close"]

    @property
    def adjclose(self):
        return self.__data["Adj Close"]

    @property
    def returns(self):
        """
        Fractional change between the current and a prior element.

        Computes the fractional change from the immediately previous row by default.
        This is useful in comparing the fraction of change in a time series of elements.

        Note
        ----
        It calculates fractional change (also known as `per unit change or relative change`)
        and `not percentage change`. If you need the percentage change, multiply these values by 100.
        """
        data = self.__data.copy()
        data["Returns"] = data["Adj Close"].pct_change()
        data = data.dropna()
        return data["Returns"]

    @property
    def volume(self):
        return self.__data["Volume"]

    def get_historical_data(
        self,
        date_from: datetime | pd.Timestamp,
        date_to: datetime | pd.Timestamp = pd.Timestamp.now(),
        utc: bool = False,
        filter: Optional[bool] = False,
        fill_na: Optional[bool | str] = False,
        lower_colnames: Optional[bool] = True,
        save_csv: Optional[bool] = False,
    ) -> Union[pd.DataFrame, None]:
        """
        Retrieves historical data within a specified date range.

        Args:
            date_from : Starting date for data retrieval.

            date_to : Ending date for data retrieval.
                Defaults to the current time.

            utc : If True, the data will be in UTC timezone.
                Defaults to False.

            filter : If True, the data will be filtered based
                on the trading sessions for the symbol.
                This is use when we want to use the data for backtesting using Zipline.

            fill_na : If True, the data will be filled with the nearest value.
                This is use only when `filter` is True and time frame is "1m" or "D1",
                this is because we use ``calendar.minutes_in_range`` or ``calendar.sessions_in_range``
                where calendar is the ``ExchangeCalendar`` from `exchange_calendars` package.
                So, for "1m" or "D1" time frame, the data will be filled with the nearest value
                because the data from MT5 will have approximately the same number of rows as the
                number of trading days or minute in the exchange calendar, so we can fill the missing
                data with the nearest value.

                But for other time frames, the data will be reindexed with the exchange calendar
                because the data from MT5 will have more rows than the number of trading days or minute
                in the exchange calendar. So we only take the data that is in the range of the exchange
                calendar sessions or minutes.

            lower_colnames : If True, the column names will be converted to lowercase.

            save_csv : File path to save the data as a CSV.
                If None, the data won't be saved.

        Returns:
            Union[pd.DataFrame, None]: A DataFrame containing historical data
                if successful, otherwise None.

        Raises:
            ValueError: If the starting date is greater than the ending date.

        Notes:
            The `filter` for this method can be use only for Admira Markets Group (AMG) symbols.
            The Datetime for this method is in Local timezone by default.
            All STK symbols are filtered based on the the exchange calendar.
            All FX symbols are filtered based on the ``us_futures`` calendar.
            All IDX symbols are filtered based on the exchange calendar of margin currency.
            All COMD symbols are filtered based on the exchange calendar of the commodity.
        """
        utc = self._check_filter(filter, utc)
        df = self._fetch_data(
            date_from, date_to, lower_colnames=lower_colnames, utc=utc
        )
        if df is None:
            return None
        if filter:
            df = self._filter_data(
                df, date_from=date_from, date_to=date_to, fill_na=fill_na
            )
        if save_csv:
            df.to_csv(f"{self.symbol}.csv")
        return df


def download_historical_data(
    symbol,
    timeframe,
    date_from,
    date_to=pd.Timestamp.now(),
    lower_colnames=True,
    utc=False,
    filter=False,
    fill_na=False,
    save_csv=False,
    **kwargs,
):
    """Download historical data from MetaTrader 5 terminal.
    See `Rates.get_historical_data` for more details.
    """
    rates = Rates(symbol, timeframe, **kwargs)
    data = rates.get_historical_data(
        date_from=date_from,
        date_to=date_to,
        save_csv=save_csv,
        utc=utc,
        filter=filter,
        lower_colnames=lower_colnames,
    )
    return data


def get_data_from_pos(
    symbol,
    timeframe,
    start_pos=0,
    fill_na=False,
    count=MAX_BARS,
    lower_colnames=False,
    utc=False,
    filter=False,
    session_duration=23.0,
    **kwargs,
):
    """Get historical data from a specific position.
    See `Rates.get_rates_from_pos` for more details.
    """
    rates = Rates(symbol, timeframe, start_pos, count, session_duration, **kwargs)
    data = rates.get_rates_from_pos(
        filter=filter, fill_na=fill_na, lower_colnames=lower_colnames, utc=utc
    )
    return data


def get_data_from_date(
    symbol,
    timeframe,
    date_from,
    count=MAX_BARS,
    fill_na=False,
    lower_colnames=False,
    utc=False,
    filter=False,
    **kwargs,
):
    """Get historical data from a specific date.
    See `Rates.get_rates_from` for more details.
    """
    rates = Rates(symbol, timeframe, **kwargs)
    data = rates.get_rates_from(
        date_from,
        count,
        filter=filter,
        fill_na=fill_na,
        lower_colnames=lower_colnames,
        utc=utc,
    )
    return data
