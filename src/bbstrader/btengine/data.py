import os.path
import time
from abc import ABCMeta, abstractmethod
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yfinance as yf
from eodhd import APIClient
from financetoolkit import Toolkit
from numpy.typing import NDArray
from pytz import timezone

from bbstrader.btengine.event import MarketEvent
from bbstrader.config import BBSTRADER_DIR
from bbstrader.metatrader.rates import download_historical_data

__all__ = [
    "DataHandler",
    "CSVDataHandler",
    "MT5DataHandler",
    "YFDataHandler",
    "EODHDataHandler",
    "FMPDataHandler",
]


def _cache_is_fresh(
    filepath: str, use_cache: bool, max_age_days: Optional[float]
) -> bool:
    """Return True if a cached file should be reused instead of re-downloading.

    Opt-in via ``use_cache``; a ``max_age_days`` of None means the cache never
    expires. Defaults keep the original always-download behavior.
    """
    if not use_cache or not os.path.exists(filepath):
        return False
    if max_age_days is None:
        return True
    age_seconds = time.time() - os.path.getmtime(filepath)
    return age_seconds <= max_age_days * 86400.0


class DataHandler(metaclass=ABCMeta):
    """
    One of the goals of an event-driven trading system is to minimise
    duplication of code between the backtesting element and the live execution
    element. Ideally it would be optimal to utilise the same signal generation
    methodology and portfolio management components for both historical testing
    and live trading. In order for this to work the Strategy object which generates
    the Signals, and the `Portfolio` object which provides Orders based on them,
    must utilise an identical interface to a market feed for both historic and live
    running.

    This motivates the concept of a class hierarchy based on a `DataHandler` object,
    which givesall subclasses an interface for providing market data to the remaining
    components within thesystem. In this way any subclass data handler can be "swapped out",
    without affecting strategy or portfolio calculation.

    Specific example subclasses could include `HistoricCSVDataHandler`,
    `YFinanceDataHandler`, `FMPDataHandler`, `IBMarketFeedDataHandler` etc.
    """

    @property
    def symbols(self) -> List[str]:
        """The list of symbols this handler serves."""
        pass

    @property
    def data(self) -> Dict[str, pd.DataFrame]:
        """The loaded market data, keyed by symbol."""
        pass

    @property
    def labels(self) -> List[str]:
        """The OHLCV column labels exposed by the handler."""
        pass

    @property
    def index(self) -> Union[str, List[str]]:
        """The name(s) of the datetime index column(s)."""
        pass

    @abstractmethod
    def get_latest_bar(self, symbol: str) -> pd.Series:
        """
        Returns the last bar updated.
        """
        raise NotImplementedError("Should implement get_latest_bar()")

    @abstractmethod
    def get_latest_bars(
        self, symbol: str, N: int = 1, df: bool = True
    ) -> Union[pd.DataFrame, List[pd.Series]]:
        """
        Returns the last N bars updated.
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def get_latest_bar_datetime(self, symbol: str) -> Union[datetime, pd.Timestamp]:
        """
        Returns a Python datetime object for the last bar.
        """
        raise NotImplementedError("Should implement get_latest_bar_datetime()")

    @abstractmethod
    def get_latest_bar_value(self, symbol: str, val_type: str) -> float:
        """
        Returns one of the Open, High, Low, Close, Adj Close, Volume or Returns
        from the last bar.
        """
        raise NotImplementedError("Should implement get_latest_bar_value()")

    @abstractmethod
    def get_latest_bars_values(self, symbol: str, val_type: str, N: int = 1) -> NDArray:
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """
        raise NotImplementedError("Should implement get_latest_bars_values()")

    @abstractmethod
    def update_bars(self) -> None:
        """
        Pushes the latest bars to the bars_queue for each symbol
        in a tuple OHLCVI format: (datetime, Open, High, Low,
        Close, Adj Close, Volume, Retruns).
        """
        raise NotImplementedError("Should implement update_bars()")


class BaseCSVDataHandler(DataHandler):
    """
    Base class for handling data loaded from CSV files.

    """

    def __init__(
        self,
        events: "Queue[MarketEvent]",
        symbol_list: List[str],
        csv_dir: str,
        columns: Optional[List[str]] = None,
        index_col: Union[str, int, List[str], List[int]] = 0,
        persist_normalized: bool = True,
    ) -> None:
        """
        Initialises the data handler by requesting the location of the CSV files
        and a list of symbols.

        Args:
            events : The Event Queue.
            symbol_list : A list of symbol strings.
            csv_dir : Absolute directory path to the CSV files.
            columns : List of column names to use for the data.
            index_col : Column to use as the index.
            persist_normalized : Whether to write the normalized frame back to
                ``csv_dir``. True for handlers that own ``csv_dir`` as a cache
                (the download handlers); False when ``csv_dir`` is the user's
                source directory, to avoid mutating it and to keep parallel
                workers that share one directory from racing on the same file.
        """
        self.events = events
        self.symbol_list = symbol_list
        self.csv_dir = csv_dir
        self.columns = columns
        self.index_col = index_col
        self._persist_normalized = persist_normalized
        self.symbol_data: Dict[str, Union[pd.DataFrame, Generator]] = {}
        self.latest_symbol_data: Dict[str, List[Any]] = {}
        # Immutable, replayable per-symbol record lists of (timestamp, Series)
        # tuples, plus a shared integer cursor. This lets a single handler be
        # replayed (see reset()) for parameter sweeps or walk-forward folds.
        self._records: Dict[str, List[Tuple[Any, pd.Series]]] = {}
        self._cursor = 0
        self.continue_backtest = True
        self._index: Optional[Union[str, List[str]]] = None
        self._load_and_process_data()

    @property
    def symbols(self) -> List[str]:
        """The list of symbols this handler serves."""
        return self.symbol_list

    @property
    def data(self) -> Dict[str, pd.DataFrame]:
        """The loaded market data, keyed by symbol."""
        return self.symbol_data  # type: ignore

    @property
    def datadir(self) -> str:
        """The directory the CSV files are read from."""
        return self.csv_dir

    @property
    def labels(self) -> List[str]:
        """The OHLCV column labels exposed by the handler."""
        return self.columns  # type: ignore

    @property
    def index(self) -> Union[str, List[str]]:
        """The name(s) of the datetime index column(s)."""
        return self._index  # type: ignore

    def _load_and_process_data(self) -> None:
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.
        """
        default_names = pd.read_csv(
            os.path.join(self.csv_dir, f"{self.symbol_list[0]}.csv")
        ).columns.to_list()
        new_names = self.columns or default_names
        new_names = [name.strip().lower().replace(" ", "_") for name in new_names]
        self.columns = new_names
        assert "adj_close" in new_names or "close" in new_names, (
            "Column names must contain 'Adj Close' and 'Close' or adj_close and close"
        )
        comb_index = None
        for s in self.symbol_list:
            # Load the CSV file with no header information,
            # indexed on date
            self.symbol_data[s] = pd.read_csv(
                os.path.join(self.csv_dir, f"{s}.csv"),
                header=0,
                index_col=self.index_col,
                parse_dates=True,
                names=new_names,
            )
            self.symbol_data[s].sort_index(inplace=True)
            # Combine the index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_data[s].index
            elif len(self.symbol_data[s].index) > len(comb_index):
                comb_index = self.symbol_data[s].index
            # Set the latest symbol_data to None
            self.latest_symbol_data[s] = []

        # Reindex the dataframes
        for s in self.symbol_list:
            self.symbol_data[s] = self.symbol_data[s].reindex(  # type: ignore
                index=comb_index, method="pad"
            )
            if "adj_close" not in new_names:
                self.columns.append("adj_close")
                self.symbol_data[s]["adj_close"] = self.symbol_data[s]["close"]  # type: ignore
            self.symbol_data[s]["returns"] = (  # type: ignore
                self.symbol_data[s][  # type: ignore
                    "adj_close" if "adj_close" in new_names else "close"
                ]
                .pct_change()
                .dropna()
            )
            self._index = self.symbol_data[s].index.name  # type: ignore
            # Persist the normalized frame only for handlers that own csv_dir as
            # a cache. For a plain CSVDataHandler csv_dir is the user's source,
            # so writing back mutates their data and races when parallel optimize
            # workers share one directory (concurrent read/write corrupts it).
            if self._persist_normalized:
                self.symbol_data[s].to_csv(  # type: ignore
                    os.path.join(self.csv_dir, f"{s}.csv")
                )
            if self.events is not None:
                # Materialize a replayable record list once. These tuples are
                # identical to what DataFrame.iterrows() yielded, so all
                # get_latest_* accessors are unchanged, but the data can now be
                # replayed by resetting the cursor instead of being exhausted.
                self._records[s] = list(self.symbol_data[s].iterrows())  # type: ignore

    @property
    def n_bars(self) -> int:
        """Total number of bars available in the replayable record set."""
        if not self._records:
            return 0
        return len(self._records[self.symbol_list[0]])

    def reset(self) -> None:
        """
        Rewinds the handler to the start so the same data can be replayed.

        Enables parameter sweeps, walk-forward folds, and Monte Carlo passes
        without reconstructing the handler.
        """
        self._cursor = 0
        self.continue_backtest = True
        for s in self.symbol_list:
            self.latest_symbol_data[s] = []

    def get_latest_bar(self, symbol: str) -> pd.Series:
        """
        Returns the last bar from the latest_symbol list.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print(f"{symbol} not available in the historical data set.")
            raise
        else:
            return bars_list[-1]

    def get_latest_bars(
        self, symbol: str, N: int = 1, df: bool = True
    ) -> Union[pd.DataFrame, List[pd.Series]]:
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print(f"{symbol} not available in the historical data set.")
            raise
        else:
            if df:
                df_ = pd.DataFrame([bar[1] for bar in bars_list[-N:]])
                df_.index.name = self._index  # type: ignore
                return df_
            return bars_list[-N:]

    def get_latest_bar_datetime(self, symbol: str) -> Union[datetime, pd.Timestamp]:
        """
        Returns a Python datetime object for the last bar.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print(f"{symbol} not available in the historical data set.")
            raise
        else:
            return bars_list[-1][0]

    def get_latest_bars_datetime(
        self, symbol: str, N: int = 1
    ) -> List[Union[datetime, pd.Timestamp]]:
        """
        Returns a list of Python datetime objects for the last N bars.
        """
        try:
            bars_list = self.get_latest_bars(symbol, N)  # type: ignore
        except KeyError:
            print(f"{symbol} not available in the historical data set for .")
            raise
        else:
            return [b[0] for b in bars_list]  # type: ignore

    def get_latest_bar_value(self, symbol: str, val_type: str) -> float:
        """
        Returns one of the Open, High, Low, Close, Volume or OI
        values from the pandas Bar series object.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print(f"{symbol} not available in the historical data set.")
            raise
        else:
            try:
                return getattr(bars_list[-1][1], val_type)
            except AttributeError:
                print(
                    f"Value type {val_type} not available in the historical data set for {symbol}."
                )
                raise

    def get_latest_bars_values(self, symbol: str, val_type: str, N: int = 1) -> NDArray:
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """
        try:
            bars_list = self.get_latest_bars(symbol, N, df=False)
        except KeyError:
            print(f"{symbol} not available in the historical data set.")
            raise
        else:
            try:
                return np.array([getattr(b[1], val_type) for b in bars_list])
            except AttributeError:
                print(
                    f"Value type {val_type} not available in the historical data set."
                )
                raise

    def update_bars(self) -> None:
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        # A MarketEvent is emitted on every call, including the exhausting one,
        # to preserve the original generator-based semantics exactly (the final
        # event fires with no new bar appended).
        if self._cursor >= self.n_bars:
            self.continue_backtest = False
        else:
            for s in self.symbol_list:
                self.latest_symbol_data[s].append(self._records[s][self._cursor])
            self._cursor += 1
        self.events.put(MarketEvent())


class CSVDataHandler(BaseCSVDataHandler):
    """
    `CSVDataHandler` is designed to read CSV files for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface.

    This class is useful when you have your own data or you want
    to cutomize specific data in some form based on your `Strategy()` .
    """

    def __init__(
        self, events: "Queue[MarketEvent]", symbol_list: List[str], **kwargs: Any
    ) -> None:
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.
        It will be assumed that all files are of the form
        `symbol.csv`, where `symbol` is a string in the list.

        Args:
            events (Queue): The Event Queue.
            symbol_list (List[str]): A list of symbol strings.
            csv_dir (str): Absolute directory path to the CSV files.

        NOTE:
        All csv fille can be stored in 'Home/.bbstrader/data/csv_data'

        """
        csv_dir = kwargs.get("csv_dir")
        csv_dir = csv_dir or BBSTRADER_DIR / "data" / "csv_data"
        super().__init__(
            events,
            symbol_list,
            str(csv_dir),
            columns=kwargs.get("columns"),
            index_col=kwargs.get("index_col", 0),
            # csv_dir is the user's own data directory; never rewrite it.
            persist_normalized=False,
        )


class MT5DataHandler(BaseCSVDataHandler):
    """
    Downloads historical data from MetaTrader 5 (MT5) and provides
    an interface for accessing this data bar-by-bar, simulating
    a live market feed for backtesting.

    Data is downloaded from MT5, saved as CSV files, and then loaded
    using the functionality inherited from `BaseCSVDataHandler`.

    This class is useful when you need to get data from specific broker
    for different time frames.
    """

    def __init__(
        self, events: "Queue[MarketEvent]", symbol_list: List[str], **kwargs: Any
    ) -> None:
        """
        Args:
            events (Queue): The Event Queue for passing market events.
            symbol_list (List[str]): A list of symbol strings to download data for.
            **kwargs: Keyword arguments for data retrieval:
                time_frame (str): MT5 time frame (e.g., 'D1' for daily).
                mt5_start (datetime): Start date for historical data.
                mt5_end (datetime): End date for historical data.
                data_dir (str): Directory for storing data .

        Note:
            Requires a working connection to an MT5 terminal.
            See `bbstrader.metatrader.rates.Rates` for other arguments.
            See `bbstrader.btengine.data.BaseCSVDataHandler` for other arguments.
        """
        self.tf = kwargs.get("time_frame", "D1")
        self.start = kwargs.get("mt5_start", datetime(2000, 1, 1))
        self.end = kwargs.get("mt5_end", datetime.now())
        self.use_utc = kwargs.get("use_utc", False)
        self.filer = kwargs.get("filter", False)
        self.fill_na = kwargs.get("fill_na", False)
        self.lower_cols = kwargs.get("lower_cols", True)
        self.data_dir = kwargs.get("data_dir")
        self.use_cache = kwargs.get("use_cache", False)
        self.cache_max_age_days = kwargs.get("cache_max_age_days")
        self.symbol_list = symbol_list
        self.kwargs = kwargs
        self.kwargs["backtest"] = (
            True  # Ensure backtest mode is set to avoid InvalidBroker errors
        )

        csv_dir = self._download_and_cache_data(self.data_dir)
        super().__init__(
            events,
            symbol_list,
            str(csv_dir),
            columns=kwargs.get("columns"),
            index_col=kwargs.get("index_col", 0),
        )

    def _download_and_cache_data(self, cache_dir: Optional[str]) -> Path:
        """Download MT5 historical data per symbol and cache it as CSV.

        Honors the handler's ``use_cache``/``cache_max_age_days`` settings to
        skip downloads whose cache is still fresh.

        Args:
            cache_dir (Optional[str]): Directory to cache CSVs in. Defaults to
                ``~/.bbstrader/data/mt5/<timeframe>``.

        Returns:
            Path: The directory the CSV files were written to.
        """
        data_dir = (
            Path(cache_dir) if cache_dir else BBSTRADER_DIR / "data" / "mt5" / self.tf
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        for symbol in self.symbol_list:
            if _cache_is_fresh(
                str(data_dir / f"{symbol}.csv"), self.use_cache, self.cache_max_age_days
            ):
                continue
            try:
                data = download_historical_data(
                    symbol=symbol,
                    timeframe=self.tf,
                    date_from=self.start,
                    date_to=self.end,
                    utc=self.use_utc,
                    filter=self.filer,
                    fill_na=self.fill_na,
                    lower_colnames=self.lower_cols,
                    **self.kwargs,
                )
                if data is None:
                    raise ValueError(f"No data found for {symbol}")
                data.to_csv(data_dir / f"{symbol}.csv")
            except Exception as e:
                raise ValueError(f"Error downloading {symbol}: {e}")
        return data_dir


class YFDataHandler(BaseCSVDataHandler):
    """
    Downloads historical data from Yahoo Finance and provides
    an interface for accessing this data bar-by-bar, simulating
    a live market feed for backtesting.

    Data is fetched using the `yfinance` library and optionally cached
    to disk to speed up subsequent runs.

    This class is useful when working with historical daily prices.
    """

    def __init__(
        self, events: "Queue[MarketEvent]", symbol_list: List[str], **kwargs: Any
    ) -> None:
        """
        Args:
            events (Queue): The Event Queue for passing market events.
            symbol_list (list[str]): List of symbols to download data for.
            yf_start (str): Start date for historical data (YYYY-MM-DD).
            yf_end (str): End date for historical data (YYYY-MM-DD).
            data_dir (str, optional): Directory for caching data .

        Note:
            See `bbstrader.btengine.data.BaseCSVDataHandler` for other arguments.
        """
        self.symbol_list = symbol_list
        self.start_date = kwargs.get("yf_start")
        self.end_date = kwargs.get("yf_end", datetime.now())
        self.cache_dir = kwargs.get("data_dir")
        self.use_cache = kwargs.get("use_cache", False)
        self.cache_max_age_days = kwargs.get("cache_max_age_days")

        csv_dir = self._download_and_cache_data(self.cache_dir)

        super().__init__(
            events,
            symbol_list,
            str(csv_dir),
            columns=kwargs.get("columns"),
            index_col=kwargs.get("index_col", 0),
        )

    def _download_and_cache_data(self, cache_dir: Optional[str]) -> str:
        """Downloads and caches historical data as CSV files."""
        _cache_dir = cache_dir or BBSTRADER_DIR / "data" / "yfinance" / "daily"
        os.makedirs(_cache_dir, exist_ok=True)
        for symbol in self.symbol_list:
            filepath = os.path.join(_cache_dir, f"{symbol}.csv")
            if _cache_is_fresh(filepath, self.use_cache, self.cache_max_age_days):
                continue
            try:
                data = yf.download(
                    symbol,
                    start=self.start_date,
                    end=self.end_date,
                    multi_level_index=False,
                    auto_adjust=True,
                    progress=False,
                )
                if "Adj Close" not in data.columns:
                    data["Adj Close"] = data["Close"]
                if data.empty:
                    raise ValueError(f"No data found for {symbol}")
                data.to_csv(filepath)
            except Exception as e:
                raise ValueError(f"Error downloading {symbol}: {e}")
        return str(_cache_dir)


class EODHDataHandler(BaseCSVDataHandler):
    """
    Downloads historical data from EOD Historical Data.
    Data is fetched using the `eodhd` library.

    To use this class, you need to sign up for an API key at
    https://eodhistoricaldata.com/ and provide the key as an argument.
    """

    def __init__(
        self, events: "Queue[MarketEvent]", symbol_list: List[str], **kwargs: Any
    ) -> None:
        """
        Args:
            events (Queue): The Event Queue for passing market events.
            symbol_list (list[str]): List of symbols to download data for.
            eodhd_start (str): Start date for historical data (YYYY-MM-DD).
            eodhd_end (str): End date for historical data (YYYY-MM-DD).
            data_dir (str, optional): Directory for caching data .
            eodhd_period (str, optional): Time period for historical data (e.g., 'd', 'w', 'm', '1m', '5m', '1h').
            eodhd_api_key (str, optional): API key for EOD Historical Data.

        Note:
            See `bbstrader.btengine.data.BaseCSVDataHandler` for other arguments.
        """
        self.symbol_list = symbol_list
        self.start_date = kwargs.get("eodhd_start")
        self.end_date = kwargs.get("eodhd_end", datetime.now().strftime("%Y-%m-%d"))
        self.period = kwargs.get("eodhd_period", "d")
        self.cache_dir = kwargs.get("data_dir")
        self.use_cache = kwargs.get("use_cache", False)
        self.cache_max_age_days = kwargs.get("cache_max_age_days")
        self.__api_key = kwargs.get("eodhd_api_key", "demo")

        csv_dir = self._download_and_cache_data(self.cache_dir)

        super().__init__(
            events,
            symbol_list,
            str(csv_dir),
            columns=kwargs.get("columns"),
            index_col=kwargs.get("index_col", 0),
        )

    def _get_data(
        self, symbol: str, period: str
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """Fetch raw EODHD price data for one symbol at a given period.

        Args:
            symbol (str): The instrument to fetch.
            period (str): The bar period code (for example ``"d"``, ``"w"``,
                ``"m"`` for daily/weekly/monthly, or an intraday code).

        Returns:
            Union[pd.DataFrame, List[Dict[str, Any]]]: The raw response, as a
            DataFrame for end-of-day periods or a list of dicts for intraday.

        Raises:
            ValueError: If no API key is configured.
        """
        if not self.__api_key:
            raise ValueError("API key is required for EODHD data.")
        client = APIClient(api_key=self.__api_key)
        if period in ["d", "w", "m"]:
            return client.get_historical_data(
                symbol=symbol,
                interval=period,
                iso8601_start=self.start_date,
                iso8601_end=self.end_date,
            )
        elif period in ["1m", "5m", "1h"]:
            hms = " 00:00:00"
            fmt = "%Y-%m-%d %H:%M:%S"
            startdt = datetime.strptime(str(self.start_date) + hms, fmt)
            enddt = datetime.strptime(str(self.end_date) + hms, fmt)
            startdt = startdt.replace(tzinfo=timezone("UTC"))
            enddt = enddt.replace(tzinfo=timezone("UTC"))
            unix_start = int(startdt.timestamp())
            unix_end = int(enddt.timestamp())
            return client.get_intraday_historical_data(
                symbol=symbol,
                interval=period,
                from_unix_time=unix_start,
                to_unix_time=unix_end,
            )
        raise ValueError(f"Unsupported period: {period}")

    def _format_data(
        self, data: Union[List[Dict[str, Any]], pd.DataFrame]
    ) -> pd.DataFrame:
        """Normalise raw EODHD data into the standard OHLCV frame.

        Args:
            data (Union[List[Dict[str, Any]], pd.DataFrame]): The raw response
                from :meth:`_get_data`.

        Returns:
            pd.DataFrame: A DataFrame with the handler's OHLCV columns.

        Raises:
            ValueError: If ``data`` is empty.
        """
        if isinstance(data, pd.DataFrame):
            if data.empty or len(data) == 0:
                raise ValueError("No data found.")
            df = data.drop(labels=["symbol", "interval"], axis=1)
            df = df.rename(columns={"adjusted_close": "adj_close"})
            return df

        elif isinstance(data, list):
            if not data or len(data) == 0:
                raise ValueError("No data found.")
            df = pd.DataFrame(data)
            df = df.drop(columns=["timestamp", "gmtoffset"], axis=1)
            df = df.rename(columns={"datetime": "date"})
            df["adj_close"] = df["close"]
            df = df[["date", "open", "high", "low", "close", "adj_close", "volume"]]
            df.date = pd.to_datetime(df.date)
            df = df.set_index("date")
            return df
        raise TypeError(f"Unsupported data type: {type(data)}")

    def _download_and_cache_data(self, cache_dir: Optional[str]) -> str:
        """Downloads and caches historical data as CSV files."""
        _cache_dir = cache_dir or BBSTRADER_DIR / "data" / "eodhd" / self.period
        os.makedirs(_cache_dir, exist_ok=True)
        for symbol in self.symbol_list:
            filepath = os.path.join(_cache_dir, f"{symbol}.csv")
            if _cache_is_fresh(filepath, self.use_cache, self.cache_max_age_days):
                continue
            try:
                data = self._get_data(symbol, self.period)
                data = self._format_data(data)
                data.to_csv(filepath)
            except Exception as e:
                raise ValueError(f"Error downloading {symbol}: {e}")
        return str(_cache_dir)


class FMPDataHandler(BaseCSVDataHandler):
    """
    Downloads historical data from Financial Modeling Prep (FMP).
    Data is fetched using the `financetoolkit` library.

    To use this class, you need to sign up for an API key at
    https://financialmodelingprep.com/developer/docs/pricing and
    provide the key as an argument.

    """

    def __init__(
        self, events: "Queue[MarketEvent]", symbol_list: List[str], **kwargs: Any
    ) -> None:
        """
        Args:
            events (Queue): The Event Queue for passing market events.
            symbol_list (list[str]): List of symbols to download data for.
            fmp_start (str): Start date for historical data (YYYY-MM-DD).
            fmp_end (str): End date for historical data (YYYY-MM-DD).
            data_dir (str, optional): Directory for caching data .
            fmp_period (str, optional): Time period for historical data
                (e.g. daily, weekly, monthly, quarterly, yearly, "1min", "5min", "15min", "30min", "1hour").
            fmp_api_key (str): API key for Financial Modeling Prep.

        Note:
            See `bbstrader.btengine.data.BaseCSVDataHandler` for other arguments.
        """
        self.symbol_list = symbol_list
        self.start_date = kwargs.get("fmp_start")
        self.end_date = kwargs.get("fmp_end", datetime.now().strftime("%Y-%m-%d"))
        self.period = kwargs.get("fmp_period", "daily")
        self.cache_dir = kwargs.get("data_dir")
        self.use_cache = kwargs.get("use_cache", False)
        self.cache_max_age_days = kwargs.get("cache_max_age_days")
        self.__api_key = kwargs.get("fmp_api_key")

        csv_dir = self._download_and_cache_data(self.cache_dir)

        super().__init__(
            events,
            symbol_list,
            str(csv_dir),
            columns=kwargs.get("columns"),
            index_col=kwargs.get("index_col", 0),
        )

    def _get_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch raw FMP price data for one symbol at a given period.

        Args:
            symbol (str): The instrument to fetch.
            period (str): The bar period (for example ``"1d"`` or a financetoolkit
                interval code).

        Returns:
            pd.DataFrame: The raw price history for the symbol.

        Raises:
            ValueError: If no API key is configured.
        """
        if not self.__api_key:
            raise ValueError("API key is required for FMP data.")
        toolkit = Toolkit(
            symbol,
            api_key=self.__api_key,
            start_date=self.start_date,
            end_date=self.end_date,
            benchmark_ticker=None,
            progress_bar=False,
        )
        if period in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            return toolkit.get_historical_data(period=period, progress_bar=False)
        elif period in ["1min", "5min", "15min", "30min", "1hour"]:
            return toolkit.get_intraday_data(period=period, progress_bar=False)
        raise ValueError(f"Unsupported period: {period}")

    def _format_data(self, data: pd.DataFrame, period: str) -> pd.DataFrame:
        """Normalise raw FMP data into the standard OHLCV frame.

        Args:
            data (pd.DataFrame): The raw price history from :meth:`_get_data`.
            period (str): The bar period, used to decide which auxiliary columns
                (return/volatility) to drop.

        Returns:
            pd.DataFrame: A DataFrame with the handler's OHLCV columns.

        Raises:
            ValueError: If ``data`` is empty.
        """
        if data.empty or len(data) == 0:
            raise ValueError("No data found.")
        if period[0].isnumeric():
            data = data.drop(columns=["Return", "Volatility", "Cumulative Return"])
        else:
            data = data.drop(
                columns=[
                    "Dividends",
                    "Return",
                    "Volatility",
                    "Excess Return",
                    "Excess Volatility",
                    "Cumulative Return",
                ]
            )
        data = data.reset_index()
        if "Adj Close" not in data.columns:
            data["Adj Close"] = data["Close"]
        data["date"] = data["date"].dt.to_timestamp()  # type: ignore
        data["date"] = pd.to_datetime(data["date"])
        data.set_index("date", inplace=True)
        return data

    def _download_and_cache_data(self, cache_dir: Optional[str]) -> str:
        """Downloads and caches historical data as CSV files."""
        _cache_dir = cache_dir or BBSTRADER_DIR / "data" / "fmp" / self.period
        os.makedirs(_cache_dir, exist_ok=True)
        for symbol in self.symbol_list:
            filepath = os.path.join(_cache_dir, f"{symbol}.csv")
            if _cache_is_fresh(filepath, self.use_cache, self.cache_max_age_days):
                continue
            try:
                data = self._get_data(symbol, self.period)
                data = self._format_data(data, self.period)
                data.to_csv(filepath)
            except Exception as e:
                raise ValueError(f"Error downloading {symbol}: {e}")
        return str(_cache_dir)
