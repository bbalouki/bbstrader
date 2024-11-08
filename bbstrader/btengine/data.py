import os.path
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import List, Dict
from queue import Queue
from abc import ABCMeta, abstractmethod
from bbstrader.metatrader.rates import download_historical_data
from bbstrader.btengine.event import MarketEvent
from bbstrader.config import BBSTRADER_DIR
from datetime import datetime


__all__ = [
    "DataHandler",
    "CSVDataHandler",
    "MT5DataHandler",
    "YFDataHandler"
]


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
        pass
    @property
    def data(self) -> Dict[str, pd.DataFrame]:
        pass
    @property
    def labels(self) -> List[str]:
        pass
    @property
    def index(self) -> str | List[str]:
        pass

    @abstractmethod
    def get_latest_bar(self, symbol) -> pd.Series:
        """
        Returns the last bar updated.
        """
        raise NotImplementedError(
            "Should implement get_latest_bar()"
        )

    @abstractmethod
    def get_latest_bars(self, symbol, N=1, df=True) -> pd.DataFrame | List[pd.Series]:
        """
        Returns the last N bars updated.
        """
        raise NotImplementedError(
            "Should implement get_latest_bars()"
        )

    @abstractmethod
    def get_latest_bar_datetime(self, symbol) -> datetime | pd.Timestamp:
        """
        Returns a Python datetime object for the last bar.
        """
        raise NotImplementedError(
            "Should implement get_latest_bar_datetime()"
        )

    @abstractmethod
    def get_latest_bar_value(self, symbol, val_type) -> float:
        """
        Returns one of the Open, High, Low, Close, Adj Close, Volume or Returns
        from the last bar.
        """
        raise NotImplementedError(
            "Should implement get_latest_bar_value()"
        )

    @abstractmethod
    def get_latest_bars_values(self, symbol, val_type, N=1) -> np.ndarray:
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """
        raise NotImplementedError(
            "Should implement get_latest_bars_values()"
        )

    @abstractmethod
    def update_bars(self):
        """
        Pushes the latest bars to the bars_queue for each symbol
        in a tuple OHLCVI format: (datetime, Open, High, Low,
        Close, Adj Close, Volume, Retruns).
        """
        raise NotImplementedError(
            "Should implement update_bars()"
        )


class BaseCSVDataHandler(DataHandler):
    """
    Base class for handling data loaded from CSV files.

    """

    def __init__(self, events: Queue, 
                 symbol_list: List[str], 
                 csv_dir: str, 
                 columns: List[str]=None,
                 index_col: str | int | List[str] | List[int] = 0):
        
        """
        Initialises the data handler by requesting the location of the CSV files
        and a list of symbols.
        
        Args:
            events : The Event Queue.
            symbol_list : A list of symbol strings.
            csv_dir : Absolute directory path to the CSV files.
            columns : List of column names to use for the data.
            index_col : Column to use as the index.
        """
        self.events = events
        self.symbol_list = symbol_list
        self.csv_dir = csv_dir
        self.columns = columns
        self.index_col = index_col
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self._index = None
        self._load_and_process_data()

    @property
    def symbols(self)-> List[str]:
        return self.symbol_list
    @property
    def data(self)-> Dict[str, pd.DataFrame]:
        return self.symbol_data
    @property
    def labels(self)-> List[str]:
        return self.columns
    @property
    def index(self)-> str | List[str]:
        return self._index
    
    def _load_and_process_data(self):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.
        """
        default_names = pd.read_csv(
                os.path.join(self.csv_dir, f'{self.symbol_list[0]}.csv')
                ).columns.to_list()
        new_names = self.columns or default_names
        new_names = [name.lower().replace(' ', '_') for name in new_names]
        self.columns = new_names
        assert 'adj_close' in new_names or 'close' in new_names, \
            "Column names must contain 'Adj Close' and 'Close' or adj_close and close"
        comb_index = None
        for s in self.symbol_list:
            # Load the CSV file with no header information,
            # indexed on date
            self.symbol_data[s] = pd.read_csv(
                os.path.join(self.csv_dir, f'{s}.csv'),
                header=0, index_col=self.index_col, parse_dates=True,
                names=new_names
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
            self.symbol_data[s] = self.symbol_data[s].reindex(
                index=comb_index, method='pad'
            )
            self.symbol_data[s]['returns'] = self.symbol_data[s][
                'adj_close' if 'adj_close' in new_names else 'close'
            ].pct_change().dropna()
            self._index = self.symbol_data[s].index.name
            if self.events is not None:
                self.symbol_data[s] = self.symbol_data[s].iterrows()

    def _get_new_bar(self, symbol: str):
        """
        Returns the latest bar from the data feed.
        """
        for b in self.symbol_data[symbol]:
            yield b

    def get_latest_bar(self, symbol: str) -> pd.Series:
        """
        Returns the last bar from the latest_symbol list.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("Symbol not available in the historical data set.")
            raise
        else:
            return bars_list[-1]

    def get_latest_bars(self, symbol: str, N=1, df=True) -> pd.DataFrame | List[pd.Series]:
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("Symbol not available in the historical data set.")
            raise
        else:
            if df:
                df = pd.DataFrame([bar[1] for bar in bars_list[-N:]])
                df.index.name = self._index
                return df
            return bars_list[-N:]

    def get_latest_bar_datetime(self, symbol: str) -> datetime | pd.Timestamp:
        """
        Returns a Python datetime object for the last bar.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("Symbol not available in the historical data set.")
            raise
        else:
            return bars_list[-1][0]

    def get_latest_bars_datetime(self, symbol: str, N=1) -> List[datetime | pd.Timestamp]:
        """
        Returns a list of Python datetime objects for the last N bars.
        """
        try:
            bars_list = self.get_latest_bars(symbol, N)
        except KeyError:
            print("Symbol not available in the historical data set.")
            raise
        else:
            return [b[0] for b in bars_list]

    def get_latest_bar_value(self, symbol: str, val_type: str) -> float:
        """
        Returns one of the Open, High, Low, Close, Volume or OI
        values from the pandas Bar series object.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("Symbol not available in the historical data set.")
            raise
        else:
            try:
                return getattr(bars_list[-1][1], val_type)
            except AttributeError:
                print(f"Value type {val_type} not available in the historical data set.")
                raise

    def get_latest_bars_values(self, symbol: str, val_type: str, N=1) -> np.ndarray:
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """
        try:
            bars_list = self.get_latest_bars(symbol, N, df=False)
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            try:
                return np.array([getattr(b[1], val_type) for b in bars_list])
            except AttributeError:
                print(f"Value type {val_type} not available in the historical data set.")
                raise

    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        for s in self.symbol_list:
            try:
                bar = next(self._get_new_bar(s))
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar)
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

    def __init__(self, events: Queue, symbol_list: List[str], **kwargs):
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
        All csv fille can be strored in 'Home/.bbstrader/csv_data'

        """
        csv_dir = kwargs.get("csv_dir")
        csv_dir =  csv_dir or BBSTRADER_DIR / 'csv_data'
        super().__init__(events, symbol_list, csv_dir)


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

    def __init__(self, events: Queue, symbol_list: List[str], **kwargs):
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
        self.tf = kwargs.get('time_frame', 'D1')
        self.start = kwargs.get('mt5_start', datetime(2000, 1, 1))
        self.end = kwargs.get('mt5_end', datetime.now())
        self.use_utc = kwargs.get('use_utc', False)
        self.filer = kwargs.get('filter', False)
        self.fill_na = kwargs.get('fill_na', False)
        self.lower_cols = kwargs.get('lower_cols', True)
        self.data_dir = kwargs.get('data_dir')
        self.symbol_list = symbol_list
        csv_dir = self._download_data(self.data_dir)
        super().__init__(events, symbol_list, csv_dir)

    def _download_data(self, cache_dir: str):
        data_dir = cache_dir or BBSTRADER_DIR /  'mt5_data' / self.tf 
        data_dir.mkdir(parents=True, exist_ok=True)
        for symbol in self.symbol_list:
            try:
                data = download_historical_data(
                    symbol=symbol,
                    time_frame=self.tf,
                    date_from=self.start, 
                    date_to=self.end, 
                    utc=self.use_utc,
                    filter=self.filer,
                    fill_na=self.fill_na,
                    lower_colnames=self.lower_cols
                )
                if data is None:
                    raise ValueError(f"No data found for {symbol}")
                data.to_csv(data_dir / f'{symbol}.csv')
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

    def __init__(self, events: Queue, symbol_list: List[str], **kwargs):
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
        self.start_date = kwargs.get('yf_start', '2000-01-01')
        self.end_date = kwargs.get('yf_end', datetime.now().strftime('%Y-%m-%d'))
        self.cache_dir = kwargs.get('data_dir')
        csv_dir = self._download_and_cache_data(self.cache_dir)
        super().__init__(events, symbol_list, csv_dir)

    def _download_and_cache_data(self, cache_dir: str):
        """Downloads and caches historical data as CSV files."""
        cache_dir = cache_dir or BBSTRADER_DIR / 'yf_data' / 'daily'
        os.makedirs(cache_dir, exist_ok=True)
        for symbol in self.symbol_list:
            filepath = os.path.join(cache_dir, f"{symbol}.csv")
            try:
                data = yf.download(
                    symbol, start=self.start_date, end=self.end_date, multi_level_index=False)
                if data.empty:
                    raise ValueError(f"No data found for {symbol}")
                data.to_csv(filepath)  # Cache the data
            except Exception as e:
                raise ValueError(f"Error downloading {symbol}: {e}")
        return cache_dir


# TODO # Get data from EODHD
# https://eodhd.com/
class EODHDataHandler(BaseCSVDataHandler):
    ...

# TODO # Get data from FMP using Financialtoolkit API
# https://github.com/bbalouki/FinanceToolkit
class FMPDataHandler(BaseCSVDataHandler):
    ...


class AlgoseekDataHandler(BaseCSVDataHandler):
    ...


class BaseFMPDataHanler(object):
    """
    This will serve as the base class for all other FMP data 
    that is not historical data and does not have an OHLC structure.
    """
    ...


class FMPFundamentalDataHandler(BaseFMPDataHanler):
    ...

# TODO Add other Handlers for FMP


# TODO Add data Handlers for Interactive Brokers
