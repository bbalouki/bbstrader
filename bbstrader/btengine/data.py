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
from eodhd import APIClient
from financetoolkit import Toolkit
from pytz import timezone


__all__ = [
    "DataHandler",
    "CSVDataHandler",
    "MT5DataHandler",
    "YFDataHandler",
    "EODHDataHandler",
    "FMPDataHandler",
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
    def datadir(self)-> str:
        return self.csv_dir
    
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
        new_names = [name.strip().lower().replace(' ', '_') for name in new_names]
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
            self.symbol_data[s].to_csv(os.path.join(self.csv_dir, f'{s}.csv'))
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
        super().__init__(
            events, 
            symbol_list, 
            csv_dir,
            columns  =kwargs.get('columns'),
            index_col=kwargs.get('index_col', 0)
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
        self.tf         = kwargs.get('time_frame', 'D1')
        self.start      = kwargs.get('mt5_start', datetime(2000, 1, 1))
        self.end        = kwargs.get('mt5_end', datetime.now())
        self.use_utc    = kwargs.get('use_utc', False)
        self.filer      = kwargs.get('filter', False)
        self.fill_na    = kwargs.get('fill_na', False)
        self.lower_cols = kwargs.get('lower_cols', True)
        self.data_dir   = kwargs.get('data_dir')
        self.symbol_list = symbol_list
        self.kwargs = kwargs
        
        csv_dir = self._download_and_cache_data(self.data_dir)
        super().__init__(
            events,
            symbol_list,
            csv_dir,
            columns  =kwargs.get('columns'),
            index_col=kwargs.get('index_col', 0)
        )

    def _download_and_cache_data(self, cache_dir: str):
        data_dir = cache_dir or BBSTRADER_DIR /  'mt5' / self.tf 
        data_dir.mkdir(parents=True, exist_ok=True)
        for symbol in self.symbol_list:
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
                    **self.kwargs
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
        self.start_date  = kwargs.get('yf_start')
        self.end_date    = kwargs.get('yf_end', datetime.now())
        self.cache_dir   = kwargs.get('data_dir')
        
        csv_dir = self._download_and_cache_data(self.cache_dir)
        
        super().__init__(
            events,
            symbol_list,
            csv_dir,
            columns  =kwargs.get('columns'),
            index_col=kwargs.get('index_col', 0)
        )

    def _download_and_cache_data(self, cache_dir: str):
        """Downloads and caches historical data as CSV files."""
        cache_dir = cache_dir or BBSTRADER_DIR / 'yfinance' / 'daily'
        os.makedirs(cache_dir, exist_ok=True)
        for symbol in self.symbol_list:
            filepath = os.path.join(cache_dir, f"{symbol}.csv")
            try:
                data = yf.download(
                    symbol, start=self.start_date, end=self.end_date, 
                    multi_level_index=False, progress=False)
                if data.empty:
                    raise ValueError(f"No data found for {symbol}")
                data.to_csv(filepath)
            except Exception as e:
                raise ValueError(f"Error downloading {symbol}: {e}")
        return cache_dir


class EODHDataHandler(BaseCSVDataHandler):
    """
    Downloads historical data from EOD Historical Data.
    Data is fetched using the `eodhd` library.

    To use this class, you need to sign up for an API key at 
    https://eodhistoricaldata.com/ and provide the key as an argument.
    """
    def __init__(self, events: Queue, symbol_list: List[str], **kwargs):
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
        self.start_date  = kwargs.get('eodhd_start')
        self.end_date    = kwargs.get('eodhd_end', datetime.now().strftime('%Y-%m-%d'))
        self.period      = kwargs.get('eodhd_period', 'd')
        self.cache_dir   = kwargs.get('data_dir')
        self.__api_key   = kwargs.get('eodhd_api_key', 'demo')

        csv_dir = self._download_and_cache_data(self.cache_dir)
        
        super().__init__(
            events, 
            symbol_list,
            csv_dir,
            columns  =kwargs.get('columns'),
            index_col=kwargs.get('index_col', 0)
        )

    def _get_data(self, symbol: str, period) -> pd.DataFrame | List[Dict]:
        if not self.__api_key:
            raise ValueError("API key is required for EODHD data.")
        client = APIClient(api_key=self.__api_key)
        if period in ['d', 'w', 'm']:
            return client.get_historical_data(
                symbol=symbol,
                interval=period, 
                iso8601_start=self.start_date,
                iso8601_end=self.end_date,
            )
        elif period in ['1m', '5m', '1h']:
            hms = ' 00:00:00'
            fmt = '%Y-%m-%d %H:%M:%S'
            startdt = datetime.strptime(self.start_date + hms, fmt)
            enddt = datetime.strptime(self.end_date + hms, fmt)
            startdt = startdt.replace(tzinfo=timezone('UTC'))
            enddt = enddt.replace(tzinfo=timezone('UTC'))
            unix_start = int(startdt.timestamp())
            unix_end = int(enddt.timestamp())
            return client.get_intraday_historical_data(
                symbol=symbol,
                interval=period, 
                from_unix_time=unix_start,
                to_unix_time=unix_end,
            )
    
    def _forma_data(self, data: List[Dict] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            if data.empty or len(data) == 0:
                raise ValueError("No data found.")
            df = data.drop(labels=['symbol', 'interval'], axis=1)
            df = df.rename(columns={'adjusted_close': 'adj_close'})
            return df
        
        elif isinstance(data, list):
            if not data or len(data) == 0:
                raise ValueError("No data found.")
            df = pd.DataFrame(data)
            df = df.drop(columns=['timestamp', 'gmtoffset'], axis=1)
            df = df.rename(columns={'datetime': 'date'})
            df['adj_close'] = df['close']
            df = df[['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
            df.date = pd.to_datetime(df.date)
            df = df.set_index('date')
            return df
    
    def _download_and_cache_data(self, cache_dir: str):
        """Downloads and caches historical data as CSV files."""
        cache_dir = cache_dir or BBSTRADER_DIR / 'eodhd' / self.period
        os.makedirs(cache_dir, exist_ok=True)
        for symbol in self.symbol_list:
            filepath = os.path.join(cache_dir, f"{symbol}.csv")
            try:
                data = self._get_data(symbol, self.period)
                data = self._forma_data(data)
                data.to_csv(filepath)
            except Exception as e:
                raise ValueError(f"Error downloading {symbol}: {e}")
        return cache_dir


class FMPDataHandler(BaseCSVDataHandler):
    """
    Downloads historical data from Financial Modeling Prep (FMP).
    Data is fetched using the `financetoolkit` library.

    To use this class, you need to sign up for an API key at
    https://financialmodelingprep.com/developer/docs/pricing and
    provide the key as an argument.

    """
    def __init__(self, events: Queue, symbol_list: List[str], **kwargs):
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
        self.start_date  = kwargs.get('fmp_start')
        self.end_date    = kwargs.get('fmp_end', datetime.now().strftime('%Y-%m-%d'))
        self.period      = kwargs.get('fmp_period', 'daily')
        self.cache_dir   = kwargs.get('data_dir')
        self.__api_key   = kwargs.get('fmp_api_key')

        csv_dir = self._download_and_cache_data(self.cache_dir)
        
        super().__init__(
            events,
            symbol_list,
            csv_dir,
            columns  =kwargs.get('columns'),
            index_col=kwargs.get('index_col', 0)
        )

    def _get_data(self, symbol: str, period: str) -> pd.DataFrame:
        if not self.__api_key:
            raise ValueError("API key is required for FMP data.")
        toolkit = Toolkit(
            symbol,
            api_key=self.__api_key,
            start_date=self.start_date, 
            end_date=self.end_date,
            benchmark_ticker=None,
            progress_bar=False
        )
        if period in ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']:
            return toolkit.get_historical_data(period=period, progress_bar=False)
        elif period in ['1min', '5min', '15min', '30min', '1hour']:
            return toolkit.get_intraday_data(period=period, progress_bar=False)
        
    def _format_data(self, data: pd.DataFrame, period: str) -> pd.DataFrame:
        if data.empty or len(data) == 0:
            raise ValueError("No data found.")
        if period[0].isnumeric():
            data = data.drop(columns=['Return', 'Volatility', 'Cumulative Return'], axis=1)
        else:
            data = data.drop(columns=['Dividends', 'Return', 'Volatility', 
                         'Excess Return', 'Excess Volatility', 
                         'Cumulative Return'], axis=1)
        data = data.reset_index()
        if 'Adj Close' not in data.columns:
            data['Adj Close'] = data['Close']
        data['date'] = data['date'].dt.to_timestamp()
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        return data
    
    def _download_and_cache_data(self, cache_dir: str):
        """Downloads and caches historical data as CSV files."""
        cache_dir = cache_dir or BBSTRADER_DIR / 'fmp' / self.period
        os.makedirs(cache_dir, exist_ok=True)
        for symbol in self.symbol_list:
            filepath = os.path.join(cache_dir, f"{symbol}.csv")
            try:
                data = self._get_data(symbol, self.period)
                data = self._format_data(data, self.period)
                data.to_csv(filepath)
            except Exception as e:
                raise ValueError(f"Error downloading {symbol}: {e}")
        return cache_dir


# TODO Add data Handlers for Interactive Brokers
class TWSDataHandler(BaseCSVDataHandler):
    ...