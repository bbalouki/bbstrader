from datetime import date
import datetime
import os.path
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import List
from queue import Queue
from abc import ABCMeta, abstractmethod
from bbstrader.metatrader.rates import Rates
from bbstrader.btengine.event import MarketEvent
from datetime import datetime


__all__ = [
    "DataHandler",
    "BaseCSVDataHandler",
    "HistoricCSVDataHandler",
    "MT5HistoricDataHandler",
    "YFHistoricDataHandler"
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

    @abstractmethod
    def get_latest_bar(self, symbol):
        """
        Returns the last bar updated.
        """
        raise NotImplementedError(
            "Should implement get_latest_bar()"
        )

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars updated.
        """
        raise NotImplementedError(
            "Should implement get_latest_bars()"
        )

    @abstractmethod
    def get_latest_bar_datetime(self, symbol):
        """
        Returns a Python datetime object for the last bar.
        """
        raise NotImplementedError(
            "Should implement get_latest_bar_datetime()"
        )

    @abstractmethod
    def get_latest_bar_value(self, symbol, val_type):
        """
        Returns one of the Open, High, Low, Close, Adj Close, Volume or Returns
        from the last bar.
        """
        raise NotImplementedError(
            "Should implement get_latest_bar_value()"
        )

    @abstractmethod
    def get_latest_bars_values(self, symbol, val_type, N=1):
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

    def __init__(self, events: Queue, symbol_list: List[str], csv_dir: str):
        self.events = events
        self.symbol_list = symbol_list
        self.csv_dir = csv_dir
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self._load_and_process_data()

    def _load_and_process_data(self):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.
        """
        comb_index = None
        for s in self.symbol_list:
            # Load the CSV file with no header information,
            # indexed on date
            self.symbol_data[s] = pd.read_csv(
                os.path.join(self.csv_dir, f'{s}.csv'),
                header=0, index_col=0, parse_dates=True,
                names=[
                    'Datetime', 'Open', 'High',
                    'Low', 'Close', 'Adj Close', 'Volume'
                ]
            )
            self.symbol_data[s].sort_index(inplace=True)
            # Combine the index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_data[s].index
            else:
                comb_index.union(self.symbol_data[s].index)
            # Set the latest symbol_data to None
            self.latest_symbol_data[s] = []

        # Reindex the dataframes
        for s in self.symbol_list:
            self.symbol_data[s] = self.symbol_data[s].reindex(
                index=comb_index, method='pad'
            )
            self.symbol_data[s]["Returns"] = self.symbol_data[s][
                "Adj Close"
            ].pct_change().dropna()
            self.symbol_data[s] = self.symbol_data[s].iterrows()

    def _get_new_bar(self, symbol: str):
        """
        Returns the latest bar from the data feed.
        """
        for b in self.symbol_data[symbol]:
            yield b

    def get_latest_bar(self, symbol: str):
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

    def get_latest_bars(self, symbol: str, N=1):
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
            return bars_list[-N:]

    def get_latest_bar_datetime(self, symbol: str):
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

    def get_latest_bar_value(self, symbol: str, val_type: str):
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
            return getattr(bars_list[-1][1], val_type)

    def get_latest_bars_values(self, symbol: str, val_type: str, N=1):
        """
        Returns the last N bar values from the
        latest_symbol list, or N-k if less available.
        """
        try:
            bars_list = self.get_latest_bars(symbol, N)
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        else:
            return np.array([getattr(b[1], val_type) for b in bars_list])

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


class HistoricCSVDataHandler(BaseCSVDataHandler):
    """
    `HistoricCSVDataHandler` is designed to read CSV files for
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
            csv_dir (str): Absolute directory path to the CSV files.
            symbol_list (List[str]): A list of symbol strings.
        """
        csv_dir = kwargs.get("csv_dir")
        super().__init__(events, symbol_list, csv_dir)


class MT5HistoricDataHandler(BaseCSVDataHandler):
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
                mt5_data (str): Directory for storing data (default: 'mt5_data').

        Note:
            Requires a working connection to an MT5 terminal.
        """
        self.tf = kwargs.get('time_frame', 'D1')
        self.start = kwargs.get('mt5_start')
        self.end = kwargs.get('mt5_end', datetime.now())
        self.data_dir = kwargs.get('mt5_data', 'mt5_data')
        self.symbol_list = symbol_list
        csv_dir = self._download_data(self.data_dir)
        super().__init__(events, symbol_list, csv_dir)

    def _download_data(self, cache_dir: str):
        data_dir = Path() / cache_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        for symbol in self.symbol_list:
            try:
                rate = Rates(symbol=symbol, time_frame=self.tf)
                data = rate.get_historical_data(
                    date_from=self.start, date_to=self.end
                )
                if data is None:
                    raise ValueError(f"No data found for {symbol}")
                data.to_csv(data_dir / f'{symbol}.csv')
            except Exception as e:
                raise ValueError(f"Error downloading {symbol}: {e}")
        return data_dir


class YFHistoricDataHandler(BaseCSVDataHandler):
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
            cache_dir (str, optional): Directory for caching data (default: 'yf_cache').
        """
        self.symbol_list = symbol_list
        self.start_date = kwargs.get('yf_start')
        self.end_date = kwargs.get('yf_end')
        self.cache_dir = kwargs.get('yf_cache', 'yf_cache')
        csv_dir = self._download_and_cache_data(self.cache_dir)
        super().__init__(events, symbol_list, csv_dir)

    def _download_and_cache_data(self, cache_dir: str):
        """Downloads and caches historical data as CSV files."""
        os.makedirs(cache_dir, exist_ok=True)
        for symbol in self.symbol_list:
            filepath = os.path.join(cache_dir, f"{symbol}.csv")
            if not os.path.exists(filepath):
                try:
                    data = yf.download(
                        symbol, start=self.start_date, end=self.end_date)
                    if data.empty:
                        raise ValueError(f"No data found for {symbol}")
                    data.to_csv(filepath)  # Cache the data
                except Exception as e:
                    raise ValueError(f"Error downloading {symbol}: {e}")
        return cache_dir


# TODO # Get data from EODHD
class EODHDHistoricDataHandler(BaseCSVDataHandler):
    ...

# TODO # Get data from FinancialModelingPrep ()
class FMPHistoricDataHandler(BaseCSVDataHandler):
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
