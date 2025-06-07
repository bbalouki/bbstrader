from datetime import datetime
from typing import Dict, List

import pandas as pd
import yfinance as yf

from bbstrader.btengine.data import EODHDataHandler, FMPDataHandler
from bbstrader.metatrader.rates import download_historical_data
from bbstrader.tseries import (
    find_cointegrated_pairs,
    select_assets,
    select_candidate_pairs,
)

__all__ = [
    "search_coint_candidate_pairs",
]

def _download_and_process_data(source, tickers, start, end, tf, path, **kwargs):
    """Download and process data for a list of tickers from the specified source."""
    data_list = []
    for ticker in tickers:
        try:
            if source == "yf":
                data = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    progress=False,
                    multi_level_index=False,
                    auto_adjust=True,
                )
                if "Adj Close" in data.columns:
                    data = data.drop(columns=["Adj Close"], axis=1)
            elif source == "mt5":
                start, end = pd.Timestamp(start), pd.Timestamp(end)
                data = download_historical_data(
                    symbol=ticker,
                    timeframe=tf,
                    date_from=start,
                    date_to=end,
                    **{"path": path},
                )
                data = data.drop(columns=["adj_close"], axis=1)
            elif source in ["fmp", "eodhd"]:
                handler_class = (
                    FMPDataHandler if source == "fmp" else EODHDataHandler
                )
                handler = handler_class(events=None, symbol_list=[ticker], **kwargs)
                data = handler.data[ticker]
            else:
                raise ValueError(f"Invalid source: {source}")

            data = data.reset_index()
            data = data.rename(columns=str.lower)
            data["ticker"] = ticker
            data_list.append(data)

        except Exception as e:
            print(f"No Data found for {ticker}: {e}")
            continue

    return pd.concat(data_list)

def _handle_date_range(start, end, window):
    """Handle start and end date generation."""
    if start is None or end is None:
        end = pd.Timestamp(datetime.now()).strftime("%Y-%m-%d")
        start = (
            pd.Timestamp(datetime.now())
            - pd.DateOffset(years=window)
            + pd.DateOffset(days=1)
        ).strftime("%Y-%m-%d")
    return start, end

def _period_search(start, end, securities, candidates, window, npairs):
    if window < 3 or (pd.Timestamp(end) - pd.Timestamp(start)).days / 365 < 3:
        raise ValueError(
            "The date range must be at least two (2) years for period search."
        )
    top_pairs = []
    p_start = pd.Timestamp(end) - pd.DateOffset(years=1)
    periods = pd.date_range(start=p_start, end=pd.Timestamp(end), freq="BQE")
    npairs = max(round(npairs / 2), 1)
    for period in periods:
        s_start = period - pd.DateOffset(years=2) + pd.DateOffset(days=1)
        print(f"Searching for pairs in period: {s_start} - {period}")
        pairs = find_cointegrated_pairs(
            securities,
            candidates,
            n=npairs,
            start=str(s_start),
            stop=str(period),
            coint=True,
        )
        pairs["period"] = period
        top_pairs.append(pairs)
    top_pairs = pd.concat(top_pairs)
    if len(top_pairs.columns) <= 1:
        raise ValueError(
            "No pairs found in the specified period."
            "Please adjust the date range or increase the number of pairs."
        )
    return top_pairs.head(npairs * 2)

def _process_asset_data(securities, candidates, universe, rolling_window):
    """Process and select assets from the data."""
    securities = select_assets(
        securities, n=universe, rolling_window=rolling_window
    )
    candidates = select_assets(
        candidates, n=universe, rolling_window=rolling_window
    )
    return securities, candidates


def search_coint_candidate_pairs(
    securities: pd.DataFrame | List[str] = None,
    candidates: pd.DataFrame | List[str] = None,
    start: str = None,
    end: str = None,
    period_search: bool = False,
    select: bool = True,
    source: str = None,
    universe: int = 100,
    window: int = 2,
    rolling_window: int = None,
    npairs: int = 10,
    tf: str = "D1",
    path: str = None,
    **kwargs,
) -> List[Dict[str, str]] | pd.DataFrame:
    """
    Searches for candidate pairs of securities based on cointegration analysis.

    This function either processes preloaded securities and candidates data
    (as pandas DataFrames) or downloads historical data from a specified
    source (e.g., Yahoo Finance, MetaTrader 5, Financial Modeling Prep, or EODHD).
    It then selects the top `npairs` based on cointegration.

    Args:
        securities (pd.DataFrame | List[str], optional):
            A DataFrame or list of tickers representing the securities for analysis.
            If using a DataFrame, it should include a MultiIndex with levels
            ['ticker', 'date'].
        candidates (pd.DataFrame | List[str], optional):
            A DataFrame or list of tickers representing the candidate securities
            for pair selection.
        start (str, optional):
            The start date for data retrieval in 'YYYY-MM-DD' format. Ignored
            if both `securities` and `candidates` are DataFrames.
        end (str, optional):
            The end date for data retrieval in 'YYYY-MM-DD' format. Ignored
            if both `securities` and `candidates` are DataFrames.
        period_search (bool, optional):
            If True, the function will perform a periodic search for cointegrated from 3 years
            to the end date by taking 2 yerars rolling window. So you need to have at least 3 years of data
            or set the `window` parameter to 3. Defaults to False.
        select (bool, optional):
            If True, the function will select the top cointegrated pairs based on the
            cointegration test results in form of List[dict].
            If False, the function will return all cointegrated pairs in form of DataFrame.
            This can be useful for further analysis or visualization.
        source (str, optional):
            The data source for historical data retrieval. Must be one of
            ['yf', 'mt5', 'fmp', 'eodhd']. Required if `securities` and
            `candidates` are lists of tickers.
        universe (int, optional):
            The maximum number of assets to retain for analysis. Defaults to 100.
        window (int, optional):
            The number of years of historical data to retrieve if `start` and `end`
            are not specified. Defaults to 2 years.
        rolling_window (int, optional):
            The size of the rolling window (in days) used for asset selection.
            Defaults to None.
        npairs (int, optional):
            The number of top cointegrated pairs to select. Defaults to 10.
        tf (str, optional):
            The timeframe for MetaTrader 5 data retrieval. Defaults to 'D1'.
        path (str, optional):
            The path to MetaTrader 5 historical data files. Required if `source='mt5'`.
        **kwargs:
            Additional parameters for data retrieval (e.g., API keys, date ranges
            for specific sources), see ``bbstrader.btengine.data.FMPDataHandler`` or
            ``bbstrader.btengine.data.EODHDataHandler`` for more details.

    Returns:
        List[dict]: A list containing the selected top cointegrated pairs if `select=True`.
        pd.DataFrame: A DataFrame containing all cointegrated pairs if `select=False`.

    Raises:
        ValueError: If the inputs are invalid or if the `source` is not one of
        the supported sources.

    Examples:
        Using preloaded DataFrames:
            >>> securities = pd.read_csv('securities.csv', index_col=['ticker', 'date'])
            >>> candidates = pd.read_csv('candidates.csv', index_col=['ticker', 'date'])
            >>> pairs = search_candidate_pairs(securities=securities, candidates=candidates)

        Using a data source (Yahoo Finance):
            >>> securities = ['SPY', 'IWM', 'XLF', 'HYG', 'XLE', 'LQD', 'GDX', 'FXI', 'EWZ', ...]
            >>> candidates = ['AAPL', 'AMZN', 'NVDA', 'MSFT', 'GOOGL', 'AMD', 'BAC', 'NFLX', ...]

            >>> pairs = search_candidate_pairs(
            ...     securities=securities,
            ...     candidates=candidates,
            ...     start='2022-12-12',
            ...     end='2024-12-10',
            ...     source='yf',
            ...     npairs=10
            ... )
            >>>    [
            ...    {'x': 'LQD', 'y': 'TMO'},
            ...    {'x': 'IEF', 'y': 'COP'},
            ...    {'x': 'WMT', 'y': 'IWM'},
            ...    {'x': 'MDT', 'y': 'OIH'},
            ...    {'x': 'EWZ', 'y': 'CMCSA'},
            ...    {'x': 'VLO', 'y': 'XOP'},
            ...    {'x': 'SHY', 'y': 'F'},
            ...    {'x': 'ABT', 'y': 'LQD'},
            ...    {'x': 'PFE', 'y': 'USO'},
            ...    {'x': 'LQD', 'y': 'MDT'}
            ...    ]

        Using MetaTrader 5:
            >>> securities = ['EURUSD', 'GBPUSD']
            >>> candidates = ['USDJPY', 'AUDUSD']
            >>> pairs = search_candidate_pairs(
            ...     securities=securities,
            ...     candidates=candidates,
            ...     source='mt5',
            ...     tf='H1',
            ...     path='/path/to/terminal64.exe',
            ... )

    Notes:
        - If `securities` and `candidates` are DataFrames, the function assumes
          the data is already preprocessed and indexed by ['ticker', 'date'].
        - When using `source='fmp'` or `source='eodhd'`, API keys and other
          required parameters should be passed via `kwargs`.

    """

    if (
        securities is not None
        and candidates is not None
        and isinstance(securities, pd.DataFrame)
        and isinstance(candidates, pd.DataFrame)
    ):
        if isinstance(securities.index, pd.MultiIndex) and isinstance(
            candidates.index, pd.MultiIndex
        ):
            securities, candidates = _process_asset_data(
                securities, candidates, universe, rolling_window
            )
        if period_search:
            start = securities.index.get_level_values("date").min()
            end = securities.index.get_level_values("date").max()
            top_pairs = _period_search(start, end, securities, candidates, window, npairs)
        else:
            top_pairs = find_cointegrated_pairs(
                securities, candidates, n=npairs, coint=True
            )
        if select:
            return select_candidate_pairs(
                top_pairs, period=True if period_search else False
            )
        else:
            return top_pairs

    elif source is not None:
        if source not in ["yf", "mt5", "fmp", "eodhd"]:
            raise ValueError("source must be either 'yf', 'mt5', 'fmp', or 'eodhd'")
        if not isinstance(securities, list) or not isinstance(candidates, list):
            raise ValueError("securities and candidates must be a list of tickers")

        start, end = _handle_date_range(start, end, window)
        if source in ["fmp", "eodhd"]:
            kwargs[f"{source}_start"] = kwargs.get(f"{source}_start") or start
            kwargs[f"{source}_end"] = kwargs.get(f"{source}_end") or end

        securities_data = _download_and_process_data(
            source, securities, start, end, tf, path, **kwargs
        )
        candidates_data = _download_and_process_data(
            source, candidates, start, end, tf, path, **kwargs
        )
        securities_data = securities_data.set_index(["ticker", "date"])
        candidates_data = candidates_data.set_index(["ticker", "date"])
        securities_data, candidates_data = _process_asset_data(
            securities_data, candidates_data, universe, rolling_window
        )
        if period_search:
            top_pairs = _period_search(
                start, end, securities_data, candidates_data, window, npairs
            ).head(npairs)
        else:
            top_pairs = find_cointegrated_pairs(
                securities_data, candidates_data, n=npairs, coint=True
            )
        if select:
            return select_candidate_pairs(
                top_pairs, period=True if period_search else False
            )
        else:
            return top_pairs

    else:
        msg = (
            "Invalid input. Either provide securities"
            "and candidates as DataFrames or specify a data source."
        )
        raise ValueError(msg)
