import os
import warnings
from itertools import product
from time import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import seaborn as sns
import yfinance as yf
from alphalens import performance as perf
from alphalens import plotting
from alphalens.tears import create_full_tear_sheet, create_summary_tear_sheet
from alphalens.utils import (
    get_clean_factor_and_forward_returns,
    rate_of_return,
    std_conversion,
)
from loguru import logger as log
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

__all__ = ["OneStepTimeSeriesSplit", "MultipleTimeSeriesCV", "LightGBModel"]


class OneStepTimeSeriesSplit:
    __author__ = "Stefan Jansen"
    """Generates tuples of train_idx, test_idx pairs
    Assumes the index contains a level labeled 'date'"""

    def __init__(self, n_splits=3, test_period_length=1, shuffle=False):
        self.n_splits = n_splits
        self.test_period_length = test_period_length
        self.shuffle = shuffle

    @staticmethod
    def chunks(l, n):  # noqa: E741
        for i in range(0, len(l), n):
            yield l[i : i + n]

    def split(self, X: pd.DataFrame, y=None, groups=None):
        unique_dates = (
            X.index.get_level_values("date")
            .unique()
            .sort_values(ascending=False)[: self.n_splits * self.test_period_length]
        )

        dates = X.reset_index()[["date"]]
        for test_date in self.chunks(unique_dates, self.test_period_length):
            train_idx = dates[dates.date < min(test_date)].index
            test_idx = dates[dates.date.isin(test_date)].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx, test_idx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


class MultipleTimeSeriesCV:
    __author__ = "Stefan Jansen"
    """
    Generates tuples of train_idx, test_idx pairs
    Assumes the MultiIndex contains levels 'symbol' and 'date'
    purges overlapping outcomes
    """

    def __init__(
        self,
        n_splits=3,
        train_period_length=126,
        test_period_length=21,
        lookahead=None,
        date_idx="date",
        shuffle=False,
    ):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle
        self.date_idx = date_idx

    def split(self, X: pd.DataFrame, y=None, groups=None):
        unique_dates = X.index.get_level_values(self.date_idx).unique()
        days = sorted(unique_dates, reverse=True)
        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            split_idx.append(
                [train_start_idx, train_end_idx, test_start_idx, test_end_idx]
            )

        dates = X.reset_index()[[self.date_idx]]
        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = dates[
                (dates[self.date_idx] > days[train_start])
                & (dates[self.date_idx] <= days[train_end])
            ].index
            test_idx = dates[
                (dates[self.date_idx] > days[test_start])
                & (dates[self.date_idx] <= days[test_end])
            ].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx.to_numpy(), test_idx.to_numpy()

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


class LightGBModel(object):
    """
    ``LightGBModel`` encapsulates a complete workflow for training and evaluating
    a ``LightGBM (Light Gradient Boosting Machine)`` model for predicting stock returns.
    It includes data acquisition, feature engineering, model tuning, and performance
    evaluation using information ``coefficient (IC)`` and Alphalens analysis.

    Key Features
    ------------
    - ``HDF5 Storage``: Utilizes ``pandas.HDFStore`` for efficient storage and retrieval
      of large datasets, which is essential for backtesting on financial time series data.

    - ``Time-Series Cross-Validation``: Employs a custom cross-validation strategy that
      respects the time series nature of the data, avoiding data leakage.

    - ``Hyperparameter Tuning``: Includes automated hyperparameter tuning using a randomized
      grid search for optimization.

    - ``Information Coefficient (IC)``: Uses IC as a core performance metric that quantifies
      the predictive power of the model, which is a standard measure for ranking models in finance.

    - ``Alphalens Integration``: Provides a comprehensive framework for validating model
      performance using Alphalens, allowing for in-depth performance analysis, like backtesting
      and return decomposition.

    Use Case
    --------
    This class is designed for quantitative finance and algorithmic trading use cases where
    the goal is to build a predictive model for stock returns based on historical data and
    technical indicators. It follows a complete cycle from data acquisition to model validation
    and provides the infrastructure needed for deployment of this model in a trading strategy.

    Notes
    -----
    The implementation is inspired by the book "Machine Learning for Algorithmic Trading"
    by Stefan Jansen.

    References
    ----------
    Stefan Jansen (2020). Machine Learning for Algorithmic Trading - Second Edition.
    Chapter 12, Boosting Your Trading Strategy.
    """

    def __init__(
        self,
        data: pd.DataFrame = None,
        datastore: pd.HDFStore = "lgbdata.h5",
        trainstore: pd.HDFStore = "lgbtrain.h5",
        outstore: pd.HDFStore = "lgbout.h5",
        logger=None,
    ):
        """
        Args:
            data (pd.DataFrame): The input data for the model. It should be a DataFrame with a MultiIndex containing
            'symbol' and 'date' levels. If not provided, the data can be downloaded using the `download_boosting_data` method.
            datastore (str): The path to the HDF5 file for storing the model data.
            trainstore (str): The path to the HDF5 file for storing the training data.
            outstore (str): The path to the HDF5 file for storing the output data.
            logger (Logger): Optional logger instance for logging messages. If not provided, a default logger will be used.
        """
        self.datastore = datastore
        self.trainstore = trainstore
        self.outstore = outstore
        self.logger = logger or log
        if data is not None:
            data.reset_index().to_hdf(path_or_buf=self.datastore, key="model_data")

    def _compute_bb(self, close):
        # Compute Bollinger Bands using pandas_ta
        bb = ta.bbands(close, length=20)
        return pd.DataFrame(
            {"bb_high": bb["BBU_20_2.0"], "bb_low": bb["BBL_20_2.0"]}, index=close.index
        )

    def _compute_atr(self, stock_data):
        # Compute ATR using pandas_ta
        atr = ta.atr(stock_data.high, stock_data.low, stock_data.close, length=14)
        return (atr - atr.mean()) / atr.std()

    def _compute_macd(self, close):
        # Compute MACD using pandas_ta
        macd = ta.macd(close)["MACD_12_26_9"]
        return (macd - macd.mean()) / macd.std()

    def _add_technical_indicators(self, prices: pd.DataFrame):
        prices = prices.copy()

        # Add RSI and normalize
        prices["rsi"] = (
            prices.groupby(level="symbol")
            .close.apply(lambda x: ta.rsi(x, length=14))
            .reset_index(level=0, drop=True)
        )

        # Add Bollinger Bands
        bb = prices.groupby(level="symbol").close.apply(self._compute_bb)
        bb = bb.reset_index(level=1, drop=True)
        prices = prices.join(bb)

        prices["bb_high"] = (
            prices.bb_high.sub(prices.close).div(prices.bb_high).apply(np.log1p)
        )
        prices["bb_low"] = (
            prices.close.sub(prices.bb_low).div(prices.close).apply(np.log1p)
        )

        # Add ATR and normalize
        prices["ATR"] = prices.groupby(level="symbol", group_keys=False).apply(
            lambda x: self._compute_atr(x)
        )

        # Add MACD and normalize
        prices["MACD"] = prices.groupby(level="symbol", group_keys=False).close.apply(
            self._compute_macd
        )

        return prices

    def download_boosting_data(self, tickers, start, end=None):
        data = []
        for ticker in tickers:
            try:
                prices = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    progress=False,
                    multi_level_index=False,
                    auto_adjust=True,
                )
                if prices.empty:
                    continue
                prices["symbol"] = ticker
                data.append(prices)
            except:  # noqa: E722
                continue
        data = pd.concat(data)
        if "Adj Close" in data.columns:
            data = data.drop(columns=["Adj Close"])
        data = (
            data.rename(columns={s: s.lower().replace(" ", "_") for s in data.columns})
            .set_index("symbol", append=True)
            .swaplevel()
            .sort_index()
            .dropna()
        )
        return data

    def download_metadata(self, tickers):
        def clean_text_column(series: pd.Series) -> pd.Series:
            return (
                series.str.lower()
                # use regex=False for literal string replacements
                .str.replace("-", "", regex=False)
                .str.replace("&", "and", regex=False)
                .str.replace(" ", "_", regex=False)
                .str.replace("__", "_", regex=False)
            )

        metadata = [
            "industry",
            "sector",
            "exchange",
            "symbol",
            "heldPercentInsiders",
            "heldPercentInstitutions",
            "overallRisk",
            "shortRatio",
            "dividendYield",
            "beta",
            "regularMarketVolume",
            "averageVolume",
            "averageVolume10days",
            "bid",
            "ask",
            "bidSize",
            "askSize",
            "marketCap",
        ]

        columns = {
            "industry": "industry",
            "sector": "sector",
            "exchange": "exchange",
            "symbol": "symbol",
            "heldPercentInsiders": "insiders",
            "heldPercentInstitutions": "institutions",
            "overallRisk": "risk",
            "shortRatio": "short_ratio",
            "dividendYield": "dyield",
            "beta": "beta",
            "regularMarketVolume": "regvolume",
            "averageVolume": "avgvolume",
            "averageVolume10days": "avgvolume10",
            "bid": "bid",
            "ask": "ask",
            "bidSize": "bidsize",
            "askSize": "asksize",
            "marketCap": "marketcap",
        }
        data = []
        for symbol in tickers:
            try:
                symbol_info = yf.Ticker(symbol).info
            except:  # noqa: E722
                continue
            infos = {}
            for info in metadata:
                infos[info] = symbol_info.get(info)
            data.append(infos)
        metadata = pd.DataFrame(data)
        metadata = metadata.rename(columns=columns)
        metadata.dyield = metadata.dyield.fillna(0)
        metadata.sector = clean_text_column(metadata.sector)
        metadata.industry = clean_text_column(metadata.industry)
        metadata = metadata.set_index("symbol")
        return metadata

    def _select_nlargest_liquidity_stocks(
        self,
        df: pd.DataFrame,
        n: int,
        volume_features,
        bid_ask_features,
        market_cap_feature,
    ):
        df = df.copy()
        scaler = StandardScaler()

        # Normalize features
        df[volume_features] = scaler.fit_transform(df[volume_features])
        df["bid_ask_spread"] = df["ask"] - df["bid"]
        df["bid_ask_spread"] = scaler.fit_transform(df[["bid_ask_spread"]])
        df[market_cap_feature] = scaler.fit_transform(df[market_cap_feature])

        # Calculate Liquidity Score
        # Assign weights to each component (these weights can be adjusted based on importance)
        weights = {"volume": 0.4, "bid_ask_spread": 0.2, "marketCap": 0.4}

        # Calculate the liquidity score by combining the normalized features
        df["liquidity_score"] = (
            weights["volume"] * df[volume_features].mean(axis=1)
            + weights["bid_ask_spread"] * df["bid_ask_spread"]
            + weights["marketCap"] * df[market_cap_feature[0]]
        )
        df_sorted = df.sort_values(by="liquidity_score", ascending=False)

        return df_sorted.nlargest(n, "liquidity_score").index

    def _encode_metadata(self, df: pd.DataFrame):
        df = df.copy()
        # Binning each numerical feature into categories
        df["insiders"] = pd.qcut(
            df["insiders"], q=4, labels=["Very Low", "Low", "High", "Very High"]
        )
        df["institutions"] = pd.qcut(
            df["institutions"], q=4, labels=["Very Low", "Low", "High", "Very High"]
        )
        df["risk"] = pd.cut(
            df["risk"],
            bins=[-float("inf"), 3, 5, 7, float("inf")],
            labels=["Low", "Medium", "High", "Very High"],
        )
        df["short_ratio"] = pd.qcut(
            df["short_ratio"], q=4, labels=["Very Low", "Low", "High", "Very High"]
        )
        df["dyield"] = pd.cut(
            df["dyield"],
            bins=[-float("inf"), 0.002, 0.005, 0.01, float("inf")],
            labels=["Very Low", "Low", "High", "Very High"],
        )
        df["beta"] = pd.cut(
            df["beta"],
            bins=[-float("inf"), 0.8, 1.0, 1.2, float("inf")],
            labels=["Low", "Moderate", "High", "Very High"],
        )

        # Encode binned features
        binned_features = [
            "insiders",
            "institutions",
            "risk",
            "short_ratio",
            "dyield",
            "beta",
            "sector",
            "industry",
            "exchange",
        ]
        label_encoders = {}

        for col in binned_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        return df, label_encoders

    def prepare_boosting_data(
        self,
        prices: pd.DataFrame,
        metadata: pd.DataFrame = None,
        min_years=7,
        universe=500,
    ):
        if metadata is None:
            mcap = False
            tickers = prices.index.get_level_values("symbol").unique()
            metadata = self.download_metadata(tickers)
        else:
            mcap = True
        YEAR = 252
        idx = pd.IndexSlice
        percentiles = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05]
        percentiles += [1 - p for p in percentiles[::-1]]
        T = [1, 5, 10, 21, 42, 63]

        prices.volume /= 1e3  # make vol figures a bit smaller
        prices.index.names = ["symbol", "date"]
        metadata.index.name = "symbol"
        prices.reset_index().to_hdf(path_or_buf=self.datastore, key="stock_data")
        metadata.reset_index().to_hdf(path_or_buf=self.datastore, key="stock_metadata")

        # Remove stocks with insufficient observations
        min_obs = min_years * YEAR
        nobs = prices.groupby(level="symbol").size()
        keep = nobs[nobs > min_obs].index
        prices = prices.loc[idx[keep, :], :]

        # # Remove duplicate symbols
        prices = prices[~prices.index.duplicated()]

        # Align price and meta data
        metadata = metadata[~metadata.index.duplicated() & metadata.sector.notnull()]
        metadata.sector = metadata.sector.str.lower().str.replace(" ", "_")
        shared = (
            prices.index.get_level_values("symbol")
            .unique()
            .intersection(metadata.index)
        )
        metadata = metadata.loc[shared, :]
        prices = prices.loc[idx[shared, :], :]

        # Limit universe
        if mcap:
            universe = metadata.marketcap.nlargest(universe).index
        else:
            volume_features = ["regvolume", "avgvolume", "avgvolume10"]
            bid_ask_features = ["bid", "ask", "bidsize", "asksize"]
            market_cap_feature = ["marketcap"]
            to_drop = volume_features + bid_ask_features + market_cap_feature
            universe = self._select_nlargest_liquidity_stocks(
                metadata,
                universe,
                volume_features,
                bid_ask_features,
                market_cap_feature,
            )
            metadata = metadata.drop(to_drop, axis=1)
        prices = prices.loc[idx[universe, :], :]
        metadata = metadata.loc[universe]
        metadata = self._encode_metadata(metadata)[0]

        prices["dollar_vol"] = prices[["close", "volume"]].prod(1).div(1e3)
        # compute dollar volume to determine universe
        dollar_vol_ma = (
            prices.dollar_vol.unstack("symbol")
            .rolling(window=21, min_periods=1)  # 1 trading month
            .mean()
        )

        # Rank stocks by moving average
        prices["dollar_vol_rank"] = (
            dollar_vol_ma.rank(axis=1, ascending=False).stack("symbol").swaplevel()
        )
        # Add some Basic Factors
        prices = self._add_technical_indicators(prices)
        # Combine Price and Meta Data
        prices = prices.join(metadata)

        # Compute Returns
        by_sym = prices.groupby(level="symbol").close
        for t in T:
            prices[f"r{t:02}"] = by_sym.pct_change(t)
        # Daily historical return deciles
        for t in T:
            # Reset the index to apply qcut by date without grouping errors
            prices[f"r{t:02}dec"] = (
                prices.reset_index(level="date")
                .groupby("date")[f"r{t:02}"]
                .apply(lambda x: pd.qcut(x, q=10, labels=False, duplicates="drop"))
                .values
            )
        # Daily sector return deciles
        for t in T:
            prices[f"r{t:02}q_sector"] = prices.groupby(["date", "sector"])[
                f"r{t:02}"
            ].transform(lambda x: pd.qcut(x, q=5, labels=False, duplicates="drop"))
        # Compute Forward Returns
        for t in [1, 5, 21]:
            prices[f"r{t:02}_fwd"] = prices.groupby(level="symbol")[f"r{t:02}"].shift(
                -t
            )

        # Remove outliers
        outliers = prices[prices.r01 > 1].index.get_level_values("symbol").unique()
        prices = prices.drop(outliers, level="symbol")
        # Create time and sector dummy variables
        prices["year"] = prices.index.get_level_values("date").year
        prices["month"] = prices.index.get_level_values("date").month
        prices["weekday"] = prices.index.get_level_values("date").weekday
        # Store Model Data
        prices = prices.drop(["open", "close", "low", "high", "volume"], axis=1)
        if "adj_close" in prices.columns:
            prices = prices.drop("adj_close", axis=1)
        prices.reset_index().to_hdf(path_or_buf=self.datastore, key="model_data")
        return prices.sort_index()

    def tickers(self):
        return pd.read_hdf(self.outstore, "lgb/tickers").tolist()

    def load_model_data(self, key="model_data"):
        return (
            pd.read_hdf(self.datastore, key=key)
            .set_index(["symbol", "date"])
            .sort_index()
        )

    def format_time(self, t):
        """Return a formatted time string 'HH:MM:SS
        based on a numeric time() value"""
        m, s = divmod(t, 60)
        h, m = divmod(m, 60)
        return f"{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}"

    def fit(self, data: pd.DataFrame, verbose=True):
        def get_fi(model):
            """Return normalized feature importance as pd.Series"""
            fi = model.feature_importance(importance_type="gain")
            return pd.Series(fi / fi.sum(), index=model.feature_name())

        def ic_lgbm(preds, train_data):
            """Custom IC eval metric for lightgbm"""
            is_higher_better = True
            return "ic", spearmanr(preds, train_data.get_label())[0], is_higher_better

        data = data.dropna()
        # Hyperparameter options
        YEAR = 252
        base_params = dict(boosting="gbdt", objective="regression", verbose=-1)

        # constraints on structure (depth) of each tree
        max_depths = [2, 3, 5, 7]
        num_leaves_opts = [2**i for i in max_depths]
        min_data_in_leaf_opts = [250, 500, 1000]

        # weight of each new tree in the ensemble
        learning_rate_ops = [0.01, 0.1, 0.3]

        # random feature selection
        feature_fraction_opts = [0.3, 0.6, 0.95]

        param_names = [
            "learning_rate",
            "num_leaves",
            "feature_fraction",
            "min_data_in_leaf",
        ]

        cv_params = list(
            product(
                learning_rate_ops,
                num_leaves_opts,
                feature_fraction_opts,
                min_data_in_leaf_opts,
            )
        )
        n_params = len(cv_params)
        print(f"# Parameters: {n_params}")

        # Train/Test Period Lengths
        lookaheads = [1, 5, 21]
        train_lengths = [int(4.5 * 252), 252]
        test_lengths = [63]
        test_params = list(product(lookaheads, train_lengths, test_lengths))
        n = len(test_params)
        test_param_sample = np.random.choice(list(range(n)), size=int(n), replace=False)
        test_params = [test_params[i] for i in test_param_sample]
        print("Train configs:", len(test_params))

        # Categorical Variables
        categoricals = ["year", "weekday", "month"]
        for feature in categoricals:
            data[feature] = pd.factorize(data[feature], sort=True)[0]

        # ### Run Cross-Validation
        labels = sorted(data.filter(like="fwd").columns)
        features = data.columns.difference(labels).tolist()
        label_dict = dict(zip(lookaheads, labels))
        num_iterations = [10, 25, 50, 75] + list(range(100, 501, 50))
        num_boost_round = num_iterations[-1]

        metric_cols = (
            param_names
            + [
                "t",
                "daily_ic_mean",
                "daily_ic_mean_n",
                "daily_ic_median",
                "daily_ic_median_n",
            ]
            + [str(n) for n in num_iterations]
        )

        for lookahead, train_length, test_length in test_params:
            # randomized grid search
            cvp = np.random.choice(
                list(range(n_params)), size=int(n_params / 2), replace=False
            )
            cv_params_ = [cv_params[i] for i in cvp]

            # set up cross-validation
            n_splits = int(2 * YEAR / test_length)
            print(
                f"Lookahead: {lookahead:2.0f} | "
                f"Train: {train_length:3.0f} | "
                f"Test: {test_length:2.0f} | "
                f"Params: {len(cv_params_):3.0f} | "
                f"Train configs: {len(test_params)}"
            )

            # time-series cross-validation
            cv = MultipleTimeSeriesCV(
                n_splits=n_splits,
                lookahead=lookahead,
                test_period_length=test_length,
                train_period_length=train_length,
            )

            label = label_dict[lookahead]
            outcome_data = data.loc[:, features + [label]].dropna()

            # binary dataset
            lgb_data = lgb.Dataset(
                data=outcome_data.drop(label, axis=1),
                label=outcome_data[label],
                categorical_feature=categoricals,
                free_raw_data=False,
            )
            T = 0
            predictions, metrics = [], []

            # iterate over (shuffled) hyperparameter combinations
            for p, param_vals in enumerate(cv_params_):
                key = f"{lookahead}/{train_length}/{test_length}/" + "/".join(
                    [str(p) for p in param_vals]
                )
                params = dict(zip(param_names, param_vals))
                params.update(base_params)

                start = time()
                cv_preds = []

                # iterate over folds
                for i, (train_idx, test_idx) in enumerate(cv.split(X=outcome_data)):
                    # select train subset
                    lgb_train = lgb_data.subset(
                        used_indices=train_idx.tolist(), params=params
                    ).construct()

                    # train model for num_boost_round
                    model = lgb.train(
                        params=params,
                        train_set=lgb_train,
                        num_boost_round=num_boost_round,
                    )
                    # log feature importance
                    if i == 0:
                        fi = get_fi(model).to_frame()
                    else:
                        fi[i] = get_fi(model)

                    # capture predictions
                    test_set = outcome_data.iloc[test_idx, :]
                    X_test = test_set.loc[:, model.feature_name()]
                    y_test = test_set.loc[:, label]
                    y_pred = {
                        str(n): model.predict(X_test, num_iteration=n)
                        for n in num_iterations
                    }

                    # record predictions for each fold
                    cv_preds.append(
                        y_test.to_frame("y_test").assign(**y_pred).assign(i=i)
                    )

                # combine fold results
                cv_preds = pd.concat(cv_preds).assign(**params)
                predictions.append(cv_preds)

                # compute IC per day
                by_day = cv_preds.groupby(level="date")
                ic_by_day = pd.concat(
                    [
                        by_day.apply(
                            lambda x: spearmanr(x.y_test, x[str(n)])[0]
                        ).to_frame(n)
                        for n in num_iterations
                    ],
                    axis=1,
                )
                daily_ic_mean = ic_by_day.mean()
                daily_ic_mean_n = daily_ic_mean.idxmax()
                daily_ic_median = ic_by_day.median()
                daily_ic_median_n = daily_ic_median.idxmax()

                # compute IC across all predictions
                ic = [
                    spearmanr(cv_preds.y_test, cv_preds[str(n)])[0]
                    for n in num_iterations
                ]
                t = time() - start
                T += t

                # collect metrics
                metrics = pd.Series(
                    list(param_vals)
                    + [
                        t,
                        daily_ic_mean.max(),
                        daily_ic_mean_n,
                        daily_ic_median.max(),
                        daily_ic_median_n,
                    ]
                    + ic,
                    index=metric_cols,
                )
                if verbose:
                    msg = f"\t{p:3.0f} | {self.format_time(T)} ({t:3.0f}) | {params['learning_rate']:5.2f} | "
                    msg += f"{params['num_leaves']:3.0f} | {params['feature_fraction']:3.0%} | {params['min_data_in_leaf']:4.0f} | "
                    msg += f" {max(ic):6.2%} | {ic_by_day.mean().max(): 6.2%} | {daily_ic_mean_n: 4.0f} | {ic_by_day.median().max(): 6.2%} | {daily_ic_median_n: 4.0f}"
                    print(msg)

                # persist results for given CV run and hyperparameter combination
                metrics.to_hdf(path_or_buf=self.trainstore, key="metrics/" + key)
                ic_by_day.assign(**params).to_hdf(
                    path_or_buf=self.trainstore, key="daily_ic/" + key
                )
                fi.T.describe().T.assign(**params).to_hdf(
                    path_or_buf=self.trainstore, key="fi/" + key
                )
                cv_preds.to_hdf(
                    path_or_buf=self.trainstore, key="predictions/" + key, append=True
                )

    def _get_lgb_metrics(self, scope_params, lgb_train_params, daily_ic_metrics):
        with pd.HDFStore(self.trainstore) as store:
            for i, key in enumerate(
                [k[1:] for k in store.keys() if k[1:].startswith("metrics")]
            ):
                _, t, train_length, test_length = key.split("/")[:4]
                attrs = {
                    "lookahead": t,
                    "train_length": train_length,
                    "test_length": test_length,
                }
                s = store[key].to_dict()
                s.update(attrs)
                if i == 0:
                    lgb_metrics = pd.Series(s).to_frame(i)
                else:
                    lgb_metrics[i] = pd.Series(s)

        id_vars = scope_params + lgb_train_params + daily_ic_metrics
        lgb_metrics = (
            pd.melt(
                lgb_metrics.T.drop("t", axis=1),
                id_vars=id_vars,
                value_name="ic",
                var_name="boost_rounds",
            )
            .dropna()
            .apply(pd.to_numeric)
        )
        return lgb_metrics

    def _get_lgb_ic(self, int_cols, scope_params, lgb_train_params, id_vars):
        lgb_ic = []
        with pd.HDFStore(self.trainstore) as store:
            keys = [k[1:] for k in store.keys()]
            for key in keys:
                _, t, train_length, test_length = key.split("/")[:4]
                if key.startswith("daily_ic"):
                    df = (
                        store[key]
                        .drop(["boosting", "objective", "verbose"], axis=1)
                        .assign(
                            lookahead=t,
                            train_length=train_length,
                            test_length=test_length,
                        )
                    )
                    lgb_ic.append(df)
            lgb_ic = pd.concat(lgb_ic).reset_index()
        lgb_ic = pd.melt(
            lgb_ic, id_vars=id_vars, value_name="ic", var_name="boost_rounds"
        ).dropna()
        lgb_ic.loc[:, int_cols] = lgb_ic.loc[:, int_cols].astype(int)
        return lgb_ic

    def _get_lgb_params(self, data, scope_params, lgb_train_params, t=5, best=0):
        param_cols = scope_params[1:] + lgb_train_params + ["boost_rounds"]
        df = data[data.lookahead == t].sort_values("ic", ascending=False).iloc[best]
        return df.loc[param_cols]

    def _get_lgb_key(self, t, p):
        key = f"{t}/{int(p.train_length)}/{int(p.test_length)}/{p.learning_rate}/"
        return (
            key + f"{int(p.num_leaves)}/{p.feature_fraction}/{int(p.min_data_in_leaf)}"
        )

    def _select_ic(self, params, ic_data, lookahead):
        return ic_data.loc[
            (ic_data.lookahead == lookahead)
            & (ic_data.train_length == params.train_length)
            & (ic_data.test_length == params.test_length)
            & (ic_data.learning_rate == params.learning_rate)
            & (ic_data.num_leaves == params.num_leaves)
            & (ic_data.feature_fraction == params.feature_fraction)
            & (ic_data.boost_rounds == params.boost_rounds),
            ["date", "ic"],
        ].set_index("date")

    def get_trade_prices(self, tickers, start, end):
        idx = pd.IndexSlice
        with pd.HDFStore(self.datastore) as store:
            data = store.select("stock_data")
            data = data.set_index(["symbol", "date"]).sort_index()
            data = data[~data.index.duplicated()]
        try:
            data = (
                data.loc[idx[tickers, start:end], "open"]
                .unstack("symbol")
                .sort_index()
                .shift(-1)
                .tz_convert("UTC")
            )
        except TypeError:
            data = (
                data.loc[idx[tickers, start:end], "open"]
                .unstack("symbol")
                .sort_index()
                .shift(-1)
                .tz_localize("UTC")
            )
        return data

    def plot_ic(self, lgb_ic, lgb_daily_ic, scope_params, lgb_train_params):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
        axes = axes.flatten()
        for i, t in enumerate([1, 21]):
            params = self._get_lgb_params(
                lgb_daily_ic, scope_params, lgb_train_params, t=t, best=0
            )
            data = self._select_ic(params, lgb_ic, lookahead=t).sort_index()
            rolling = data.rolling(63).ic.mean().dropna()
            avg = data.ic.mean()
            med = data.ic.median()
            rolling.plot(
                ax=axes[i],
                title=f"Horizon: {t} Day(s) | IC: Mean={avg * 100:.2f}   Median={med * 100:.2f}",
            )
            axes[i].axhline(avg, c="darkred", lw=1)
            axes[i].axhline(0, ls="--", c="k", lw=1)

        fig.suptitle("3-Month Rolling Information Coefficient", fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)

    def plot_metrics(self, lgb_metrics, lgb_daily_ic, t=1):
        # Visualization
        sns.jointplot(x=lgb_metrics.daily_ic_mean, y=lgb_metrics.ic)

        sns.catplot(
            x="lookahead",
            y="ic",
            col="train_length",
            row="test_length",
            data=lgb_metrics,
            kind="box",
        )
        sns.catplot(
            x="boost_rounds",
            y="ic",
            col="train_length",
            row="test_length",
            data=lgb_daily_ic[lgb_daily_ic.lookahead == t],
            kind="box",
        )

    def get_best_predictions(
        self, lgb_daily_ic, scope_params, lgb_train_params, lookahead=1, topn=10
    ):
        for best in range(topn):
            best_params = self._get_lgb_params(
                lgb_daily_ic, scope_params, lgb_train_params, t=lookahead, best=best
            )
            key = self._get_lgb_key(lookahead, best_params)
            rounds = str(int(best_params.boost_rounds))
            if best == 0:
                best_predictions = pd.read_hdf(self.trainstore, "predictions/" + key)
                best_predictions = best_predictions[rounds].to_frame(best)
            else:
                best_predictions[best] = pd.read_hdf(
                    self.trainstore, "predictions/" + key
                )[rounds]
        best_predictions = best_predictions.sort_index()
        best_predictions.reset_index().to_hdf(
            path_or_buf=self.outstore, key=f"lgb/train/{lookahead:02}"
        )
        return best_predictions

    def apply_alphalen_analysis(self, factor_data, tearsheet=True, verbose=True):
        # Compute Alphalens metrics
        mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
            factor_data,
            by_date=True,
            by_group=False,
            demeaned=True,
            group_adjust=False,
        )
        factor_returns = perf.factor_returns(factor_data)
        mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
            factor_data, by_group=False, demeaned=True
        )

        mean_quant_rateret = mean_quant_ret.apply(
            rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
        )

        mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
            factor_data,
            by_date=True,
            by_group=False,
            demeaned=True,
            group_adjust=False,
        )

        mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
            rate_of_return,
            base_period=mean_quant_ret_bydate.columns[0],
        )

        compstd_quant_daily = std_quant_daily.apply(
            std_conversion, base_period=std_quant_daily.columns[0]
        )

        alpha_beta = perf.factor_alpha_beta(factor_data, demeaned=True)

        mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
            mean_quant_rateret_bydate,
            factor_data["factor_quantile"].max(),
            factor_data["factor_quantile"].min(),
            std_err=compstd_quant_daily,
        )
        if verbose:
            print(
                mean_ret_spread_quant.mean()
                .mul(10000)
                .to_frame("Mean Period Wise Spread (bps)")
                .join(alpha_beta.T)
                .T
            )

        fig, axes = plt.subplots(ncols=3, figsize=(18, 4))

        plotting.plot_quantile_returns_bar(mean_quant_rateret, ax=axes[0])
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=0)
        axes[0].set_xlabel("Quantile")

        plotting.plot_cumulative_returns_by_quantile(
            mean_quant_ret_bydate["1D"],
            freq=pd.tseries.offsets.BDay(),
            period="1D",
            ax=axes[1],
        )
        axes[1].set_title("Cumulative Return by Quantile (1D Period)")

        title = "Cumulative Return - Factor-Weighted Long/Short PF (1D Period)"
        plotting.plot_cumulative_returns(
            factor_returns["1D"],
            period="1D",
            freq=pd.tseries.offsets.BDay(),
            title=title,
            ax=axes[2],
        )

        fig.suptitle("Alphalens - Validation Set Performance", fontsize=14)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

        # Summary Tearsheet
        create_summary_tear_sheet(factor_data)
        create_full_tear_sheet(factor_data)

    def evaluate(self, remove_instore=False, lookahead=1, verbose=True):
        scope_params = ["lookahead", "train_length", "test_length"]
        daily_ic_metrics = [
            "daily_ic_mean",
            "daily_ic_mean_n",
            "daily_ic_median",
            "daily_ic_median_n",
        ]
        lgb_train_params = [
            "learning_rate",
            "num_leaves",
            "feature_fraction",
            "min_data_in_leaf",
        ]

        lgb_metrics = self._get_lgb_metrics(
            scope_params, lgb_train_params, daily_ic_metrics
        )
        # Summary Metrics by Fold
        lgb_metrics.to_hdf(path_or_buf=self.outstore, key="lgb/metrics")

        # Information Coefficient by Day
        int_cols = ["lookahead", "train_length", "test_length", "boost_rounds"]
        id_vars = ["date"] + scope_params + lgb_train_params
        lgb_ic = self._get_lgb_ic(int_cols, scope_params, lgb_train_params, id_vars)
        lgb_ic.to_hdf(path_or_buf=self.outstore, key="lgb/ic")
        lgb_daily_ic = (
            lgb_ic.groupby(id_vars[1:] + ["boost_rounds"])
            .ic.mean()
            .to_frame("ic")
            .reset_index()
        )
        lgb_daily_ic.to_hdf(path_or_buf=self.outstore, key="lgb/daily_ic")

        # Cross-validation Result: Best Hyperparameters
        if verbose:
            print(
                lgb_daily_ic.groupby("lookahead", group_keys=False).apply(
                    lambda x: x.nlargest(3, "ic")
                )
            )
        lgb_metrics.groupby("lookahead", group_keys=False).apply(
            lambda x: x.nlargest(3, "ic")
        )
        lgb_metrics.groupby("lookahead", group_keys=False).apply(
            lambda x: x.nlargest(3, "ic")
        ).to_hdf(path_or_buf=self.outstore, key="lgb/best_model")
        if verbose:
            print(
                lgb_metrics.groupby("lookahead", group_keys=False).apply(
                    lambda x: x.nlargest(3, "daily_ic_mean")
                )
            )

        # Visualization
        if verbose:
            self.plot_metrics(lgb_metrics, lgb_daily_ic, t=lookahead)

        # AlphaLens Analysis - Validation Performance
        lgb_daily_ic = pd.read_hdf(self.outstore, "lgb/daily_ic")
        best_params = self._get_lgb_params(
            lgb_daily_ic, scope_params, lgb_train_params, t=lookahead, best=0
        )
        best_params.to_hdf(path_or_buf=self.outstore, key="lgb/best_params")

        if verbose:
            self.plot_ic(lgb_ic, lgb_daily_ic, scope_params, lgb_train_params)

        # Get Predictions for Validation Period
        best_predictions = self.get_best_predictions(
            lgb_daily_ic, scope_params, lgb_train_params, lookahead=lookahead, topn=10
        )
        test_tickers = best_predictions.index.get_level_values("symbol").unique()
        start = best_predictions.index.get_level_values("date").min()
        end = best_predictions.index.get_level_values("date").max()
        trade_prices = self.get_trade_prices(test_tickers, start, end)
        pd.Series(test_tickers).to_hdf(path_or_buf=self.outstore, key="lgb/tickers")
        # We average the top five models and provide the corresponding prices to Alphalens,
        # in order to compute the mean period-wise
        # return earned on an equal-weighted portfolio invested in the daily factor quintiles
        # for various holding periods:
        try:
            factor = (
                best_predictions.iloc[:, :5]
                .mean(1)
                .dropna()
                .tz_convert("UTC", level="date")
                .swaplevel()
            )
        except TypeError:
            factor = (
                best_predictions.iloc[:, :5]
                .mean(1)
                .dropna()
                .tz_localize("UTC", level="date")
                .swaplevel()
            )
        # Create AlphaLens Inputs
        if verbose:
            factor_data = get_clean_factor_and_forward_returns(
                factor=factor,
                prices=trade_prices,
                quantiles=5,
                periods=(1, 5, 10, 21),
                max_loss=1,
            )
            self.apply_alphalen_analysis(factor_data, tearsheet=True, verbose=True)
        # Delete the temporary files
        if remove_instore:
            os.remove(self.trainstore)

    def make_predictions(
        self, data: pd.DataFrame, mode="test", lookahead=1, verbose=True
    ):
        data = data.copy()
        YEAR = 252
        scope_params = ["lookahead", "train_length", "test_length"]
        lgb_train_params = [
            "learning_rate",
            "num_leaves",
            "feature_fraction",
            "min_data_in_leaf",
        ]

        base_params = dict(boosting="gbdt", objective="regression", verbose=-1)

        categoricals = ["year", "month", "weekday"]
        labels = sorted(data.filter(like="_fwd").columns)
        features = data.columns.difference(labels).tolist()
        label = f"r{lookahead:02}_fwd"
        for feature in categoricals:
            data[feature] = pd.factorize(data[feature], sort=True)[0]

        if mode == "test":
            data = data.dropna().sort_index()
        elif mode == "live":
            data[labels] = data[labels].fillna(0)
            data = data.sort_index().dropna()
        else:
            raise ValueError("Mode must be either 'test' or 'live'.")

        lgb_data = lgb.Dataset(
            data=data[features],
            label=data[label],
            categorical_feature=categoricals,
            free_raw_data=False,
        )
        # Generate predictions
        lgb_daily_ic = pd.read_hdf(self.outstore, "lgb/daily_ic")

        for position in range(10):
            params = self._get_lgb_params(
                lgb_daily_ic, scope_params, lgb_train_params, t=lookahead, best=position
            )

            params = params.to_dict()

            for p in ["min_data_in_leaf", "num_leaves"]:
                params[p] = int(params[p])
            train_length = int(params.pop("train_length"))
            test_length = int(params.pop("test_length"))
            num_boost_round = int(params.pop("boost_rounds"))
            params.update(base_params)
            if verbose:
                print(f"\nPosition: {position:02}")

            # 1-year out-of-sample period
            n_splits = int(YEAR / test_length)
            cv = MultipleTimeSeriesCV(
                n_splits=n_splits,
                test_period_length=test_length,
                lookahead=lookahead,
                train_period_length=train_length,
            )

            predictions = []
            for i, (train_idx, test_idx) in enumerate(cv.split(X=data), 1):
                if verbose:
                    print(i, end=" ", flush=True)
                lgb_train = lgb_data.subset(
                    used_indices=train_idx.tolist(), params=params
                ).construct()

                model = lgb.train(
                    params=params,
                    train_set=lgb_train,
                    num_boost_round=num_boost_round,
                )

                test_set = data.iloc[test_idx, :]
                y_test = test_set.loc[:, label].to_frame("y_test")
                y_pred = model.predict(test_set.loc[:, model.feature_name()])
                predictions.append(y_test.assign(prediction=y_pred))

            if position == 0:
                test_predictions = pd.concat(predictions).rename(
                    columns={"prediction": position}
                )
            else:
                test_predictions[position] = pd.concat(predictions).prediction

        by_day = test_predictions.groupby(level="date")
        for position in range(10):
            if position == 0:
                ic_by_day = by_day.apply(
                    lambda x: spearmanr(x.y_test, x[position])[0]
                ).to_frame()
            else:
                ic_by_day[position] = by_day.apply(
                    lambda x: spearmanr(x.y_test, x[position])[0]
                )
        if verbose:
            print(ic_by_day.describe())
        test_predictions.reset_index().to_hdf(
            path_or_buf=self.outstore, key=f"lgb/test/{lookahead:02}"
        )
        return test_predictions

    def load_predictions(self, predictions=None, lookahead=1):
        if predictions is None:
            predictions = pd.concat(
                [
                    pd.read_hdf(self.outstore, f"lgb/train/{lookahead:02}"),
                    pd.read_hdf(self.outstore, f"lgb/test/{lookahead:02}").drop(
                        "y_test", axis=1
                    ),
                ]
            )
            predictions = predictions.set_index(["symbol", "date"])

        predictions = (
            predictions.loc[~predictions.index.duplicated()]
            .iloc[:, :10]
            .mean(1)
            .sort_index()
            .dropna()
            .to_frame("prediction")
        )
        tickers = predictions.index.get_level_values("symbol").unique().tolist()
        try:
            return (predictions.unstack("symbol").prediction.tz_convert("UTC")), tickers
        except TypeError:
            return (
                predictions.unstack("symbol").prediction.tz_localize("UTC")
            ), tickers

    def assert_last_date(self, predictions: pd.DataFrame):
        """
        Usefull in Live Trading to ensure that the last date in the predictions
        is the previous day, so it predicts today's returns.
        """
        last_date: pd.Timestamp
        last_date = predictions.index.get_level_values("date").max()
        now = pd.Timestamp.now(tz="UTC")
        try:
            if last_date.tzinfo is None:
                last_date = last_date.tz_localize("UTC")
            else:
                last_date = last_date.tz_convert("UTC")
            last_date = last_date.normalize()
        except Exception as e:
            self.logger.error(f"Error getting last date: {e}")
        try:
            if now.weekday() == 0:  # Monday
                expected_date = (now - pd.Timedelta(days=3)).normalize()  # last Friday
            else:
                expected_date = (now - pd.Timedelta(days=1)).normalize()  # yesterday

            assert last_date == expected_date or last_date == now.normalize()
            return True
        except AssertionError:
            yesterday = (now - pd.Timedelta(days=1)).normalize()
            last_friday = (now - pd.Timedelta(days=now.weekday() + 3)).normalize()
            self.logger.debug(
                f"Last date in predictions ({last_date}) is not equal to "
                f"yesterday ({yesterday}) or last Friday ({last_friday})"
            )
            return False

    def clean_stores(self, *stores):
        for store in stores:
            if os.path.exists(store):
                os.remove(store)
