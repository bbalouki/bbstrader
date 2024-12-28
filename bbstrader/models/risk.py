import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from hmmlearn.hmm import GaussianHMM
from abc import ABCMeta, abstractmethod
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from typing import Optional, Dict
from bbstrader.metatrader.rates import Rates
sns.set_theme()

__all__ = [
    "RiskModel",
    "HMMRiskManager",
    "build_hmm_models"
]


class RiskModel(metaclass=ABCMeta):
    """
    The RiskModel class serves as an abstract base for implementing 
    risk management strategies in financial markets. It is designed 
    to assist in the decision-making process regarding which trades 
    are permissible under current market conditions and how to allocate 
    assets effectively to optimize the risk-reward ratio.

    Risk management is a critical component in trading and investment 
    strategies, aiming to minimize potential losses without significantly 
    reducing the potential for gains. This class encapsulates the core 
    principles of risk management by providing a structured approach to 
    evaluate market conditions and manage asset allocation.

    Implementing classes are required to define two key methods:

    - `which_trade_allowed`: 
      Determines the types of trades that are permissible based on the 
      analysis of current market conditions and the risk profile of the portfolio. 
      This method should analyze the provided `returns_val` parameter, 
      which could represent historical returns, volatility measures, 
      or other financial metrics, to decide on the suitability of executing 
      certain trades.

    - `which_quantity_allowed`: 
      Defines how assets should be allocated across the portfolio to maintain 
      an optimal balance between risk and return. This involves determining 
      the quantity of each asset that can be held, considering factors such as 
      diversification, liquidity, and the asset's volatility. This method ensures 
      that the portfolio adheres to predefined risk tolerance levels and 
      investment objectives.

    Note:
        Implementing these methods requires a deep understanding of risk 
        management theories, market analysis, and portfolio management principles. 
        The implementation should be tailored to the specific needs of the 
        investment strategy and the risk tolerance of the investor or the fund 
        being managed.
    """

    @abstractmethod
    def which_trade_allowed(self, returns_val):
        """
        Determines the types of trades permissible under current market conditions.

        Parameters:
            returns_val: A parameter representing financial metrics 
            such as historical returns or volatility, used to 
            assess market conditions.
        """
        raise NotImplementedError("Should implement which_trade_allowed()")

    @abstractmethod
    def which_quantity_allowed(self):
        """
        Defines the strategy for asset allocation within 
        the portfolio to optimize risk-reward ratio.
        """
        raise NotImplementedError("Should implement which_quantity_allowed()")


class HMMRiskManager(RiskModel):
    """
    This class represents a risk management model using Hidden Markov Models (HMM)
    to identify and manage market risks. It inherits from a generic RiskModel class
    and utilizes Gaussian HMM to model the financial market's hidden states. These states
    are used to make decisions on permissible trading actions based on the identified market
    trends, thus facilitating a risk-aware trading strategy.

    To learn more about about the Hidden Markov model,
    See https://en.wikipedia.org/wiki/Hidden_Markov_model

    Exemple:
        >>> # Assuming `data` is a DataFrame containing your market data
        >>> risk_manager = HMMRiskManager(data=data, states=3, iterations=200, verbose=True)
        >>> current_regime = risk_manager.get_current_regime(data['Returns'].values)
        >>> print(f"Current Market Regime: {current_regime}")

    """

    def __init__(
            self,
            data: Optional[pd.DataFrame] = None,
            states: int = 2,
            iterations: int = 100,
            end_date: Optional[datetime] = None,
            csv_filepath: Optional[str] = None,
            **kwargs):
        """
        Initializes the HMMRiskManager with market data and model parameters.

        Args:
            `data` : DataFrame containing market data.
                If not provided, data must be loaded via csv_filepath.

            `states` (int): The number of hidden states in the HMM.
            `iterations` (int): The number of iterations for the HMM to converge.
            `end_date` (datetime): The end date for the market data to be considered.

            `csv_filepath` (str): Path to the CSV file containing 
                market data if data is not provided directly.

            kwarg (dict): Additional arguments
                - `model_filename` (str): Filename to save the trained HMM model.
                - `verbose` (bool): If True, prints additional model information.

                - `cov_variance` (str): Type of covariance to use in the HMM.
                    possibles values are "spherical", "tied", "diag", "full".
                    see https://hmmlearn.readthedocs.io/en/latest/api.html#gaussianhmm for more details.

        """
        self.data = data
        self.states = states
        self.iterations = iterations
        self.end_date = end_date
        self.csv_filepath = csv_filepath

        self.model_filename = kwargs.get("model_filename")
        self.verbose = kwargs.get("verbose", True)
        self.cov_variance = kwargs.get("cov_variance", "diag")

        self.df = self._get_data()
        self.hmm_model = self._fit_model()
        if self.verbose:
            self.show_hidden_states()
        trends = self.identify_market_trends()
        self.bullish_state = trends['bullish']
        self.bearish_state = trends['bearish']
        self.allowed_regimes = [s for s in trends.values()]

    def _get_data(self):
        """
        Retrieves market data for the model either 
            from a provided DataFrame or a CSV file.
        """
        if self.data is not None:
            return self.data
        elif self.csv_filepath is not None:
            return self.read_csv_file(self.csv_filepath)
        else:
            raise ValueError("No data source provided.")

    def _fit_model(self):
        """
        Fits the HMM model to the market data 
            and saves the model if a filename is provided.
        """
        df = self.df.copy()
        data = self.obtain_prices_df(df, end=self.end_date)
        returns = np.column_stack([data["Returns"]])

        hmm_model = GaussianHMM(
            n_components=self.states, covariance_type=self.cov_variance,
            n_iter=self.iterations
        ).fit(returns)

        if self.verbose:
            print(
                f"Hidden Markov model (HMM) Score: {hmm_model.score(returns)}")
        if self.model_filename is not None:
            self.save_hmm_model(hmm_model, self.model_filename)

        return hmm_model

    def get_states(self):
        """
        Predicts the hidden states for the market data 
        and calculates the mean returns and volatility for each state.

        Returns:
            DataFrame containing mean returns 
            and volatility for each hidden state.
        """
        df = self.df.copy()
        data = self.obtain_prices_df(df, end=self.end_date)
        returns = np.column_stack([data["Returns"]])
        states = self.hmm_model.predict(returns)
        data['State'] = states
        # Calculating mean and volatility for each state
        state_stats = data.groupby(
            'State'
        )['Returns'].agg(['mean', 'std']).rename(
            columns={'mean': 'Mean Returns', 'std': 'Volatility'})
        return state_stats

    def identify_market_trends(self):
        """
        Identifies bullish and bearish market trends
          based on the mean returns and volatility of each state.

        Returns:
            A dictionary with keys 'bullish' and 'bearish' 
            indicating the identified states.
        """
        df = self.get_states()
        # Sort the df based on Mean Returns and then by lower Volatility
        sorted_df = df.sort_values(by=['Mean Returns', 'Volatility'],
                                   ascending=[False, True])

        # The first row after sorting will be the bullish state
        # (highest mean return, lower volatility preferred)
        bullish_state = sorted_df.index[0]

        # The last row will be the bearish state
        # (as it has the lowest mean return)
        bearish_state = sorted_df.index[-1]

        return {"bullish": bullish_state, "bearish": bearish_state}

    def get_current_regime(self, returns_val):
        """
        Determines the current market regime based on the latest returns.

        Args:
            returns_val : Array of recent return values 
            to predict the current state.

        Returns:
            The predicted current market state.
        """
        returns = returns_val[~(np.isnan(returns_val) | np.isinf(returns_val))]
        features = np.array(returns).reshape(-1, 1)
        current_regime = self.hmm_model.predict(features)[-1]
        return current_regime

    def which_trade_allowed(self, returns_val):
        """
        Decides whether a long or short trade 
        is allowed based on the current market regime.
        This method override the which_trade_allowed() from 
        RiskModel class . 

        Args:
            returns_val : Array of recent return values 
            to assess the permissible trade.

        Returns:
            A string indicating "LONG" or "SHORT" 
                if a trade is allowed, or None if no trade is permitted.
        """
        state = self.get_current_regime(returns_val)
        if state in self.allowed_regimes:
            trade = "LONG" if state == self.bullish_state else "SHORT"
            return trade
        else:
            return None

    def which_quantity_allowed(self):
        ...

    def save_hmm_model(self, hmm_model, filename):
        """
        Saves the trained HMM model to a pickle file.

        Args:
            hmm_model : The trained GaussianHMM model to be saved.
            filename : The filename under which to save the model.
        """
        print("Pickling HMM model...")
        pickle.dump(hmm_model, open(f"{filename}.pkl", "wb"))
        print("...HMM model pickled.")

    def read_csv_file(self, csv_filepath):
        """
        Reads market data from a CSV file.

        Args:
            csv_filepath : Path to the CSV file containing the market data.

        Returns:
            DataFrame containing the parsed market data.
        """
        df = pd.read_csv(csv_filepath, header=0,
                         names=["Date", "Open", "High", "Low",
                                "Close", "Adj Close", "Volume"],
                         index_col="Date", parse_dates=True)
        return df

    def obtain_prices_df(self, data_frame, end=None):
        """
        Processes the market data to calculate returns 
        and optionally filters data up to a specified end date.

        Args:
            data_frame : DataFrame containing the market data.
            end : Optional datetime object specifying the end date for the data.

        Returns:
            DataFrame with returns calculated 
                and data filtered up to the end date if specified.
        """
        df = data_frame.copy()
        if 'Returns' or 'returns' not in df.columns:
            if 'Close'  in df.columns:
                df['Returns'] = df['Close'].pct_change()
            elif 'Adj Close' in df.columns:
                df['Returns'] = df['Adj Close'].pct_change()
            else:
                raise ValueError("No 'Close' or 'Adj Close' columns found.")
        elif 'returns' in df.columns:
            df.rename(columns={'returns': 'Returns'}, inplace=True)
        if end is not None:
            df = df[:end.strftime('%Y-%m-%d')]
        df.dropna(inplace=True)
        return df

    def show_hidden_states(self):
        """
        Visualizes the market data 
            and the predicted hidden states from the HMM model.
        """
        data = self.df.copy()
        df = self.obtain_prices_df(data, end=self.end_date)
        self.plot_hidden_states(self.hmm_model, df)

    def plot_hidden_states(self, hmm_model, df):
        """
        Plots the adjusted closing prices masked 
        by the in-sample hidden states as a mechanism 
        to understand the market regimes.

        Args:
            hmm_model : The trained GaussianHMM model used for predicting states.
            df : DataFrame containing the market data with calculated returns.
        """
        hidden_states = hmm_model.predict(df[['Returns']])
        fig, axs = plt.subplots(hmm_model.n_components,
                                sharex=True, sharey=True)
        colours = cm.rainbow(np.linspace(0, 1, hmm_model.n_components))

        for i, (ax, colour) in enumerate(zip(axs, colours)):
            mask = hidden_states == i
            ax.plot_date(df.index[mask], df['Adj Close']
                         [mask], ".", linestyle='none', c=colour)
            ax.set_title(f"Hidden State #{i}")
            ax.xaxis.set_major_locator(YearLocator())
            ax.xaxis.set_minor_locator(MonthLocator())
            ax.grid(True)
        plt.show()

def build_hmm_models(symbol_list=None, **kwargs
                     ) -> Dict[str, HMMRiskManager]:
    mt5_data = kwargs.get("use_mt5_data", False)
    data = kwargs.get("hmm_data")
    tf = kwargs.get("time_frame", 'D1')
    hmm_end =  kwargs.get("hmm_end", 0)
    sd = kwargs.get("session_duration", 23.0)
    hmm_tickers = kwargs.get("hmm_tickers")
    if hmm_tickers is not None:
        symbols = hmm_tickers
    else:
        symbols = symbol_list
    hmm_models = {symbol: None for symbol in symbols}
    if data is not None:
        if isinstance(data, pd.DataFrame):
            hmm_data = data
            hmm = HMMRiskManager(
                data=hmm_data, verbose=True, iterations=1000, **kwargs)
            for symbol in symbols:
                hmm_models[symbol] = hmm
        elif isinstance(data, dict):
            for symbol, data in data.items():
                hmm = HMMRiskManager(
                    data=data, verbose=True, iterations=1000, **kwargs)
                hmm_models[symbol] = hmm
    if mt5_data:
        for symbol in symbols:
            rates = Rates(symbol, timeframe=tf, start_pos=hmm_end, session_duration=sd, **kwargs)
            data = rates.get_rates_from_pos()
            assert data is not None, f"No data for {symbol}"
            hmm = HMMRiskManager(
                data=data, verbose=True, iterations=1000, **kwargs)
            hmm_models[symbol] = hmm
    return hmm_models