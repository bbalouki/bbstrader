import pickle

import numpy as np
import pandas as pd
import seaborn as sns

from hmmlearn.hmm import GaussianHMM
from risk_models.risk import RiskModel
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

sns.set_theme()



class HMMRiskManager(RiskModel):
    """
    This class represents a risk management model using Hidden Markov Models (HMM)
    to identify and manage market risks. It inherits from a generic RiskModel class
    and utilizes Gaussian HMM to model the financial market's hidden states. These states
    are used to make decisions on permissible trading actions based on the identified market
    trends, thus facilitating a risk-aware trading strategy.
    """

    def __init__(self, **kwargs):
        """
        Initializes the HMMRiskManager with market data and model parameters.

        Parameters
        ==========
        :param data: DataFrame containing market data. 
            If not provided, data must be loaded via csv_filepath.
        :param states (int): The number of hidden states in the HMM. Default is 2.
        :param iterations (int): The number of iterations for the HMM to converge. Default is 100.
        :param end_date (datetime): The end date for the market data to be considered.
        :param csv_filepath (str): Path to the CSV file containing market data if data is not provided directly.
        :param model_filename (str): Filename to save the trained HMM model.
        :param verbose (bool): If True, prints additional model information. Default is False.
        :param cov_variance (str): Type of covariance to use in the HMM. Default is "diag".
            possibles values are "spherical", "tied", "diag", "full".
            see https://hmmlearn.readthedocs.io/en/latest/api.html#gaussianhmm for more details.
        """
        self.data = kwargs.get("data")
        self.states = kwargs.get("states", 2)
        self.iterations = kwargs.get("iterations", 100)
        self.end_date = kwargs.get("end_date")
        self.csv_filepath = kwargs.get("csv_filepath")
        self.model_filename = kwargs.get("model_filename")
        self.verbose = kwargs.get("verbose", False)
        self.cov_variance = kwargs.get("cov_variance", "diag")
        self.df = self._get_data()
        self.hmm_model = self._fit_model()
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
            print(f"Model Score: {hmm_model.score(returns)}")
        if self.model_filename is not None:
            self.save_hmm_model(hmm_model, self.model_filename)

        return hmm_model

    def get_states(self):
        """
        Predicts the hidden states for the market data 
        and calculates the mean returns and volatility for each state.

        Returns
        =======
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

        Parameters
        ==========
        :param returns_val: Array of recent return values 
            to predict the current state.

        Returns
        =======
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

        Parameters
        ==========
        :param returns_val: Array of recent return values 
            to assess the permissible trade.

        Returns
        =======
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

        Parameters
        ==========
        :param hmm_model: The trained GaussianHMM model to be saved.
        :param filename: The filename under which to save the model.
        """
        print("Pickling HMM model...")
        pickle.dump(hmm_model, open(f"{filename}.pkl", "wb"))
        print("...HMM model pickled.")

    def read_csv_file(self, csv_filepath):
        """
        Reads market data from a CSV file.

        Parameters
        ==========
        :param csv_filepath: Path to the CSV file containing the market data.

        Returns
        =======
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

        Parameters
        ==========
        :param data_frame: DataFrame containing the market data.
        :param end: Optional datetime object specifying the end date for the data.

        Returns
        =======
            DataFrame with returns calculated 
                and data filtered up to the end date if specified.
        """
        df = data_frame.copy()
        df['Returns'] = df['Adj Close'].pct_change()
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

        Parameters
        ==========
        :param hmm_model: The trained GaussianHMM 
            model used for predicting states.
        :param df: DataFrame containing 
            the market data with calculated returns.
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
