import numpy as np
import seaborn as sns
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima
sns.set_theme()


class ArimaGarchStrategy():
    """
    This class implements a trading strategy 
    that combines `ARIMA (AutoRegressive Integrated Moving Average)` 
    and `GARCH (Generalized Autoregressive Conditional Heteroskedasticity)` models
    to predict future returns based on historical price data. 
    It inherits from a abstract `Strategy` class and implement `calculate_signals()`.

    The strategy is implemented in the following steps:
    1. Data Preparation: Load and prepare the historical price data.
    2. Modeling: Fit the ARIMA model to the data and then fit the GARCH model to the residuals.
    3. Prediction: Predict the next return using the ARIMA model and the next volatility using the GARCH model.
    4. Trading Strategy: Execute the trading strategy based on the predictions.
    5. Vectorized Backtesting: Backtest the trading strategy using the historical data.
    """

    def __init__(self, symbol, data, k: int = 252):
        """
        Initializes the ArimaGarchStrategy class.

        Parameters
        ==========
        :param symbol (str): The ticker symbol for the financial instrument.
        :param data (pd.DataFrame): `The raw dataset containing at least the 'Close' prices`.
        :param k (int): The window size for rolling prediction in backtesting.
        :param jobs (int): The number of parallel jobs to run for ARIMA model fitting.
        """
        self.symbol = symbol
        self.data = self.load_and_prepare_data(data)
        self.k = k

    # Step 1: Data Preparation
    def load_and_prepare_data(self, df):
        """
        Prepares the dataset by calculating logarithmic returns 
            and differencing if necessary.

        Parameters
        ==========
        :param df (pd.DataFrame): `The raw dataset containing at least the 'Close' prices`.

        Returns
        ========
            pd.DataFrame: The dataset with additional columns 
                for log returns and differenced log returns.
        """
        # Load data
        data = df.copy()
        # Calculate logarithmic returns
        data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
        # Differencing if necessary
        data['diff_log_return'] = data['log_return'].diff()
        data.fillna(0., inplace=True)
        return data

    # Step 2: Modeling (ARIMA + GARCH)
    def fit_best_arima(self, window_data):
        """
        Fits the ARIMA model to the provided window of data, 
            selecting the best model based on AIC.

        Parameters
        ==========
        :param window_data (np.array): The dataset for a specific window period.

        Returns
        ========
            ARIMA model: The best fitted ARIMA model based on AIC.
        """
        model = auto_arima(
            window_data,
            start_p=1,
            start_q=1,
            max_p=6,
            max_q=6,
            seasonal=False,
            stepwise=True
        )
        final_order = model.order
        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        best_arima_model = ARIMA(window_data, order=final_order, missing='drop').fit()
        return best_arima_model

    def fit_garch(self, window_data):
        """
        Fits the GARCH model to the residuals of the best ARIMA model.

        Parameters
        ==========
        :param window_data (np.array): The dataset for a specific window period.

        Returns
        =======
            tuple: Contains the ARIMA result and GARCH result.
        """
        arima_result = self.fit_best_arima(window_data)
        resid = np.asarray(arima_result.resid)
        resid = resid[~(np.isnan(resid) | np.isinf(resid))]
        garch_model = arch_model(resid, p=1, q=1, rescale=False)
        garch_result = garch_model.fit(disp='off')
        return arima_result, garch_result

    def show_arima_garch_results(self, window_data, acf=True, test_resid=True):
        """
        Displays the ARIMA and GARCH model results, including plotting 
        ACF of residuals and conducting , Box-Pierce and Ljung-Box tests.

        Parameters
        ==========
        :param window_data (np.array): The dataset for a specific window period.
        :param acf (bool, optional): If True, plot the ACF of residuals. Defaults to True.
        :param test_resid (bool, optional): If True, 
            conduct Box-Pierce and Ljung-Box tests on residuals. Defaults to True.
        """
        arima_result = self.fit_best_arima(window_data)
        resid = np.asarray(arima_result.resid)
        resid = resid[~(np.isnan(resid) | np.isinf(resid))]
        garch_model = arch_model(resid, p=1, q=1, rescale=False)
        garch_result = garch_model.fit(disp='off')
        residuals = garch_result.resid

        # TODO : Plot the ACF of the residuals
        if acf:
            fig = plt.figure(figsize=(12, 8))
            # Plot the ACF of ARIMA residuals
            ax1 = fig.add_subplot(211, ylabel='ACF')
            plot_acf(resid, alpha=0.05, ax=ax1, title='ACF of ARIMA Residuals')
            ax1.set_xlabel('Lags')
            ax1.grid(True)

            # Plot the ACF of GARCH residuals on the same axes
            ax2 = fig.add_subplot(212, ylabel='ACF')
            plot_acf(residuals, alpha=0.05, ax=ax2,
                     title='ACF of GARCH  Residuals')
            ax2.set_xlabel('Lags')
            ax2.grid(True)

            # Plot the figure
            plt.tight_layout()
            plt.show()

        # TODO : Conduct Box-Pierce and Ljung-Box Tests of the residuals
        if test_resid:
            print(arima_result.summary())
            print(garch_result.summary())
            bp_test = acorr_ljungbox(resid, return_df=True)
            print("Box-Pierce and Ljung-Box Tests Results  for ARIMA:\n", bp_test)
    
    # Step 3: Prediction
    def predict_next_return(self, arima_result, garch_result):
        """
        Predicts the next return using the ARIMA model 
            and the next volatility using the GARCH model.

        Parameters
        ==========
        :param arima_result (ARIMA model): The ARIMA model result.
        :param garch_result (GARCH model): The GARCH model result.

        Returns
        =======
            float: The predicted next return.
        """
        # Predict next value with ARIMA
        arima_pred = arima_result.forecast(steps=1)
        # Predict next volatility with GARCH
        garch_pred = garch_result.forecast(horizon=1)
        next_volatility = garch_pred.variance.iloc[-1, 0]

        # Combine predictions (return + volatility)
        next_return = arima_pred.values[0] + next_volatility
        return next_return

    def get_prediction(self, window_data):
        """
        Generates a prediction for the next return based on a window of data.

        Parameters
        ==========
        :param window_data (np.array): The dataset for a specific window period.

        Returns
        =======
            float: The predicted next return.
        """
        arima_result, garch_result = self.fit_garch(window_data)
        prediction = self.predict_next_return(arima_result, garch_result)
        return prediction

    def calculate_signals(self, window_data):
        """
        Calculates the trading signal based on the prediction.

        Parameters
        ==========
        :param window_data (np.array): The dataset for a specific window period.

        Returns
        =======
            str: The trading signal ('LONG', 'SHORT', or None).
        """
        prediction = self.get_prediction(window_data)
        if prediction > 0:
            signal = "LONG"
        elif prediction < 0:
            signal = "SHORT"
        else:
            signal = None
        return signal

    # Step 4: Trading Strategy

    def execute_trading_strategy(self, predictions):
        """
        Executes the trading strategy based on a list 
        of predictions, determining positions to take.

        Parameters
        ==========
        :param predictions (list): A list of predicted returns.

        Returns
        =======
            list: A list of positions (1 for 'LONG', -1 for 'SHORT', 0 for 'HOLD').
        """
        positions = []  # Long if 1, Short if -1
        previous_position = 0  # Initial position
        for prediction in predictions:
            if prediction > 0:
                current_position = 1  # Long
            elif prediction < 0:
                current_position = -1  # Short
            else:
                current_position = previous_position  # Hold previous position
            positions.append(current_position)
            previous_position = current_position

        return positions

    # Step 5: Vectorized Backtesting
    def generate_predictions(self):
        """
        Generator that yields predictions one by one.
        """
        data = self.data
        window_size = self.k
        for i in range(window_size, len(data)):
            print(
                f"Processing window {i - window_size + 1}/{len(data) - window_size}...")
            window_data = data['diff_log_return'].iloc[i-window_size:i]
            next_return = self.get_prediction(window_data)
            yield next_return

    def backtest_strategy(self):
        """
        Performs a backtest of the strategy over 
        the entire dataset, plotting cumulative returns.
        """
        data = self.data
        window_size = self.k
        print(
            f"Starting backtesting for {self.symbol}\n"
            f"Window size {window_size}.\n"
            f"Total iterations: {len(data) - window_size}.\n")
        predictions_generator = self.generate_predictions()
        
        positions = self.execute_trading_strategy(predictions_generator)

        strategy_returns = np.array(
            positions[:-1]) * data['log_return'].iloc[window_size+1:].values
        buy_and_hold = data['log_return'].iloc[window_size+1:].values
        buy_and_hold_returns = np.cumsum(buy_and_hold)
        cumulative_returns = np.cumsum(strategy_returns)
        dates = data.index[window_size+1:]
        self.plot_cumulative_returns(
            cumulative_returns, buy_and_hold_returns, dates)
        
        print("\nBacktesting completed !!")

    # Function to plot the cumulative returns
    def plot_cumulative_returns(self, strategy_returns, buy_and_hold_returns, dates):
        """
        Plots the cumulative returns of the ARIMA+GARCH strategy against 
            a buy-and-hold strategy.

        Parameters
        ==========
        :param strategy_returns (np.array): Cumulative returns from the strategy.
        :param buy_and_hold_returns (np.array): Cumulative returns from a buy-and-hold strategy.
        :param dates (pd.Index): The dates corresponding to the returns.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(dates, strategy_returns, label='ARIMA+GARCH ', color='blue')
        plt.plot(dates, buy_and_hold_returns, label='Buy & Hold', color='red')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')
        plt.title(f'ARIMA+GARCH Strategy vs. Buy & Hold on ({self.symbol})')
        plt.legend()
        plt.grid(True)
        plt.show()
