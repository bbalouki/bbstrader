import numpy as np
import seaborn as sns
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
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
        data['diff_log_return'] = data['log_return'].diff().dropna()
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
        final_aic = np.inf
        final_order = (0, 0, 0)
        for p in range(6):  # Range of p values
            for q in range(6):  # Range of q values
                if p == 0 and q == 0:
                    continue  # Skip the (0,0,0) combination
                try:
                    model = ARIMA(window_data, order=(p, 0, q))
                    results = model.fit()
                    current_aic = results.aic
                    if current_aic < final_aic:
                        final_aic = current_aic
                        final_order = (p, 0, q)
                except:  # Catching all exceptions to continue the loop
                    continue
        # Fit ARIMA with the best order
        best_arima_model = ARIMA(window_data, order=final_order).fit()
        return best_arima_model

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
        resid = arima_result.resid
        resid = resid[~(np.isnan(resid) | np.isinf(resid))]
        garch_model = arch_model(resid, p=1, q=1)
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
            bp_test = acorr_ljungbox(resid, lags=[10], return_df=True)
            print("Box-Pierce and Ljung-Box Tests Results  for ARIMA:\n", bp_test)

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
        resid = arima_result.resid
        resid = resid[~(np.isnan(resid) | np.isinf(resid))]
        garch_model = arch_model(resid, p=1, q=1)
        garch_result = garch_model.fit(disp='off')
        return arima_result, garch_result

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
        for i in range(1, len(predictions)):
            if predictions[i] > 0:
                positions.append(1)  # Long
            elif predictions[i] < 0:
                positions.append(-1)  # Short
            else:
                positions.append(positions[-1])  # Hold previous position

        # Adjust for the initial position based on the first prediction
        initial_position = 1 if predictions[0] > 0 else -1
        positions = [initial_position] + positions
        return positions

    # Step 5: Vectorized Backtesting
    def backtest_strategy(self):
        """
        Performs a backtest of the strategy over 
        the entire dataset, plotting cumulative returns.
        """
        # Calculate strategy returns
        predictions = []
        data = self.data
        window_size = self.k
        total_iterations = len(data) - window_size
        print(
            f"Starting backtesting for {self.symbol} "
            f"\nWindow size {window_size}. "
            f"\nTotal iterations: {total_iterations}.\n")
        for i in range(window_size, len(data)):
            print(
                f"Processing window {i - window_size + 1}/{total_iterations}...")
            window_data = data['diff_log_return'].iloc[i-window_size:i]
            next_return = self.get_prediction(window_data)
            predictions.append(next_return)

        positions = self.execute_trading_strategy(predictions)
        # calculate strategy returns
        strategy_returns = np.array(
            positions[:-1]) * data['log_return'].iloc[window_size+1:].values
        # Calculate buy and hold returns
        buy_and_hold = data['log_return'].iloc[window_size+1:].values
        # Calculate cumulative returns
        buy_and_hold_returns = np.cumsum(buy_and_hold)
        cumulative_returns = np.cumsum(strategy_returns)
        # Extract the dates for plotting
        dates = data.index[window_size+1:]
        self.plot_cumulative_returns(
            cumulative_returns, buy_and_hold_returns, dates)
        print("Backtesting completed.")

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

