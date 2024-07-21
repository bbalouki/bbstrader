
# ArimaGarchStrategy Class Documentation

The `ArimaGarchStrategy` class implements a trading strategy by combining ARIMA (AutoRegressive Integrated Moving Average) and GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models. This strategy aims to predict future returns based on historical price data and is built upon a generic Strategy class, which it extends with specific strategy logic.

The strategy is implemented in the following steps:
1. Data Preparation: Load and prepare the historical price data.
2. Modeling: Fit the ARIMA model to the data and then fit the GARCH model to the residuals.
3. Prediction: Predict the next return using the ARIMA model and the next volatility using the GARCH model.
4. Trading Strategy: Execute the trading strategy based on the predictions.
5. Vectorized Backtesting: Backtest the trading strategy using the historical data.

## Class Attributes

- `symbol` (str): The ticker symbol for the financial instrument being analyzed.
- `data` (pd.DataFrame): The dataset prepared for model training and prediction, including log returns and possibly differenced log returns.
- `k` (int): The window size for rolling prediction, indicating how many past observations are considered for each prediction.

## Methods

### `__init__(self, symbol, data, k)`

Initializes the `ArimaGarchStrategy` class with a symbol, raw data, and window size.

#### Parameters

- `symbol` (str): The ticker symbol for the financial instrument.
- `data` (pd.DataFrame): The raw dataset containing at least 'Close' prices.
- `k` (int): The window size for rolling prediction.

### `load_and_prepare_data(self, df)`

Prepares the dataset by calculating logarithmic returns and differencing if necessary.

#### Parameters

- `df` (pd.DataFrame): The raw dataset containing at least 'Close' prices.

#### Returns

- `pd.DataFrame`: The dataset with additional columns for log returns and differenced log returns.

### `fit_best_arima(self, window_data)`

Fits the ARIMA model to the provided window of data, selecting the best model based on AIC.

#### Parameters

- `window_data` (pd.DataFrame): The dataset for a specific window period.

#### Returns

- ARIMA model: The best fitted ARIMA model based on AIC.

### `show_arima_garch_results(self, arima_result, acf=True, test_resid=True)`

Displays the ARIMA and GARCH model results, including plotting ACF of residuals and conducting Box-Pierce and Ljung-Box tests.

#### Parameters

- `arima_result` (ARIMA model): The ARIMA model result.
- `acf` (bool, optional): If True, plot the ACF of residuals. Defaults to True.
- `test_resid` (bool, optional): If True, conduct Box-Pierce and Ljung-Box tests on residuals. Defaults to True.

### `fit_garch(self, window_data)`

Fits the GARCH model to the residuals of the best ARIMA model.

#### Parameters

- `window_data` (pd.DataFrame): The dataset for a specific window period.

#### Returns

- tuple: Contains the ARIMA result and GARCH result.

### `predict_next_return(self, arima_result, garch_result)`

Predicts the next return using the ARIMA model and the next volatility using the GARCH model.

#### Parameters

- `arima_result` (ARIMA model): The ARIMA model result.
- `garch_result` (GARCH model): The GARCH model result.

#### Returns

- float: The predicted next return.

### `get_prediction(self, window_data)`

Generates a prediction for the next return based on a window of data.

#### Parameters

- `window_data` (pd.DataFrame): The dataset for a specific window period.

#### Returns

- float: The predicted next return.

### `calculate_signals(self, window_data)`

Calculates the trading signal based on the prediction: 'LONG' if positive, 'SHORT' if negative.

#### Parameters

- `window_data` (pd.DataFrame): The dataset for a specific window period.

#### Returns

- str: The trading signal ('LONG', 'SHORT', or None).

### `execute_trading_strategy(self, predictions)`

Executes the trading strategy based on a list of predictions, determining positions to take.

#### Parameters

- `predictions` (list): A list of predicted returns.

#### Returns

- list: A list of positions (1 for 'LONG', -1 for 'SHORT', 0 for 'HOLD').

### `backtest_strategy(self)`

Performs a backtest of the strategy over the entire dataset, plotting cumulative returns.

### `plot_cumulative_returns(self, strategy_returns, buy_and_hold_returns, dates)`

Plots the cumulative returns of the ARIMA+GARCH strategy against a buy-and-hold strategy.

#### Parameters

- `strategy_returns` (np.array): Cumulative returns from the strategy.
- `buy_and_hold_returns` (np.array): Cumulative returns from a buy-and-hold strategy.
- `dates` (pd.Index): The dates corresponding to

 the returns.


# KLFStrategy Class Documentation

The `KLFStrategy` class implements a trading strategy based on the Kalman Filter. This filter is a recursive algorithm used for estimating the state of a linear dynamic system from a series of noisy measurements. The strategy is designed to process market data, estimate dynamic parameters such as the slope and intercept of price relationships, and generate trading signals based on those estimates.

## Initialization

```python
def __init__(self, tickers: list | tuple, **kwargs):
```

Initializes the Kalman Filter strategy with the given tickers and optional parameters.

### Parameters

- `tickers`: A list or tuple of ticker symbols representing financial instruments.
- `**kwargs`: Keyword arguments for additional parameters. Accepts `delta` and `vt` as optional parameters.

## Public Methods

### `_init_kalman`

```python
def _init_kalman(self):
```

Initializes and returns a Kalman Filter configured for the trading strategy. The filter is set up with initial state and covariance, state transition matrix, process noise, and measurement noise covariances.

### `calc_slope_intercep`

```python
def calc_slope_intercep(self, prices: np.ndarray):
```

Calculates and returns the slope and intercept of the relationship between the provided prices using the Kalman Filter. This method updates the filter with the latest price and returns the estimated slope and intercept.

#### Parameters

- `prices`: A numpy array of prices for two financial instruments.

#### Returns

- A tuple containing the slope and intercept of the relationship.

### `calculate_xy_signals`

```python
def calculate_xy_signals(self, et, std):
```

Generates trading signals based on the forecast error and standard deviation of the predictions. It returns signals for exiting, going long, or shorting positions based on the comparison of the forecast error with the standard deviation.

#### Parameters

- `et`: The forecast error.
- `std`: The standard deviation of the predictions.

#### Returns

- A tuple containing the trading signals for the two financial instruments.

### `calculate_signals`

```python
def calculate_signals(self, prices: np.ndarray):
```

Calculates trading signals based on the latest prices and the Kalman Filter's estimates. It updates the filter's state with the latest prices, computes the slope and intercept, and generates trading signals based on the forecast error and prediction standard deviation.

#### Parameters

- `prices`: A numpy array of prices for two financial instruments.

#### Returns

- A dictionary containing trading signals for the two financial instruments.

## Attributes

- `tickers`: The list or tuple of ticker symbols for the financial instruments.
- `latest_prices`: A numpy array storing the latest prices for the financial instruments.
- `delta`: A small constant used in the calculation of process noise covariance.
- `wt`: Process noise covariance matrix.
- `vt`: Measurement noise variance.
- `theta`: A numpy array storing the estimated slope and intercept.
- `P`: The error covariance matrix.
- `R`: Measurement noise covariance.
- `kf`: The Kalman Filter instance used in the strategy.

This class provides a comprehensive framework for implementing a trading strategy based on the Kalman Filter, allowing for the dynamic estimation of market parameters and the generation of trading signals based on those estimates.

## Example

```python
# Define the tickers
tickers = ('AAPL', 'MSFT')

# Initialize the KLFStrategy with tickers and optional parameters
klf_strategy = KLFStrategy(tickers, delta=1e-4, vt=1e-3)

# Example prices for AAPL and MSFT
prices = np.array([150.0, 250.0])

# Calculate trading signals based on current prices
signals = klf_strategy.calculate_signals(prices)

print(signals)
````

# Ornstein-Uhlenbeck Process Model Documentation

## Overview

The Ornstein-Uhlenbeck (OU) process model is a sophisticated financial model that describes the mean-reverting behavior of an asset's price. It captures the essence of how asset prices fluctuate around a long-term mean, making it a vital tool for modeling price dynamics in various financial applications.

## Features

- **Mean-Reverting Stochastic Process**: Models asset price dynamics that tend to revert to a long-term mean.
- **Parameter Estimation**: Estimates the drift (θ), volatility (σ), and long-term mean (μ) based on historical price data.
- **Simulation**: Simulates the OU process using the estimated parameters to predict future price movements.

## Class `OrnsteinUhlenbeck`

### Initialization

- **Parameters**:
  - `prices` (`np.ndarray`): Historical close prices of the asset.
  - `returns` (`bool`, default=`True`): Indicates whether to simulate the returns or the raw data.
  - `timeframe` (`str`, default=`"D1"`): The timeframe for the historical prices, supported values include "1m", "5m", "15m", "30m", "1h", "4h", "D1".

- **Example**:

  ```python
  ou_model = OrnsteinUhlenbeck(prices=np.array([...]), returns=True, timeframe="D1")
  ```

### Methods

#### `ornstein_uhlenbeck(mu, theta, sigma, dt, X0, n)`

Simulates the OU process over `n` time steps.

- **Parameters**:
  - `mu` (`float`): Estimated long-term mean.
  - `theta` (`float`): Estimated drift.
  - `sigma` (`float`): Estimated volatility.
  - `dt` (`float`): Time step.
  - `X0` (`float`): Initial value.
  - `n` (`int`): Number of time steps.

- **Returns**: Simulated OU process as `np.ndarray`.

#### `estimate_parameters()`

Estimates the OU process parameters (μ, θ, σ) using the historical price data.

- **Returns**: Tuple of estimated parameters (μ, θ, σ).

#### `simulate_process(rts=None, n=100, p=None)`

Simulates the OU process multiple times.

- **Parameters**:
  - `rts` (`np.ndarray`): Historical returns. Optional.
  - `n` (`int`): Number of simulations to perform.
  - `p` (`int`): Number of time steps.

- **Returns**: 2D array representing simulated processes.

#### `calculate_signals(rts, p, n=10, th=1)`

Calculates trading signals based on the deviation from the mean of the last values in the simulated processes.

- **Parameters**:
  - `rts` (`np.ndarray`): Historical returns.
  - `p` (`int`): Number of time steps.
  - `n` (`int`): Number of simulations to perform.
  - `th` (`int`): Threshold for signal generation.

- **Returns**: Trading signal as `str`.

## Application

The Ornstein-Uhlenbeck process model is a powerful tool for traders and portfolio managers to understand and predict price dynamics. By quantifying the mean-reverting nature of asset prices, it provides a foundation for developing sophisticated trading strategies and risk management practices.

For more detailed information, refer to the [Wikipedia page on the Ornstein–Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process).