
# HMMRiskManager Class Documentation

## Overview

`HMMRiskManager` is a class designed for managing market risks by utilizing Hidden Markov Models (HMM) to analyze financial market data. It inherits from the adbstract `RiskModel` class and employs a Gaussian HMM to identify hidden states within market dynamics. These states assist in making informed decisions about permissible trading actions, aligning with a risk-aware trading strategy.

## Initialization

The class is initialized with several parameters that allow customization of the model according to specific data inputs, model complexity, and operational verbosity.

```python
HMMRiskManager(data=None, states=2, iterations=100, end_date=None, csv_filepath=None, model_filename=None, verbose=False, cov_variance="diag")
```

### Parameters

- `data`: DataFrame containing market data. If not provided, data must be loaded from a CSV file specified by `csv_filepath`.
- `states`: The number of hidden states for the HMM to identify. Defaults to 2.
- `iterations`: The number of iterations for the HMM fitting process. Defaults to 100.
- `end_date`: The end date for the market data to be analyzed.
- `csv_filepath`: The file path to a CSV file containing market data, used if `data` is not directly provided.
- `model_filename`: The filename to save the trained HMM model. If not provided, the model is not saved.
- `verbose`: If set to True, additional information about the model fitting process is printed.
- `cov_variance`: Specifies the type of covariance parameter to be used by the HMM. Defaults to "diag".

## Methods

### `_get_data()`

Retrieves market data from the provided DataFrame or CSV file.

### `_fit_model()`

Fits the Gaussian HMM to the market data and optionally saves the model.

### `get_states()`

Predicts hidden states for the market data and calculates mean returns and volatility for each state.

### `identify_market_trends()`

Identifies bullish and bearish market trends based on the mean returns and volatility of each state.

### `get_current_regime(returns_val)`

Determines the current market regime based on the latest returns.

### `which_trade_allowed(returns_val)`

Decides whether a long or short trade is allowed based on the current market regime.

### `save_hmm_model(hmm_model, filename)`

Saves the trained HMM model to a pickle file.

### `read_csv_file(csv_filepath)`

Reads market data from a specified CSV file.

### `obtain_prices_df(data_frame, end=None)`

Processes the market data to calculate returns and optionally filters data up to a specified end date.

### `show_hidden_states()`

Visualizes the market data and the predicted hidden states from the HMM model.

### `plot_hidden_states(hmm_model, df)`

Plots the adjusted closing prices masked by the in-sample hidden states as a mechanism to understand the market regimes.

## Usage Example

```python
# Assuming `data` is a DataFrame containing your market data
risk_manager = HMMRiskManager(data=data, states=3, iterations=200, verbose=True)
current_regime = risk_manager.get_current_regime(data['Returns'].values)
print(f"Current Market Regime: {current_regime}")
```