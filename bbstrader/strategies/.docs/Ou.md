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