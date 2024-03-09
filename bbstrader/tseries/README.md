# tseries: Time Series Analysis Toolkit

The `tseries` module is a Python package designed for conducting advanced time series analysis in financial markets. It leverages statistical models and algorithms to perform tasks such as cointegration testing, volatility modeling, and filter-based estimation to assist in trading strategy development, market analysis, and financial data exploration.

## Features

- **Cointegration Testing**: Utilize Johansen and Augmented Dickey-Fuller (CADF) tests to identify long-term relationships between financial instruments.
- **Volatility Modeling**: Apply ARCH and GARCH models for volatility forecasting, crucial for risk management and derivative pricing.
- **Hurst Exponent Calculation**: Determine the nature of financial time series (trending, mean-reverting, or random) using the Hurst exponent.
- **Kalman Filters**: Implement Kalman Filters for dynamic regression and state estimation in time-varying systems.
- **Financial Data Acquisition**: Integrate with `yfinance` to easily fetch historical market data.

## Installation

To install the `tseries` module, clone this repository to your local machine. Ensure you have Python 3.6+ installed. Navigate to the cloned directory and run:

```bash
pip install -r requirements.txt
```

This command installs all necessary dependencies, including `pandas`, `numpy`, `matplotlib`, `yfinance`, `statsmodels`, and `arch`.

## Usage

The `tseries` module consists of several components, each designed for specific types of time series analysis. Below is a brief overview of how to use each component:

### Cointegration Testing

- **Johansen Test**: `johensen.py` performs the Johansen cointegration test on a pair of stock tickers.
- **CADF Test**: `cadf.py` conducts the CADF test and visualizes the price series and residuals.

### Volatility Modeling

- **ARCH/GARCH Models**: `arch.py` fits ARCH and GARCH models to financial time series data to forecast volatility.

### Hurst Exponent

- **Hurst Exponent Calculation**: `hurst.py` calculates the Hurst exponent to analyze the time series behavior.

### Kalman Filters

- **Kalman Filter Application**: `kalman.py` demonstrates the use of Kalman Filters for estimating the dynamic relationship between ETF pairs.

### Getting Started

To use a specific functionality, import the corresponding Python file and call its main function with appropriate parameters. For example, to test cointegration between two tickers:

```python
from johensen import run_test as johansen_test

johansen_test(['AAPL', 'MSFT'], '2020-01-01', '2021-01-01')
```

## Contributing

We welcome contributions to the `tseries` module. If you have suggestions for improvement or want to contribute code, please:

1. Fork the repository.
2. Create a new branch for your features or fixes.
3. Submit a pull request with a detailed description of your changes.

## License

The `tseries` module is open-sourced under the MIT License. See the LICENSE file for more details.
