# `bbstrader`: High-Performance Trading with C++ and Python

[![Documentation Status](https://readthedocs.org/projects/bbstrader/badge/?version=latest)](https://bbstrader.readthedocs.io/en/latest/?badge=latest)
[![PYPI Version](https://img.shields.io/pypi/v/bbstrader)](https://pypi.org/project/bbstrader/)
[![PyPi status](https://img.shields.io/pypi/status/bbstrader.svg?maxAge=60)](https://pypi.python.org/pypi/bbstrader)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/bbstrader)](https://pypi.org/project/bbstrader/)
[![PyPI Downloads](https://static.pepy.tech/badge/bbstrader)](https://pepy.tech/projects/bbstrader)
[![CodeFactor](https://www.codefactor.io/repository/github/bbalouki/bbstrader/badge)](https://www.codefactor.io/repository/github/bbalouki/bbstrader)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-grey?logo=Linkedin&logoColor=white)](https://www.linkedin.com/in/bertin-balouki-s-15b17a1a6)

### **[Get Premium Data Packs]([bertin@bbs-trading.com])** | **[Need Custom Strategies](mailto:[bertin@bbs-trading.com])** | ☕ **[Support the Dev](https://paypal.me/bertinbalouki?country.x=SN&locale.x=en_US)**

## Overview

`bbstrader` is not just another trading library. It is an institutional-grade, high-performance toolkit designed to give you a decisive edge in the financial markets. Built on a unique C++/Python architecture, it empowers you to execute complex strategies with the speed of compiled C++ and the flexibility of Python.

Whether you're a Python quant looking for more speed, or a C++ developer who needs access to the MetaTrader 5 ecosystem, `bbstrader` is your bridge to professional-grade algorithmic trading.

## The `bbstrader` Philosophy: Best of Both Worlds

The markets are unforgiving. To succeed, you need speed, precision, and flexibility. This is why `bbstrader` was built to seamlessly blend the raw power of C++ with the expressive simplicity of Python.

### Why C++ for Trading? The MQL5 Bottleneck

MetaTrader 5 is a fantastic platform, but its native MQL5 language has limitations:
- **Execution Speed:** As an interpreted language, MQL5 can be too slow for strategies that require rapid calculations or reactions to market events.
- **Limited Libraries:** You don't have access to the vast ecosystem of numerical, statistical, and machine learning libraries available in other languages.
- **Complexity:** Implementing sophisticated mathematical models in MQL5 is often cumbersome and error-prone.

`bbstrader` shatters these limitations by allowing you to write your performance-critical logic in C++, the language of choice for high-frequency trading and institutional systems.

### The Power of Python

Python is the undisputed king of data science and quantitative analysis for a reason. With `bbstrader`, you can leverage Python's world-class libraries (like NumPy, pandas, and scikit-learn) to:
- **Rapidly Prototype & Backtest:** Develop and test your ideas with ease.
- **Orchestrate Your Strategies:** Use Python as the "brain" of your trading operation, making high-level decisions.
- **Analyze Your Results:** Dive deep into your trading performance with powerful data analysis tools.

## The C++/Python Bridge: How It Works

`bbstrader` provides a C++ `MetaTraderClient` that mirrors the official MetaTrader 5 Python API. The magic is in how they connect:

1.  **Python Handles:** The Python `MetaTrader5` library functions are passed into the C++ client as "handlers".
2.  **C++ Execution:** Your C++ strategy code calls these handlers to interact with the MT5 terminal (e.g., to get market data or send an order).
3.  **The Flow:** Your Python code calls a function in your C++ strategy. The C++ code then executes its high-speed logic, and when it needs to talk to the market, it calls the Python handler it was given.

This creates a powerful feedback loop: **Python (Orchestration) -> C++ (High-Speed Logic) -> Python (MT5 Communication) -> C++ (Result)**

## Installation

### For the Python Developer
Get started in seconds with `pip`. We recommend using a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install bbstrader with MetaTrader 5 support
pip install bbstrader[MT5]
```

### For the C++ Developer
If you want to build your own C++ strategies, you can use `vcpkg` to install the `bbstrader` library.

```bash
# Install vcpkg (if you haven't already)
git clone https://github.com/microsoft/vcpkg
./vcpkg/bootstrap-vcpkg.sh

# Install bbstrader
./vcpkg/vcpkg install bbstrader
```

## Usage Examples

Here are a few patterns to illustrate the power of `bbstrader`'s hybrid architecture.

### Pattern 1: C++ Core, Python Shell (High-Performance)
In this pattern, your core strategy logic lives in C++. Python is used to instantiate the strategy and feed it data. This is ideal for speed-sensitive strategies.

**C++ Side (`MovingAverageStrategy.hpp`):**
```cpp
#include "bbstrader/metatrader.hpp"
#include <numeric>
#include <iostream>

class MovingAverageStrategy : public MT5::MetaTraderClient {
public:
    using MetaTraderClient::MetaTraderClient; // Inherit constructors

    // This is where your custom C++ logic lives
    void on_tick(const std::string& symbol) {
        // 1. Get the latest rates (C++ logic calling mapped Python function)
        auto rates_opt = copy_rates_from_pos(symbol, 1, 0, 20); // M1 timeframe, 20 bars

        if (!rates_opt || rates_opt->size() < 20) return;

        const auto& rates = *rates_opt;

        // 2. Calculate SMA in C++ (Fast!)
        double sum = std::accumulate(rates.begin(), rates.end(), 0.0,
                                     [](double a, const MT5::RateInfo& b) { return a + b.close; });
        double sma = sum / rates.size();
        double current_price = rates.back().close;

        // 3. Execution Logic
        if (current_price > sma) {
            std::cout << "Price above SMA. Sending Buy Order for " << symbol << std::endl;
            // order_send(...) would be called here
        }
    }
};
```

**Python Side (`main.py`):**
```python
from bbstrader.api import Mt5Handlers
import MetaTrader5 as mt5
import time

# You would need to compile the C++ code and create Python bindings.
# This is a simplified example.
# Let's assume you have a compiled module called `my_strategies`
from my_strategies import MovingAverageStrategy

# Create the strategy instance, injecting the Python MT5 handlers
strategy = MovingAverageStrategy(Mt5Handlers)

# Main Loop
while True:
    strategy.on_tick("EURUSD")
    time.sleep(1) # Check every second
```

### Pattern 2: Python-driven with C++ Acceleration
In this pattern, your strategy is primarily written in Python, but you can call the C++ `MetaTraderClient` for direct, high-performance access to the MT5 API.

```python
import bbstrader
import MetaTrader5 as mt5

# 1. Create a Python class that inherits from the C++ Client
class MyStrategyClient(bbstrader.api.MetaTraderClient):
    def __init__(self, handlers):
        # Initialize the C++ base class
        super().__init__(handlers)

# 2. Define the Handlers (The bridge)
h = bbstrader.api.Handlers

# 3. Use the Client
client = MyStrategyClient(h)
if client.initialize():
    # This call goes: Python -> C++ Logic -> Python MT5 Lib -> C++ Result
    rates = client.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M1, 0, 100)
    print(f"Retrieved {len(rates)} rates via C++ Bridge")
```

## Core Components

- **Backtesting Engine (`btengine`):** An event-driven backtester to rigorously test your strategies with historical data.
- **MetaTrader5 Module (`metatrader`):** The C++/Python bridge that facilitates direct, high-speed interaction with the MT5 platform.
- **Models Module (`models`):** A framework for financial modeling, including statistical models and risk management tools.
- **Time Series Module (`tseries`):** Specialized tools for advanced time series analysis.

## Documentation

For comprehensive information, including detailed API references and mathematical explanations, please visit our official documentation:
[**View the Full Documentation**](https://bbstrader.readthedocs.io/en/latest/)

## Contributing

We welcome contributions! Whether you are a developer or a trader:

1.  Fork the repository.
2.  Create a feature branch.
3.  Submit a Pull Request.

_Note: If you find a security vulnerability, please email the maintainer directly._

## Disclaimer

Trading financial instruments involves a high level of risk. The developers of `bbstrader` are not responsible for any financial losses incurred from the use of this software.

## License

`bbstrader` is open source and available under the MIT License.
