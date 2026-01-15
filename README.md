# `bbstrader`: High-Performance Algorithmic Trading with C++ and Python

[![Documentation Status](https://readthedocs.org/projects/bbstrader/badge/?version=latest)](https://bbstrader.readthedocs.io/en/latest/?badge=latest)
[![PYPI Version](https://img.shields.io/pypi/v/bbstrader)](https://pypi.org/project/bbstrader/)
[![PyPi status](https://img.shields.io/pypi/status/bbstrader.svg?maxAge=60)](https://pypi.python.org/pypi/bbstrader)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/bbstrader)](https://pypi.org/project/bbstrader/)
[![PyPI Downloads](https://static.pepy.tech/badge/bbstrader)](https://pepy.tech/projects/bbstrader)

Welcome to `bbstrader`, an institutional-grade, high-performance algorithmic trading toolkit. This library is architected for serious traders and quants who require the raw speed of C++ for execution and the rich, expressive power of Python for analysis and orchestration.

`bbstrader` is not a simple wrapper. It is a sophisticated, dual-language ecosystem that provides a decisive edge in the competitive financial markets. Whether you are a C++ developer seeking seamless integration with MetaTrader 5, or a Python quant hitting performance bottlenecks, `bbstrader` is the definitive solution.

## Why `bbstrader` Stands Out

In a crowded field of trading libraries, `bbstrader` is architected to solve the most challenging problems in algorithmic trading: performance, flexibility, and platform limitations.

*   **True High-Performance Core:** Unlike pure Python libraries that rely on JIT compilers or wrappers, `bbstrader` allows you to write your core logic in native, compiled C++ for maximum, deterministic speed. This is the architecture used by professional high-frequency trading firms.
*   **Seamless C++/Python Interoperability:** Our innovative architectural bridge allows for bidirectional, low-latency communication between C++ and Python, letting you use the best language for the job without compromise.
*   **Escape the MQL5 Sandbox:** Move beyond the limitations of MQL5's interpreted language and constrained ecosystem. With `bbstrader`, you can build sophisticated, multi-threaded, and computationally intensive strategies that are simply not possible in MQL5.
*   **Institutional-Grade Architecture:** From its event-driven backtester to its modular design, `bbstrader` is built with the principles of professional trading systems in mind, providing a robust foundation for serious strategy development.

## The `bbstrader` Edge: Uniting C++ Speed with Python Flexibility

Modern algorithmic trading demands both speed and intelligence. `bbstrader` is built on the philosophy that you shouldn't have to choose between them.

### Overcoming the MQL5 Bottleneck
MetaTrader 5 is a world-class trading platform, but its native MQL5 language presents significant limitations for complex, high-frequency strategies:
- **Performance Ceilings:** As an interpreted language, MQL5 struggles with the computationally intensive logic required for advanced statistical models, machine learning, and rapid-fire order execution.
- **Ecosystem Constraints:** MQL5 lacks access to the vast, mature ecosystems of libraries for numerical computation, data science, and AI that C++ and Python offer.
- **Architectural Rigidity:** Implementing sophisticated, multi-threaded, or event-driven architectures in MQL5 is often a complex and error-prone endeavor.

`bbstrader` eradicates these barriers. By moving your core strategy logic to C++, you can unlock the full potential of your trading ideas, executing them with the microsecond-level precision demanded by institutional trading.

### A Symphony of Languages
`bbstrader` creates a powerful symbiosis between C++ and Python:
- **C++ for the Core:** Implement your performance-critical strategy logic, signal generation, and risk management in compiled C++ for maximum speed and control.
- **Python for the Command:** Use Python's unparalleled data science stack (NumPy, pandas, SciPy, scikit-learn) to research, backtest, and orchestrate your C++ strategies.

## Architectural Deep Dive: The C++/Python Bridge
The power of `bbstrader` lies in its innovative C++/Python bridge, which allows for seamless, bidirectional communication between the two languages.

At the core of this architecture is the C++ `MetaTraderClient`, a powerful class that mirrors the official MetaTrader 5 Python API. The magic lies in how it's connected:

1.  **Dependency Injection of Handlers:** The Python `MetaTrader5` library functions are passed into the C++ `MetaTraderClient` as `std::function` handlers. This means the C++ code isn't just calling Python; it's treating Python functions as native, callable objects.
2.  **High-Speed C++ Execution:** Your C++ strategy can execute complex, computationally intensive logic at native speeds. When it needs to interact with the market, it simply invokes one of the injected Python handlers.
3.  **The Data Flow:** The result is a clean, efficient, and powerful execution loop:
    `Python (Orchestration & Analysis) -> C++ (High-Speed Signal Generation) -> Python (MT5 Communication) -> C++ (Receives Market Data)`

This architecture provides the best of both worlds: the raw power of C++ for your core logic, and the flexibility and rich library support of Python for everything else.

## Installation
`bbstrader` is designed for both Python and C++ developers. Follow the instructions that best suit your needs.

### For the Python Quant
Get started in minutes using `pip`. We strongly recommend using a virtual environment.
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # on Linux/macOS
venv\Scripts\activate     # on Windows

# Install bbstrader with MetaTrader 5 support
pip install bbstrader[MT5]
```

### For the C++ Developer
To develop your own C++ strategies, you can use `vcpkg` to install the `bbstrader` library and its dependencies.
```bash
# If you don't have vcpkg, clone and bootstrap it
git clone https://github.com/microsoft/vcpkg
./vcpkg/bootstrap-vcpkg.sh

# Install bbstrader
./vcpkg/vcpkg install bbstrader
```

## Usage Patterns & Sophisticated Examples

### Pattern 1: C++ Core, Python Orchestrator (Maximum Performance)
This is the recommended pattern for latency-sensitive strategies, such as statistical arbitrage, market making, or any strategy where execution speed is a critical component of your edge. By compiling your core logic, you minimize interpretation overhead and gain direct control over memory and execution.

**Use this pattern when:**
- Your strategy involves complex mathematical calculations that are slow in Python.
- You need to react to market data in the shortest possible time.
- Your production environment demands deterministic, low-latency performance.

**C++ Side (`MovingAverageStrategy.hpp`):**
```cpp
#include "bbstrader/metatrader.hpp"
#include <numeric>
#include <iostream>

class MovingAverageStrategy : public MT5::MetaTraderClient {
public:
    using MetaTraderClient::MetaTraderClient; // Inherit constructors

    void on_tick(const std::string& symbol) {
        auto rates_opt = copy_rates_from_pos(symbol, 1, 0, 20); // M1 timeframe, 20 bars

        if (!rates_opt || rates_opt->size() < 20) return;

        const auto& rates = *rates_opt;

        double sum = std::accumulate(rates.begin(), rates.end(), 0.0,
                                     [](double a, const MT5::RateInfo& b) { return a + b.close; });
        double sma = sum / rates.size();
        double current_price = rates.back().close;

        if (current_price > sma) {
            std::cout << "Price is above SMA. Sending Buy Order for " << symbol << std::endl;
            // In a real strategy, you would call order_send(...) here
        }
    }
};
```
*This C++ class would then be exposed to Python using `pybind11`.*

**Python Side (`main.py`):**
```python
from bbstrader.api import Mt5Handlers
import MetaTrader5 as mt5
import time
from my_strategies import MovingAverageStrategy # Assuming you've compiled your C++ code

# 1. Instantiate the C++ strategy, injecting the Python MT5 handlers
strategy = MovingAverageStrategy(Mt5Handlers)

# 2. Main execution loop
if strategy.initialize():
    while True:
        strategy.on_tick("EURUSD")
        time.sleep(1)
```

### Pattern 2: Python-Driven with C++ Acceleration
This pattern is ideal for strategies that benefit from Python's rich ecosystem for data analysis, machine learning, or complex event orchestration, but still require high-performance access to market data and the trading API.

**Use this pattern when:**
- Your strategy relies heavily on Python libraries like `pandas`, `scikit-learn`, or `tensorflow`.
- Rapid prototyping and iteration are more important than absolute minimum latency.
- Your core logic is more about decision-making based on pre-processed data than it is about raw computation speed.

```python
import bbstrader
import MetaTrader5 as mt5

# 1. Inherit from the C++ MetaTraderClient in Python
class MyStrategyClient(bbstrader.api.MetaTraderClient):
    def __init__(self, handlers):
        super().__init__(handlers)

# 2. Instantiate your client
client = MyStrategyClient(bbstrader.api.Mt5Handlers)

# 3. Interact with the MT5 terminal via the C++ bridge
if client.initialize():
    rates = client.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M1, 0, 100)
    print(f"Retrieved {len(rates)} rates via the C++ bridge.")
```

## A Tour of the Core Components
`bbstrader` is a comprehensive toolkit, with each module engineered to provide a specific, powerful capability.

### `btengine`: The Backtesting Powerhouse
Go beyond simple backtests. The `btengine` is a fully event-driven backtester that allows you to simulate your strategy's performance with a high degree of realism. It handles the complexities of portfolio construction, risk management, and performance attribution, giving you a true picture of your strategy's viability.

### `metatrader`: The High-Speed Bridge
This is the heart of `bbstrader`'s interoperability. The `metatrader` module provides the C++ `MetaTraderClient` and the Python `Mt5Handlers` that connect the two languages, enabling the high-performance trading patterns described above.

### `models`: Your Quantitative Toolkit
Build sophisticated, data-driven strategies. The `models` module provides a framework for integrating advanced statistical and machine learning models into your trading logic. From ARIMA+GARCH models for volatility forecasting to NLP models for sentiment analysis, this module is your gateway to the world of quantitative finance.

### `tseries`: Mastering Time Series Analysis
Financial markets are all about time series. The `tseries` module provides a suite of specialized tools for advanced time series analysis, including cointegration tests to identify mean-reverting pairs, volatility modeling, and advanced filtering techniques to separate signal from noise.

### `trading`: Orchestrating Live Execution
This module provides a high-level interface for managing your live trading operations. It's the conductor of the orchestra, coordinating the signals from your strategies, the risk parameters from your models, and the execution logic of the `metatrader` bridge to manage your portfolio in real-time.

## Documentation
For a deep dive into the API, advanced tutorials, and more, please visit our full documentation:
[**View the Full Documentation on ReadTheDocs**](https://bbstrader.readthedocs.io/en/latest/)

## Contributing
`bbstrader` is an open-source project, and we welcome contributions from the community. Whether you're interested in adding new features, improving the documentation, or fixing bugs, we encourage you to get involved. Please see our `CONTRIBUTING.md` file for more details.

## Disclaimer
Trading financial instruments involves a high level of risk. The developers of `bbstrader` are not responsible for any financial losses incurred through the use of this software. Always trade responsibly.

## License
`bbstrader` is licensed under the MIT License.
