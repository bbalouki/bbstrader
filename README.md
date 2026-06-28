# Simplified Investment & Trading Toolkit with Python & C++

[![Build](https://github.com/bbalouki/bbstrader/actions/workflows/build.yml/badge.svg)](https://github.com/bbalouki/bbstrader/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/bbstrader/badge/?version=latest)](https://bbstrader.readthedocs.io/en/latest/?badge=latest)
[![PyPi status](https://img.shields.io/pypi/status/bbstrader.svg?maxAge=60)](https://pypi.python.org/pypi/bbstrader)
![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)
![macOS](https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-0078D6?logo=windows&logoColor=white)
[![PYPI Version](https://img.shields.io/pypi/v/bbstrader)](https://pypi.org/project/bbstrader/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/bbstrader)](https://pypi.org/project/bbstrader/)
![vcpkg version](https://img.shields.io/vcpkg/v/bbstrader)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/std/the-standard)
[![CMake](https://img.shields.io/badge/CMake-3.20+-blue.svg)](https://cmake.org/)
[![PyPI Downloads](https://static.pepy.tech/badge/bbstrader)](https://pepy.tech/projects/bbstrader)
[![CodeFactor](https://www.codefactor.io/repository/github/bbalouki/bbstrader/badge)](https://www.codefactor.io/repository/github/bbalouki/bbstrader)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-grey?logo=Linkedin&logoColor=white)](https://www.linkedin.com/in/bertin-balouki-s-15b17a1a6)

## Welcome to `bbstrader` – The Ultimate C++ & Python Trading Powerhouse!

## Table of Contents

- [Overview](#overview)
- [Why `bbstrader` Stands Out](#why-bbstrader-stands-out)
- [Trusted by Traders Worldwide](#trusted-by-traders-worldwide)
- [The `bbstrader` Edge: Uniting C++ Speed with Python Flexibility](#the-bbstrader-edge-uniting-c-speed-with-python-flexibility)
  - [Overcoming the MQL5 Bottleneck](#overcoming-the-mql5-bottleneck)
- [Key Modules](#key-modules)
  - [1. `btengine`: Event-Driven Backtesting Beast](#1-btengine-event-driven-backtesting-beast)
  - [2. `metatrader`: The C++/Python Bridge to MT5](#2-metatrader-the-cpython-bridge-to-mt5)
    - [Pattern 1: C++ Core, Python Orchestrator (Maximum Performance)](#pattern-1-c-core-python-orchestrator-maximum-performance)
    - [Pattern 2: Python-Driven with C++ Acceleration](#pattern-2-python-driven-with-c-acceleration)
    - [Account API](#account-api)
  - [3. `trading`: Live Execution & Strategy Orchestrator](#3-trading-live-execution--strategy-orchestrator)
  - [4. `models`: Quant Toolkit for Signals & Risk](#4-models-quant-toolkit-for-signals--risk)
- [What's New in 2.1.0](#whats-new-in-210)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [For the Python Quant](#for-the-python-quant)
  - [For the C++ Developer](#for-the-c-developer)
- [CLI workflow](#cli-workflow)
- [Community & Support](#-community--support)
- [Professional Services](#professional-services)

### Overview

Imagine having the raw, blistering speed of C++ for your high-frequency trades, combined with Python's ecosystem for lightning-fast prototyping, advanced AI models, and seamless data analysis. That's `bbstrader` – not just a library, but a game-changing toolkit designed for quants, algo traders, and institutional pros who demand an edge in volatile markets. Whether you're scalping forex pairs, backtesting complex strategies, or copying trades across accounts in real-time, `bbstrader` empowers you to build, test, and deploy with unmatched efficiency.

Forget the frustrations of slow Python bottlenecks or MQL5's rigid sandbox. `bbstrader` bridges worlds: C++ for mission-critical performance and Python for intelligent orchestration. It's open-source, battle-tested across platforms, and ready to supercharge your trading arsenal.

## **Why `bbstrader` Stands Out**

In a crowded field of trading libraries, `bbstrader` is architected to solve the most challenging problems in algorithmic trading: performance, flexibility, and platform limitations.

- **Blazing Speed with C++ Core**: Compile your strategy logic in native C++ for deterministic, low-latency execution. Perfect for HFT, arbitrage, or compute-heavy models that Python alone can't handle.
- **Python's Powerhouse Ecosystem**: Leverage `NumPy`, `pandas`, `scikit-learn`, `TensorFlow`, and more for research, ML-driven signals, and backtesting – all seamlessly integrated with your C++ core.
- **Institutional-Grade Architecture:** From its event-driven backtester to its modular design, `bbstrader` is built with the principles of professional trading systems in mind, providing a robust foundation for serious strategy development.
  In today's hyper-fast financial landscape, every microsecond counts. `bbstrader` isn't another lightweight wrapper – it's an institutional-grade powerhouse engineered to tackle real-world trading challenges head-on.
- **Break Free from MQL5 Limits**: Ditch interpreted code and ecosystem constraints. Build multi-threaded, AI-infused strategies that execute orders in microseconds via MetaTrader 5 (MT5) integration.
  **Flexible Interface**: CLI & GUI
  `bbstrader` adapts to your workflow.
  - **Automation Fanatics**: Use the CLI for headless scripts, cron jobs, and server deployments.
  - **Visual Traders**: Launch the Desktop GUI (currently for Copy Trading) to monitor your master and slave accounts, check replication status, and manage connections visually.
  - **Cross-Platform & Future-Proof**: Works on Windows, macOS, Linux. (IBKR integration in development).

## **Trusted by Traders Worldwide**

With thousands of downloads, `bbstrader` is trusted by traders worldwide. It's not just code – it's your ticket to profitable, scalable strategies.

## **The `bbstrader` Edge: Uniting C++ Speed with Python Flexibility**

bbstrader's hybrid design is its secret weapon. At the heart is a bidirectional C++/Python bridge via `client` module:

1. **C++ for Speed**: Core classes like `MetaTraderClient` handle high-performance tasks. Inject Python handlers for MT5 interactions, enabling native-speed signal generation and risk checks.
2. **Python for Smarts**: Orchestrate everything with modules like `trading` and `btengine`.
3. **The Data Flow:** The result is a clean, efficient, and powerful execution loop:
   `Python (Orchestration & Analysis) -> C++ (High-Speed Signal Generation) -> Python (MT5 Communication) -> C++ (Receives Market Data)`

This setup crushes performance ceilings: run ML models in Python and execute trades in C++. The backtester is an event-driven simulator with a replayable, columnar data feed built for fidelity (faithful order state and accounting) rather than raw throughput. A fully vectorized research fast-path for screening millions of bars per second is on the [roadmap](FEATURES.md#2-ultra-fast-hybrid-backtesting-engine-flagship).

### **Overcoming the MQL5 Bottleneck**

MetaTrader 5 is a world-class trading platform, but its native MQL5 language presents significant limitations for complex, high-frequency strategies:

- **Performance Ceilings:** As an interpreted language, MQL5 struggles with the computationally intensive logic required for advanced statistical models, machine learning, and rapid-fire order execution.
- **Ecosystem Constraints:** MQL5 lacks access to the vast, mature ecosystems of libraries for numerical computation, data science, and AI that C++ and Python offer.
- **Architectural Rigidity:** Implementing sophisticated, multi-threaded, or event-driven architectures in MQL5 is often a complex and error-prone endeavor.

`bbstrader` eradicates these barriers. By moving your core strategy logic to C++, you can unlock the full potential of your trading ideas, executing them with the microsecond-level precision demanded by institutional trading.

## **Key Modules**

bbstrader is modular, with each component laser-focused.

### 1. **btengine**: Event-Driven Backtesting Beast

- **Purpose**: Simulate strategies with historical data across multi-asset portfolios, with commission modeling and metrics like Sharpe Ratio, Drawdown, and CAGR. Pluggable slippage/market-impact friction models are on the [roadmap](FEATURES.md#23-realistic-execution-institution-grade-friction).
- **Features**: Event queue for ticks/orders, a replayable columnar data feed (re-run the same data for parameter sweeps and walk-forward), and integration with models for signal generation. Note: the engine is event-driven for fidelity today; a vectorized research mode is on the [roadmap](FEATURES.md#2-ultra-fast-hybrid-backtesting-engine-flagship).
- **Example**: Backtest a StockIndexSTBOTrading from the example strategies.

```Python
# Inside the examples/
from strategies import test_strategy
if __name__ == '__main__':
    # Run backtesting for Stock Index Short Term Buy Only Strategy
    test_strategy(strategy='sistbo')
```

#### Research & realism toolkit (btengine)

The engine ships a batteries-included research stack on top of the shared
strategy API:

- **Execution realism (opt-in, defaults unchanged):** pluggable slippage
  (fixed-spread, percent, volatility, volume-participation), square-root
  **market impact**, commission models, partial fills, **time-frontier**
  (next-bar) fills and order-to-fill **latency** — so backtests survive the jump
  to live.
- **Vectorized research fast-path:** `vectorized_backtest(...)` screens
  entry/exit signal arrays across the whole history at once for fast "does this
  have alpha?" hypothesis testing, alongside the high-fidelity event engine.
- **Built-in indicators & strategy templates:** vectorized SMA/EMA/RSI/ATR/
  Bollinger/MACD/z-score and ready-made trend / mean-reversion / breakout
  templates on the same API used for live trading.
- **Optimization & validation:** parallel `optimize(...)` parameter sweeps and
  walk-forward (replayable columnar data + `reset()`), plus overfitting
  diagnostics — **deflated/probabilistic Sharpe, PBO (CSCV), combinatorial
  purged CV**.
- **Risk analytics:** historical & parametric **VaR/CVaR**, **Monte Carlo**
  equity-curve confidence bands, volatility-**regime** detection, and
  **factor/beta** exposure.
- **Multi-strategy & multi-timeframe:** several strategies sharing one
  portfolio/clock, and on-the-fly higher-timeframe resampling (e.g. daily
  signals on a 1m feed, no look-ahead).
- **Reproducibility:** a cached **data catalog** (Parquet) and an
  **experiment store** that persists params, metrics and equity curves for
  leaderboard-style comparison. A `benchmarks/` script backs the performance
  claims.
- **Broker abstraction:** a venue-neutral `Broker` interface (with an in-memory
  `PaperBroker`) so strategies can target MT5 today and other brokers later.

### Backtesting Results

![Backtesting Results1](https://github.com/bbalouki/bbstrader/blob/main/bbstrader/assets/bbs_.png?raw=true)
![Backtesting Results2](https://github.com/bbalouki/bbstrader/blob/main/bbstrader/assets/qs_metrics_1.png?raw=true)

### 2. **metatrader**: The C++/Python Bridge to MT5

- **Purpose**: High-speed MT5 integration. C++ MetaTraderClient mirrors MT5 API for orders, rates, and account management.
- **Features**: Bidirectional callbacks, error handling, real-time tick processing.
- **Strategy Patterns**: Two main patterns to build strategies:

#### Pattern 1: C++ Core, Python Orchestrator (Maximum Performance)

This is the recommended pattern for latency-sensitive strategies, such as statistical arbitrage, market making, or any strategy where execution speed is a critical component of your edge. By compiling your core logic, you minimize interpretation overhead and gain direct control over memory and execution.

**Use this pattern when:**

- Your strategy involves complex mathematical calculations that are slow in Python.
- You need to react to market data in the shortest possible time.
- Your production environment demands deterministic, low-latency performance.

**C++ Side (`MovingAverageStrategy.cpp`):**

```cpp
#include "bbstrader/metatrader.hpp"
#include <numeric>
#include <iostream>

class MovingAverageStrategy : public MT5::MetaTraderClient {
public:
    using MetaTraderClient::MetaTraderClient;

    void on_tick(const std::string& symbol) {
        auto rates_opt = copy_rates_from_pos(symbol, 1, 0, 20);

        if (!rates_opt || rates_opt->size() < 20) return;

        const auto& rates = *rates_opt;

        double sum = std::accumulate(rates.begin(), rates.end(), 0.0,
                                     [](double a, const MT5::RateInfo& b) { return a + b.close; });
        double sma = sum / rates.size();
        double current_price = rates.back().close;

        if (current_price > sma) {
            std::cout << "Price is above SMA. Sending Buy Order for " << symbol << '\n';
            MT5::TradeRequest request;
            request.action = MT5::TradeAction::DEAL;
            request.symbol = symbol;
            request.volume = 0.1;
            request.type = MT5::OrderType::BUY;
            request.type_filling = MT5::OrderFilling::FOK;
            request.type_time = MT5::OrderTime::GTC;
            send_order(request);
        }
    }
};
```

_This C++ class would then be exposed to Python using `pybind11`._

```cpp
// Inside bindings.cpp
#include <pybind11/pybind11.h>
#include "MovingAverageStrategy.hpp"

namespace py = pybind11;

PYBIND11_MODULE(my_strategies, m){
py::class_<MovingAverageStrategy, MT5::MetaTraderClient>(m, "MovingAverageStrategy")
    .def(py::init<MT5::MetaTraderClient::Handlers>())
    .def("on_tick", &MovingAverageStrategy::on_tick);
}
```

**Python Side (`main.py`):**

```python
from bbstrader.api import Mt5Handlers
import MetaTrader5 as mt5
import time
from my_strategies import MovingAverageStrategy

# 1. Instantiate the C++ strategy, injecting the Python MT5 handlers
strategy = MovingAverageStrategy(Mt5Handlers)

# 2. Main execution loop
if strategy.initialize():
    while True:
        strategy.on_tick("EURUSD")
        time.sleep(1)
```

#### Pattern 2: Python-Driven with C++ Acceleration

This pattern is ideal for strategies that benefit from Python's rich ecosystem for data analysis, machine learning, or complex event orchestration, but still require high-performance access to market data and the trading API.

**Use this pattern when:**

- Your strategy relies heavily on Python libraries like `pandas`, `scikit-learn`, or `tensorflow`.
- Rapid prototyping and iteration are more important than absolute minimum latency.
- Your core logic is more about decision-making based on pre-processed data than it is about raw computation speed.

```python
import MetaTrader5 as mt5
from bbstrader.api import Mt5Handlers
from bbstrader.api.client import MetaTraderClient

# 1. Inherit from the C++ MetaTraderClient in Python
class MyStrategyClient(MetaTraderClient):
    def __init__(self, handlers):
        super().__init__(handlers)

# 2. Instantiate your client
strategy = MyStrategyClient(Mt5Handlers)

# 3. Interact with the MT5 terminal via the C++ bridge
if strategy.initialize():
    rates = strategy.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M1, 0, 100)
    print(f"Retrieved {len(rates)} rates via the C++ bridge.")
```

#### Account API

The `Account` class provides a high-level, safe interface to live MT5 account data.

**Context manager** — automatically shuts down the connection on exit:

```python
from bbstrader.metatrader import Account

with Account() as acc:
    print(acc)                  # Account #12345 on MyBroker | Balance: 10000.0 USD
    print(repr(acc))            # Account(number=12345, server='MyBroker', balance=10000.0)
    rates = acc.get_currency_rates("EURUSD")
```

**`refresh()`** — reload live account data mid-session without creating a new object:

```python
acc.refresh()   # re-fetches balance, equity, margin from the terminal
print(acc.balance)
```

**Symbol info cache** — `get_symbol_info()` caches results to avoid repeated round-trips to the terminal. Call `clear_symbol_cache()` to invalidate when needed.

**`retry_on_disconnect` decorator** — wrap any MT5 call to auto-retry on connection drops:

```python
from bbstrader.metatrader.utils import retry_on_disconnect

@retry_on_disconnect(max_retries=3, delay=1.0)
def fetch_tick(symbol):
    return client.symbol_info_tick(symbol)
```

---

### 3. **`trading`: Live Execution & Strategy Orchestrator**

- **Purpose**: Manages live sessions, coordinates signals from strategies, risk from models, and execution via metatrader.
- **Features**: Multi-account support, position hedging, trailing stops.

### 4. `models`: Quant Toolkit for Signals & Risk

- **Purpose**: Build/test models like NLP sentiment, VaR/CVaR risk, optimization.
- **Features**: Currently Sentiment analysis, and Topic Modeling.
- **Example**: Sentiment-Based Entry:

```python
from bbstrader.models import SentimenSentimentAnalyzer

model = SentimenSentimentAnalyzer()  # Loads pre-trained NLP
score = model.analyze_sentiment("Fed hikes rates – markets soar!")
if score > 0.7:  # Bullish? Buy!
    print("Go long!")
```

### **Other Modules:**

`core`: Utilities (data structs, logging).
`config`: Manages JSON configs in ~/.bbstrader/.
`api`: Handler injections for bridges.

## What's New in 2.1.0

### Python improvements

| Area | Change |
|---|---|
| `Account` | New `refresh()` method, context manager (`with Account() as acc:`), `__repr__` / `__str__`, and symbol info cache with `clear_symbol_cache()` |
| `utils` | New `retry_on_disconnect(max_retries, delay)` decorator for automatic retry on `InternalFailConnect` / `InternalFailTimeout` |
| `trade` | Fixed `or` → `and` logic bug in market order type guard (was always `True`, skipped the guard) |
| `rates` | Fixed `get_data_from_pos` passing `session_duration` as a positional argument to `Rates.__init__` (caused `TypeError` at runtime) |
| `core/data` | Fixed `assert symbol is None, ValueError(...)` antipattern — `assert` does not raise the given exception; replaced with an explicit `if/raise` |
| `api/handlers` | Fixed `_build_request` filter that silently dropped `magic=0`, `deviation=0`, and `sl/tp=0.0` from trade requests |
| All modules | Modernized type hints to `X \| Y`, `X \| None`, `list[x]`, `dict[x, y]` (PEP 604/585); removed `Optional`, `Union`, `List`, `Dict` from `typing` |

### C++ improvements

| Area | Change |
|---|---|
| `metatrader.hpp` | Added `[[nodiscard]]` to all value-returning virtual methods (32 methods) |
| `metatrader.hpp` | Fixed six methods returning `0` / empty struct instead of `std::nullopt` when the handler is missing: `orders_total`, `positions_total`, `order_check`, `order_send`, `history_orders_total`, `history_deals_total` |
| `metatrader.hpp` | Added `noexcept` to `shutdown()` |
| `metatrader.cpp` | `TradeRequest` dict constructor now catches `py::cast_error` and raises a Python `ValueError` with a descriptive field name, instead of propagating a raw C++ exception |
| `objects.hpp` | Fixed invalid C++ identifier `int64_t_ONLY` → `LONG_ONLY` in the `PositionType` enum |

---

## Getting Started

### Prerequisites

- **Python**: Python 3.12+ is required.
- **MetaTrader 5 (MT5)**: Required for live execution (Windows).
- **MT5 Broker**: [Admirals](https://one.justmarkets.link/a/tufvj0xugm/registration/trader), [JustMarkets](https://one.justmarkets.link/a/tufvj0xugm/registration/trader), [FTMO](https://trader.ftmo.com/?affiliates=JGmeuQqepAZLMcdOEQRp).

## Installation

`bbstrader` is designed for both Python and C++ developers. Follow the instructions that best suit your needs.

### For the Python Quant

Get started in minutes using `pip`. We strongly recommend using a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # on Linux/macOS
venv\Scripts\activate     # on Windows

# Install bbstrader (lean core: numpy/pandas/yfinance + the backtesting engine)
pip install bbstrader

# Add MetaTrader 5 live trading (Windows)
pip install "bbstrader[mt5]"

# Optional extras (install only what you need):
#   nlp      -> NLP/sentiment stack (nltk, spacy, textblob, vaderSentiment, sumy)
#   social   -> social-media feeds (tweepy, praw)
#   viz      -> extra plotting (plotly, seaborn)
#   catalog  -> Parquet-backed cached data catalog (pyarrow)
#   all      -> everything above
pip install "bbstrader[all]"
```

> **Lean core:** as of the latest release the base install no longer pulls the
> heavy NLP/social/viz stacks. If you use those modules, install the matching
> extra (the code raises an actionable error telling you which one).

### For the C++ Developer

To develop your own C++ strategies, you can use `vcpkg` to install the `bbstrader` library and its dependencies.

```bash
# If you don't have vcpkg, clone and bootstrap it
git clone https://github.com/microsoft/vcpkg
./vcpkg/bootstrap-vcpkg.sh or ./vcpkg/bootstrap-vcpkg.bat

# Install bbstrader
./vcpkg/vcpkg install bbstrader
```

## CLI workflow

`bbstrader` shines via CLI – launch everything from one command!

| Action             | Command                                                                                                               |
| :----------------- | :-------------------------------------------------------------------------------------------------------------------- |
| **Run Backtest**   | `python -m bbstrader --run backtest --strategy SMAStrategy --account MY_ACCOUNT --config backtest.json`               |
| **Live Execution** | `python -m bbstrader --run execution --strategy KalmanFilter --account MY_ACCOUNT --config execution.json --parallel` |
| **Copy Trades**    | `python -m bbstrader --run copier --source "S1" --destination "D1"`                             |
| **Get Help**       | `python -m bbstrader --help`                                                                                          |

**Config Example** (`~/.bbstrader/execution/execution.json`):

```json
{
  "SMAStrategy": {
    "MY_MT5_ACCOUNT_1": {
      "symbol_list": ["EURUSD", "GBPUSD"],
      "trades_kwargs": { "magic": 12345, "comment": "SMA_Live" },
      "short_window": 20,
      "long_window": 50
    }
  }
}
```

## 🌍 Community & Support

- **[Read the Docs](https://bbstrader.readthedocs.io/en/latest/)**: Full API reference and tutorials.
- **[GitHub Issues](https://github.com/bbalouki/bbstrader/issues)**: Report bugs or request features.
- **[LinkedIn](https://www.linkedin.com/in/bertin-balouki-s-15b17a1a6)**: Connect with the creator.

---

### Professional Services

If you need a custom trading strategy, a proprietary risk model, advanced data pipelines, or a dedicated copy trading server setup, professional services are available.

**Contact the Developer:**  
📧 [bertin@bbs-trading.com](mailto:bertin@bbs-trading.com)

---

### Support the Project

If you find this project useful and would like to support its continued development, you can contribute here:

☕ [Support the Developer](https://paypal.me/bertinbalouki?country.x=SN&locale.x=en_US)

---

_Disclaimer: Trading involves significant risk. `bbstrader` provides the tools, but you provide the strategy. Test thoroughly on demo accounts before deploying real capital._
