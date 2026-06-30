# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import types
from importlib.metadata import PackageNotFoundError, version
from unittest.mock import MagicMock

# Import bbstrader from the source tree without building it on ReadTheDocs.
sys.path.insert(0, os.path.abspath("../src"))

# The compiled C++ extension `bbstrader.api.client` is not built here, yet it is
# imported via `from bbstrader.api.client import *` at package import time. A
# generic mock does not bind names through a star-import, so we inject a fake
# module that exposes the extension's public symbols via `__all__`.
_client_symbols = [
    "AccountInfo",
    "BookInfo",
    "MetaTraderClient",
    "MetaTraderHandlers",
    "OrderCheckResult",
    "OrderSentResult",
    "RateInfo",
    "SymbolInfo",
    "TerminalInfo",
    "TickInfo",
    "TradeDeal",
    "TradeOrder",
    "TradePosition",
    "TradeRequest",
]
_client = types.ModuleType("bbstrader.api.client")
for _symbol in _client_symbols:
    setattr(_client, _symbol, MagicMock())
_client.__all__ = _client_symbols
sys.modules["bbstrader.api.client"] = _client

project = "bbstrader"
copyright = "2023 - 2026, Bertin Balouki SIMYELI"
author = "Bertin Balouki SIMYELI"

try:
    release = version("bbstrader")
except PackageNotFoundError:
    release = "unknown"
version = ".".join(release.split(".")[:2])

# General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
]

# Every third-party package imported at module top level must be either
# installed (see docs/requirements.txt) or mocked here; otherwise autodoc fails
# to import the module and renders an empty section. A failure inside any
# submodule also cascades through its package __init__ star-imports, blanking
# the whole package. Mocking keeps the ReadTheDocs build fast and avoids
# compiling/installing heavy or platform-specific dependencies.
#
# This list must cover the full set of third-party imports in src/bbstrader,
# minus the few installed for real (numpy, pandas, pytz). Use *import* names,
# not distribution names (e.g. bs4 not beautifulsoup4, telegram not
# python-telegram-bot, notifypy not notify_py, pypfopt not pyportfolioopt).
autodoc_mock_imports = [
    # Platform-specific.
    "MetaTrader5",
    # Networking / scraping (core/data, etc.).
    "requests",
    "certifi",
    "bs4",
    # Market data and quant libraries.
    "eodhd",
    "financetoolkit",
    "yfinance",
    "exchange_calendars",
    "pypfopt",
    "quantstats",
    "scipy",
    # Plotting.
    "matplotlib",
    "plotly",
    "seaborn",
    "PIL",
    # NLP / social (optional extras).
    "spacy",
    "nltk",
    "sumy",
    "textblob",
    "vaderSentiment",
    "praw",
    "tweepy",
    # CLI / notifications / misc.
    "tabulate",
    "telegram",
    "notifypy",
    "loguru",
    "colorama",
    "pyfiglet",
]
# NB: pyarrow is intentionally NOT mocked. The real pandas imports pyarrow and
# reads pyarrow.__version__ at import time, which a mock breaks; leaving it
# absent is safe because catalog.py imports it lazily inside a function and
# falls back to CSV when it is missing.


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
