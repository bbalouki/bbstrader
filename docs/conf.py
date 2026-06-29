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

# Heavy or platform-specific dependencies imported at module top level. Mocking
# them keeps the ReadTheDocs build fast and avoids compiling/installing them.
autodoc_mock_imports = [
    "MetaTrader5",
    "financetoolkit",
    "eodhd",
    "pypfopt",
    "exchange_calendars",
    "quantstats",
    "yfinance",
    "spacy",
    "nltk",
    "sumy",
    "textblob",
    "vaderSentiment",
    "praw",
    "tweepy",
    "plotly",
    "seaborn",
]
# NB: pyarrow is intentionally NOT mocked. The real pandas imports pyarrow and
# reads pyarrow.__version__ at import time, which a mock breaks; leaving it
# absent is safe because bbstrader's catalog falls back to CSV without it.


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
