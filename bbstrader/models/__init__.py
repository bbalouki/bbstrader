"""
Overview
========

The Models Module provides a collection of quantitative models for financial analysis and decision-making.
It includes tools for portfolio optimization and natural language processing (NLP) to extract insights
from financial text data. This module is designed to support quantitative trading strategies by
providing a robust framework for financial modeling.

Features
========

- **Portfolio Optimization**: Implements techniques to optimize portfolio allocation, helping to maximize returns and manage risk.
- **Natural Language Processing (NLP)**: Provides tools for analyzing financial news and other text-based data to gauge market sentiment.
- **Extensible Design**: Structured to allow for the easy addition of new quantitative models and algorithms.

Components
==========

- **Optimization**: Contains portfolio optimization models and related utilities.
- **NLP**: Includes tools and models for natural language processing tailored for financial applications.

Examples
========

>>> from bbstrader.models import PortfolioOptimizer
>>> # Assuming 'returns' is a DataFrame of asset returns
>>> optimizer = PortfolioOptimizer(returns)
>>> optimal_weights = optimizer.optimize()
>>> print(optimal_weights)

Notes
=====

This module is focused on providing the analytical tools for quantitative analysis. The models
can be integrated into trading strategies to provide data-driven signals.
"""
from bbstrader.models.optimization import *  # noqa: F403
from bbstrader.models.nlp import *  # noqa: F403
