"""Institutional risk analytics: VaR/CVaR, Monte Carlo, regimes, factor exposure.

These functions extend the quantstats-backed :mod:`bbstrader.btengine.performance`
metrics with ex-ante risk and attribution tools. They operate on plain return
series (NumPy arrays or pandas Series) and are deterministic -- the Monte Carlo
routines take an explicit seed -- so they are safe for reproducible research.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

__all__ = [
    "historical_var",
    "parametric_var",
    "historical_cvar",
    "parametric_cvar",
    "MonteCarloResult",
    "monte_carlo_bootstrap",
    "cusum_change_points",
    "volatility_regimes",
    "factor_exposure",
    "rolling_beta",
]

ReturnsLike = Union[NDArray[np.float64], pd.Series, list]


def _clean(returns: ReturnsLike) -> NDArray[np.float64]:
    arr = np.asarray(returns, dtype=np.float64)
    return arr[~np.isnan(arr)]


def historical_var(returns: ReturnsLike, level: float = 0.95) -> float:
    """Historical Value-at-Risk as a positive loss fraction at ``level``."""
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    return float(-np.quantile(r, 1.0 - level))


def parametric_var(returns: ReturnsLike, level: float = 0.95) -> float:
    """Gaussian (parametric) Value-at-Risk as a positive loss fraction."""
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    z = stats.norm.ppf(1.0 - level)
    return float(-(r.mean() + r.std(ddof=1) * z))


def historical_cvar(returns: ReturnsLike, level: float = 0.95) -> float:
    """Historical Conditional VaR (expected shortfall) beyond the VaR threshold."""
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    threshold = np.quantile(r, 1.0 - level)
    tail = r[r <= threshold]
    if tail.size == 0:
        return float(-threshold)
    return float(-tail.mean())


def parametric_cvar(returns: ReturnsLike, level: float = 0.95) -> float:
    """Gaussian Conditional VaR (expected shortfall)."""
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    alpha = 1.0 - level
    z = stats.norm.ppf(alpha)
    es = r.mean() - r.std(ddof=1) * stats.norm.pdf(z) / alpha
    return float(-es)


@dataclass
class MonteCarloResult:
    """Monte Carlo simulated terminal-return distribution and equity bands."""

    terminal_returns: NDArray[np.float64]
    bands: Dict[str, NDArray[np.float64]]
    horizon: int

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.terminal_returns, q))

    @property
    def mean_terminal(self) -> float:
        return float(self.terminal_returns.mean())

    @property
    def prob_loss(self) -> float:
        return float(np.mean(self.terminal_returns < 0.0))


def monte_carlo_bootstrap(
    returns: ReturnsLike,
    n_sims: int = 1000,
    horizon: Optional[int] = None,
    seed: int = 0,
    quantiles=(0.05, 0.5, 0.95),
) -> MonteCarloResult:
    """Bootstrap the return series into equity-curve confidence bands.

    Resamples the historical returns with replacement to build ``n_sims`` equity
    paths over ``horizon`` steps, then reports per-step percentile bands and the
    terminal-return distribution. Deterministic given ``seed``.
    """
    r = _clean(returns)
    if r.size == 0:
        raise ValueError("returns must contain at least one finite value.")
    h = horizon or r.size
    rng = np.random.default_rng(seed)
    # (n_sims, h) sampled returns -> cumulative equity paths.
    sampled = rng.choice(r, size=(n_sims, h), replace=True)
    equity_paths = np.cumprod(1.0 + sampled, axis=1)
    bands = {
        f"q{int(q * 100)}": np.quantile(equity_paths, q, axis=0) for q in quantiles
    }
    terminal = equity_paths[:, -1] - 1.0
    return MonteCarloResult(terminal_returns=terminal, bands=bands, horizon=h)


def cusum_change_points(
    series: ReturnsLike, threshold: float = 1.0
) -> NDArray[np.int_]:
    """Detect mean-shift change points with a two-sided CUSUM filter.

    ``threshold`` is in units of the series' standard deviation. Returns the
    indices at which the cumulative sum breaches the threshold (and resets).
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.array([], dtype=int)
    mean = x.mean()
    sd = x.std(ddof=1) or 1.0
    thr = threshold * sd
    s_pos = s_neg = 0.0
    points = []
    for i, val in enumerate(x):
        diff = val - mean
        s_pos = max(0.0, s_pos + diff)
        s_neg = min(0.0, s_neg + diff)
        if s_pos > thr or s_neg < -thr:
            points.append(i)
            s_pos = s_neg = 0.0
    return np.array(points, dtype=int)


def volatility_regimes(
    returns: ReturnsLike, window: int = 20, n_states: int = 2
) -> NDArray[np.int_]:
    """Label each bar by its volatility regime (0 = lowest vol .. n_states-1).

    A lightweight, dependency-free alternative to an HMM: rolling volatility is
    bucketed into ``n_states`` quantile bins. Useful for conditional-performance
    analysis (how a strategy behaves in calm vs. turbulent regimes).
    """
    r = pd.Series(np.asarray(returns, dtype=np.float64))
    vol = r.rolling(window, min_periods=1).std().fillna(0.0).to_numpy()
    # Bucket by quantile edges so each state holds a comparable share of bars.
    edges = np.quantile(vol, np.linspace(0, 1, n_states + 1)[1:-1])
    return np.digitize(vol, edges).astype(int)


def factor_exposure(
    returns: ReturnsLike, factors: Union[pd.DataFrame, pd.Series, NDArray]
) -> Dict[str, float]:
    """OLS factor regression: alpha, factor betas and R-squared.

    ``factors`` may be a single series (e.g. the market) or a DataFrame of
    factor returns aligned to ``returns``. Returns alpha, one beta per factor
    and the regression R-squared.
    """
    y = _clean(returns)
    F = np.asarray(factors, dtype=np.float64)
    if F.ndim == 1:
        F = F.reshape(-1, 1)
    n = min(len(y), len(F))
    y, F = y[:n], F[:n]
    X = np.column_stack([np.ones(n), F])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    names = (
        list(factors.columns)
        if isinstance(factors, pd.DataFrame)
        else [f"factor_{i}" for i in range(F.shape[1])]
    )
    result = {"alpha": float(coef[0]), "r_squared": 1.0 - ss_res / ss_tot}
    for name, beta in zip(names, coef[1:]):
        result[f"beta_{name}"] = float(beta)
    return result


def rolling_beta(
    returns: ReturnsLike, market: ReturnsLike, window: int = 60
) -> NDArray[np.float64]:
    """Rolling market beta (cov/var) over ``window`` bars; NaN until warmed up."""
    r = pd.Series(np.asarray(returns, dtype=np.float64))
    m = pd.Series(np.asarray(market, dtype=np.float64))
    cov = r.rolling(window).cov(m)
    var = m.rolling(window).var()
    return (cov / var).to_numpy()
