"""Overfitting diagnostics for strategy research.

Implements the Bailey & Lopez de Prado toolkit for distinguishing real alpha
from selection bias:

* Probabilistic and Deflated Sharpe ratios adjust an observed Sharpe for
  sample length, non-normality, and the number of trials that produced it.
* CSCV PBO the probability of backtest overfitting from combinatorially
  symmetric cross-validation.
* Combinatorial purged cross-validation splits multiple train/test folds
  with purging/embargo for leakage-free out-of-sample evaluation.

All routines are deterministic and pure NumPy/SciPy.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, Iterator, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats

__all__ = [
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "cscv_pbo",
    "combinatorial_splits",
]

_EULER_MASCHERONI = 0.5772156649015329


def _sharpe(returns: NDArray[np.float64]) -> float:
    """Return the per-period Sharpe ratio of a return series.

    Args:
        returns (NDArray[np.float64]): The periodic returns. Uses the sample
            standard deviation (``ddof=1``); returns 0 for degenerate inputs.

    Returns:
        float: The unannualised Sharpe ratio, or ``0.0`` when undefined.
    """
    sd = returns.std(ddof=1) if returns.size > 1 else 0.0
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(returns.mean() / sd)


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability that the true Sharpe exceeds ``benchmark`` (PSR).

    ``sharpe`` and ``benchmark`` are per-observation (non-annualized) Sharpe
    ratios. ``skew``/``kurtosis`` are the return distribution's moments
    (kurtosis 3 == normal).
    """
    if n_obs < 2:
        return 0.0
    denom = math.sqrt(1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2)
    if denom == 0:
        return 0.0
    z = (sharpe - benchmark) * math.sqrt(n_obs - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum of ``n_trials`` independent Sharpe estimates.

    The benchmark a strategy must beat to be considered non-random when it was
    selected from ``n_trials`` candidates (Bailey & Lopez de Prado).
    """
    if n_trials < 2 or sharpe_variance <= 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sharpe_variance) * (
        (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
    )


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    n_trials: int,
    sharpe_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (DSR).

    PSR computed against the expected maximum Sharpe across ``n_trials``, i.e.
    the probability the strategy's Sharpe is real after accounting for multiple
    testing. ``sharpe``/``sharpe_variance`` are per-observation.
    """
    benchmark = expected_max_sharpe(n_trials, sharpe_variance)
    return probabilistic_sharpe_ratio(sharpe, n_obs, benchmark, skew, kurtosis)


def cscv_pbo(
    performance: NDArray[np.float64],
    n_splits: int = 10,
    metric: Optional[Callable[[NDArray[np.float64]], float]] = None,
) -> float:
    """Probability of Backtest Overfitting via combinatorially symmetric CV.

    Args:
        performance: A (T, N) matrix of per-observation returns for N candidate
            configurations over T observations.
        n_splits: Number of disjoint row blocks S (must be even); IS/OOS are all
            C(S, S/2) balanced partitions.
        metric: Per-configuration score from a sub-matrix of returns. Defaults to
            the Sharpe ratio.

    Returns:
        PBO in [0, 1]: the fraction of partitions where the in-sample best
        configuration ranks below the out-of-sample median.
    """
    perf = np.asarray(performance, dtype=np.float64)
    if perf.ndim != 2:
        raise ValueError("performance must be a 2-D (T, N) matrix.")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even.")
    score = metric or _sharpe
    n_obs, n_cfg = perf.shape
    blocks = np.array_split(np.arange(n_obs), n_splits)

    logits = []
    for combo in itertools.combinations(range(n_splits), n_splits // 2):
        is_rows = np.concatenate([blocks[b] for b in combo])
        oos_rows = np.concatenate(
            [blocks[b] for b in range(n_splits) if b not in combo]
        )
        is_scores = np.array([score(perf[is_rows, c]) for c in range(n_cfg)])
        oos_scores = np.array([score(perf[oos_rows, c]) for c in range(n_cfg)])
        best = int(np.argmax(is_scores))
        # Relative rank of the IS-best config among OOS scores.
        rank = float(stats.rankdata(oos_scores)[best])
        omega = rank / (n_cfg + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(math.log(omega / (1.0 - omega)))

    logits_arr = np.array(logits)
    return float(np.mean(logits_arr <= 0.0))


def combinatorial_splits(
    n_obs: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo: int = 0,
) -> Iterator[Tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """Yield combinatorial purged cross-validation (CPCV) train/test splits.

    Observations are partitioned into ``n_groups`` contiguous blocks; every
    combination of ``n_test_groups`` blocks forms a test set, with the remaining
    blocks (minus an ``embargo`` band around each test block, to prevent
    leakage) as the training set. Yields C(n_groups, n_test_groups) folds.
    """
    if n_test_groups >= n_groups:
        raise ValueError("n_test_groups must be smaller than n_groups.")
    groups = np.array_split(np.arange(n_obs), n_groups)
    for combo in itertools.combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([groups[g] for g in combo])
        train_mask = np.ones(n_obs, dtype=bool)
        train_mask[test_idx] = False
        if embargo > 0:
            for g in combo:
                start, end = groups[g][0], groups[g][-1]
                lo = max(0, start - embargo)
                hi = min(n_obs, end + 1 + embargo)
                train_mask[lo:hi] = False
        train_idx = np.where(train_mask)[0]
        yield train_idx, test_idx
