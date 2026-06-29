import warnings

import scipy.cluster.hierarchy as _sch
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.black_litterman import BlackLittermanModel

# PyPortfolioOpt 1.5.6 (the latest release) validates the linkage method against
# the private ``scipy.cluster.hierarchy._LINKAGE_METHODS`` dict, which newer
# SciPy releases removed. Restore the canonical mapping so HRP keeps working
# across SciPy versions without pinning users to an old SciPy.
if not hasattr(_sch, "_LINKAGE_METHODS"):
    _sch._LINKAGE_METHODS = {
        "single": 0,
        "complete": 1,
        "average": 2,
        "weighted": 3,
        "centroid": 4,
        "median": 5,
        "ward": 6,
    }

__all__ = [
    "markowitz_weights",
    "hierarchical_risk_parity",
    "black_litterman_weights",
    "equal_weighted",
    "optimized_weights",
]


def markowitz_weights(prices=None, rfr=0.0, freq=252, min_vol=False):
    """
    Calculates optimal portfolio weights using Markowitz's mean-variance optimization (Max Sharpe Ratio or Min Volatility) with multiple solvers.

    Parameters
    ----------
    prices : pd.DataFrame, optional
        Price data for assets, where rows represent time periods and columns represent assets.
    rfr : float, optional
        Risk-free rate (default is 0.0).
    freq : int, optional
        Frequency of the data, such as 252 for daily returns in a year (default is 252).
    min_vol : bool, optional
        If True, optimizes for minimum volatility instead of maximum Sharpe ratio (default is False).

    Returns
    -------
    dict
        Dictionary containing the optimal asset weights for maximizing the Sharpe ratio or minimizing volatility, normalized to sum to 1.

    Notes
    -----
    This function attempts to maximize the Sharpe ratio by iterating through various solvers ('SCS', 'ECOS', 'OSQP')
    from the PyPortfolioOpt library. If a solver fails, it proceeds to the next one. If none succeed, an error message
    is printed for each solver that fails.

    This function is useful for portfolio with a small number of assets, as it may not scale well for large portfolios.

    Raises
    ------
    Exception
        If all solvers fail, each will print an exception error message during runtime.
    """
    returns = expected_returns.mean_historical_return(prices, frequency=freq)
    cov = risk_models.sample_cov(prices, frequency=freq)

    # Try different solvers to maximize Sharpe ratio
    for solver in ["SCS", "ECOS", "OSQP"]:
        ef = EfficientFrontier(
            expected_returns=returns,
            cov_matrix=cov,
            weight_bounds=(0, 1),
            solver=solver,
        )
        try:
            if min_vol:
                ef.min_volatility()
            else:
                ef.max_sharpe(risk_free_rate=rfr)
            return ef.clean_weights()
        except Exception as e:
            print(f"Solver {solver} failed with error: {e}")
    # Default to equal weighted if all solvers fail
    return equal_weighted(prices=prices)


def hierarchical_risk_parity(prices=None, returns=None, freq=252):
    """
    Computes asset weights using Hierarchical Risk Parity (HRP) for risk-averse portfolio allocation.

    Parameters
    ----------
    prices : pd.DataFrame, optional
        Price data for assets; if provided, daily returns will be calculated.
    returns : pd.DataFrame, optional
        Daily returns for assets. One of `prices` or `returns` must be provided.
    freq : int, optional
        Number of days to consider in calculating portfolio weights (default is 252).

    Returns
    -------
    dict
        Optimized asset weights using the HRP method, with asset weights summing to 1.

    Raises
    ------
    ValueError
        If neither `prices` nor `returns` are provided.

    Notes
    -----
    Hierarchical Risk Parity is particularly useful for portfolios with a large number of assets,
    as it mitigates issues of multicollinearity and estimation errors in covariance matrices by
    using hierarchical clustering.
    """
    warnings.filterwarnings("ignore")
    if returns is None and prices is None:
        raise ValueError("Either prices or returns must be provided")
    if returns is None:
        returns = prices.pct_change().dropna(how="all")
    # Remove duplicate columns and index
    returns = returns.loc[:, ~returns.columns.duplicated()]
    returns = returns.loc[~returns.index.duplicated(keep="first")]
    hrp = HRPOpt(returns=returns.iloc[-freq:])
    return hrp.optimize()


def equal_weighted(prices=None, returns=None, round_digits=5):
    """
    Generates an equal-weighted portfolio by assigning an equal proportion to each asset.

    Parameters
    ----------
    prices : pd.DataFrame, optional
        Price data for assets, where each column represents an asset.
    returns : pd.DataFrame, optional
        Return data for assets. One of `prices` or `returns` must be provided.
    round_digits : int, optional
        Number of decimal places to round each weight to (default is 5).

    Returns
    -------
    dict
        Dictionary with equal weights assigned to each asset, summing to 1.

    Raises
    ------
    ValueError
        If neither `prices` nor `returns` are provided.

    Notes
    -----
    Equal weighting is a simple allocation method that assumes equal importance across all assets,
    useful as a baseline model and when no strong views exist on asset return expectations or risk.
    """

    if returns is None and prices is None:
        raise ValueError("Either prices or returns must be provided")
    if returns is None:
        n = len(prices.columns)
        columns = prices.columns
    else:
        n = len(returns.columns)
        columns = returns.columns
    return {col: round(1 / n, round_digits) for col in columns}


def black_litterman_weights(
    prices=None,
    rfr=0.0,
    freq=252,
    views=None,
    view_confidences=None,
    pi=None,
    market_caps=None,
):
    """
    Computes portfolio weights using the Black-Litterman model.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data for assets.
    rfr : float, optional
        Risk-free rate (default is 0.0).
    freq : int, optional
        Frequency of the data (default is 252).
    views : dict, optional
        Investor's views on asset returns.
    view_confidences : list or np.array, optional
        Confidence levels for each view.
    pi : pd.Series, optional
        Market-implied prior returns.
    market_caps : pd.Series, optional
        Market capitalization of assets.

    Returns
    -------
    dict
        Optimal asset weights based on the Black-Litterman model.
    """
    cov_matrix = risk_models.sample_cov(prices, frequency=freq)
    if pi is None:
        if market_caps is not None:
            # If market caps are provided, we can use them to compute the prior
            # This requires a benchmark, which we don't have here easily.
            # For simplicity, we use the mean historical return if pi is not provided.
            pi = expected_returns.mean_historical_return(prices, frequency=freq)
        else:
            pi = expected_returns.mean_historical_return(prices, frequency=freq)

    bl = BlackLittermanModel(
        cov_matrix,
        pi=pi,
        absolute_views=views,
        omega=None,
        view_confidences=view_confidences,
    )
    ret_bl = bl.bl_returns()
    ef = EfficientFrontier(ret_bl, cov_matrix)
    ef.max_sharpe(risk_free_rate=rfr)
    return ef.clean_weights()


def optimized_weights(
    prices=None, returns=None, rfr=0.0, freq=252, method="equal", **kwargs
):
    """
    Selects an optimization method to calculate portfolio weights based on user preference.

    Parameters
    ----------
    prices : pd.DataFrame, optional
        Price data for assets, required for certain methods.
    returns : pd.DataFrame, optional
        Returns data for assets, an alternative input for certain methods.
    freq : int, optional
        Number of days for calculating portfolio weights, such as 252 for a year's worth of daily returns (default is 252).
    method : str, optional
        Optimization method to use ('markowitz', 'hrp', or 'equal') (default is 'equal').

    Returns
    -------
    dict
        Dictionary containing optimized asset weights based on the chosen method.

    Raises
    ------
    ValueError
        If an unknown optimization method is specified.

    Notes
    -----
    This function integrates different optimization methods:
    - 'markowitz': mean-variance optimization with max Sharpe ratio
    - 'hrp': Hierarchical Risk Parity, for risk-based clustering of assets
    - 'equal': Equal weighting across all assets
    """
    if method == "markowitz":
        return markowitz_weights(prices=prices, rfr=rfr, freq=freq)
    elif method == "min_vol":
        return markowitz_weights(prices=prices, rfr=rfr, freq=freq, min_vol=True)
    elif method == "hrp":
        return hierarchical_risk_parity(prices=prices, returns=returns, freq=freq)
    elif method == "black_litterman":
        return black_litterman_weights(prices=prices, rfr=rfr, freq=freq, **kwargs)
    elif method == "equal":
        return equal_weighted(prices=prices, returns=returns)
    else:
        raise ValueError(f"Unknown method: {method}")
