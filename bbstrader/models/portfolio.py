import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from bbstrader.models.optimization import (
    equal_weighted,
    hierarchical_risk_parity,
    markowitz_weights,
)

__all__ = ["EigenPortfolios"]


class EigenPortfolios(object):
    """
    The `EigenPortfolios` class applies Principal Component Analysis (PCA) to a covariance matrix of normalized asset returns
    to derive portfolios (eigenportfolios) that capture distinct risk factors in the asset returns. Each eigenportfolio
    represents a principal component of the return covariance matrix, ordered by the magnitude of its eigenvalue. These
    portfolios capture most of the variance in asset returns and are mutually uncorrelated.

    Notes
    -----
    The implementation is inspired by the book "Machine Learning for Algorithmic Trading" by Stefan Jansen.

    References
    ----------
    Stefan Jansen (2020). Machine Learning for Algorithmic Trading - Second Edition.
    chapter 13, Data-Driven Risk Factors and Asset Allocation with Unsupervised Learning.

    """

    def __init__(self):
        self.returns = None
        self.n_portfolios = None
        self._portfolios = None
        self._fit_called = False

    def get_portfolios(self) -> pd.DataFrame:
        """
        Returns the computed eigenportfolios (weights of assets in each portfolio).

        Returns
        -------
        pd.DataFrame
            DataFrame containing eigenportfolio weights for each asset.

        Raises
        ------
        ValueError
            If `fit()` has not been called before retrieving portfolios.
        """
        if not self._fit_called:
            raise ValueError("fit() must be called first")
        return self._portfolios

    def fit(self, returns: pd.DataFrame, n_portfolios: int = 4) -> pd.DataFrame:
        """
        Computes the eigenportfolios based on PCA of the asset returns' covariance matrix.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns of assets to be used for PCA.
        n_portfolios : int, optional
            Number of eigenportfolios to compute (default is 4).

        Returns
        -------
        pd.DataFrame
            DataFrame containing normalized weights for each eigenportfolio.

        Notes
        -----
        This method performs winsorization and normalization on returns to reduce the impact of outliers
        and achieve zero mean and unit variance. It uses the first `n_portfolios` principal components
        as portfolio weights.
        """
        # Winsorize and normalize the returns
        normed_returns = scale(
            returns.clip(
                lower=returns.quantile(q=0.025), upper=returns.quantile(q=0.975), axis=1
            ).apply(lambda x: x.sub(x.mean()).div(x.std()))
        )
        returns = returns.dropna(thresh=int(normed_returns.shape[0] * 0.95), axis=1)
        returns = returns.dropna(thresh=int(normed_returns.shape[1] * 0.95))

        cov = returns.cov()
        cov.columns = cov.columns.astype(str)
        pca = PCA()
        pca.fit(cov)

        top_portfolios = pd.DataFrame(
            pca.components_[:n_portfolios], columns=cov.columns
        )
        eigen_portfolios = top_portfolios.div(top_portfolios.sum(axis=1), axis=0)
        eigen_portfolios.index = [f"Portfolio {i}" for i in range(1, n_portfolios + 1)]
        self._portfolios = eigen_portfolios
        self.returns = returns
        self.n_portfolios = n_portfolios
        self._fit_called = True

    def plot_weights(self):
        """
        Plots the weights of each asset in each eigenportfolio as bar charts.

        Notes
        -----
        Each subplot represents one eigenportfolio, showing the contribution of each asset.
        """
        eigen_portfolios = self.get_portfolios()
        n_cols = 2
        n_rows = (self.n_portfolios + 1) // n_cols
        figsize = (n_cols * 10, n_rows * 5)
        axes = eigen_portfolios.T.plot.bar(
            subplots=True, layout=(n_rows, n_cols), figsize=figsize, legend=False
        )
        for ax in axes.flatten():
            ax.set_ylabel("Portfolio Weight")
            ax.set_xlabel("")

        sns.despine()
        plt.tight_layout()
        plt.show()

    def plot_performance(self):
        """
        Plots the cumulative returns of each eigenportfolio over time.

        Notes
        -----
        This method calculates the historical cumulative performance of each eigenportfolio
        by weighting asset returns according to eigenportfolio weights.
        """
        eigen_portfolios = self.get_portfolios()
        returns = self.returns.copy()

        n_cols = 2
        n_rows = (self.n_portfolios + 1 + n_cols - 1) // n_cols
        figsize = (n_cols * 10, n_rows * 5)
        fig, axes = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=figsize, sharex=True
        )
        axes = axes.flatten()
        returns.mean(1).add(1).cumprod().sub(1).plot(title="The Market", ax=axes[0])

        for i in range(self.n_portfolios):
            rc = returns.mul(eigen_portfolios.iloc[i]).sum(1).add(1).cumprod().sub(1)
            rc.plot(title=f"Portfolio {i+1}", ax=axes[i + 1], lw=1, rot=0)

        for j in range(self.n_portfolios + 1, len(axes)):
            fig.delaxes(axes[j])

        for i in range(self.n_portfolios + 1):
            axes[i].set_xlabel("")

        sns.despine()
        fig.tight_layout()
        plt.show()

    def optimize(
        self,
        portfolio: int = 1,
        optimizer: str = "hrp",
        prices=None,
        freq=252,
        plot=True,
    ):
        """
        Optimizes the chosen eigenportfolio based on a specified optimization method.

        Parameters
        ----------
        portfolio : int, optional
            Index of the eigenportfolio to optimize (default is 1).
        optimizer : str, optional
            Optimization method: 'markowitz', 'hrp' (Hierarchical Risk Parity), or 'equal' (default is 'hrp').
        prices : pd.DataFrame, optional
            Asset prices used for Markowitz optimization (required if optimizer is 'markowitz').
        freq : int, optional
            Frequency of returns (e.g., 252 for daily returns).
        plot : bool, optional
            Whether to plot the performance of the optimized portfolio (default is True).

        Returns
        -------
        dict
            Dictionary of optimized asset weights.

        Raises
        ------
        ValueError
            If an unknown optimizer is specified, or if prices are not provided when using Markowitz optimization.

        Notes
        -----
        The optimization method varies based on risk-return assumptions, with options for traditional Markowitz optimization,
        Hierarchical Risk Parity, or equal weighting.
        """
        portfolio = self.get_portfolios().iloc[portfolio - 1]
        returns = self.returns.loc[:, portfolio.index]
        returns = returns.loc[:, ~returns.columns.duplicated()]
        returns = returns.loc[~returns.index.duplicated(keep="first")]
        if optimizer == "markowitz":
            if prices is None:
                raise ValueError("prices must be provided for markowitz optimization")
            prices = prices.loc[:, returns.columns]
            weights = markowitz_weights(prices=prices, freq=freq)
        elif optimizer == "hrp":
            weights = hierarchical_risk_parity(returns=returns, freq=freq)
        elif optimizer == "equal":
            weights = equal_weighted(returns=returns)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        if plot:
            # plot the optimized potfolio performance
            returns = returns.filter(weights.keys())
            rc = returns.mul(weights).sum(1).add(1).cumprod().sub(1)
            rc.plot(title=f"Optimized {portfolio.name}", lw=1, rot=0)
            sns.despine()
            plt.show()
        return weights
