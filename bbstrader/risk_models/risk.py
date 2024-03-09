from abc import ABCMeta, abstractmethod


class RiskModel(metaclass=ABCMeta):
    """
    The RiskModel class serves as an abstract base 
    for implementing risk management strategies in financial markets. 
    It is designed to assist in the decision-making process regarding 
    which trades are permissible under current market 
    conditions and how to allocate assets effectively 
    to optimize the risk-reward ratio.

    Risk management is a critical component in trading 
    and investment strategies, aiming to minimize potential losses 
    without significantly reducing the potential for gains. 
    This class encapsulates the core principles of risk management 
    by providing a structured approach to evaluate market conditions 
    and manage asset allocation.

    Implementing classes are required to define two key methods:
    - `which_trade_allowed`: Determines the types of trades 
      that are permissible based on the analysis of current market 
      conditions and the risk profile of the portfolio. 
      This method should analyze the provided returns_val parameter, 
      which could represent historical returns, volatility measures, 
      or other financial metrics, to decide on the 
      suitability of executing certain trades.

    - `which_quantity_allowed`: Defines how assets should be allocated 
      across the portfolio to maintain an optimal balance 
      between risk and return. This involves determining the quantity 
      of each asset that can be held, considering factors 
      such as diversification, liquidity, and the asset's volatility. 
      This method ensures that the portfolio adheres to 
      predefined risk tolerance levels and investment objectives.

    Note:
    Implementing these methods requires a deep understanding of risk management theories,
      market analysis, and portfolio management principles. 
      The implementation should be tailored to the specific needs of 
      the investment strategy and the risk tolerance of the investor 
      or the fund being managed.
    """

    @abstractmethod
    def which_trade_allowed(self, returns_val):
        """
        Determines the types of trades permissible under current market conditions.

        Parameters:
            returns_val: A parameter representing financial metrics 
            such as historical returns or volatility, used to 
            assess market conditions.
        """
        raise NotImplementedError("Should implement which_trade_allowed()")

    @abstractmethod
    def which_quantity_allowed(self):
        """
        Defines the strategy for asset allocation within 
        the portfolio to optimize risk-reward ratio.
        """
        raise NotImplementedError("Should implement which_quantity_allowed()")
