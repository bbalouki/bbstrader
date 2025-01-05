from financetoolkit import Toolkit

__all__ = [
    'FMP',
]


class FMP(Toolkit):
    """
    FMPData class for fetching data from Financial Modeling Prep API
    using the Toolkit class from financetoolkit package.

    See `financetoolkit` for more details.

    """

    def __init__(self, api_key: str = '', symbols: str | list = 'AAPL'):
        super().__init__(tickers=symbols, api_key=api_key)


class DataBendo:
    ...
