from abc import ABCMeta, abstractmethod
from typing import Dict, Union


class Strategy(metaclass=ABCMeta):
    """
    A `Strategy()` object encapsulates all calculation on market data 
    that generate advisory signals to a `Portfolio` object. Thus all of 
    the "strategy logic" resides within this class. We opted to separate 
    out the `Strategy` and `Portfolio` objects for this backtester, 
    since we believe this is more amenable to the situation of multiple 
    strategies feeding "ideas" to a larger `Portfolio`, which then can handle 
    its own risk (such as sector allocation, leverage). In higher frequency trading, 
    the strategy and portfolio concepts will be tightly coupled and extremely 
    hardware dependent.

    At this stage in the event-driven backtester development there is no concept of 
    an indicator or filter, such as those found in technical trading. These are also 
    good candidates for creating a class hierarchy.

    The strategy hierarchy is relatively simple as it consists of an abstract 
    base class with a single pure virtual method for generating `SignalEvent` objects. 
    """

    @abstractmethod
    def calculate_signals(self, *args, **kwargs) -> Dict[str, Union[str, None]]:
        """
        Provides the mechanisms to calculate the list of signals.
        This methods should return a dictionary of symbols and their respective signals.
        """
        raise NotImplementedError(
            "Should implement calculate_signals()"
        )
