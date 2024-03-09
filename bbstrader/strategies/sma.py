import numpy as np
from btengine.strategy import Strategy

class SMAStrategy(Strategy):
    """
    Carries out a basic Moving Average Crossover strategy with a
    short/long simple weighted moving average. Default short/long
    windows are 50/200 periods respectively and uses Hiden Markov Model 
    as risk Managment system for filteering signals.
    """

    def __init__(
        self, **kwargs
    ):
        self.short_window = kwargs.get("short_window", 50)
        self.long_window = kwargs.get("long_window", 200)

    def get_data(self, prices):
        assert len(prices) >= self.long_window
        short_sma = np.mean(prices[-self.short_window:])
        long_sma = np.mean(prices[-self.long_window:])
        return short_sma, long_sma

    def create_signal(self, prices):
        signal = None
        data = self.get_data(prices)
        short_sma, long_sma = data
        if short_sma > long_sma :
            signal = 'LONG'
        elif short_sma < long_sma:
            signal = 'SHORT'
        return signal
    
    def calculate_signals(self, prices):
            return self.create_signal(prices)