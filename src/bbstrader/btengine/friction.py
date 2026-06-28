"""Pluggable execution-friction models for the simulated backtester.

The default ``SimExecutionHandler`` fills instantly at the bar price with no
trading costs, which optimistically biases results. These models add realistic
friction -- slippage, market impact, commission, and partial fills -- so a
backtest survives the jump to live trading. They are all **opt-in**: a handler
constructed without friction models behaves exactly as before.

All models are plain, deterministic functions of the order and recent bar data,
so backtests stay reproducible.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from bbstrader.btengine.data import DataHandler

__all__ = [
    "SlippageModel",
    "NoSlippage",
    "FixedSpreadSlippage",
    "PercentSlippage",
    "VolatilitySlippage",
    "VolumeParticipationSlippage",
    "MarketImpactModel",
    "NoImpact",
    "SquareRootImpact",
    "CommissionModel",
    "ZeroCommission",
    "FixedCommission",
    "PerShareCommission",
    "PercentCommission",
    "IBCommission",
]


def _direction_sign(direction: str) -> int:
    """Return +1 for a BUY (pays up) and -1 for a SELL (receives less)."""
    return 1 if direction.upper() == "BUY" else -1


# --------------------------------------------------------------------------- #
# Slippage models                                                             #
# --------------------------------------------------------------------------- #
class SlippageModel(ABC):
    """Adjusts the execution price to account for adverse price movement."""

    @abstractmethod
    def adjusted_price(
        self,
        base_price: float,
        direction: str,
        quantity: float,
        symbol: str,
        bardata: DataHandler,
    ) -> float:
        """Return the slippage-adjusted execution price."""


class NoSlippage(SlippageModel):
    """Fills at the unadjusted base price."""

    def adjusted_price(self, base_price, direction, quantity, symbol, bardata) -> float:
        return base_price


class FixedSpreadSlippage(SlippageModel):
    """Charges half of a fixed spread (in price units) on each fill."""

    def __init__(self, spread: float) -> None:
        if spread < 0:
            raise ValueError("spread must be non-negative.")
        self.spread = float(spread)

    def adjusted_price(self, base_price, direction, quantity, symbol, bardata) -> float:
        return base_price + _direction_sign(direction) * self.spread / 2.0


class PercentSlippage(SlippageModel):
    """Applies a fixed percentage slippage to the base price."""

    def __init__(self, pct: float) -> None:
        if pct < 0:
            raise ValueError("pct must be non-negative.")
        self.pct = float(pct)

    def adjusted_price(self, base_price, direction, quantity, symbol, bardata) -> float:
        return base_price * (1.0 + _direction_sign(direction) * self.pct)


class VolatilitySlippage(SlippageModel):
    """Slippage scaled by recent return volatility.

    ``slippage = coef * sigma * base_price`` where ``sigma`` is the rolling
    standard deviation of returns over ``window`` bars.
    """

    def __init__(self, coef: float = 1.0, window: int = 20) -> None:
        self.coef = float(coef)
        self.window = int(window)

    def adjusted_price(self, base_price, direction, quantity, symbol, bardata) -> float:
        try:
            returns = bardata.get_latest_bars_values(symbol, "returns", N=self.window)
            sigma = float(np.nanstd(returns)) if len(returns) else 0.0
        except (AttributeError, KeyError, ValueError):
            sigma = 0.0
        return base_price * (1.0 + _direction_sign(direction) * self.coef * sigma)


class VolumeParticipationSlippage(SlippageModel):
    """Slippage proportional to the order's share of bar volume."""

    def __init__(self, coef: float = 0.1) -> None:
        self.coef = float(coef)

    def adjusted_price(self, base_price, direction, quantity, symbol, bardata) -> float:
        try:
            volume = float(bardata.get_latest_bar_value(symbol, "volume"))
        except (AttributeError, KeyError, ValueError):
            volume = 0.0
        if volume <= 0:
            return base_price
        participation = abs(quantity) / volume
        return base_price * (
            1.0 + _direction_sign(direction) * self.coef * participation
        )


# --------------------------------------------------------------------------- #
# Market-impact models                                                        #
# --------------------------------------------------------------------------- #
class MarketImpactModel(ABC):
    """Adds price impact from consuming liquidity."""

    @abstractmethod
    def impact(self, base_price: float, quantity: float, direction: str) -> float:
        """Return the per-unit price impact (always adverse)."""


class NoImpact(MarketImpactModel):
    """No market impact."""

    def impact(self, base_price, quantity, direction) -> float:
        return 0.0


class SquareRootImpact(MarketImpactModel):
    """The square-root impact model: impact proportional to sqrt(size / ADV).

    ``impact = coef * base_price * sqrt(|quantity| / adv)``. Suitable for
    institution-scale sizing where impact grows sub-linearly with order size.
    """

    def __init__(self, coef: float = 0.1, adv: float = 1_000_000.0) -> None:
        if adv <= 0:
            raise ValueError("adv (average daily volume) must be positive.")
        self.coef = float(coef)
        self.adv = float(adv)

    def impact(self, base_price, quantity, direction) -> float:
        magnitude = self.coef * base_price * math.sqrt(abs(quantity) / self.adv)
        return _direction_sign(direction) * magnitude


# --------------------------------------------------------------------------- #
# Commission models                                                           #
# --------------------------------------------------------------------------- #
class CommissionModel(ABC):
    """Computes commission for a fill."""

    @abstractmethod
    def commission(self, symbol: str, quantity: float, price: float) -> float:
        """Return the commission charged for the fill."""


class ZeroCommission(CommissionModel):
    """No commission."""

    def commission(self, symbol, quantity, price) -> float:
        return 0.0


class FixedCommission(CommissionModel):
    """A flat fee per fill."""

    def __init__(self, amount: float) -> None:
        self.amount = float(amount)

    def commission(self, symbol, quantity, price) -> float:
        return self.amount


class PerShareCommission(CommissionModel):
    """A per-share/contract fee with an optional minimum."""

    def __init__(self, per_share: float = 0.005, minimum: float = 1.0) -> None:
        self.per_share = float(per_share)
        self.minimum = float(minimum)

    def commission(self, symbol, quantity, price) -> float:
        return max(self.minimum, self.per_share * abs(quantity))


class PercentCommission(CommissionModel):
    """A commission as a percentage of notional with an optional minimum."""

    def __init__(self, pct: float = 0.001, minimum: float = 0.0) -> None:
        self.pct = float(pct)
        self.minimum = float(minimum)

    def commission(self, symbol, quantity, price) -> float:
        return max(self.minimum, self.pct * abs(quantity) * price)


class IBCommission(CommissionModel):
    """The Interactive Brokers tiered share commission used by ``FillEvent``."""

    def commission(self, symbol, quantity, price) -> float:
        qty = abs(quantity)
        if qty <= 500:
            return max(1.30, 0.013 * qty)
        return max(1.30, 0.008 * qty)


def apply_friction(
    base_price: float,
    direction: str,
    quantity: float,
    symbol: str,
    bardata: DataHandler,
    slippage: Optional[SlippageModel],
    impact: Optional[MarketImpactModel],
) -> float:
    """Return the effective fill price after slippage and market impact."""
    price = base_price
    if slippage is not None:
        price = slippage.adjusted_price(price, direction, quantity, symbol, bardata)
    if impact is not None:
        price = price + impact.impact(base_price, quantity, direction)
    return price
