"""Pluggable execution-friction models for the simulated backtester.

The default ``SimExecutionHandler`` fills instantly at the bar price with no
trading costs, which optimistically biases results. These models add realistic
friction slippage, market impact, commission, and partial fills so a
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
    "FundingModel",
    "NoFunding",
    "FixedRateFunding",
    "BrokerSwapFunding",
]


def _direction_sign(direction: str) -> int:
    """Return +1 for a BUY (pays up) and -1 for a SELL (receives less)."""
    return 1 if direction.upper() == "BUY" else -1


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
        """Return ``base_price`` unchanged (see :meth:`SlippageModel.adjusted_price`)."""
        return base_price


class FixedSpreadSlippage(SlippageModel):
    """Charges half of a fixed spread (in price units) on each fill."""

    def __init__(self, spread: float) -> None:
        """Initialise the model with a fixed spread.

        Args:
            spread (float): The full bid-ask spread in price units; half is
                charged on each fill. Must be non-negative.

        Raises:
            ValueError: If ``spread`` is negative.
        """
        if spread < 0:
            raise ValueError("spread must be non-negative.")
        self.spread = float(spread)

    def adjusted_price(self, base_price, direction, quantity, symbol, bardata) -> float:
        """Return ``base_price`` shifted adversely by half the fixed spread."""
        return base_price + _direction_sign(direction) * self.spread / 2.0


class PercentSlippage(SlippageModel):
    """Applies a fixed percentage slippage to the base price."""

    def __init__(self, pct: float) -> None:
        """Initialise the model with a fractional slippage.

        Args:
            pct (float): The slippage as a fraction of price (for example
                ``0.001`` for 10 bps). Must be non-negative.

        Raises:
            ValueError: If ``pct`` is negative.
        """
        if pct < 0:
            raise ValueError("pct must be non-negative.")
        self.pct = float(pct)

    def adjusted_price(self, base_price, direction, quantity, symbol, bardata) -> float:
        """Return ``base_price`` moved adversely by the configured percentage."""
        return base_price * (1.0 + _direction_sign(direction) * self.pct)


class VolatilitySlippage(SlippageModel):
    """Slippage scaled by recent return volatility.

    ``slippage = coef * sigma * base_price`` where ``sigma`` is the rolling
    standard deviation of returns over ``window`` bars.
    """

    def __init__(self, coef: float = 1.0, window: int = 20) -> None:
        """Initialise the model with a volatility coefficient and window.

        Args:
            coef (float): Multiplier applied to the rolling return standard
                deviation to size the slippage.
            window (int): Number of recent bars used to estimate volatility.
        """
        self.coef = float(coef)
        self.window = int(window)

    def adjusted_price(self, base_price, direction, quantity, symbol, bardata) -> float:
        """Return ``base_price`` shifted by ``coef * sigma`` of recent returns."""
        try:
            returns = bardata.get_latest_bars_values(symbol, "returns", N=self.window)
            sigma = float(np.nanstd(returns)) if len(returns) else 0.0
        except (AttributeError, KeyError, ValueError):
            sigma = 0.0
        return base_price * (1.0 + _direction_sign(direction) * self.coef * sigma)


class VolumeParticipationSlippage(SlippageModel):
    """Slippage proportional to the order's share of bar volume."""

    def __init__(self, coef: float = 0.1) -> None:
        """Initialise the model with a participation coefficient.

        Args:
            coef (float): Multiplier applied to the order's share of bar volume
                to size the slippage.
        """
        self.coef = float(coef)

    def adjusted_price(self, base_price, direction, quantity, symbol, bardata) -> float:
        """Return ``base_price`` shifted by the order's share of bar volume."""
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


class MarketImpactModel(ABC):
    """Adds price impact from consuming liquidity."""

    @abstractmethod
    def impact(self, base_price: float, quantity: float, direction: str) -> float:
        """Return the per-unit price impact (always adverse)."""


class NoImpact(MarketImpactModel):
    """No market impact."""

    def impact(self, base_price, quantity, direction) -> float:
        """Return ``0.0`` regardless of order size."""
        return 0.0


class SquareRootImpact(MarketImpactModel):
    """The square-root impact model: impact proportional to sqrt(size / ADV).

    ``impact = coef * base_price * sqrt(|quantity| / adv)``. Suitable for
    institution-scale sizing where impact grows sub-linearly with order size.
    """

    def __init__(self, coef: float = 0.1, adv: float = 1_000_000.0) -> None:
        """Initialise the model with an impact coefficient and ADV.

        Args:
            coef (float): Scales the impact; larger values model thinner books.
            adv (float): Average daily volume used to normalise order size. Must
                be positive.

        Raises:
            ValueError: If ``adv`` is not positive.
        """
        if adv <= 0:
            raise ValueError("adv (average daily volume) must be positive.")
        self.coef = float(coef)
        self.adv = float(adv)

    def impact(self, base_price, quantity, direction) -> float:
        """Return the adverse per-unit impact ``coef * price * sqrt(|qty|/adv)``."""
        magnitude = self.coef * base_price * math.sqrt(abs(quantity) / self.adv)
        return _direction_sign(direction) * magnitude


class CommissionModel(ABC):
    """Computes commission for a fill."""

    @abstractmethod
    def commission(self, symbol: str, quantity: float, price: float) -> float:
        """Return the commission charged for the fill."""


class ZeroCommission(CommissionModel):
    """No commission."""

    def commission(self, symbol, quantity, price) -> float:
        """Return ``0.0`` for every fill."""
        return 0.0


class FixedCommission(CommissionModel):
    """A flat fee per fill."""

    def __init__(self, amount: float) -> None:
        """Initialise the model with the flat per-fill fee.

        Args:
            amount (float): The fee charged on every fill, in account currency.
        """
        self.amount = float(amount)

    def commission(self, symbol, quantity, price) -> float:
        """Return the flat fee regardless of symbol, quantity or price."""
        return self.amount


class PerShareCommission(CommissionModel):
    """A per-share/contract fee with an optional minimum."""

    def __init__(self, per_share: float = 0.005, minimum: float = 1.0) -> None:
        """Initialise the model with a per-share rate and floor.

        Args:
            per_share (float): Fee charged per share or contract filled.
            minimum (float): Minimum commission applied to any fill.
        """
        self.per_share = float(per_share)
        self.minimum = float(minimum)

    def commission(self, symbol, quantity, price) -> float:
        """Return ``per_share * |quantity|`` floored at ``minimum``."""
        return max(self.minimum, self.per_share * abs(quantity))


class PercentCommission(CommissionModel):
    """A commission as a percentage of notional with an optional minimum."""

    def __init__(self, pct: float = 0.001, minimum: float = 0.0) -> None:
        """Initialise the model with a notional percentage and floor.

        Args:
            pct (float): Fraction of traded notional charged as commission.
            minimum (float): Minimum commission applied to any fill.
        """
        self.pct = float(pct)
        self.minimum = float(minimum)

    def commission(self, symbol, quantity, price) -> float:
        """Return ``pct * |quantity| * price`` floored at ``minimum``."""
        return max(self.minimum, self.pct * abs(quantity) * price)


class IBCommission(CommissionModel):
    """The Interactive Brokers tiered share commission used by ``FillEvent``."""

    def commission(self, symbol, quantity, price) -> float:
        """Return the IB tiered per-share commission for the fill."""
        qty = abs(quantity)
        if qty <= 500:
            return max(1.30, 0.013 * qty)
        return max(1.30, 0.008 * qty)


class FundingModel(ABC):
    """Charges the per-bar carrying cost of holding an open position.

    Unlike slippage, impact and commission - which apply once at the fill - a
    funding model is evaluated every bar a position is held, capturing the
    overnight/swap financing that dominates the cost of leveraged CFD and FX
    positions. The returned value is a cash flow (positive = a cost debited from
    the account, negative = a credit) so a model can charge longs while crediting
    shorts, or vice versa.
    """

    @abstractmethod
    def carry(self, symbol: str, quantity: float, price: float) -> float:
        """Return the per-bar carry cash flow for an open position.

        Args:
            symbol (str): The instrument the position is held in.
            quantity (float): The signed position size; positive for a long,
                negative for a short.
            price (float): The current mark-to-market price of one unit.

        Returns:
            float: The cash flow for holding the position over one bar. A
            positive number is a cost debited from cash; a negative number is a
            credit added to cash.
        """


class NoFunding(FundingModel):
    """Applies no carrying cost; positions are free to hold."""

    def carry(self, symbol: str, quantity: float, price: float) -> float:
        """Return zero carry regardless of the position.

        Args:
            symbol (str): Unused; present for interface compatibility.
            quantity (float): Unused; present for interface compatibility.
            price (float): Unused; present for interface compatibility.

        Returns:
            float: Always ``0.0``.
        """
        return 0.0


class FixedRateFunding(FundingModel):
    """A simple cost-of-carry charged as an annual rate on notional.

    The per-bar cost is ``(annual_rate / periods) * quantity * price``. Because
    ``quantity`` is signed, a long position is debited and a short position is
    credited at the same rate, mirroring a basic financing model where the
    holder of a long leveraged position pays to borrow. Supply ``short_rate`` to
    charge shorts at a different annual rate (for example a borrow fee that makes
    shorting a net cost rather than a credit).
    """

    def __init__(
        self,
        annual_rate: float,
        periods: int = 252,
        short_rate: Optional[float] = None,
    ) -> None:
        """Initialise the model with annualised financing rates.

        Args:
            annual_rate (float): The annual financing rate applied to long
                notional (for example ``0.05`` for 5% per year).
            periods (int): The number of bars per year used to convert the
                annual rate to a per-bar rate (for example ``252`` for daily
                bars). Must be positive.
            short_rate (Optional[float]): The annual rate applied to short
                notional. When ``None`` the long ``annual_rate`` is reused, so a
                short earns the symmetric credit; supply an explicit value to
                model an asymmetric borrow cost.

        Raises:
            ValueError: If ``periods`` is not positive.
        """
        if periods <= 0:
            raise ValueError("periods must be a positive number of bars per year.")
        self.annual_rate = float(annual_rate)
        self.periods = int(periods)
        self.short_rate = annual_rate if short_rate is None else float(short_rate)

    def carry(self, symbol: str, quantity: float, price: float) -> float:
        """Return the per-bar financing cash flow for the position.

        Args:
            symbol (str): Unused; the rate is instrument independent.
            quantity (float): The signed position size.
            price (float): The current mark-to-market price of one unit.

        Returns:
            float: ``(rate / periods) * quantity * price`` using the long rate
            for positive quantities and ``short_rate`` for negative ones. A
            positive result is debited from cash.
        """
        if quantity == 0:
            return 0.0
        rate = self.annual_rate if quantity > 0 else self.short_rate
        return (rate / self.periods) * quantity * price


class BrokerSwapFunding(FundingModel):
    """Per-unit swap points charged each bar, mirroring MT5 swap semantics.

    Brokers quote a long and a short swap per lot/unit; this model charges
    ``points * |quantity| * point_value`` each bar, using the long or short
    points according to the sign of the position. Points are expressed as a
    cost: positive points are debited from cash and negative points (a positive
    swap) are credited.
    """

    def __init__(
        self,
        long_points: float,
        short_points: float,
        point_value: float = 1.0,
    ) -> None:
        """Initialise the model with the broker's long/short swap points.

        Args:
            long_points (float): The swap cost per unit per bar applied to long
                positions. Positive debits cash; negative credits it.
            short_points (float): The swap cost per unit per bar applied to short
                positions, with the same sign convention as ``long_points``.
            point_value (float): The cash value of one swap point per unit, used
                to convert points to account currency.
        """
        self.long_points = float(long_points)
        self.short_points = float(short_points)
        self.point_value = float(point_value)

    def carry(self, symbol: str, quantity: float, price: float) -> float:
        """Return the per-bar swap cash flow for the position.

        Args:
            symbol (str): Unused; swap points are supplied per model instance.
            quantity (float): The signed position size.
            price (float): Unused; swap is charged per unit, not on notional.

        Returns:
            float: ``points * |quantity| * point_value`` where ``points`` is the
            long or short swap selected by the sign of ``quantity``. A positive
            result is debited from cash.
        """
        if quantity == 0:
            return 0.0
        points = self.long_points if quantity > 0 else self.short_points
        return points * abs(quantity) * self.point_value


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
