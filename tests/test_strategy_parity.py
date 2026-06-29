"""Backtest <-> live strategy parity tests.

The shared ``BaseStrategy`` / ``TradeSignal`` model is the project's key asset:
a strategy written once must behave identically in backtest and live. These
tests assert that ``BacktestStrategy`` and ``LiveStrategy`` inherit the same
shared helpers (so signal-handling logic cannot silently diverge) and that those
helpers produce the documented results.
"""

import pytest

from bbstrader.btengine.strategy import BacktestStrategy
from bbstrader.core.strategy import BaseStrategy, Strategy
from bbstrader.trading.strategy import LiveStrategy

SHARED_HELPERS = [
    "calculate_pct_change",
    "is_signal_time",
    "get_current_dt",
    "convert_time_zone",
    "stop_time",
    "get_quantity",
    "get_quantities",
    "apply_risk_management",
]


def test_both_subclass_base_strategy():
    assert issubclass(BacktestStrategy, BaseStrategy)
    assert issubclass(LiveStrategy, BaseStrategy)
    assert issubclass(BaseStrategy, Strategy)


@pytest.mark.parametrize("name", SHARED_HELPERS)
def test_shared_helpers_are_the_same_object(name):
    # Neither subclass overrides the shared helpers, so they are guaranteed to
    # behave identically in backtest and live.
    base = getattr(BaseStrategy, name)
    assert getattr(BacktestStrategy, name) is base
    assert getattr(LiveStrategy, name) is base


@pytest.mark.parametrize("name", ["calculate_signals", "get_asset_values"])
def test_both_implement_the_abstract_contract(name):
    # These differ by venue (the parity *points*); both must provide them.
    assert callable(getattr(BacktestStrategy, name))
    assert callable(getattr(LiveStrategy, name))


def test_calculate_pct_change_value():
    assert BaseStrategy.calculate_pct_change(110.0, 100.0) == pytest.approx(10.0)
    assert BaseStrategy.calculate_pct_change(90.0, 100.0) == pytest.approx(-10.0)


@pytest.mark.parametrize(
    "count,interval,expected",
    [(0, 5, True), (None, 5, True), (5, 5, True), (10, 5, True), (3, 5, False)],
)
def test_is_signal_time(count, interval, expected):
    assert BaseStrategy.is_signal_time(count, interval) is expected


class _MiniStrategy(BaseStrategy):
    """Minimal concrete BaseStrategy for exercising shared instance helpers."""

    def calculate_signals(self, *args, **kwargs):
        return None

    def get_asset_values(self, *args, **kwargs):
        return None

    @property
    def cash(self) -> float:
        return 1000.0


def test_get_quantities_parity_forms():
    strat = _MiniStrategy(["AAA", "BBB"])
    assert strat.get_quantities(None) == {"AAA": None, "BBB": None}
    assert strat.get_quantities(5) == {"AAA": 5, "BBB": 5}
    assert strat.get_quantities({"AAA": 1}) == {"AAA": 1}
    with pytest.raises(TypeError):
        strat.get_quantities("bad")
