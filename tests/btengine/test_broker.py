"""Tests for the broker abstraction and the PaperBroker adapter."""

import pytest

from bbstrader.core.broker import (
    AccountInfo,
    BrokerOrder,
    OrderSide,
    OrderType,
    PaperBroker,
)


def test_connect_and_account():
    b = PaperBroker(cash=10000.0)
    assert b.connect() is True
    acct = b.account()
    assert isinstance(acct, AccountInfo)
    assert acct.cash == pytest.approx(10000.0)
    assert acct.equity == pytest.approx(10000.0)


def test_market_buy_updates_cash_and_position():
    b = PaperBroker(cash=10000.0)
    b.set_price("AAA", 100.0)
    b.submit_order(BrokerOrder("AAA", OrderSide.BUY, 10))
    pos = b.positions()[0]
    assert pos.quantity == pytest.approx(10)
    assert pos.avg_price == pytest.approx(100.0)
    assert b.account().cash == pytest.approx(10000.0 - 1000.0)


def test_average_price_on_add():
    b = PaperBroker(cash=100000.0)
    b.set_price("AAA", 100.0)
    b.submit_order(BrokerOrder("AAA", OrderSide.BUY, 10))
    b.set_price("AAA", 120.0)
    b.submit_order(BrokerOrder("AAA", OrderSide.BUY, 10))
    pos = b.positions()[0]
    assert pos.quantity == pytest.approx(20)
    assert pos.avg_price == pytest.approx(110.0)


def test_realized_pnl_on_close():
    b = PaperBroker(cash=100000.0)
    b.set_price("AAA", 100.0)
    b.submit_order(BrokerOrder("AAA", OrderSide.BUY, 10))
    b.set_price("AAA", 130.0)
    b.submit_order(BrokerOrder("AAA", OrderSide.SELL, 10))
    assert b.realized_pnl == pytest.approx(300.0)  # (130-100)*10
    assert b.positions() == []


def test_equity_marks_to_market():
    b = PaperBroker(cash=10000.0)
    b.set_price("AAA", 100.0)
    b.submit_order(BrokerOrder("AAA", OrderSide.BUY, 10))
    b.set_price("AAA", 150.0)
    # cash 9000 + 10*150 = 10500.
    assert b.account().equity == pytest.approx(10500.0)


def test_limit_order_rests_until_crossed():
    b = PaperBroker(cash=100000.0)
    b.set_price("AAA", 100.0)
    b.submit_order(BrokerOrder("AAA", OrderSide.BUY, 10, OrderType.LIMIT, price=95.0))
    # Not crossed yet -> resting, no position.
    assert b.positions() == []
    assert len(b.orders()) == 1
    b.set_price("AAA", 94.0)  # crosses the limit
    assert b.positions()[0].quantity == pytest.approx(10)
    assert b.orders() == []


def test_stop_order_triggers_on_breakout():
    b = PaperBroker(cash=100000.0)
    b.set_price("AAA", 100.0)
    b.submit_order(BrokerOrder("AAA", OrderSide.BUY, 5, OrderType.STOP, price=110.0))
    assert b.positions() == []
    b.set_price("AAA", 111.0)
    assert b.positions()[0].quantity == pytest.approx(5)


def test_invalid_quantity_raises():
    b = PaperBroker()
    b.set_price("AAA", 100.0)
    with pytest.raises(ValueError):
        b.submit_order(BrokerOrder("AAA", OrderSide.BUY, 0))
