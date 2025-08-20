# tests/test_adapter_mock.py
import pytest
from execution.mock_adapter import MockExecutionAdapter
from execution.adapter import Order

def test_send_order_and_position_tracking():
    adapter = MockExecutionAdapter(initial_market_price=1.0, initial_balance=1000.0)
    order = Order(symbol="EURUSD", side="buy", volume=1.0)
    res = adapter.send_order(order)
    assert res.success is True
    assert res.position_id is not None
    positions = adapter.get_positions()
    assert len(positions) == 1
    pos = positions[0]
    assert pos.symbol == "EURUSD"
    assert pos.side == "buy"
    assert pos.volume == 1.0
    assert pos.open_price == pytest.approx(1.0)

def test_update_market_price_and_unrealized_profit():
    adapter = MockExecutionAdapter(initial_market_price=1.0, initial_balance=1000.0)
    res = adapter.send_order(Order(symbol="EURUSD", side="buy", volume=2.0))
    pid = res.position_id
    # increase market price
    adapter.update_market_price(1.5)
    acc = adapter.get_account()
    # unrealized = (1.5 - 1.0) * 2 = 1.0
    assert acc["unrealized"] == pytest.approx(1.0)
    # equity = balance + unrealized
    assert acc["equity"] == pytest.approx(adapter.balance + acc["unrealized"])

def test_close_position_and_profit_applied_to_balance():
    adapter = MockExecutionAdapter(initial_market_price=10.0, initial_balance=1000.0)
    send = adapter.send_order(Order(symbol="BTCUSD", side="buy", volume=0.5))
    pid = send.position_id
    # price goes to 12.0
    adapter.update_market_price(12.0)
    before_balance = adapter.balance
    close_res = adapter.close_position(pid)  # closes at current market price
    assert close_res.success is True
    # expected profit = (12.0 - 10.0) * 0.5 = 1.0
    assert "profit=1.0" in close_res.message or "profit=1.0" in str(close_res.message)
    after_balance = adapter.balance
    assert after_balance == pytest.approx(before_balance + 1.0)
    # ensure no positions remain
    assert len(adapter.get_positions()) == 0

def test_close_all_positions():
    adapter = MockExecutionAdapter(initial_market_price=2.0, initial_balance=500.0)
    p1 = adapter.send_order(Order(symbol="AAA", side="buy", volume=1.0))
    p2 = adapter.send_order(Order(symbol="AAA", side="sell", volume=2.0))
    assert len(adapter.get_positions()) == 2
    adapter.update_market_price(1.5)
    results = adapter.close_all_positions()
    assert isinstance(results, dict)
    assert len(results) == 2
    assert len(adapter.get_positions()) == 0
