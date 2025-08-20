# tests/test_adapter_mock_contractsize.py
from execution.mock_adapter import MockExecutionAdapter
from execution.adapter import Order

def test_mock_adapter_contract_size_profit():
    # Suppose symbol with contract_size 100, volume 0.5, price diff 2.0 -> profit = 2.0*100*0.5 = 100.0
    specs = {"SYM": {"contract_size": 100.0}}
    adapter = MockExecutionAdapter(initial_market_price=10.0, initial_balance=1000.0, symbol_specs=specs)
    res = adapter.send_order(Order(symbol="SYM", side="buy", volume=0.5))
    pid = res.position_id
    adapter.update_market_price(12.0)  # price up by 2.0
    close = adapter.close_position(pid)
    # profit = (12 - 10) * 100 * 0.5 = 100.0
    assert "profit=100.0" in close.message or "profit=100.0" in str(close.message)
    assert adapter.balance == 1100.0
