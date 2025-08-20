# tests/test_backtest_engine.py
import numpy as np
from backtest.engine import BacktestEngine
from execution.mock_adapter import MockExecutionAdapter
from execution.adapter import Order

def test_backtest_engine_linear_price_buy_and_hold():
    # Create strictly increasing price series so buy-and-hold yields profit
    price_series = np.linspace(1.0, 2.0, num=10)  # length 10
    adapter = MockExecutionAdapter(initial_market_price=price_series[0], initial_balance=1000.0)
    engine = BacktestEngine(adapter, price_series, symbol="SYM")

    # Simple signal generator: buy 1 unit on first step, do nothing else
    def signal_gen(step, ctx):
        if step == 0:
            return [Order(symbol=ctx["symbol"], side="buy", volume=1.0)]
        return []

    history = engine.run(signal_gen)
    # final snapshot is last element
    final = history[-1]
    final_balance = final["account"]["balance"]
    # price increased from 1.0 to 2.0, profit = (2.0 - 1.0) * 1.0 = 1.0
    assert final_balance > 1000.0, "Final balance should be greater than initial after buy-and-hold"
    assert round(final_balance - 1000.0, 6) == 1.0
