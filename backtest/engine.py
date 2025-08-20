# backtest/engine.py
from typing import Callable, List, Dict, Any
from execution.mock_adapter import MockExecutionAdapter
from execution.adapter import Order, OrderResult
import time

class BacktestEngine:
    """
    Simple bar-level backtest engine.
    - adapter: ExecutionAdapter (MockExecutionAdapter recommended)
    - price_series: 1D iterable/sequence of market prices (length T)
    - dataset_len: number of windows/steps (optional). If provided, engine will run for that many steps.
    - symbol: symbol name used for Orders
    """
    def __init__(self, adapter: MockExecutionAdapter, price_series, symbol: str = "SYM"):
        self.adapter = adapter
        self.price_series = list(price_series)
        self.symbol = symbol
        self.history = []  # account snapshots

    def run(self, signal_generator: Callable[[int, Dict[str, Any]], List[Order]]):
        """
        Runs the backtest.
        - signal_generator(step_index, context) -> list of Order objects to send at that step.
          context is a dict with keys: 'step', 'price', 'adapter', 'symbol'.
        Engine behavior (per step index i):
          - update adapter market price to price_series[i]
          - call signal_generator(i, context) -> get orders -> send each order via adapter
          - record account snapshot after orders
        After steps complete, engine will call close_all_positions (market price set to last price) and record final snapshot.
        Returns history: list of account snapshots (dicts) for each step and final closing step.
        """
        T = len(self.price_series)
        for i in range(T):
            price = float(self.price_series[i])
            # update adapter market price to this bar's price
            self.adapter.update_market_price(price)
            # prepare context
            context = {"step": i, "price": price, "adapter": self.adapter, "symbol": self.symbol}
            orders = signal_generator(i, context) or []
            for o in orders:
                self.adapter.send_order(o)
            # snapshot account
            acc = self.adapter.get_account()
            snapshot = {"step": i, "price": price, "account": acc, "open_positions": [p.__dict__ for p in self.adapter.get_positions()]}
            self.history.append(snapshot)
        # after loop close all remaining positions at last market price
        last_price = float(self.price_series[-1])
        self.adapter.update_market_price(last_price)
        closed = self.adapter.close_all_positions(price=last_price)
        final_acc = self.adapter.get_account()
        final_snapshot = {"step": T, "price": last_price, "account": final_acc, "closed_results": {k: v.__dict__ for k, v in closed.items()}}
        self.history.append(final_snapshot)
        return self.history
