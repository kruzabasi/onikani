# execution/mock_adapter.py
from __future__ import annotations
from typing import Dict, Optional, List
import uuid
import time
from dataclasses import asdict
from execution.adapter import ExecutionAdapter, Order, OrderResult, Position


class MockExecutionAdapter(ExecutionAdapter):
    """
    In-memory mock adapter for testing/backtesting with symbol-specific contract sizes/pip handling.
    - symbol_specs: optional dict like {"EURUSD": {"contract_size": 100000, "pip": 0.0001}}
    - profit = (price - open_price) * contract_size * volume  for buy
    """

    def __init__(self, initial_market_price: float = 1.0, initial_balance: float = 10000.0, symbol_specs: Optional[Dict[str, Dict]] = None):
        self.positions: Dict[str, Position] = {}
        self.market_price = float(initial_market_price)
        self.balance = float(initial_balance)
        self._last_closed_profits: Dict[str, float] = {}
        self.symbol_specs = symbol_specs or {}

    def send_order(self, order: Order) -> OrderResult:
        if order.side not in ("buy", "sell"):
            return OrderResult(False, None, None, f"Unknown side {order.side}")
        fill_price = order.price if (order.price is not None) else self.market_price
        pid = str(uuid.uuid4())
        pos = Position(
            position_id=pid,
            symbol=order.symbol,
            side=order.side,
            volume=float(order.volume),
            open_price=float(fill_price),
            open_time=time.time(),
            sl=order.sl,
            tp=order.tp,
            comment=order.client_id,
        )
        self.positions[pid] = pos
        return OrderResult(True, pid, float(fill_price), "filled (mock)")

    def close_position(self, position_id: str, price: Optional[float] = None) -> OrderResult:
        pos = self.positions.get(position_id)
        if pos is None:
            return OrderResult(False, None, None, "position not found")
        close_price = price if (price is not None) else self.market_price
        profit = self._profit_for_position(pos, close_price)
        # update balance
        self.balance += profit
        # store last closed profits
        self._last_closed_profits[position_id] = profit
        # remove position
        del self.positions[position_id]
        return OrderResult(True, position_id, float(close_price), f"closed profit={profit}")

    def get_positions(self) -> List[Position]:
        return list(self.positions.values())

    def get_account(self) -> dict:
        # compute unrealized P&L for open positions
        unreal = sum(self._profit_for_position(p, self.market_price) for p in self.positions.values())
        equity = self.balance + unreal
        return {"balance": self.balance, "equity": equity, "unrealized": unreal, "positions": len(self.positions)}

    def update_market_price(self, new_price: float):
        self.market_price = float(new_price)

    def _profit_for_position(self, pos: Position, price: float) -> float:
        # symbol-based contract size
        spec = self.symbol_specs.get(pos.symbol, {})
        contract_size = float(spec.get("contract_size", 1.0))
        # compute price difference
        if pos.side == "buy":
            diff = price - pos.open_price
        else:
            diff = pos.open_price - price
        profit = diff * contract_size * pos.volume
        return profit

    def close_all_positions(self, price: Optional[float] = None) -> Dict[str, OrderResult]:
        results = {}
        # close every position using snapshot of keys to avoid mutation during iteration
        keys = list(self.positions.keys())
        for pid in keys:
            res = self.close_position(pid, price=price)
            results[pid] = res
        return results
