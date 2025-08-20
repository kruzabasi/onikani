# execution/adapter.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import time


@dataclass
class Order:
    symbol: str
    side: str  # 'buy' or 'sell'
    volume: float
    price: Optional[float] = None  # limit/price to fill at; if None, fill at market_price
    sl: Optional[float] = None
    tp: Optional[float] = None
    client_id: Optional[str] = None  # optional user-provided id


@dataclass
class OrderResult:
    success: bool
    position_id: Optional[str]
    filled_price: Optional[float]
    message: Optional[str] = None


@dataclass
class Position:
    position_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    volume: float
    open_price: float
    open_time: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    comment: Optional[str] = None


class ExecutionAdapter(ABC):
    @abstractmethod
    def send_order(self, order: Order) -> OrderResult:
        """Send an order. Returns OrderResult describing fill status / created position."""
        raise NotImplementedError()

    @abstractmethod
    def close_position(self, position_id: str, price: Optional[float] = None) -> OrderResult:
        """Close a position by id at specified price (or current market price)."""
        raise NotImplementedError()

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Return a list of open positions."""
        raise NotImplementedError()

    @abstractmethod
    def get_account(self) -> dict:
        """Return a dict with simple account info (balance/equity)."""
        raise NotImplementedError()
