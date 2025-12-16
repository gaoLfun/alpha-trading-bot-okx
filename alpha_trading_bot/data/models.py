"""
数据模型定义
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum


class SignalType(Enum):
    """信号类型"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradeStatus(Enum):
    """交易状态"""
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class AISignalRecord:
    """AI信号记录"""
    id: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    provider: str = ""
    signal: str = ""
    confidence: float = 0.0
    reason: str = ""
    market_price: float = 0.0
    market_data: Dict[str, Any] = field(default_factory=dict)
    used_in_trade: bool = False
    trade_result: Optional[str] = None
    pnl: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'provider': self.provider,
            'signal': self.signal,
            'confidence': self.confidence,
            'reason': self.reason,
            'market_price': self.market_price,
            'market_data': self.market_data,
            'used_in_trade': self.used_in_trade,
            'trade_result': self.trade_result,
            'pnl': self.pnl
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AISignalRecord':
        """从字典创建实例"""
        return cls(
            id=data.get('id'),
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
            provider=data.get('provider', ''),
            signal=data.get('signal', ''),
            confidence=data.get('confidence', 0.0),
            reason=data.get('reason', ''),
            market_price=data.get('market_price', 0.0),
            market_data=data.get('market_data', {}),
            used_in_trade=data.get('used_in_trade', False),
            trade_result=data.get('trade_result'),
            pnl=data.get('pnl')
        )


@dataclass
class TradeRecord:
    """交易记录"""
    id: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = ""
    side: str = ""  # buy/sell
    price: float = 0.0
    amount: float = 0.0
    cost: float = 0.0
    fee: float = 0.0
    status: TradeStatus = TradeStatus.PENDING
    order_id: str = ""
    signal_source: str = ""  # ai, strategy, manual
    signal_confidence: float = 0.0
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    holding_period: Optional[int] = None  # 持仓时间（秒）
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'side': self.side,
            'price': self.price,
            'amount': self.amount,
            'cost': self.cost,
            'fee': self.fee,
            'status': self.status.value,
            'order_id': self.order_id,
            'signal_source': self.signal_source,
            'signal_confidence': self.signal_confidence,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'holding_period': self.holding_period,
            'notes': self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeRecord':
        """从字典创建实例"""
        return cls(
            id=data.get('id'),
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
            symbol=data.get('symbol', ''),
            side=data.get('side', ''),
            price=data.get('price', 0.0),
            amount=data.get('amount', 0.0),
            cost=data.get('cost', 0.0),
            fee=data.get('fee', 0.0),
            status=TradeStatus(data.get('status', 'pending')),
            order_id=data.get('order_id', ''),
            signal_source=data.get('signal_source', ''),
            signal_confidence=data.get('signal_confidence', 0.0),
            pnl=data.get('pnl'),
            pnl_percent=data.get('pnl_percent'),
            exit_price=data.get('exit_price'),
            exit_time=datetime.fromisoformat(data['exit_time']) if data.get('exit_time') else None,
            holding_period=data.get('holding_period'),
            notes=data.get('notes', '')
        )


@dataclass
class MarketDataRecord:
    """市场数据记录"""
    id: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = ""
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: float = 0.0
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    close: float = 0.0
    change_percent: float = 0.0
    market_state: str = ""
    technical_indicators: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'price': self.price,
            'bid': self.bid,
            'ask': self.ask,
            'volume': self.volume,
            'high': self.high,
            'low': self.low,
            'open': self.open,
            'close': self.close,
            'change_percent': self.change_percent,
            'market_state': self.market_state,
            'technical_indicators': self.technical_indicators
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketDataRecord':
        """从字典创建实例"""
        return cls(
            id=data.get('id'),
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
            symbol=data.get('symbol', ''),
            price=data.get('price', 0.0),
            bid=data.get('bid', 0.0),
            ask=data.get('ask', 0.0),
            volume=data.get('volume', 0.0),
            high=data.get('high', 0.0),
            low=data.get('low', 0.0),
            open=data.get('open', 0.0),
            close=data.get('close', 0.0),
            change_percent=data.get('change_percent', 0.0),
            market_state=data.get('market_state', ''),
            technical_indicators=data.get('technical_indicators', {})
        )


@dataclass
class EquityRecord:
    """资产记录"""
    id: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    total_equity: float = 0.0
    available_balance: float = 0.0
    used_margin: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    position_count: int = 0
    open_position_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'total_equity': self.total_equity,
            'available_balance': self.available_balance,
            'used_margin': self.used_margin,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'total_pnl_percent': self.total_pnl_percent,
            'position_count': self.position_count,
            'open_position_value': self.open_position_value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EquityRecord':
        """从字典创建实例"""
        return cls(
            id=data.get('id'),
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
            total_equity=data.get('total_equity', 0.0),
            available_balance=data.get('available_balance', 0.0),
            used_margin=data.get('used_margin', 0.0),
            unrealized_pnl=data.get('unrealized_pnl', 0.0),
            realized_pnl=data.get('realized_pnl', 0.0),
            total_pnl=data.get('total_pnl', 0.0),
            total_pnl_percent=data.get('total_pnl_percent', 0.0),
            position_count=data.get('position_count', 0),
            open_position_value=data.get('open_position_value', 0.0)
        )