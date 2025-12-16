"""
数据管理模块 - 负责数据持久化和历史记录管理
"""

from .manager import DataManager, create_data_manager, get_data_manager
from .models import (
    AISignalRecord,
    TradeRecord,
    MarketDataRecord,
    EquityRecord
)

__all__ = [
    'DataManager',
    'create_data_manager',
    'get_data_manager',
    'AISignalRecord',
    'TradeRecord',
    'MarketDataRecord',
    'EquityRecord'
]