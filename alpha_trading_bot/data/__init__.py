"""
Alpha Trading Bot 数据模块

包含：
- kline_persistence: K 线数据持久化管理
- manager: 数据管理器统一接口
"""

from .kline_persistence import (
    KLinePersistenceManager,
    KLineFileMetadata,
    OHLCVData,
    get_kline_manager,
)

from .manager import (
    DataManager,
    DataManagerConfig,
    create_data_manager,
    get_data_manager,
)

__all__ = [
    "KLinePersistenceManager",
    "KLineFileMetadata",
    "OHLCVData",
    "get_kline_manager",
    "DataManager",
    "DataManagerConfig",
    "create_data_manager",
    "get_data_manager",
]
