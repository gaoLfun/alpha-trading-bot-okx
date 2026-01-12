"""
核心基础模块 - 提供统一的数据结构和基础组件
"""

from .base import (
    BaseConfig,
    BaseComponent,
    SignalData,
    MarketData,
    TradingResult
)

from .bot import TradingBot, BotConfig
from .exceptions import TradingBotException, ConfigurationError, ExchangeError

__all__ = [
    # 基础数据结构
    'BaseConfig',
    'BaseComponent',
    'SignalData',
    'MarketData',
    'TradingResult',

    # 交易机器人
    'TradingBot',
    'BotConfig',

    # 异常类
    'TradingBotException',
    'ConfigurationError',
    'ExchangeError'
]
# 新增：重构后的组件
from .scheduler import TradeScheduler, SchedulerConfig
from .risk_monitor import RiskMonitor, RiskMonitorConfig
