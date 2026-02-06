"""
自适应交易模块

提供实时参数自适应和市场环境感知能力，
使交易系统能够根据当前市场状况自动调整交易参数。
"""

from .parameter_manager import AdaptiveParameterManager, AdaptiveConfig
from .market_regime import MarketRegimeDetector, MarketRegime
from .performance_tracker import PerformanceTracker
from .rules_engine import AdaptiveRulesEngine

__all__ = [
    "AdaptiveParameterManager",
    "AdaptiveConfig",
    "MarketRegimeDetector",
    "MarketRegime",
    "PerformanceTracker",
    "AdaptiveRulesEngine",
]
