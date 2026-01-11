"""
工具模块 - 提供各种工具函数和类
"""

from .logging import setup_logging, get_logger
from .cache import CacheManager, memory_cache, cache_manager
from .price_position import (
    PricePositionCalculator,
    PricePositionResult,
    calculate_price_position,
    calculate_price_position_from_market_data,
)
from .price_calculator import (
    PriceCalculator,
    PricePosition,
    StopLossCalculation,
)
from .profit_calculator import (
    ProfitCalculator,
    ProfitAnalysis,
    TradeResult,
    PositionSide,
)
from .config_accessor import (
    ConfigAccessor,
    config_accessor,
)
from .cache_strategy import (
    CacheKeyGenerator,
    CacheExpirationManager,
    CacheStrategy,
    cache_key_generator,
    cache_expiration_manager,
    cache_strategy,
)
from .technical_cache import (
    TechnicalIndicatorsCache,
    CachedTechnicalIndicators,
    technical_cache,
    cached_technical_indicators,
)

__all__ = [
    # 日志
    "setup_logging",
    "get_logger",
    # 缓存
    "CacheManager",
    "memory_cache",
    "cache_manager",
    # 价格位置计算
    "PricePositionCalculator",
    "PricePositionResult",
    "calculate_price_position",
    "calculate_price_position_from_market_data",
    # 统一价格计算器
    "PriceCalculator",
    "PricePosition",
    "StopLossCalculation",
    # 利润计算器
    "ProfitCalculator",
    "ProfitAnalysis",
    "TradeResult",
    "PositionSide",
    # 配置访问器
    "ConfigAccessor",
    "config_accessor",
    # 缓存策略
    "CacheKeyGenerator",
    "CacheExpirationManager",
    "CacheStrategy",
    "cache_key_generator",
    "cache_expiration_manager",
    "cache_strategy",
    # 技术指标缓存
    "TechnicalIndicatorsCache",
    "CachedTechnicalIndicators",
    "technical_cache",
    "cached_technical_indicators",
]
