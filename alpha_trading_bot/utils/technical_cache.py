"""
技术指标缓存机制 - 优化技术指标计算的性能
"""

import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

from .cache import CacheManager
from .technical import TechnicalIndicators

logger = logging.getLogger(__name__)


class TechnicalIndicatorsCache:
    """技术指标缓存管理器"""

    def __init__(self, cache_ttl: int = 300):  # 默认5分钟缓存
        """
        初始化技术指标缓存

        Args:
            cache_ttl: 缓存生存时间（秒）
        """
        self.cache_manager = CacheManager()
        self.cache_ttl = cache_ttl
        self.cache_hits = 0
        self.cache_misses = 0

    def _generate_cache_key(
        self,
        high: List[float],
        low: List[float],
        close: List[float],
        indicator_type: str,
        **kwargs,
    ) -> str:
        """
        生成缓存键

        Args:
            high: 最高价列表
            low: 最低价列表
            close: 收盘价列表
            indicator_type: 指标类型
            **kwargs: 其他参数

        Returns:
            str: 缓存键
        """
        try:
            # 使用数据的最后几个值和长度来生成唯一键
            data_str = f"{indicator_type}_"
            data_str += f"len_{len(close)}_"
            data_str += f"last_h_{high[-1] if high else 0}_"
            data_str += f"last_l_{low[-1] if low else 0}_"
            data_str += f"last_c_{close[-1] if close else 0}_"

            # 添加其他参数
            for key, value in sorted(kwargs.items()):
                data_str += f"_{key}_{value}"

            # 生成MD5哈希
            return hashlib.md5(data_str.encode()).hexdigest()

        except Exception as e:
            logger.error(f"生成缓存键失败: {e}")
            # 生成简单键作为备用
            return (
                f"{indicator_type}_{len(close)}_{datetime.now().strftime('%Y%m%d%H%M')}"
            )

    def get_cached_atr(
        self, high: List[float], low: List[float], close: List[float], period: int = 14
    ) -> Optional[List[float]]:
        """
        获取缓存的ATR值

        Args:
            high: 最高价列表
            low: 最低价列表
            close: 收盘价列表
            period: 计算周期

        Returns:
            List[float] or None: 缓存的ATR值
        """
        cache_key = self._generate_cache_key(high, low, close, "atr", period=period)

        cached_value = self.cache_manager.get(cache_key)
        if cached_value is not None:
            self.cache_hits += 1
            logger.debug(f"ATR缓存命中: period={period}, data_len={len(close)}")
            return cached_value

        self.cache_misses += 1
        return None

    def cache_atr(
        self,
        high: List[float],
        low: List[float],
        close: List[float],
        period: int = 14,
        atr_values: List[float] = None,
    ) -> None:
        """
        缓存ATR值

        Args:
            high: 最高价列表
            low: 最低价列表
            close: 收盘价列表
            period: 计算周期
            atr_values: ATR值列表
        """
        if atr_values is None:
            return

        cache_key = self._generate_cache_key(high, low, close, "atr", period=period)
        self.cache_manager.set(cache_key, atr_values, self.cache_ttl)
        logger.debug(f"ATR已缓存: period={period}, data_len={len(close)}")

    def get_cached_rsi(
        self, close: List[float], period: int = 14
    ) -> Optional[List[float]]:
        """获取缓存的RSI值"""
        cache_key = self._generate_cache_key([], [], close, "rsi", period=period)

        cached_value = self.cache_manager.get(cache_key)
        if cached_value is not None:
            self.cache_hits += 1
            logger.debug(f"RSI缓存命中: period={period}, data_len={len(close)}")
            return cached_value

        self.cache_misses += 1
        return None

    def cache_rsi(
        self, close: List[float], period: int = 14, rsi_values: List[float] = None
    ) -> None:
        """缓存RSI值"""
        if rsi_values is None:
            return

        cache_key = self._generate_cache_key([], [], close, "rsi", period=period)
        self.cache_manager.set(cache_key, rsi_values, self.cache_ttl)
        logger.debug(f"RSI已缓存: period={period}, data_len={len(close)}")

    def get_cached_macd(
        self, close: List[float], fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Optional[Tuple[List[float], List[float], List[float]]]:
        """获取缓存的MACD值"""
        cache_key = self._generate_cache_key(
            [], [], close, "macd", fast=fast, slow=slow, signal=signal
        )

        cached_value = self.cache_manager.get(cache_key)
        if cached_value is not None:
            self.cache_hits += 1
            logger.debug(
                f"MACD缓存命中: fast={fast}, slow={slow}, signal={signal}, data_len={len(close)}"
            )
            return cached_value

        self.cache_misses += 1
        return None

    def cache_macd(
        self,
        close: List[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        macd_result: Tuple[List[float], List[float], List[float]] = None,
    ) -> None:
        """缓存MACD值"""
        if macd_result is None:
            return

        cache_key = self._generate_cache_key(
            [], [], close, "macd", fast=fast, slow=slow, signal=signal
        )
        self.cache_manager.set(cache_key, macd_result, self.cache_ttl)
        logger.debug(
            f"MACD已缓存: fast={fast}, slow={slow}, signal={signal}, data_len={len(close)}"
        )

    def get_cached_bollinger_bands(
        self, close: List[float], period: int = 20, std_dev: float = 2.0
    ) -> Optional[Dict[str, List[float]]]:
        """获取缓存的布林带值"""
        cache_key = self._generate_cache_key(
            [], [], close, "bb", period=period, std_dev=std_dev
        )

        cached_value = self.cache_manager.get(cache_key)
        if cached_value is not None:
            self.cache_hits += 1
            logger.debug(
                f"布林带缓存命中: period={period}, std_dev={std_dev}, data_len={len(close)}"
            )
            return cached_value

        self.cache_misses += 1
        return None

    def cache_bollinger_bands(
        self,
        close: List[float],
        period: int = 20,
        std_dev: float = 2.0,
        bb_result: Dict[str, List[float]] = None,
    ) -> None:
        """缓存布林带值"""
        if bb_result is None:
            return

        cache_key = self._generate_cache_key(
            [], [], close, "bb", period=period, std_dev=std_dev
        )
        self.cache_manager.set(cache_key, bb_result, self.cache_ttl)
        logger.debug(
            f"布林带已缓存: period={period}, std_dev={std_dev}, data_len={len(close)}"
        )

    def get_cached_adx(
        self, high: List[float], low: List[float], close: List[float], period: int = 14
    ) -> Optional[List[float]]:
        """获取缓存的ADX值"""
        cache_key = self._generate_cache_key(high, low, close, "adx", period=period)

        cached_value = self.cache_manager.get(cache_key)
        if cached_value is not None:
            self.cache_hits += 1
            logger.debug(f"ADX缓存命中: period={period}, data_len={len(close)}")
            return cached_value

        self.cache_misses += 1
        return None

    def cache_adx(
        self,
        high: List[float],
        low: List[float],
        close: List[float],
        period: int = 14,
        adx_values: List[float] = None,
    ) -> None:
        """缓存ADX值"""
        if adx_values is None:
            return

        cache_key = self._generate_cache_key(high, low, close, "adx", period=period)
        self.cache_manager.set(cache_key, adx_values, self.cache_ttl)
        logger.debug(f"ADX已缓存: period={period}, data_len={len(close)}")

    def get_cached_all_indicators(
        self, high: List[float], low: List[float], close: List[float]
    ) -> Optional[Dict[str, Any]]:
        """
        获取所有指标的缓存值

        Args:
            high: 最高价列表
            low: 最低价列表
            close: 收盘价列表

        Returns:
            Dict or None: 缓存的所有指标
        """
        cache_key = self._generate_cache_key(high, low, close, "all_indicators")

        cached_value = self.cache_manager.get(cache_key)
        if cached_value is not None:
            self.cache_hits += 1
            logger.debug(f"所有指标缓存命中: data_len={len(close)}")
            return cached_value

        self.cache_misses += 1
        return None

    def cache_all_indicators(
        self,
        high: List[float],
        low: List[float],
        close: List[float],
        indicators: Dict[str, Any],
    ) -> None:
        """
        缓存所有指标

        Args:
            high: 最高价列表
            low: 最低价列表
            close: 收盘价列表
            indicators: 计算出的所有指标
        """
        cache_key = self._generate_cache_key(high, low, close, "all_indicators")
        self.cache_manager.set(cache_key, indicators, self.cache_ttl)
        logger.debug(f"所有指标已缓存: data_len={len(close)}")

    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache_manager.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("技术指标缓存已清空")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total_requests,
            "hit_rate_percent": hit_rate,
            "cache_ttl": self.cache_ttl,
            "cache_items": len(self.cache_manager.cache),
        }

    def log_cache_stats(self) -> None:
        """记录缓存统计信息"""
        stats = self.get_cache_stats()
        logger.info("📊 技术指标缓存统计:")
        logger.info(f"  🎯 缓存命中: {stats['cache_hits']}")
        logger.info(f"  ❌ 缓存未命中: {stats['cache_misses']}")
        logger.info(f"  📈 命中率: {stats['hit_rate_percent']:.1f}%")
        logger.info(f"  🗄️  缓存项数: {stats['cache_items']}")
        logger.info(f"  ⏰ 缓存TTL: {stats['cache_ttl']}秒")


class CachedTechnicalIndicators(TechnicalIndicators):
    """带缓存的技术指标计算器"""

    def __init__(self, cache_ttl: int = 300):
        """
        初始化带缓存的技术指标计算器

        Args:
            cache_ttl: 缓存生存时间（秒）
        """
        super().__init__()
        self.cache = TechnicalIndicatorsCache(cache_ttl)

    def calculate_atr(
        self, high: List[float], low: List[float], close: List[float], period: int = 14
    ) -> List[float]:
        """计算ATR（带缓存）"""
        # 尝试从缓存获取
        cached_atr = self.cache.get_cached_atr(high, low, close, period)
        if cached_atr is not None:
            return cached_atr

        # 计算新的ATR
        atr_values = super().calculate_atr(high, low, close, period)

        # 缓存结果
        self.cache.cache_atr(high, low, close, period, atr_values)

        return atr_values

    def calculate_rsi(self, close: List[float], period: int = 14) -> List[float]:
        """计算RSI（带缓存）"""
        # 尝试从缓存获取
        cached_rsi = self.cache.get_cached_rsi(close, period)
        if cached_rsi is not None:
            return cached_rsi

        # 计算新的RSI
        rsi_values = super().calculate_rsi(close, period)

        # 缓存结果
        self.cache.cache_rsi(close, period, rsi_values)

        return rsi_values

    def calculate_macd(
        self, close: List[float], fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[List[float], List[float], List[float]]:
        """计算MACD（带缓存）"""
        # 尝试从缓存获取
        cached_macd = self.cache.get_cached_macd(close, fast, slow, signal)
        if cached_macd is not None:
            return cached_macd

        # 计算新的MACD
        macd_result = super().calculate_macd(close, fast, slow, signal)

        # 缓存结果
        self.cache.cache_macd(close, fast, slow, signal, macd_result)

        return macd_result

    def calculate_bollinger_bands(
        self, close: List[float], period: int = 20, std_dev: float = 2.0
    ) -> Dict[str, List[float]]:
        """计算布林带（带缓存）"""
        # 尝试从缓存获取
        cached_bb = self.cache.get_cached_bollinger_bands(close, period, std_dev)
        if cached_bb is not None:
            return cached_bb

        # 计算新的布林带
        bb_result = super().calculate_bollinger_bands(close, period, std_dev)

        # 缓存结果
        self.cache.cache_bollinger_bands(close, period, std_dev, bb_result)

        return bb_result

    def calculate_adx(
        self, high: List[float], low: List[float], close: List[float], period: int = 14
    ) -> List[float]:
        """计算ADX（带缓存）"""
        # 尝试从缓存获取
        cached_adx = self.cache.get_cached_adx(high, low, close, period)
        if cached_adx is not None:
            return cached_adx

        # 计算新的ADX
        adx_values = super().calculate_adx(high, low, close, period)

        # 缓存结果
        self.cache.cache_adx(high, low, close, period, adx_values)

        return adx_values

    def calculate_all_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算所有指标（带缓存）"""
        try:
            high = market_data.get("high_prices", [])
            low = market_data.get("low_prices", [])
            close = market_data.get("close_prices", [])

            # 尝试从缓存获取所有指标
            cached_indicators = self.cache.get_cached_all_indicators(high, low, close)
            if cached_indicators is not None:
                logger.info("使用缓存的技术指标")
                return cached_indicators

            # 计算所有指标
            indicators = super().calculate_all_indicators(market_data)

            # 缓存所有指标
            self.cache.cache_all_indicators(high, low, close, indicators)

            return indicators

        except Exception as e:
            logger.error(f"计算所有技术指标失败: {e}")
            return {}


# 全局缓存实例
technical_cache = TechnicalIndicatorsCache()
cached_technical_indicators = CachedTechnicalIndicators()
