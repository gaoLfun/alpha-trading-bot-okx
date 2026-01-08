"""
AI缓存管理器 - 专门负责信号的缓存和智能失效管理
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .dynamic_cache import DynamicCacheManager, cache_manager
from .cache_monitor import cache_monitor

logger = logging.getLogger(__name__)


class AICacheManager:
    """AI缓存管理器 - 专门负责缓存逻辑"""

    def __init__(self, config=None):
        self.cache: Dict[str, Any] = {}
        self.dynamic_cache = cache_manager
        self.enable_dynamic_cache = True
        self.cache_duration = 900  # 默认15分钟

        if config:
            self.enable_dynamic_cache = getattr(config, "enable_dynamic_cache", True)
            self.cache_duration = getattr(config, "cache_duration", 900)
            # 同步动态缓存配置
            self.dynamic_cache.config.base_duration = self.cache_duration

    def generate_cache_key(self, market_data: Dict[str, Any]) -> str:
        """生成缓存键"""
        if self.enable_dynamic_cache:
            return self.dynamic_cache.generate_cache_key_v2(market_data)
        else:
            return self._generate_simple_cache_key(market_data)

    def _generate_simple_cache_key(self, market_data: Dict[str, Any]) -> str:
        """生成简单缓存键（传统方式）"""
        price = market_data.get("price", 0)
        rsi = market_data.get("technical_data", {}).get("rsi", 50)
        macd = market_data.get("technical_data", {}).get("macd_histogram", 0)

        # 价格和主要技术指标的组合
        price_bucket = int(price / 100) * 100  # 100美元价格区间
        rsi_bucket = int(rsi / 5) * 5  # 5点RSI区间
        macd_sign = "pos" if macd > 0 else "neg" if macd < 0 else "zero"

        return f"ai_signal_{price_bucket}_{rsi_bucket}_{macd_sign}"

    def check_cache(
        self, market_data: Dict[str, Any]
    ) -> tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
        """
        检查缓存

        Returns:
            tuple: (cached_signals, cache_metadata)
        """
        try:
            cache_key = self.generate_cache_key(market_data)

            if cache_key not in self.cache:
                self.dynamic_cache.record_cache_miss()
                cache_monitor.record_miss(cache_key)
                return None, None

            cached_result = self.cache[cache_key]

            # 确定缓存持续时间
            if self.enable_dynamic_cache:
                atr_percentage = market_data.get("atr_percentage", 0)
                dynamic_duration = self.dynamic_cache.get_dynamic_cache_duration(
                    atr_percentage
                )
                logger.info(
                    f"🔄 使用动态缓存系统 - ATR: {atr_percentage:.2f}%, 缓存时间: {dynamic_duration}秒"
                )
            else:
                dynamic_duration = self.cache_duration

            # 检查缓存是否过期
            cache_age = (datetime.now() - cached_result["timestamp"]).total_seconds()
            if cache_age >= dynamic_duration:
                # 缓存过期，删除
                del self.cache[cache_key]
                self.dynamic_cache.record_cache_miss()
                cache_monitor.record_miss(cache_key)
                logger.info(f"缓存已过期，删除缓存键: {cache_key}")
                return None, None

            # 智能失效检测
            if self.enable_dynamic_cache:
                should_invalidate, reason = self.dynamic_cache.should_invalidate_cache(
                    market_data, cached_result.get("market_snapshot", {})
                )
                if should_invalidate:
                    logger.info(f"🔄 智能缓存失效: {reason}")
                    del self.cache[cache_key]
                    self.dynamic_cache.record_cache_eviction()
                    cache_monitor.record_eviction(cache_key, reason)
                    return None, None

            # 缓存有效
            self.dynamic_cache.record_cache_hit()
            cache_monitor.record_hit(cache_key, cache_age)

            # 返回缓存的信号和元数据
            signals = cached_result.get("signals", [])
            for signal in signals:
                signal["_from_cache"] = True  # 标记为缓存信号

            metadata = {
                "success_count": cached_result.get("success_count", 0),
                "fail_count": cached_result.get("fail_count", 0),
                "success_providers": cached_result.get("success_providers", []),
                "cache_age": cache_age,
                "from_cache": True,
            }

            return signals, metadata

        except Exception as e:
            logger.error(f"检查缓存失败: {e}")
            return None, None

    def save_to_cache(
        self,
        signals: List[Dict[str, Any]],
        market_data: Dict[str, Any],
        success_count: int = 0,
        fail_count: int = 0,
        success_providers: List[str] = None,
    ) -> None:
        """保存信号到缓存"""
        try:
            cache_key = self.generate_cache_key(market_data)

            cache_data = {
                "signals": signals,
                "success_count": success_count,
                "fail_count": fail_count,
                "success_providers": success_providers or [],
                "timestamp": datetime.now(),
            }

            # 如果使用动态缓存，保存市场快照用于智能失效检测
            if self.enable_dynamic_cache:
                cache_data["market_snapshot"] = {
                    "price": market_data.get("price", 0),
                    "volume": market_data.get("volume", 0),
                    "atr": market_data.get("atr", 0),
                    "atr_percentage": market_data.get("atr_percentage", 0),
                    "technical_data": market_data.get("technical_data", {}),
                }

            self.cache[cache_key] = cache_data
            logger.info(f"信号已保存到缓存: {cache_key}")

        except Exception as e:
            logger.error(f"保存到缓存失败: {e}")

    def log_cache_stats(self, metadata: Dict[str, Any]) -> None:
        """记录缓存统计信息"""
        if not metadata or not metadata.get("from_cache"):
            return

        success_count = metadata.get("success_count", 0)
        fail_count = metadata.get("fail_count", 0)
        success_providers = metadata.get("success_providers", [])
        cache_age = metadata.get("cache_age", 0)

        total = success_count + fail_count
        logger.info(
            f"📊 多AI信号获取统计: 成功={success_count}, 失败={fail_count}, 总计={total}"
        )
        logger.info(
            f"✅ 成功提供商: {success_providers if success_providers else '无'}"
        )
        logger.info(f"🕐 缓存年龄: {cache_age:.1f}秒")

    def clear_cache(self) -> None:
        """清空缓存"""
        cache_count = len(self.cache)
        self.cache.clear()
        logger.info(f"已清空缓存，删除了 {cache_count} 个缓存项")

    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        total_count = len(self.cache)
        expired_count = sum(
            1
            for item in self.cache.values()
            if (datetime.now() - item["timestamp"]).total_seconds()
            > self.cache_duration
        )

        return {
            "total_cache_items": total_count,
            "expired_items": expired_count,
            "valid_items": total_count - expired_count,
            "dynamic_cache_enabled": self.enable_dynamic_cache,
            "cache_duration": self.cache_duration,
        }
