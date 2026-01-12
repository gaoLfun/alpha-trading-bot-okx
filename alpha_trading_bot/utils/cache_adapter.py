"""
缓存适配器 - 统一缓存接口
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import asyncio
import logging
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class CacheAdapter(ABC):
    """缓存适配器基类"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 900) -> None:
        """设置缓存值"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """清空缓存"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        pass


class MemoryCacheAdapter(CacheAdapter):
    """内存缓存适配器"""

    def __init__(self):
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._lock = asyncio.Lock()
        self._default_ttl = 900  # 15分钟

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self._cache:
                return None

            value, expiry = self._cache[key]
            if datetime.now() > expiry:
                del self._cache[key]
                return None

            return value

    async def set(self, key: str, value: Any, ttl: int = 900) -> None:
        expiry = datetime.now() + timedelta(seconds=ttl)
        async with self._lock:
            self._cache[key] = (value, expiry)
        logger.debug(f"缓存已设置: {key}, TTL: {ttl}秒")

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
            logger.info("缓存已清空")

    async def exists(self, key: str) -> bool:
        async with self._lock:
            if key not in self._cache:
                return False

            value, expiry = self._cache[key]
            if datetime.now() > expiry:
                del self._cache[key]
                return False

            return True

    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        return {
            "size": len(self._cache),
            "valid_count": sum(
                1 for v, e in self._cache.values() if datetime.now() <= e
            ),
        }


class DynamicCacheAdapter(CacheAdapter):
    """动态缓存适配器"""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._access_count: Dict[str, int] = {}
        self._hit_count: Dict[str, int] = {}

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            self._access_count[key] = self._access_count.get(key, 0) + 1

            if key not in self._cache:
                return None

            self._hit_count[key] = self._hit_count.get(key, 0) + 1
            return self._cache[key]

    async def set(self, key: str, value: Any, ttl: int = 900) -> None:
        async with self._lock:
            self._cache[key] = value
        logger.debug(f"动态缓存已设置: {key}")

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_count.pop(key, None)
                self._hit_count.pop(key, None)
                return True
            return False

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
            self._access_count.clear()
            self._hit_count.clear()
            logger.info("动态缓存已清空")

    async def exists(self, key: str) -> bool:
        async with self._lock:
            return key in self._cache

    async def get_dynamic_ttl(self, key: str) -> int:
        """根据访问动态计算TTL"""
        async with self._lock:
            hits = self._hit_count.get(key, 0)
            accesses = self._access_count.get(key, 0)

            if accesses == 0:
                return 900

            hit_rate = hits / accesses

            if hit_rate > 0.8:
                return 1800  # 高命中率，延长TTL
            elif hit_rate > 0.5:
                return 900  # 中等命中率
            else:
                return 300  # 低命中率，缩短TTL

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "size": len(self._cache),
            "total_accesses": sum(self._access_count.values()),
            "total_hits": sum(self._hit_count.values()),
        }


class CacheManager:
    """统一缓存管理器"""

    def __init__(self, adapter: Optional[CacheAdapter] = None):
        self._adapter: CacheAdapter = adapter or MemoryCacheAdapter()

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        return await self._adapter.get(key)

    async def set(self, key: str, value: Any, ttl: int = 900) -> None:
        """设置缓存值"""
        await self._adapter.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        return await self._adapter.delete(key)

    async def clear(self) -> None:
        """清空缓存"""
        await self._adapter.clear()

    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        return await self._adapter.exists(key)

    def set_adapter(self, adapter: CacheAdapter) -> None:
        """设置缓存适配器"""
        self._adapter = adapter
        logger.info(f"缓存适配器已切换: {adapter.__class__.__name__}")

    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        if hasattr(self._adapter, "get_stats"):
            return self._adapter.get_stats()
        return {"adapter_type": self._adapter.__class__.__name__}
