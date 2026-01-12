"""
可观察配置基类 - 支持配置热重载
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable
import asyncio


class ConfigObserver(ABC):
    """配置观察者接口"""

    @abstractmethod
    async def on_config_changed(
        self, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ):
        """配置变更时的回调"""
        pass


class ObservableConfig:
    """可观察配置基类"""

    def __init__(self):
        self._observers: List[ConfigObserver] = []
        self._config: Dict[str, Any] = {}
        self._config_lock = asyncio.Lock()

    async def register_observer(self, observer: ConfigObserver):
        """注册观察者"""
        async with self._config_lock:
            if observer not in self._observers:
                self._observers.append(observer)

    async def unregister_observer(self, observer: ConfigObserver):
        """注销观察者"""
        async with self._config_lock:
            if observer in self._observers:
                self._observers.remove(observer)

    async def notify_observers(
        self, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ):
        """通知所有观察者"""
        async with self._config_lock:
            for observer in self._observers:
                try:
                    await observer.on_config_changed(old_config, new_config)
                except Exception as e:
                    import logging

                    logging.error(f"通知观察者失败: {e}")

    async def update_config(self, new_config: Dict[str, Any]):
        """更新配置并通知观察者"""
        old_config = self._config.copy()
        self._config.update(new_config)
        await self.notify_observers(old_config, self._config)

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self._config.copy()

    async def reload_config(self) -> Dict[str, Any]:
        """重新加载配置"""
        from ..config import load_config

        old_config = self._config.copy()
        new_config = load_config()
        self._config = new_config
        await self.notify_observers(old_config, self._config)
        return new_config
