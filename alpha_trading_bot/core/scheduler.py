"""
任务调度器 - 洪责交易任务的调度和时间管理
"""

import asyncio
import random
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from .base import BaseComponent


@dataclass
class SchedulerConfig:
    """调度器配置"""

    cycle_minutes: int = 15
    random_offset_enabled: bool = True
    random_offset_range: int = 180  # 秒，默认±3分钟


class TradeScheduler(BaseComponent):
    """交易任务调度器
    负责：
    - 周期性任务调度
    - 随机时间偏移计算
    - 执行时间管理
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        if config is None:
            config = SchedulerConfig()
        super().__init__(config)
        self._running = False
        self._last_execution_time: Optional[datetime] = None
        self._next_execution_time: Optional[datetime] = None
        self._random_offset = 0

    async def initialize(self) -> bool:
        """初始化调度器"""
        self.logger.info("初始化交易任务调度器...")
        self._initialized = True
        self.logger.info("调度器初始化成功")
        return True

    async def cleanup(self) -> None:
        """清理资源"""
        self._running = False
        self.logger.info("调度器已清理")

    def calculate_next_execution_time(self) -> datetime:
        """计算下次执行时间"""
        base_time = datetime.now()

        # 应用随机偏移
        if self.config.random_offset_enabled:
            offset_seconds = random.randint(
                -self.config.random_offset_range,
                self.config.random_offset_range
            )
            self._random_offset = offset_seconds
            base_time = base_time + timedelta(seconds=offset_seconds)
            self.logger.info(
                f"应用随机偏移: {offset_seconds}秒，下次执行时间: {base_time}"
            )
        else:
            self._random_offset = 0

        # 计算下一个周期时间
        next_time = base_time + timedelta(minutes=self.config.cycle_minutes)
        self._next_execution_time = next_time
        return next_time

    async def wait_for_next_cycle(self) -> None:
        """等待下一个交易周期"""
        if self._last_execution_time is None:
            self._last_execution_time = datetime.now()

        next_time = self.calculate_next_execution_time()
        now = datetime.now()

        if next_time > now:
            wait_seconds = (next_time - now).total_seconds()
            self.logger.info(
                f"等待下一个周期: {wait_seconds:.1f}秒，下次执行时间: {next_time}"
            )
            await asyncio.sleep(wait_seconds)
        else:
            self.logger.warning("下次执行时间已过，立即执行")
            await asyncio.sleep(1)

        self._last_execution_time = datetime.now()

    def get_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        return {
            "name": self.config.name,
            "enabled": self.config.enabled,
            "initialized": self._initialized,
            "running": self._running,
            "next_execution_time": self._next_execution_time.isoformat()
            if self._next_execution_time else None,
            "random_offset": self._random_offset,
            "last_execution_time": self._last_execution_time.isoformat()
            if self._last_execution_time else None,
        }
