"""
事件总线 - 解耦组件通信
"""

import asyncio
import logging
from typing import Dict, Any, Callable, List, Optional
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json


logger = logging.getLogger(__name__)


@dataclass
class Event:
    """事件基类"""

    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source,
        }


class EventHandler(ABC):
    """事件处理器接口"""

    @abstractmethod
    async def handle(self, event: Event) -> None:
        """处理事件"""
        pass


class EventBus:
    """事件总线
    负责组件间的解耦通信
    """

    def __init__(self):
        self._subscribers: Dict[str, List[EventHandler]] = {}
        self._lock = asyncio.Lock()
        self._event_queue: asyncio.Queue[Event] = asyncio.Queue()
        self._running = False
        self._event_count = 0

    async def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """订阅事件类型"""
        async with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)
                logger.info(
                    f"订阅事件: {event_type}, 处理器: {handler.__class__.__name__}"
                )

    async def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """取消订阅"""
        async with self._lock:
            if event_type in self._subscribers:
                if handler in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(handler)
                    logger.info(
                        f"取消订阅: {event_type}, 处理器: {handler.__class__.__name__}"
                    )

    async def publish(self, event: Event) -> None:
        """发布事件"""
        await self._event_queue.put(event)
        logger.debug(
            f"事件已发布: {event.event_type}, 队列长度: {self._event_queue.qsize()}"
        )

    async def _process_event(self, event: Event) -> None:
        """处理单个事件"""
        handlers = self._subscribers.get(event.event_type, [])

        if not handlers:
            logger.warning(f"没有事件处理器: {event.event_type}")
            return

        logger.info(f"处理事件: {event.event_type}, 订阅者数量: {len(handlers)}")

        for handler in handlers:
            try:
                await handler.handle(event)
            except Exception as e:
                logger.error(
                    f"事件处理器异常: {e}, 处理器: {handler.__class__.__name__}"
                )

        self._event_count += 1

    async def start(self) -> None:
        """启动事件总线"""
        if self._running:
            logger.warning("事件总线已在运行")
            return

        self._running = True
        logger.info("事件总线已启动")

        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"事件处理异常: {e}")

    async def stop(self) -> None:
        """停止事件总线"""
        self._running = False
        logger.info(f"事件总线已停止, 共处理: {self._event_count} 个事件")

    def get_stats(self) -> Dict[str, Any]:
        """获取事件总线统计"""
        return {
            "running": self._running,
            "event_count": self._event_count,
            "subscriber_count": {
                event_type: len(handlers)
                for event_type, handlers in self._subscribers.items()
            },
            "queue_size": self._event_queue.qsize(),
        }


class EventBuilder:
    """事件构建器"""

    @staticmethod
    def build_event(
        event_type: str, data: Dict[str, Any], source: Optional[str] = None
    ) -> Event:
        """构建事件"""
        return Event(event_type=event_type, data=data, source=source)

    @staticmethod
    def signal_event(signal: str, confidence: float, provider: str) -> Event:
        """构建信号事件"""
        return Event(
            event_type="signal_generated",
            data={"signal": signal, "confidence": confidence, "provider": provider},
            source="ai_manager",
        )

    @staticmethod
    def trade_event(
        success: bool, order_id: Optional[str], error: Optional[str] = None
    ) -> Event:
        """构建交易事件"""
        return Event(
            event_type="trade_executed",
            data={"success": success, "order_id": order_id, "error": error},
            source="execution_controller",
        )

    @staticmethod
    def risk_event(risk_score: float, allowed: bool) -> Event:
        """构建风险事件"""
        return Event(
            event_type="risk_assessed",
            data={"risk_score": risk_score, "allowed": allowed},
            source="risk_monitor",
        )


class DefaultEventHandler(EventHandler):
    """默认事件处理器"""

    async def handle(self, event: Event) -> None:
        """处理事件 - 记录日志"""
        logger.info(f"事件处理: {event.event_type}")
        logger.debug(f"事件数据: {json.dumps(event.data, ensure_ascii=False)}")


event_bus = EventBus()
"""全局事件总线实例"""
