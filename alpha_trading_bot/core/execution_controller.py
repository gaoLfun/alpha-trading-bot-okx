"""
交易执行控制器 - 洪责交易执行的协调
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from .base import BaseComponent, BaseConfig


@dataclass
class ExecutionControllerConfig:
    """执行控制器配置"""

    enable_trading: bool = True
    max_concurrent_trades: int = 3
    trade_timeout: int = 30


class ExecutionController(BaseComponent):
    """交易执行控制器
    负责：
    - 交易执行协调
    - 订单管理
    - 仓位更新
    """

    def __init__(self, config: Optional[ExecutionControllerConfig] = None):
        if config is None:
            config = ExecutionControllerConfig(name="ExecutionController")
        super().__init__(config)

        self._trading_engine = None
        self._active_orders = {}

    async def initialize(self, trading_engine) -> bool:
        """初始化执行控制器"""
        self.logger.info("初始化交易执行控制器...")
        self._trading_engine = trading_engine
        self._initialized = True
        self.logger.info("交易执行控制器初始化成功")
        return True

    async def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("清理交易执行控制器资源")
        self._active_orders.clear()

    async def execute_trade(self, trade_request: Dict[str, Any]) -> Dict[str, Any]:
        """执行交易"""
        if not self._trading_engine:
            raise RuntimeError("交易引擎未初始化")

        self.logger.info(f"执行交易: {trade_request}")

        try:
            result = await self._trading_engine.execute_trade(trade_request)
            if result.success:
                self._active_orders[result.order_id] = {
                    "request": trade_request,
                    "timestamp": datetime.now()
                }
                self.logger.info(f"交易执行成功: {result.order_id}")
            else:
                self.logger.error(f"交易执行失败: {result.error_message}")

            return result

        except Exception as e:
            self.logger.error(f"交易执行异常: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        self.logger.info(f"取消订单: {order_id}")

        if not self._trading_engine:
            raise RuntimeError("交易引擎未初始化")

        result = await self._trading_engine.cancel_order(order_id)

        if result.get("success", False):
            self._active_orders.pop(order_id, None)
            self.logger.info(f"订单已取消: {order_id}")

        return result.get("success", False)

    async def get_active_orders(self) -> Dict[str, Dict[str, Any]]:
        """获取活动订单"""
        return self._active_orders

    def get_status(self) -> Dict[str, Any]:
        """获取执行控制器状态"""
        return {
            "name": self.config.name,
            "enabled": self.config.enabled,
            "initialized": self._initialized,
            "active_orders": len(self._active_orders),
        }
