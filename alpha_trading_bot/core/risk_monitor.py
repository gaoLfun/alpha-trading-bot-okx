"""
风险监控器 - 洪责风险控制和止盈止损管理
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from .base import BaseComponent, BaseConfig


@dataclass
class RiskMonitorConfig:
    """风险监控器配置"""

    max_daily_loss: float = 100.0
    max_position_risk: float = 0.05
    stop_loss_enabled: bool = True
    take_profit_enabled: bool = True
    trailing_stop_enabled: bool = True
    trailing_distance: float = 0.015


class RiskMonitor(BaseComponent):
    """风险监控器
    负责：
    - 风险评估
    - 止盈止损管理
    - 仓位监控
    """

    def __init__(self, config: Optional[RiskMonitorConfig] = None):
        if config is None:
            config = RiskMonitorConfig(name="RiskMonitor")
        super().__init__(config)

        # 风险状态
        self._daily_loss = 0.0
        self._consecutive_losses = 0
        self._managed_positions = set()

        # 止盈止损状态
        self._tp_orders = {}
        self._sl_orders = {}

    async def initialize(self) -> bool:
        """初始化风险监控器"""
        self.logger.info("初始化风险监控器...")
        self._initialized = True
        self.logger.info("风险监控器初始化成功")
        return True

    async def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("风险监控器已清理")

    def assess_risk(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估交易风险"""
        risk_score = 0.0
        risk_factors = []

        # 检查日亏损
        if self._daily_loss >= self.config.max_daily_loss:
            risk_score += 0.3
            risk_factors.append("max_daily_loss")

        # 检查连续亏损
        if self._consecutive_losses >= 3:
            risk_score += 0.4
            risk_factors.append("consecutive_losses")

        # 检查价格位置
        price_position = market_data.get("price_position", 0.5)
        if price_position > 0.85:
            risk_score += 0.3
            risk_factors.append("high_price_position")
        elif price_position < 0.15:
            risk_score -= 0.1
            risk_factors.append("low_price_position")

        allowed = risk_score < 0.7

        return {
            "allowed": allowed,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
        }

    def update_daily_loss(self, loss: float) -> None:
        """更新日亏损"""
        self._daily_loss += loss
        self.logger.info(f"更新日亏损: {loss:.2f}，累计: {self._daily_loss:.2f}")

    def update_consecutive_losses(self, is_win: bool) -> None:
        """更新连续亏损次数"""
        if is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
        self.logger.info(f"更新连续亏损: {self._consecutive_losses}")

    def add_managed_position(self, position_id: str) -> None:
        """添加已管理仓位"""
        self._managed_positions.add(position_id)

    def remove_managed_position(self, position_id: str) -> None:
        """移除已管理仓位"""
        self._managed_positions.discard(position_id)

    def is_position_managed(self, position_id: str) -> bool:
        """检查仓位是否已管理"""
        return position_id in self._managed_positions

    def get_status(self) -> Dict[str, Any]:
        """获取风险监控器状态"""
        return {
            "name": self.config.name,
            "enabled": self.config.enabled,
            "initialized": self._initialized,
            "daily_loss": self._daily_loss,
            "consecutive_losses": self._consecutive_losses,
            "managed_positions": len(self._managed_positions),
            "tp_orders": len(self._tp_orders),
            "sl_orders": len(self._sl_orders),
        }
