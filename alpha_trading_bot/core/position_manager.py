"""
仓位管理器
处理开仓、平仓、止损止盈、仓位计算
"""

import logging
from typing import Optional
from dataclasses import dataclass

from ..config.models import Config

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """持仓信息"""

    symbol: str
    side: str
    amount: float
    entry_price: float
    unrealized_pnl: float = 0.0


class PositionManager:
    """仓位管理器"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.from_env()
        self._position: Optional[Position] = None
        self._entry_price: float = 0.0
        self._stop_order_id: Optional[str] = None

    @property
    def position(self) -> Optional[Position]:
        """获取当前持仓"""
        return self._position

    @property
    def entry_price(self) -> float:
        """获取入场价"""
        return self._entry_price

    @property
    def stop_order_id(self) -> Optional[str]:
        """获取止损单ID"""
        return self._stop_order_id

    def has_position(self) -> bool:
        """是否有持仓"""
        return self._position is not None and self._entry_price > 0

    def update_from_exchange(self, position_data: dict) -> None:
        """从交易所数据更新持仓信息"""
        if position_data:
            self._position = Position(
                symbol=position_data["symbol"],
                side=position_data["side"],
                amount=position_data["amount"],
                entry_price=position_data["entry_price"],
            )
            self._entry_price = position_data["entry_price"]
            logger.info(
                f"[仓位更新] 从交易所更新持仓: {self._position.symbol}, "
                f"方向:{self._position.side}, 数量:{self._position.amount}, 入场价:{self._position.entry_price}"
            )
        else:
            self._position = None
            logger.info("[仓位更新] 交易所返回无持仓信息")

    def calculate_stop_price(self, current_price: float) -> float:
        """
        计算止损价

        Args:
            current_price: 当前价格

        Returns:
            止损价
        """
        if self._position is None or self._entry_price == 0:
            return 0.0

        if current_price < self._entry_price:
            # 亏损时使用较大的止损比例
            stop_percent = self.config.stop_loss.stop_loss_percent
            stop_price = self._entry_price * (1 - stop_percent)
            logger.debug(
                f"[止损计算] 亏损状态: 当前价({current_price}) < 入场价({self._entry_price}), "
                f"止损比例:{stop_percent * 100}%, 止损价:{stop_price}"
            )
            return stop_price
        else:
            # 盈利时使用较小的止损比例（锁定利润）
            stop_percent = self.config.stop_loss.stop_loss_profit_percent
            stop_price = self._entry_price * (1 - stop_percent)
            logger.debug(
                f"[止损计算] 盈利状态: 当前价({current_price}) > 入场价({self._entry_price}), "
                f"止损比例:{stop_percent * 100}%, 止损价:{stop_price}"
            )
            return stop_price

    def log_stop_loss_info(self, current_price: float, new_stop: float) -> None:
        """记录止损信息"""
        if current_price < self._entry_price:
            pnl = (current_price - self._entry_price) / self._entry_price * 100
            logger.info(
                f"[止损监控] 亏损持仓: 当前价={current_price}, 入场价={self._entry_price}, "
                f"亏损={pnl:.2f}%, 止损价={new_stop}"
            )
        else:
            pnl = (current_price - self._entry_price) / self._entry_price * 100
            logger.info(
                f"[止损监控] 盈利持仓: 当前价={current_price}, 入场价={self._entry_price}, "
                f"盈利={pnl:.2f}%, 止损价={new_stop}"
            )

    def set_stop_order(self, stop_order_id: str) -> None:
        """设置止损单ID"""
        self._stop_order_id = stop_order_id
        logger.debug(f"[止损单] 设置止损单ID: {stop_order_id}")

    def clear_position(self) -> None:
        """清空持仓信息"""
        logger.info(f"[清仓] 清空持仓信息，原止损单: {self._stop_order_id}")
        self._position = None
        self._entry_price = 0.0
        self._stop_order_id = None

    def update_position(self, amount: float, entry_price: float, symbol: str) -> None:
        """更新持仓信息（开仓后调用）"""
        self._entry_price = entry_price
        self._position = Position(
            symbol=symbol,
            side="long",
            amount=amount,
            entry_price=entry_price,
        )
        logger.info(
            f"[持仓更新] 开仓成功: {symbol}, 方向:long, 数量:{amount}, 入场价:{entry_price}"
        )


def create_position_manager(config: Optional[Config] = None) -> PositionManager:
    """创建仓位管理器实例"""
    return PositionManager(config)
