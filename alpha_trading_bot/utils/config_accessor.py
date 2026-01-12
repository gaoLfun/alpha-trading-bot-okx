"""
利润计算统一模块
集中管理所有利润相关的计算逻辑
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class PositionSide(Enum):
    """持仓方向枚举"""

    LONG = "long"
    SHORT = "short"


@dataclass
class ProfitAnalysis:
    """利润分析结果"""

    profit_percentage: float  # 利润百分比
    profit_amount: float  # 利润金额
    is_profitable: bool  # 是否盈利
    unrealized_pnl: float  # 未实现盈亏
    realized_pnl: float  # 已实现盈亏
    total_pnl: float  # 总盈亏
    win_rate: float  # 胜率
    avg_win: float  # 平均盈利
    avg_loss: float  # 平均亏损
    profit_factor: float  # 盈亏比
    max_drawdown: float  # 最大回撤
    sharpe_ratio: float  # 夏普比率


@dataclass
class TradeResult:
    """交易结果"""

    entry_price: float
    exit_price: float
    position_size: float
    position_side: PositionSide
    profit_amount: float
    profit_percentage: float
    is_win: bool
    timestamp: float


class ProfitCalculator:
    """统一利润计算器"""

    @staticmethod
    def calculate_profit_percentage(
        entry_price: float,
        current_price: float,
        position_side: PositionSide = PositionSide.LONG,
    ) -> float:
        """
        统一的利润百分比计算
        避免在多个文件中重复实现相同的计算逻辑
        """
        if entry_price <= 0:
            return 0.0

        if position_side == PositionSide.LONG:
            return ((current_price - entry_price) / entry_price) * 100
        else:  # SHORT
            return ((entry_price - current_price) / entry_price) * 100

    @staticmethod
    def calculate_profit_amount(
        entry_price: float,
        current_price: float,
        position_size: float,
        position_side: PositionSide = PositionSide.LONG,
    ) -> float:
        """
        统一的利润金额计算
        """
        if position_side == PositionSide.LONG:
            return (current_price - entry_price) * position_size
        else:  # SHORT
            return (entry_price - current_price) * position_size

    @staticmethod
    def calculate_unrealized_pnl(
        entry_price: float,
        current_price: float,
        position_size: float,
        position_side: PositionSide = PositionSide.LONG,
    ) -> float:
        """
        计算未实现盈亏
        """
        return ProfitCalculator.calculate_profit_amount(
            entry_price, current_price, position_size, position_side
        )
