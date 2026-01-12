"""
动态交易冷却管理器 - 基于市场条件和交易表现的智能冷却
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CoolingLevel(Enum):
    """冷却等级"""

    NONE = "none"  # 无冷却
    LIGHT = "light"  # 轻度冷却（5-15分钟）
    MEDIUM = "medium"  # 中等冷却（15-30分钟）
    HEAVY = "heavy"  # 重度冷却（30-60分钟）
    EXTREME = "extreme"  # 极度冷却（60分钟以上）


@dataclass
class CoolingState:
    """冷却状态"""

    level: CoolingLevel
    remaining_seconds: int
    reason: str
    last_trade_time: Optional[datetime]
    cooldown_until: Optional[datetime]


class DynamicTradeCoolingManager:
    """动态交易冷却管理器"""

    def __init__(self):
        # 冷却配置
        self.cooling_configs = {
            CoolingLevel.NONE: {
                "min_minutes": 0,
                "max_minutes": 0,
                "description": "无冷却",
            },
            CoolingLevel.LIGHT: {
                "min_minutes": 5,
                "max_minutes": 15,
                "description": "轻度冷却",
            },
            CoolingLevel.MEDIUM: {
                "min_minutes": 15,
                "max_minutes": 30,
                "description": "中等冷却",
            },
            CoolingLevel.HEAVY: {
                "min_minutes": 30,
                "max_minutes": 60,
                "description": "重度冷却",
            },
            CoolingLevel.EXTREME: {
                "min_minutes": 60,
                "max_minutes": 120,
                "description": "极度冷却",
            },
        }

        # 当前冷却状态
        self.current_cooling: Dict[str, CoolingState] = {
            "buy": CoolingState(CoolingLevel.NONE, 0, "初始化", None, None),
            "sell": CoolingState(CoolingLevel.NONE, 0, "初始化", None, None),
        }

        # 交易历史记录
        self.trade_history: List[Dict] = []

        # 配置参数
        self.max_trades_per_hour = 3  # 每小时最大交易次数
        self.max_trades_per_day = 12  # 每日最大交易次数

        # 市场条件权重
        self.volatility_weight = 0.4  # 波动率权重
        self.trend_weight = 0.3  # 趋势权重
        self.performance_weight = 0.3  # 表现权重

    def can_trade(
        self, trade_side: str, market_conditions: Dict[str, Any]
    ) -> Tuple[bool, str, int]:
        """
        检查是否可以交易

        Args:
            trade_side: 交易方向 ("buy" 或 "sell")
            market_conditions: 市场条件

        Returns:
            (can_trade, reason, remaining_seconds)
        """
        current_time = datetime.now()
        cooling_state = self.current_cooling[trade_side]

        # 检查是否还在冷却期
        if cooling_state.cooldown_until and current_time < cooling_state.cooldown_until:
            remaining = int(
                (cooling_state.cooldown_until - current_time).total_seconds()
            )
            return False, f"冷却中: {cooling_state.reason}", remaining

        # 检查交易频率限制
        frequency_check, frequency_reason, frequency_cooldown = (
            self._check_trade_frequency(trade_side, current_time)
        )
        if not frequency_check:
            self._apply_cooling(
                trade_side, frequency_cooldown, frequency_reason, current_time
            )
            return False, frequency_reason, frequency_cooldown * 60

        # 检查市场条件
        market_check, market_reason, market_cooldown = self._evaluate_market_conditions(
            trade_side, market_conditions
        )
        if not market_check:
            self._apply_cooling(
                trade_side, market_cooldown, market_reason, current_time
            )
            return False, market_reason, market_cooldown * 60

        return True, "可以交易", 0

    def record_trade(self, trade_side: str, trade_result: Dict[str, Any]):
        """
        记录交易结果并调整冷却策略

        Args:
            trade_side: 交易方向
            trade_result: 交易结果
        """
        current_time = datetime.now()

        # 记录交易历史
        self.trade_history.append(
            {
                "timestamp": current_time,
                "side": trade_side,
                "result": trade_result,
                "market_conditions": trade_result.get("market_conditions", {}),
            }
        )

        # 根据交易结果调整冷却
        self._adjust_cooling_based_on_result(trade_side, trade_result, current_time)

        # 清理过期历史记录（保留7天）
        self._cleanup_old_history(current_time)

    def _check_trade_frequency(
        self, trade_side: str, current_time: datetime
    ) -> Tuple[bool, str, int]:
        """
        检查交易频率是否合理

        Returns:
            (can_trade, reason, cooldown_minutes)
        """
        # 检查最近1小时的交易次数
        recent_trades = [
            t
            for t in self.trade_history
            if t["side"] == trade_side
            and (current_time - t["timestamp"]).total_seconds() < 3600
        ]

        if len(recent_trades) >= self.max_trades_per_hour:
            return (
                False,
                f"1小时内已交易{len(recent_trades)}次，超过上限{self.max_trades_per_hour}",
                30,
            )

        # 检查最近1天的交易次数
        daily_trades = [
            t
            for t in self.trade_history
            if t["side"] == trade_side
            and (current_time - t["timestamp"]).total_seconds() < 86400
        ]

        if len(daily_trades) >= self.max_trades_per_day:
            return (
                False,
                f"今日已交易{len(daily_trades)}次，超过上限{self.max_trades_per_day}",
                60,
            )

        # 检查最小交易间隔
        if recent_trades:
            last_trade_time = max(t["timestamp"] for t in recent_trades)
            time_since_last = (current_time - last_trade_time).total_seconds() / 60

            # 根据交易表现调整最小间隔
            performance_score = self._calculate_recent_performance(
                trade_side, current_time
            )
            min_interval = self._get_min_interval_based_on_performance(
                performance_score
            )

            if time_since_last < min_interval:
                return (
                    False,
                    f"距离上次交易仅{time_since_last:.1f}分钟，需要{min_interval}分钟冷却",
                    min_interval,
                )

        return True, "交易频率正常", 0

    def _evaluate_market_conditions(
        self, trade_side: str, market_conditions: Dict[str, Any]
    ) -> Tuple[bool, str, int]:
        """
        评估市场条件是否适合交易

        Returns:
            (can_trade, reason, cooldown_minutes)
        """
        volatility = market_conditions.get("volatility", 0.02)
        trend_strength = market_conditions.get("trend_strength", 0)
        trend_direction = market_conditions.get("trend_direction", "neutral")

        # 低波动市场：增加冷却时间
        if volatility < 0.01:  # ATR < 1%
            if trade_side == "buy":
                return False, f"极低波动市场(ATR={volatility:.2%})不适合主动买入", 45
            else:
                return (
                    False,
                    f"极低波动市场(ATR={volatility:.2%})，建议等待更好时机",
                    30,
                )

        # 高波动市场：减少冷却时间
        elif volatility > 0.05:  # ATR > 5%
            return True, f"高波动市场(ATR={volatility:.2%})适合捕捉机会", 0

        # 强烈趋势：根据方向判断
        if abs(trend_strength) > 0.7:
            if trade_side == "buy" and trend_direction == "up":
                return True, f"强烈上升趋势，适合买入", 0
            elif trade_side == "sell" and trend_direction == "down":
                return True, f"强烈下跌趋势，适合卖出", 0
            elif trade_side == "buy" and trend_direction == "down":
                return False, f"强烈下跌趋势，不适合买入", 60
            elif trade_side == "sell" and trend_direction == "up":
                return False, f"强烈上升趋势，不适合卖出", 60

        return True, "市场条件正常", 0

    def _adjust_cooling_based_on_result(
        self, trade_side: str, trade_result: Dict[str, Any], current_time: datetime
    ):
        """
        根据交易结果调整冷却策略
        """
        pnl = trade_result.get("pnl", 0)
        pnl_percentage = trade_result.get("pnl_percentage", 0)
        execution_quality = trade_result.get("execution_quality", "normal")

        # 盈利交易：减少冷却时间
        if pnl > 0:
            if pnl_percentage > 0.02:  # 盈利 > 2%
                new_level = CoolingLevel.LIGHT
                reason = f"盈利{pnl_percentage:.2%}，轻度冷却"
            else:  # 小幅盈利
                new_level = CoolingLevel.NONE
                reason = f"小幅盈利{pnl_percentage:.2%}，无冷却"

        # 亏损交易：增加冷却时间
        elif pnl < 0:
            if pnl_percentage < -0.02:  # 亏损 > 2%
                new_level = CoolingLevel.HEAVY
                reason = f"亏损{pnl_percentage:.2%}，重度冷却"
            elif pnl_percentage < -0.01:  # 中等亏损
                new_level = CoolingLevel.MEDIUM
                reason = f"中等亏损{pnl_percentage:.2%}，中等冷却"
            else:  # 小幅亏损
                new_level = CoolingLevel.LIGHT
                reason = f"小幅亏损{pnl_percentage:.2%}，轻度冷却"

        # 执行质量差：增加冷却时间
        elif execution_quality in ["poor", "timeout"]:
            new_level = CoolingLevel.MEDIUM
            reason = f"执行质量差({execution_quality})，中等冷却"

        else:
            new_level = CoolingLevel.LIGHT
            reason = "正常交易，轻度冷却"

        self._apply_cooling(trade_side, new_level, reason, current_time)

    def _apply_cooling(
        self,
        trade_side: str,
        level_or_minutes: int | CoolingLevel,
        reason: str,
        current_time: datetime,
    ):
        """
        应用冷却策略
        """
        if isinstance(level_or_minutes, CoolingLevel):
            level = level_or_minutes
            config = self.cooling_configs[level]
            # 随机选择冷却时间
            import random

            cooldown_minutes = random.randint(
                config["min_minutes"], config["max_minutes"]
            )
        else:
            cooldown_minutes = level_or_minutes
            level = self._get_level_from_minutes(cooldown_minutes)

        cooldown_until = current_time + timedelta(minutes=cooldown_minutes)

        self.current_cooling[trade_side] = CoolingState(
            level=level,
            remaining_seconds=cooldown_minutes * 60,
            reason=reason,
            last_trade_time=current_time,
            cooldown_until=cooldown_until,
        )

        logger.info(
            f"📊 {trade_side.upper()} 冷却设置: {level.value} ({cooldown_minutes}分钟) - {reason}"
        )

    def _get_level_from_minutes(self, minutes: int) -> CoolingLevel:
        """根据分钟数获取冷却等级"""
        if minutes >= 60:
            return CoolingLevel.EXTREME
        elif minutes >= 30:
            return CoolingLevel.HEAVY
        elif minutes >= 15:
            return CoolingLevel.MEDIUM
        elif minutes >= 5:
            return CoolingLevel.LIGHT
        else:
            return CoolingLevel.NONE

    def _calculate_recent_performance(
        self, trade_side: str, current_time: datetime, hours: int = 4
    ) -> float:
        """
        计算最近几小时的交易表现

        Returns:
            表现评分 (0-1, 1为最佳)
        """
        cutoff_time = current_time - timedelta(hours=hours)
        recent_trades = [
            t
            for t in self.trade_history
            if t["side"] == trade_side and t["timestamp"] > cutoff_time
        ]

        if not recent_trades:
            return 0.5  # 无历史数据，返回中等表现

        total_trades = len(recent_trades)
        profitable_trades = sum(
            1 for t in recent_trades if t["result"].get("pnl", 0) > 0
        )

        win_rate = profitable_trades / total_trades if total_trades > 0 else 0.5

        # 计算平均盈利/亏损比率
        profitable_pnl = [
            t["result"].get("pnl", 0)
            for t in recent_trades
            if t["result"].get("pnl", 0) > 0
        ]
        losing_pnl = [
            abs(t["result"].get("pnl", 0))
            for t in recent_trades
            if t["result"].get("pnl", 0) < 0
        ]

        avg_win = sum(profitable_pnl) / len(profitable_pnl) if profitable_pnl else 0
        avg_loss = sum(losing_pnl) / len(losing_pnl) if losing_pnl else 1

        profit_factor = avg_win / avg_loss if avg_loss > 0 else 1

        # 综合评分：胜率权重0.6，利润因子权重0.4
        performance_score = (win_rate * 0.6) + (min(profit_factor / 3, 1) * 0.4)

        return min(performance_score, 1.0)

    def _get_min_interval_based_on_performance(self, performance_score: float) -> int:
        """
        根据表现评分获取最小交易间隔

        Args:
            performance_score: 表现评分 (0-1)

        Returns:
            最小间隔分钟数
        """
        if performance_score >= 0.8:  # 优秀表现
            return 5  # 5分钟
        elif performance_score >= 0.6:  # 良好表现
            return 10  # 10分钟
        elif performance_score >= 0.4:  # 一般表现
            return 15  # 15分钟
        elif performance_score >= 0.2:  # 差表现
            return 25  # 25分钟
        else:  # 很差表现
            return 40  # 40分钟

    def _cleanup_old_history(self, current_time: datetime):
        """清理7天前的交易历史"""
        cutoff_time = current_time - timedelta(days=7)
        self.trade_history = [
            t for t in self.trade_history if t["timestamp"] > cutoff_time
        ]

    def get_cooling_status(self) -> Dict[str, Dict[str, Any]]:
        """获取当前冷却状态"""
        result = {}
        current_time = datetime.now()

        for side, state in self.current_cooling.items():
            remaining_seconds = 0
            if state.cooldown_until and current_time < state.cooldown_until:
                remaining_seconds = int(
                    (state.cooldown_until - current_time).total_seconds()
                )

            result[side] = {
                "level": state.level.value,
                "remaining_seconds": remaining_seconds,
                "reason": state.reason,
                "last_trade_time": state.last_trade_time.isoformat()
                if state.last_trade_time
                else None,
                "can_trade": remaining_seconds == 0,
            }

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        current_time = datetime.now()

        # 最近24小时统计
        last_24h = current_time - timedelta(hours=24)
        trades_24h = [t for t in self.trade_history if t["timestamp"] > last_24h]

        # 最近1小时统计
        last_1h = current_time - timedelta(hours=1)
        trades_1h = [t for t in self.trade_history if t["timestamp"] > last_1h]

        # 计算胜率
        profitable_24h = sum(1 for t in trades_24h if t["result"].get("pnl", 0) > 0)
        profitable_1h = sum(1 for t in trades_1h if t["result"].get("pnl", 0) > 0)

        return {
            "total_trades": len(self.trade_history),
            "trades_24h": len(trades_24h),
            "trades_1h": len(trades_1h),
            "win_rate_24h": profitable_24h / len(trades_24h) if trades_24h else 0,
            "win_rate_1h": profitable_1h / len(trades_1h) if trades_1h else 0,
            "cooling_status": self.get_cooling_status(),
        }

    def reset_for_new_day(self):
        """新的一天开始时重置状态"""
        logger.info("冷却管理器重置为新的一天")

        # 重置为无冷却状态
        for side in ["buy", "sell"]:
            self.current_cooling[side] = CoolingState(
                CoolingLevel.NONE, 0, "新的一天", None, None
            )

        # 保留历史记录但可以适当清理
        # 这里不清理历史记录，保留完整的历史
