"""
智能止损优化器 - 基于市场条件和风险偏好的动态止损管理
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class StopLossType(Enum):
    """止损类型"""

    FIXED = "fixed"  # 固定百分比止损
    ATR_BASED = "atr_based"  # ATR基础止损
    DYNAMIC = "dynamic"  # 动态止损
    TRAILING = "trailing"  # 追踪止损
    TIME_BASED = "time_based"  # 时间基础止损
    VOLATILITY_ADAPTIVE = "volatility_adaptive"  # 波动率自适应


@dataclass
class StopLossConfig:
    """止损配置"""

    type: StopLossType
    initial_percentage: float  # 初始止损百分比
    max_percentage: float  # 最大止损百分比
    min_percentage: float  # 最小止损百分比
    atr_multiplier: float  # ATR倍数
    trailing_distance: float  # 追踪距离百分比
    time_limit_hours: int  # 时间限制（小时）
    volatility_threshold: float  # 波动率阈值
    profit_lock_threshold: float  # 利润锁定阈值


@dataclass
class StopLossState:
    """止损状态"""

    current_price: float
    entry_price: float
    stop_loss_price: float
    take_profit_price: Optional[float]
    last_update_time: datetime
    atr_value: float
    volatility: float
    unrealized_pnl: float
    unrealized_pnl_percentage: float


class IntelligentStopLossOptimizer:
    """智能止损优化器"""

    def __init__(self):
        # 默认配置
        self.default_config = StopLossConfig(
            type=StopLossType.DYNAMIC,
            initial_percentage=0.015,  # 1.5%
            max_percentage=0.05,  # 5%
            min_percentage=0.005,  # 0.5%
            atr_multiplier=2.5,
            trailing_distance=0.01,  # 1%
            time_limit_hours=24,
            volatility_threshold=0.02,  # 2%
            profit_lock_threshold=0.02,  # 2%
        )

        # 市场条件权重
        self.volatility_weight = 0.3
        self.trend_weight = 0.25
        self.time_weight = 0.2
        self.performance_weight = 0.25

        # 止损历史记录（用于学习和优化）
        self.stop_loss_history: List[Dict] = []

    def calculate_optimal_stop_loss(
        self,
        position_info: Dict[str, Any],
        market_data: Dict[str, Any],
        config: Optional[StopLossConfig] = None,
    ) -> Dict[str, Any]:
        """
        计算最优止损价格

        Args:
            position_info: 持仓信息
            market_data: 市场数据
            config: 止损配置（可选）

        Returns:
            止损建议结果
        """
        if config is None:
            config = self.default_config

        current_time = datetime.now()

        # 提取关键数据
        entry_price = position_info["entry_price"]
        current_price = position_info["current_price"]
        position_side = position_info.get("side", "long")
        entry_time = position_info.get("entry_time", current_time - timedelta(hours=1))

        # 计算基本指标
        atr_value = market_data.get("atr", 0)
        volatility = market_data.get("volatility", 0.02)
        trend_strength = market_data.get("trend_strength", 0)
        trend_direction = market_data.get("trend_direction", "neutral")

        # 计算未实现盈亏
        if position_side == "long":
            unrealized_pnl_percentage = (current_price - entry_price) / entry_price
        else:  # short
            unrealized_pnl_percentage = (entry_price - current_price) / entry_price

        unrealized_pnl = unrealized_pnl_percentage * position_info.get(
            "position_value", 1000
        )

        # 创建止损状态
        state = StopLossState(
            current_price=current_price,
            entry_price=entry_price,
            stop_loss_price=position_info.get("current_stop_loss", 0),
            take_profit_price=position_info.get("take_profit_price"),
            last_update_time=current_time,
            atr_value=atr_value,
            volatility=volatility,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_percentage=unrealized_pnl_percentage,
        )

        # 根据配置类型计算止损
        if config.type == StopLossType.FIXED:
            result = self._calculate_fixed_stop_loss(state, config)
        elif config.type == StopLossType.ATR_BASED:
            result = self._calculate_atr_based_stop_loss(state, config, market_data)
        elif config.type == StopLossType.DYNAMIC:
            result = self._calculate_dynamic_stop_loss(state, config, market_data)
        elif config.type == StopLossType.TRAILING:
            result = self._calculate_trailing_stop_loss(state, config, market_data)
        elif config.type == StopLossType.VOLATILITY_ADAPTIVE:
            result = self._calculate_volatility_adaptive_stop_loss(
                state, config, market_data
            )
        else:
            result = self._calculate_fixed_stop_loss(state, config)

        # 添加时间限制检查
        time_based_check = self._check_time_based_stop_loss(
            entry_time, config, current_time
        )
        if time_based_check["should_stop"]:
            result["recommended_action"] = "CLOSE_POSITION"
            result["reason"] = time_based_check["reason"]
            result["stop_loss_price"] = current_price * (1 - 0.001)  # 略低于当前价格

        # 添加利润锁定逻辑
        profit_lock_check = self._check_profit_lock_logic(state, config)
        if profit_lock_check["should_lock"]:
            result["take_profit_price"] = profit_lock_check["lock_price"]
            result["reason"] += f" | {profit_lock_check['reason']}"

        # 记录历史
        self._record_stop_loss_decision(
            position_info, market_data, result, current_time
        )

        return result

    def _calculate_fixed_stop_loss(
        self, state: StopLossState, config: StopLossConfig
    ) -> Dict[str, Any]:
        """计算固定百分比止损"""
        if state.unrealized_pnl_percentage > 0:
            # 盈利中，使用更紧的止损
            stop_percentage = min(
                config.initial_percentage * 0.7, config.min_percentage
            )
        else:
            stop_percentage = config.initial_percentage

        # 确保在合理范围内
        stop_percentage = max(
            config.min_percentage, min(config.max_percentage, stop_percentage)
        )

        if state.unrealized_pnl_percentage >= 0:
            # 多头止损
            stop_loss_price = state.entry_price * (1 - stop_percentage)
        else:
            # 空头止损
            stop_loss_price = state.entry_price * (1 + stop_percentage)

        return {
            "stop_loss_price": stop_loss_price,
            "stop_loss_percentage": stop_percentage,
            "type": "fixed",
            "reason": f"固定止损: {stop_percentage:.2%}",
            "recommended_action": "UPDATE_STOP_LOSS",
            "confidence": 0.8,
        }

    def _calculate_atr_based_stop_loss(
        self, state: StopLossState, config: StopLossConfig, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算ATR基础止损"""
        if state.atr_value <= 0:
            # ATR无效，回退到固定止损
            return self._calculate_fixed_stop_loss(state, config)

        # 根据ATR计算止损距离
        atr_percentage = state.atr_value / state.entry_price
        stop_percentage = min(
            atr_percentage * config.atr_multiplier, config.max_percentage
        )

        # 确保不低于最小值
        stop_percentage = max(config.min_percentage, stop_percentage)

        # 根据盈利状态调整
        if state.unrealized_pnl_percentage > 0.01:  # 盈利1%以上
            stop_percentage *= 0.8  # 收紧止损
        elif state.unrealized_pnl_percentage < -0.005:  # 亏损0.5%以上
            stop_percentage *= 1.2  # 放宽止损

        stop_percentage = max(
            config.min_percentage, min(config.max_percentage, stop_percentage)
        )

        if state.unrealized_pnl_percentage >= 0:
            stop_loss_price = state.entry_price * (1 - stop_percentage)
        else:
            stop_loss_price = state.entry_price * (1 + stop_percentage)

        return {
            "stop_loss_price": stop_loss_price,
            "stop_loss_percentage": stop_percentage,
            "type": "atr_based",
            "reason": f"ATR止损: {stop_percentage:.2%} (ATR: {atr_percentage:.2%} × {config.atr_multiplier})",
            "recommended_action": "UPDATE_STOP_LOSS",
            "confidence": 0.85,
        }

    def _calculate_dynamic_stop_loss(
        self, state: StopLossState, config: StopLossConfig, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算动态止损（结合多种因素）"""
        # 基础ATR止损
        atr_result = self._calculate_atr_based_stop_loss(state, config, market_data)
        stop_percentage = atr_result["stop_loss_percentage"]

        # 根据市场条件调整
        volatility = state.volatility
        trend_strength = market_data.get("trend_strength", 0)

        # 高波动期：放宽止损
        if volatility > config.volatility_threshold * 1.5:
            stop_percentage *= 1.3
            reason_add = "高波动期放宽止损"
        # 低波动期：收紧止损
        elif volatility < config.volatility_threshold * 0.5:
            stop_percentage *= 0.7
            reason_add = "低波动期收紧止损"
        else:
            reason_add = "正常波动期"

        # 强趋势：根据趋势方向调整
        if abs(trend_strength) > 0.6:
            if (trend_strength > 0 and state.unrealized_pnl_percentage >= 0) or (
                trend_strength < 0 and state.unrealized_pnl_percentage <= 0
            ):
                # 同向趋势：收紧止损
                stop_percentage *= 0.8
                reason_add += "，同向强趋势收紧止损"
            else:
                # 反向趋势：放宽止损
                stop_percentage *= 1.2
                reason_add += "，反向强趋势放宽止损"

        # 确保在合理范围内
        stop_percentage = max(
            config.min_percentage, min(config.max_percentage, stop_percentage)
        )

        if state.unrealized_pnl_percentage >= 0:
            stop_loss_price = state.entry_price * (1 - stop_percentage)
        else:
            stop_loss_price = state.entry_price * (1 + stop_percentage)

        return {
            "stop_loss_price": stop_loss_price,
            "stop_loss_percentage": stop_percentage,
            "type": "dynamic",
            "reason": f"动态止损: {stop_percentage:.2%} | {reason_add}",
            "recommended_action": "UPDATE_STOP_LOSS",
            "confidence": 0.9,
        }

    def _calculate_trailing_stop_loss(
        self, state: StopLossState, config: StopLossConfig, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算追踪止损"""
        current_stop = state.stop_loss_price or (
            state.entry_price * (1 - config.initial_percentage)
        )

        # 计算新的追踪止损价格
        if state.unrealized_pnl_percentage >= 0:
            # 多头盈利中
            trailing_distance = max(config.trailing_distance, state.volatility * 2)
            new_stop = state.current_price * (1 - trailing_distance)

            # 只在价格上涨时调整止损
            if new_stop > current_stop:
                stop_loss_price = new_stop
                action = "UPDATE_STOP_LOSS"
                reason = f"追踪止损更新: 价格上涨至{state.current_price:.2f}"
            else:
                stop_loss_price = current_stop
                action = "KEEP_CURRENT"
                reason = f"追踪止损保持: 当前止损{current_stop:.2f}优于新计算值{new_stop:.2f}"
        else:
            # 亏损中，保持现有止损
            stop_loss_price = current_stop
            action = "KEEP_CURRENT"
            reason = "亏损中保持现有止损"

        return {
            "stop_loss_price": stop_loss_price,
            "stop_loss_percentage": abs(stop_loss_price - state.entry_price)
            / state.entry_price,
            "type": "trailing",
            "reason": reason,
            "recommended_action": action,
            "confidence": 0.85,
        }

    def _calculate_volatility_adaptive_stop_loss(
        self, state: StopLossState, config: StopLossConfig, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算波动率自适应止损"""
        volatility = state.volatility
        base_percentage = config.initial_percentage

        # 根据波动率调整止损百分比
        if volatility > config.volatility_threshold * 2:
            # 极高波动
            adjusted_percentage = base_percentage * 2.0
            reason = f"极高波动(>{config.volatility_threshold * 200:.0%})，放宽止损"
        elif volatility > config.volatility_threshold * 1.5:
            # 高波动
            adjusted_percentage = base_percentage * 1.5
            reason = f"高波动(>{config.volatility_threshold * 150:.0%})，适度放宽止损"
        elif volatility < config.volatility_threshold * 0.5:
            # 低波动
            adjusted_percentage = base_percentage * 0.6
            reason = f"低波动(<{config.volatility_threshold * 50:.0%})，收紧止损"
        else:
            adjusted_percentage = base_percentage
            reason = f"正常波动，标准止损"

        # 确保在合理范围内
        adjusted_percentage = max(
            config.min_percentage, min(config.max_percentage, adjusted_percentage)
        )

        if state.unrealized_pnl_percentage >= 0:
            stop_loss_price = state.entry_price * (1 - adjusted_percentage)
        else:
            stop_loss_price = state.entry_price * (1 + adjusted_percentage)

        return {
            "stop_loss_price": stop_loss_price,
            "stop_loss_percentage": adjusted_percentage,
            "type": "volatility_adaptive",
            "reason": reason,
            "recommended_action": "UPDATE_STOP_LOSS",
            "confidence": 0.88,
        }

    def _check_time_based_stop_loss(
        self, entry_time: datetime, config: StopLossConfig, current_time: datetime
    ) -> Dict[str, Any]:
        """检查时间基础止损"""
        time_held = current_time - entry_time
        hours_held = time_held.total_seconds() / 3600

        if hours_held > config.time_limit_hours:
            return {
                "should_stop": True,
                "reason": f"持仓时间超过{hours_held:.1f}小时，达到限制{config.time_limit_hours}小时",
            }

        return {"should_stop": False, "reason": ""}

    def _check_profit_lock_logic(
        self, state: StopLossState, config: StopLossConfig
    ) -> Dict[str, Any]:
        """检查利润锁定逻辑"""
        if state.unrealized_pnl_percentage < config.profit_lock_threshold:
            return {"should_lock": False, "reason": "", "lock_price": None}

        # 计算利润锁定价格
        # 锁定大部分利润，只保留小部分波动空间
        lock_buffer = min(
            state.unrealized_pnl_percentage * 0.3, 0.005
        )  # 保留30%利润或0.5%

        if state.unrealized_pnl_percentage >= 0:
            lock_price = state.current_price * (1 - lock_buffer)
        else:
            lock_price = state.current_price * (1 + lock_buffer)

        return {
            "should_lock": True,
            "reason": f"利润超过{config.profit_lock_threshold:.1%}，锁定利润至{lock_price:.2f}",
            "lock_price": lock_price,
        }

    def _record_stop_loss_decision(
        self,
        position_info: Dict[str, Any],
        market_data: Dict[str, Any],
        result: Dict[str, Any],
        timestamp: datetime,
    ):
        """记录止损决策历史"""
        record = {
            "timestamp": timestamp,
            "position_info": position_info,
            "market_data": market_data,
            "result": result,
        }

        self.stop_loss_history.append(record)

        # 保留最近1000条记录
        if len(self.stop_loss_history) > 1000:
            self.stop_loss_history = self.stop_loss_history[-1000:]

    def get_stop_loss_statistics(self) -> Dict[str, Any]:
        """获取止损统计信息"""
        if not self.stop_loss_history:
            return {"total_decisions": 0, "avg_confidence": 0, "success_rate": 0}

        total_decisions = len(self.stop_loss_history)
        avg_confidence = (
            sum(d["result"].get("confidence", 0) for d in self.stop_loss_history)
            / total_decisions
        )

        # 计算成功率（避免过早止损导致的虚假亏损）
        successful_decisions = sum(
            1
            for d in self.stop_loss_history
            if d["result"]["recommended_action"] in ["UPDATE_STOP_LOSS", "KEEP_CURRENT"]
        )
        success_rate = (
            successful_decisions / total_decisions if total_decisions > 0 else 0
        )

        return {
            "total_decisions": total_decisions,
            "avg_confidence": avg_confidence,
            "success_rate": success_rate,
            "type_distribution": self._get_type_distribution(),
        }

    def _get_type_distribution(self) -> Dict[str, int]:
        """获取止损类型分布"""
        distribution = {}
        for decision in self.stop_loss_history:
            stop_type = decision["result"].get("type", "unknown")
            distribution[stop_type] = distribution.get(stop_type, 0) + 1
        return distribution

    def optimize_stop_loss_config(
        self, historical_performance: List[Dict]
    ) -> StopLossConfig:
        """
        基于历史表现优化止损配置

        Args:
            historical_performance: 历史交易表现数据

        Returns:
            优化后的止损配置
        """
        # 分析历史数据，找出最佳参数
        if not historical_performance:
            return self.default_config

        # 简单的优化逻辑（实际可以更复杂）
        profitable_trades = [t for t in historical_performance if t.get("pnl", 0) > 0]
        losing_trades = [t for t in historical_performance if t.get("pnl", 0) < 0]

        if profitable_trades and losing_trades:
            avg_win = sum(t["pnl"] for t in profitable_trades) / len(profitable_trades)
            avg_loss = abs(sum(t["pnl"] for t in losing_trades) / len(losing_trades))

            # 根据盈亏比调整止损
            profit_factor = avg_win / avg_loss
            if profit_factor > 2:
                # 高盈亏比，可以使用更紧的止损
                optimal_percentage = 0.01  # 1%
            elif profit_factor > 1.5:
                optimal_percentage = 0.015  # 1.5%
            else:
                optimal_percentage = 0.02  # 2%
        else:
            optimal_percentage = self.default_config.initial_percentage

        # 返回优化后的配置
        optimized_config = StopLossConfig(
            type=StopLossType.DYNAMIC,
            initial_percentage=optimal_percentage,
            max_percentage=min(optimal_percentage * 3, 0.08),
            min_percentage=max(optimal_percentage * 0.3, 0.003),
            atr_multiplier=2.5,
            trailing_distance=optimal_percentage * 0.7,
            time_limit_hours=24,
            volatility_threshold=0.02,
            profit_lock_threshold=optimal_percentage * 1.5,
        )

        logger.info(f"止损配置已优化: 初始止损={optimal_percentage:.2%}")
        return optimized_config

    def reset_history(self):
        """重置历史记录"""
        self.stop_loss_history = []
        logger.info("止损优化器历史已重置")
