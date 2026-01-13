"""
市场波动率适配器 - 根据市场波动率动态调整交易策略
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    """波动率制度"""

    EXTREMELY_LOW = "extremely_low"  # 极低波动 (< 0.5%)
    LOW = "low"  # 低波动 (0.5%-1.5%)
    NORMAL = "normal"  # 正常波动 (1.5%-3%)
    HIGH = "high"  # 高波动 (3%-5%)
    EXTREMELY_HIGH = "extremely_high"  # 极高波动 (> 5%)
    CHAOTIC = "chaotic"  # 混乱波动 (异常波动)


@dataclass
class VolatilityMetrics:
    """波动率指标"""

    regime: VolatilityRegime
    atr_percentage: float  # ATR百分比
    realized_volatility: float  # 已实现波动率
    implied_volatility: float  # 隐含波动率（如果可用）
    volume_volatility: float  # 成交量波动率
    price_range_percentage: float  # 价格区间百分比
    volatility_trend: str  # 波动率趋势: increasing/decreasing/stable
    confidence: float  # 计算置信度


@dataclass
class AdaptiveStrategyParameters:
    """自适应策略参数"""

    signal_threshold: float  # 信号阈值
    position_size_multiplier: float  # 仓位大小倍数
    stop_loss_percentage: float  # 止损百分比
    take_profit_percentage: float  # 止盈百分比
    cooling_minutes: int  # 冷却时间（分钟）
    max_trades_per_hour: int  # 每小时最大交易次数
    order_type_preference: str  # 订单类型偏好: market/limit/conditional
    risk_multiplier: float  # 风险倍数


class MarketVolatilityAdapter:
    """市场波动率适配器"""

    def __init__(self):
        # 波动率阈值配置
        self.volatility_thresholds = {
            VolatilityRegime.EXTREMELY_LOW: {
                "min": 0.0,
                "max": 0.005,
                "description": "极低波动",
            },
            VolatilityRegime.LOW: {"min": 0.005, "max": 0.015, "description": "低波动"},
            VolatilityRegime.NORMAL: {
                "min": 0.015,
                "max": 0.03,
                "description": "正常波动",
            },
            VolatilityRegime.HIGH: {"min": 0.03, "max": 0.05, "description": "高波动"},
            VolatilityRegime.EXTREMELY_HIGH: {
                "min": 0.05,
                "max": 0.10,
                "description": "极高波动",
            },
            VolatilityRegime.CHAOTIC: {
                "min": 0.10,
                "max": float("inf"),
                "description": "混乱波动",
            },
        }

        # 默认策略参数映射（优化后：降低低波动市场的信号阈值，允许更多交易）
        self.default_strategy_map = {
            VolatilityRegime.EXTREMELY_LOW: AdaptiveStrategyParameters(
                signal_threshold=0.65,  # 原为0.85，降低以允许低波动市场交易
                position_size_multiplier=0.5,  # 原为0.3，增加仓位
                stop_loss_percentage=0.008,
                take_profit_percentage=0.04,
                cooling_minutes=45,  # 原为60，减少冷却时间
                max_trades_per_hour=2,  # 原为1，增加交易频率
                order_type_preference="limit",
                risk_multiplier=0.7,  # 原为0.5，增加风险容忍
            ),
            VolatilityRegime.LOW: AdaptiveStrategyParameters(
                signal_threshold=0.60,  # 原为0.75，降低以允许交易
                position_size_multiplier=0.7,  # 原为0.5，增加仓位
                stop_loss_percentage=0.012,
                take_profit_percentage=0.06,
                cooling_minutes=25,  # 原为30，减少冷却
                max_trades_per_hour=3,  # 原为2，增加交易频率
                order_type_preference="limit",
                risk_multiplier=0.85,  # 原为0.7，增加风险容忍
            ),
            VolatilityRegime.NORMAL: AdaptiveStrategyParameters(
                signal_threshold=0.55,  # 原为0.65
                position_size_multiplier=1.0,
                stop_loss_percentage=0.015,
                take_profit_percentage=0.08,
                cooling_minutes=15,
                max_trades_per_hour=3,
                order_type_preference="market",
                risk_multiplier=1.0,
            ),
            VolatilityRegime.HIGH: AdaptiveStrategyParameters(
                signal_threshold=0.50,  # 原为0.55
                position_size_multiplier=1.5,
                stop_loss_percentage=0.025,
                take_profit_percentage=0.12,
                cooling_minutes=10,
                max_trades_per_hour=4,
                order_type_preference="market",
                risk_multiplier=1.3,
            ),
            VolatilityRegime.EXTREMELY_HIGH: AdaptiveStrategyParameters(
                signal_threshold=0.45,  # 原为0.45
                position_size_multiplier=2.0,
                stop_loss_percentage=0.04,
                take_profit_percentage=0.15,
                cooling_minutes=5,
                max_trades_per_hour=6,
                order_type_preference="market",
                risk_multiplier=1.5,
            ),
            VolatilityRegime.CHAOTIC: AdaptiveStrategyParameters(
                signal_threshold=0.70,  # 原为0.90，降低以允许一定交易
                position_size_multiplier=0.3,  # 原为0.1，增加仓位
                stop_loss_percentage=0.02,
                take_profit_percentage=0.05,
                cooling_minutes=60,  # 原为120，减少冷却
                max_trades_per_hour=2,  # 原为1，增加交易频率
                order_type_preference="limit",
                risk_multiplier=0.5,  # 原为0.3，增加风险容忍
            ),
        }

        # 历史波动率数据
        self.volatility_history: List[VolatilityMetrics] = []

        # 自适应学习参数
        self.learning_enabled = True
        self.performance_memory_days = 30

    def analyze_volatility(
        self, market_data: Dict[str, Any], historical_prices: List[float]
    ) -> VolatilityMetrics:
        """
        分析当前市场波动率

        Args:
            market_data: 当前市场数据
            historical_prices: 历史价格数据

        Returns:
            波动率分析结果
        """
        # 计算ATR百分比
        atr = market_data.get("atr", 0)
        current_price = market_data.get("price", 0)
        atr_percentage = atr / current_price if current_price > 0 else 0

        # 计算已实现波动率（基于历史价格）
        realized_volatility = self._calculate_realized_volatility(historical_prices)

        # 计算成交量波动率
        volume_volatility = self._calculate_volume_volatility(market_data)

        # 计算价格区间百分比
        price_range_percentage = self._calculate_price_range_percentage(market_data)

        # 确定波动率制度
        regime = self._determine_volatility_regime(atr_percentage, realized_volatility)

        # 分析波动率趋势
        volatility_trend = self._analyze_volatility_trend()

        # 计算置信度
        confidence = self._calculate_analysis_confidence(market_data, historical_prices)

        metrics = VolatilityMetrics(
            regime=regime,
            atr_percentage=atr_percentage,
            realized_volatility=realized_volatility,
            implied_volatility=0.0,  # 暂时不支持
            volume_volatility=volume_volatility,
            price_range_percentage=price_range_percentage,
            volatility_trend=volatility_trend,
            confidence=confidence,
        )

        # 记录历史
        self._record_volatility_metrics(metrics)

        return metrics

    def get_adaptive_strategy(
        self,
        volatility_metrics: VolatilityMetrics,
        current_performance: Optional[Dict] = None,
    ) -> AdaptiveStrategyParameters:
        """
        获取自适应策略参数

        Args:
            volatility_metrics: 波动率指标
            current_performance: 当前表现数据（可选）

        Returns:
            自适应策略参数
        """
        # 获取基础策略参数
        base_params = self.default_strategy_map[volatility_metrics.regime]

        # 如果启用学习，根据历史表现调整参数
        if self.learning_enabled and current_performance:
            adjusted_params = self._learn_from_performance(
                base_params, volatility_metrics, current_performance
            )
        else:
            adjusted_params = base_params

        return adjusted_params

    def _calculate_realized_volatility(
        self, prices: List[float], window: int = 20
    ) -> float:
        """
        计算已实现波动率

        Args:
            prices: 价格列表
            window: 计算窗口

        Returns:
            已实现波动率（年化）
        """
        if len(prices) < window + 1:
            return 0.0

        # 计算收益率
        returns = []
        for i in range(1, min(len(prices), window + 1)):
            ret = (prices[i] - prices[i - 1]) / prices[i - 1]
            returns.append(ret)

        if not returns:
            return 0.0

        # 计算波动率（日收益率标准差）
        volatility_daily = float(np.std(returns))

        # 年化（假设252个交易日）
        volatility_annualized = float(volatility_daily * np.sqrt(252))

        return volatility_annualized

    def _calculate_volume_volatility(self, market_data: Dict[str, Any]) -> float:
        """
        计算成交量波动率
        """
        volume_data = market_data.get("volume_history", [])
        if len(volume_data) < 10:
            return 0.0

        # 计算成交量变化率的标准差
        volume_changes = []
        for i in range(1, len(volume_data)):
            if volume_data[i - 1] > 0:
                change = (volume_data[i] - volume_data[i - 1]) / volume_data[i - 1]
                volume_changes.append(change)

        if not volume_changes:
            return 0.0

        return float(np.std(volume_changes))

    def _calculate_price_range_percentage(self, market_data: Dict[str, Any]) -> float:
        """
        计算价格区间百分比
        """
        high_24h = market_data.get("high_24h", 0)
        low_24h = market_data.get("low_24h", 0)
        current_price = market_data.get("price", 0)

        if current_price <= 0 or high_24h <= 0 or low_24h >= high_24h:
            return 0.0

        price_range = (high_24h - low_24h) / current_price
        return price_range

    def _determine_volatility_regime(
        self, atr_percentage: float, realized_volatility: float
    ) -> VolatilityRegime:
        """
        确定波动率制度
        """
        # 使用ATR百分比作为主要指标，结合已实现波动率
        primary_metric = atr_percentage

        # 特殊情况：检查是否为混乱波动
        if atr_percentage > 0.08 or realized_volatility > 0.15:
            return VolatilityRegime.CHAOTIC

        # 正常情况：根据阈值确定制度
        for regime, thresholds in self.volatility_thresholds.items():
            if thresholds["min"] <= primary_metric < thresholds["max"]:
                return regime

        # 默认返回正常波动
        return VolatilityRegime.NORMAL

    def _analyze_volatility_trend(self) -> str:
        """
        分析波动率趋势
        """
        if len(self.volatility_history) < 5:
            return "stable"

        # 获取最近5次的波动率
        recent_volatilities = [v.atr_percentage for v in self.volatility_history[-5:]]

        # 计算趋势
        if len(recent_volatilities) >= 3:
            # 简单线性回归斜率
            x = list(range(len(recent_volatilities)))
            slope = float(np.polyfit(x, recent_volatilities, 1)[0])

            if slope > 0.001:
                return "increasing"
            elif slope < -0.001:
                return "decreasing"
            else:
                return "stable"

        return "stable"

    def _calculate_analysis_confidence(
        self, market_data: Dict[str, Any], historical_prices: List[float]
    ) -> float:
        """
        计算分析置信度
        """
        confidence = 1.0

        # 数据完整性检查
        if not historical_prices or len(historical_prices) < 20:
            confidence *= 0.7

        if not market_data.get("atr"):
            confidence *= 0.8

        if not market_data.get("volume_history"):
            confidence *= 0.9

        # 数据质量检查
        price_variation = (
            float(np.std(historical_prices) / np.mean(historical_prices))
            if historical_prices
            else 0.0
        )
        if price_variation < 0.001:  # 价格几乎不变
            confidence *= 0.8

        return min(1.0, confidence)

    def _record_volatility_metrics(self, metrics: VolatilityMetrics):
        """
        记录波动率指标历史
        """
        self.volatility_history.append(metrics)

        # 保留最近1000条记录
        if len(self.volatility_history) > 1000:
            self.volatility_history = self.volatility_history[-1000:]

    def _learn_from_performance(
        self,
        base_params: AdaptiveStrategyParameters,
        volatility_metrics: VolatilityMetrics,
        performance: Dict[str, Any],
    ) -> AdaptiveStrategyParameters:
        """
        从历史表现中学习，调整策略参数
        """
        # 分析最近表现
        win_rate = performance.get("win_rate", 0.5)
        profit_factor = performance.get("profit_factor", 1.0)
        max_drawdown = performance.get("max_drawdown", 0.05)
        total_trades = performance.get("total_trades", 0)

        if total_trades < 10:
            # 交易样本不足，使用基础参数
            return base_params

        # 根据表现调整参数
        adjustment_factor = 1.0

        # 胜率调整
        if win_rate > 0.7:
            adjustment_factor *= 1.1  # 表现好，可以稍微激进
        elif win_rate < 0.4:
            adjustment_factor *= 0.9  # 表现差，需要保守

        # 利润因子调整
        if profit_factor > 1.5:
            adjustment_factor *= 1.05
        elif profit_factor < 0.8:
            adjustment_factor *= 0.95

        # 最大回撤调整
        if max_drawdown > 0.1:
            adjustment_factor *= 0.9  # 回撤大，需要更保守

        # 应用调整
        adjusted_params = AdaptiveStrategyParameters(
            signal_threshold=min(
                0.95, base_params.signal_threshold * adjustment_factor
            ),
            position_size_multiplier=max(
                0.1, base_params.position_size_multiplier * adjustment_factor
            ),
            stop_loss_percentage=base_params.stop_loss_percentage,
            take_profit_percentage=base_params.take_profit_percentage
            * adjustment_factor,
            cooling_minutes=max(
                5, int(base_params.cooling_minutes * (2 - adjustment_factor))
            ),
            max_trades_per_hour=max(
                1, int(base_params.max_trades_per_hour * adjustment_factor)
            ),
            order_type_preference=base_params.order_type_preference,
            risk_multiplier=max(0.1, base_params.risk_multiplier * adjustment_factor),
        )

        return adjusted_params

    def get_volatility_statistics(self) -> Dict[str, Any]:
        """
        获取波动率统计信息
        """
        if not self.volatility_history:
            return {"total_observations": 0}

        # 制度分布
        regime_counts = {}
        for metrics in self.volatility_history:
            regime = metrics.regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # 波动率趋势
        trend_distribution = {}
        for metrics in self.volatility_history:
            trend = metrics.volatility_trend
            trend_distribution[trend] = trend_distribution.get(trend, 0) + 1

        # 平均指标
        avg_atr = sum(m.atr_percentage for m in self.volatility_history) / len(
            self.volatility_history
        )
        avg_realized_vol = sum(
            m.realized_volatility for m in self.volatility_history
        ) / len(self.volatility_history)

        return {
            "total_observations": len(self.volatility_history),
            "regime_distribution": regime_counts,
            "trend_distribution": trend_distribution,
            "avg_atr_percentage": avg_atr,
            "avg_realized_volatility": avg_realized_vol,
            "current_regime": self.volatility_history[-1].regime.value
            if self.volatility_history
            else None,
        }

    def predict_volatility_regime(self, hours_ahead: int = 1) -> Dict[str, float]:
        """
        预测未来波动率制度

        Args:
            hours_ahead: 预测小时数

        Returns:
            各制度的预测概率
        """
        if len(self.volatility_history) < 10:
            # 数据不足，返回当前制度概率为1
            current_regime = (
                self.volatility_history[-1].regime
                if self.volatility_history
                else VolatilityRegime.NORMAL
            )
            return {current_regime.value: 1.0}

        # 简单的马尔可夫链预测
        transitions = self._calculate_regime_transitions()

        current_regime = self.volatility_history[-1].regime
        prediction = {regime.value: 0.0 for regime in VolatilityRegime}

        # 当前制度的转移概率
        if current_regime in transitions:
            prediction.update(transitions[current_regime])

        return prediction

    def _calculate_regime_transitions(self) -> Dict[VolatilityRegime, Dict[str, float]]:
        """
        计算制度转移概率
        """
        if len(self.volatility_history) < 2:
            return {}

        transitions = {}
        regime_sequence = [m.regime for m in self.volatility_history]

        for i in range(len(regime_sequence) - 1):
            current = regime_sequence[i]
            next_regime = regime_sequence[i + 1]

            if current not in transitions:
                transitions[current] = {}

            transitions[current][next_regime.value] = (
                transitions[current].get(next_regime.value, 0) + 1
            )

        # 转换为概率
        for current, next_counts in transitions.items():
            total = sum(next_counts.values())
            for regime, count in next_counts.items():
                next_counts[regime] = count / total

        return transitions

    def get_adaptation_recommendations(self) -> List[str]:
        """
        获取适应性建议
        """
        recommendations = []
        stats = self.get_volatility_statistics()

        if stats["total_observations"] == 0:
            return ["📊 暂无波动率历史数据，建议继续收集数据"]

        # 基于当前制度给出建议
        current_regime = stats.get("current_regime")
        if current_regime == "extremely_low":
            recommendations.extend(
                [
                    "🐌 当前极低波动，建议大幅降低交易频率",
                    "🎯 使用极严格的信号过滤标准",
                    "⏰ 延长冷却时间至1小时以上",
                    "📏 缩小止损范围至0.5%-1%",
                ]
            )
        elif current_regime == "low":
            recommendations.extend(
                [
                    "🐌 当前低波动，建议降低交易频率",
                    "🎯 提高信号置信度阈值",
                    "⏰ 适当延长冷却时间",
                    "📏 使用保守止损策略",
                ]
            )
        elif current_regime == "high":
            recommendations.extend(
                [
                    "⚡ 当前高波动，适合积极交易",
                    "🎯 可以降低信号阈值",
                    "⏰ 缩短冷却时间",
                    "📏 适当放宽止损范围",
                ]
            )
        elif current_regime == "extremely_high":
            recommendations.extend(
                [
                    "🌪️ 当前极高波动，注意风险控制",
                    "🎯 保持较高信号标准",
                    "⏰ 保持适中冷却时间",
                    "📏 使用更宽松的止损",
                ]
            )
        elif current_regime == "chaotic":
            recommendations.extend(
                [
                    "⚠️ 当前市场混乱，建议暂停交易",
                    "🎯 大幅提高信号阈值",
                    "⏰ 极长冷却时间",
                    "📏 极小仓位或暂停交易",
                ]
            )

        # 基于历史数据给出一般建议
        regime_dist = stats.get("regime_distribution", {})
        most_common_regime = (
            max(regime_dist.items(), key=lambda x: x[1])[0] if regime_dist else None
        )

        if most_common_regime:
            recommendations.append(f"📊 历史最常见波动制度: {most_common_regime}")

        return recommendations

    def reset_history(self):
        """重置历史记录"""
        self.volatility_history = []
        logger.info("波动率适配器历史已重置")
