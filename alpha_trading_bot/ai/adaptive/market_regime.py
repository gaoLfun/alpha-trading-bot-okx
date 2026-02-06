"""
市场环境识别模块

功能：
- 基于技术指标识别当前市场状态（趋势/震荡/高波动/低波动）
- 识别市场 regime 转换
- 为参数自适应提供市场环境信息
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """市场环境类型"""

    TREND_UP = "trend_up"  # 上升趋势
    TREND_DOWN = "trend_down"  # 下降趋势
    TREND_SIDEWAYS = "trend_sideways"  # 震荡趋势
    HIGH_VOLATILITY = "high_volatility"  # 高波动
    LOW_VOLATILITY = "low_volatility"  # 低波动
    OVERSOLD = "oversold"  # 超卖
    OVERBOUGHT = "overbought"  # 超买
    NORMAL = "normal"  # 正常状态


@dataclass
class MarketRegimeState:
    """当前市场环境状态"""

    regime: MarketRegime
    confidence: float  # 0-1, 识别置信度
    trend_strength: float  # -1 到 1
    volatility_level: float  # 0 到 1
    rsi_level: float  # 0 到 100
    atr_percent: float  # ATR 百分比
    trend_direction: str  # "up", "down", "sideways"
    regime_changes: int  # 近期 regime 转换次数
    timestamp: str


class MarketRegimeDetector:
    """
    市场环境检测器

    基于多个技术指标综合判断当前市场环境
    """

    # 阈值配置
    TREND_STRENGTH_THRESHOLDS = {
        "strong_up": 0.5,
        "weak_up": 0.2,
        "strong_down": -0.5,
        "weak_down": -0.2,
    }

    VOLATILITY_THRESHOLDS = {
        "high": 0.03,  # 3% ATR
        "low": 0.015,  # 1.5% ATR
    }

    RSI_THRESHOLDS = {
        "oversold": 30,
        "overbought": 70,
        "neutral_low": 40,
        "neutral_high": 60,
    }

    def __init__(self, lookback_candles: int = 20):
        """
        初始化检测器

        Args:
            lookback_candles: 回溯K线条数
        """
        self.lookback = lookback_candles
        self._regime_history: list[MarketRegime] = []
        self._regime_change_count = 0

    def detect(self, market_data: Dict[str, Any]) -> MarketRegimeState:
        """
        检测当前市场环境

        Args:
            market_data: 市场数据字典

        Returns:
            MarketRegimeState: 市场环境状态
        """
        technical = market_data.get("technical", {})
        rsi = technical.get("rsi", 50)
        atr_percent = technical.get("atr_percent", 0.02)
        trend_strength = technical.get("trend_strength", 0)
        trend_direction = technical.get("trend_direction", "sideways")

        # 计算综合指标
        trend_score = self._calculate_trend_score(trend_strength, trend_direction)
        volatility_score = self._calculate_volatility_score(atr_percent)
        rsi_score = self._calculate_rsi_score(rsi)

        # 判断 regime
        regime, confidence = self._determine_regime(
            trend_score, volatility_score, rsi_score, rsi, atr_percent
        )

        # 追踪 regime 变化
        self._track_regime_history(regime)

        return MarketRegimeState(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_level=volatility_score,
            rsi_level=rsi,
            atr_percent=atr_percent,
            trend_direction=trend_direction,
            regime_changes=self._regime_change_count,
            timestamp=market_data.get("timestamp", ""),
        )

    def _calculate_trend_score(self, trend_strength: float, direction: str) -> float:
        """计算趋势得分"""
        # 方向权重
        direction_multiplier = {"up": 1.0, "down": -1.0, "sideways": 0}.get(
            direction, 0
        )

        # 趋势强度已经包含了方向，这里做标准化处理
        if direction == "up":
            return min(1.0, max(0, trend_strength))
        elif direction == "down":
            return max(-1.0, min(0, trend_strength))
        else:
            return 0.0

    def _calculate_volatility_score(self, atr_percent: float) -> float:
        """计算波动率得分 (0-1)"""
        if atr_percent >= self.VOLATILITY_THRESHOLDS["high"]:
            return 1.0
        elif atr_percent <= self.VOLATILITY_THRESHOLDS["low"]:
            return 0.0
        else:
            return (atr_percent - self.VOLATILITY_THRESHOLDS["low"]) / (
                self.VOLATILITY_THRESHOLDS["high"] - self.VOLATILITY_THRESHOLDS["low"]
            )

    def _calculate_rsi_score(self, rsi: float) -> float:
        """计算RSI得分 (0-1, 0.5为中性)"""
        # RSI 30-70 范围映射到 0-1
        normalized = (rsi - 30) / 40
        return max(0, min(1, normalized))

    def _determine_regime(
        self,
        trend_score: float,
        volatility_score: float,
        rsi_score: float,
        rsi: float,
        atr_percent: float,
    ) -> Tuple[MarketRegime, float]:
        """
        综合判断市场环境

        Returns:
            (regime, confidence)
        """
        confidence = 0.5  # 基础置信度

        # 1. 超卖检测 (高优先级)
        if rsi < self.RSI_THRESHOLDS["oversold"]:
            if volatility_score > 0.7:
                return MarketRegime.HIGH_VOLATILITY, 0.9
            return MarketRegime.OVERSOLD, 0.85

        # 2. 超买检测
        if rsi > self.RSI_THRESHOLDS["overbought"]:
            if volatility_score > 0.7:
                return MarketRegime.HIGH_VOLATILITY, 0.9
            return MarketRegime.OVERBOUGHT, 0.85

        # 3. 强趋势检测
        if abs(trend_score) > 0.5:
            confidence += 0.2
            if trend_score > 0:
                return MarketRegime.TREND_UP, min(0.95, confidence)
            return MarketRegime.TREND_DOWN, min(0.95, confidence)

        # 4. 弱趋势
        if abs(trend_score) > 0.2:
            confidence += 0.1
            return MarketRegime.TREND_SIDEWAYS, min(0.8, confidence)

        # 5. 波动率检测
        if volatility_score > 0.8:
            return MarketRegime.HIGH_VOLATILITY, 0.75
        if volatility_score < 0.3:
            return MarketRegime.LOW_VOLATILITY, 0.7

        # 6. 默认正常状态
        return MarketRegime.NORMAL, 0.6

    def _track_regime_history(self, current_regime: MarketRegime) -> None:
        """追踪 regime 历史变化"""
        if self._regime_history and self._regime_history[-1] != current_regime:
            self._regime_change_count += 1

        self._regime_history.append(current_regime)

        # 保持历史长度
        if len(self._regime_history) > 100:
            self._regime_history.pop(0)

    def get_recent_regimes(self, count: int = 10) -> list[MarketRegime]:
        """获取最近的 regime 历史"""
        return self._regime_history[-count:]

    def reset(self) -> None:
        """重置状态"""
        self._regime_history = []
        self._regime_change_count = 0
