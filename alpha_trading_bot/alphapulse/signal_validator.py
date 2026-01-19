"""
AlphaPulse 信号验证器
综合技术指标，决定是否触发AI分析
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import AlphaPulseConfig
from .market_monitor import TechnicalIndicatorResult, SignalCheckResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """验证结果"""

    passed: bool
    signal_type: str  # "buy", "sell", "hold"
    confidence: float
    score_details: Dict[str, float]
    warnings: List[str]
    recommendations: List[str]
    final_message: str


class SignalValidator:
    """
    信号验证器

    功能:
    - 综合多个维度的指标验证
    - 趋势方向确认
    - 风险评估
    - 决定是否需要AI介入
    """

    def __init__(self, config: AlphaPulseConfig):
        """
        初始化信号验证器

        Args:
            config: AlphaPulse配置
        """
        self.config = config
        self._thresholds = config.get_thresholds()
        self._indicator_params = config.get_indicator_params()

    async def validate(
        self,
        symbol: str,
        signal_result: SignalCheckResult,
        market_summary: Dict[str, Any] = None,
    ) -> ValidationResult:
        """
        验证交易信号

        Args:
            symbol: 交易对
            signal_result: MarketMonitor的信号检查结果
            market_summary: 市场摘要（可选）

        Returns:
            验证结果
        """
        # 基础分数
        buy_score = signal_result.buy_score
        sell_score = signal_result.sell_score
        indicator = signal_result.indicator_result

        # 初始化详细信息
        score_details = {}
        warnings = []
        recommendations = []

        # 1. 基础阈值检查
        base_threshold_passed = False
        if signal_result.signal_type == "buy":
            base_threshold_passed = buy_score >= self.config.buy_threshold
            score_details["基础BUY分数"] = buy_score
            score_details["BUY阈值"] = self.config.buy_threshold
        elif signal_result.signal_type == "sell":
            base_threshold_passed = sell_score >= self.config.sell_threshold
            score_details["基础SELL分数"] = sell_score
            score_details["SELL阈值"] = self.config.sell_threshold

        if not base_threshold_passed:
            return ValidationResult(
                passed=False,
                signal_type="hold",
                confidence=0,
                score_details=score_details,
                warnings=warnings,
                recommendations=recommendations,
                final_message=f"基础分数未达到阈值 ({buy_score:.2f}/{sell_score:.2f} vs {self.config.buy_threshold:.2f})",
            )

        # 2. 趋势方向确认
        trend_score = self._validate_trend_direction(
            signal_result.signal_type, indicator
        )
        score_details["趋势确认分数"] = trend_score
        warnings.extend(self._get_trend_warnings(indicator))

        if trend_score < 0.3:
            return ValidationResult(
                passed=False,
                signal_type="hold",
                confidence=trend_score,
                score_details=score_details,
                warnings=warnings,
                recommendations=["等待趋势更加明确"],
                final_message=f"趋势方向不确认 ({indicator.trend_direction})",
            )

        # 3. RSI验证
        rsi_score = self._validate_rsi(signal_result.signal_type, indicator.rsi)
        score_details["RSI验证分数"] = rsi_score
        if rsi_score < 0.1:
            warnings.append(f"RSI状态不理想: {indicator.rsi:.1f}")

        # 4. 波动率验证
        volatility_score = self._validate_volatility(indicator.atr_percent)
        score_details["波动率分数"] = volatility_score

        if volatility_score < 0.1:
            warnings.append(f"市场波动率过低: {indicator.atr_percent:.2f}%")
            recommendations.append("等待更大波动再入场")

        # 5. 位置验证
        position_score = self._validate_position(
            signal_result.signal_type,
            indicator.price_position_24h,
            indicator.price_position_7d,
        )
        score_details["位置分数"] = position_score

        # 6. 布林带验证
        bb_score = self._validate_bollinger_band(
            signal_result.signal_type, indicator.bb_position
        )
        score_details["布林带分数"] = bb_score

        # 7. ADX趋势强度验证
        adx_score = self._validate_adx(indicator.adx)
        score_details["ADX分数"] = adx_score
        if adx_score < 0.1 and indicator.adx < 20:
            warnings.append(f"ADX过低，趋势不明显: {indicator.adx:.1f}")

        # 计算综合分数
        # 权重: 基础50% + 趋势20% + RSI10% + 波动率5% + 位置5% + 布林带5% + ADX5%
        final_confidence = (
            (buy_score if signal_result.signal_type == "buy" else sell_score) * 0.50
            + trend_score * 0.20
            + rsi_score * 0.10
            + volatility_score * 0.05
            + position_score * 0.05
            + bb_score * 0.05
            + adx_score * 0.05
        )

        # 最终验证
        passed = (
            final_confidence >= 0.5 and trend_score >= 0.3 and base_threshold_passed
        )

        if passed:
            recommendations.extend(self._get_recommendations(signal_result, indicator))

        final_message = self._generate_final_message(
            signal_result.signal_type,
            final_confidence,
            passed,
            warnings,
            recommendations,
        )

        return ValidationResult(
            passed=passed,
            signal_type=signal_result.signal_type,
            confidence=final_confidence,
            score_details=score_details,
            warnings=warnings,
            recommendations=recommendations,
            final_message=final_message,
        )

    def _validate_trend_direction(
        self, signal_type: str, indicator: TechnicalIndicatorResult
    ) -> float:
        """验证趋势方向与信号方向是否一致"""
        trend = indicator.trend_direction
        strength = indicator.trend_strength

        if signal_type == "buy":
            if trend in ["up", "sideways"]:
                return strength if trend == "up" else strength * 0.5
            return 0.1  # 逆趋势
        else:  # sell
            if trend in ["down", "sideways"]:
                return strength if trend == "down" else strength * 0.5
            return 0.1  # 逆趋势

    def _get_trend_warnings(self, indicator: TechnicalIndicatorResult) -> List[str]:
        """获取趋势相关警告"""
        warnings = []

        if indicator.trend_strength < 0.3:
            warnings.append(f"趋势强度较弱: {indicator.trend_strength:.2f}")

        if indicator.trend_direction == "sideways":
            warnings.append("市场处于横盘状态")

        return warnings

    def _validate_rsi(self, signal_type: str, rsi: float) -> float:
        """验证RSI状态"""
        if signal_type == "buy":
            if rsi < 30:
                return 1.0  # 超卖，非常好
            elif rsi < 40:
                return 0.7  # 偏弱
            elif rsi < 50:
                return 0.3  # 中性
            return 0.0  # 不适合买入
        else:  # sell
            if rsi > 70:
                return 1.0  # 超买，非常好
            elif rsi > 60:
                return 0.7  # 偏强
            elif rsi > 50:
                return 0.3  # 中性
            return 0.0  # 不适合卖出

    def _validate_volatility(self, atr_percent: float) -> float:
        """验证波动率是否足够"""
        if atr_percent < 0.2:
            return 0.1  # 波动太低
        elif atr_percent < 0.5:
            return 0.5  # 适中
        elif atr_percent < 1.0:
            return 0.8  # 良好
        return 1.0  # 高波动，适合交易

    def _validate_position(
        self, signal_type: str, pos_24h: float, pos_7d: float
    ) -> float:
        """验证价格位置是否合适"""
        avg_position = (pos_24h + pos_7d) / 2

        if signal_type == "buy":
            if avg_position < 20:
                return 1.0  # 非常好的买入位置
            elif avg_position < 35:
                return 0.7  # 不错
            elif avg_position < 50:
                return 0.3  # 一般
            return 0.0  # 位置太高
        else:  # sell
            if avg_position > 80:
                return 1.0  # 非常好的卖出位置
            elif avg_position > 65:
                return 0.7  # 不错
            elif avg_position > 50:
                return 0.3  # 一般
            return 0.0  # 位置太低

    def _validate_bollinger_band(self, signal_type: str, bb_position: float) -> float:
        """验证布林带位置"""
        if signal_type == "buy":
            if bb_position < 10:
                return 1.0  # 触及下轨
            elif bb_position < 25:
                return 0.7  # 靠近下轨
            elif bb_position < 40:
                return 0.3  # 中性
            return 0.0
        else:  # sell
            if bb_position > 90:
                return 1.0  # 触及上轨
            elif bb_position > 75:
                return 0.7  # 靠近上轨
            elif bb_position > 60:
                return 0.3  # 中性
            return 0.0

    def _validate_adx(self, adx: float) -> float:
        """验证ADX趋势强度"""
        if adx < 20:
            return 0.1  # 无趋势
        elif adx < 25:
            return 0.3  # 弱趋势
        elif adx < 40:
            return 0.7  # 中等趋势
        return 1.0  # 强趋势

    def _get_recommendations(
        self, signal_result: SignalCheckResult, indicator: TechnicalIndicatorResult
    ) -> List[str]:
        """生成交易建议"""
        recommendations = []

        # 止盈止损建议
        tp_percent = 2.0 if indicator.atr_percent < 1.0 else indicator.atr_percent * 2
        sl_percent = 1.0 if indicator.atr_percent < 1.0 else indicator.atr_percent

        recommendations.append(f"建议止盈: {tp_percent:.1f}%")
        recommendations.append(f"建议止损: {sl_percent:.1f}%")

        # 仓位建议
        if indicator.trend_strength < 0.5:
            recommendations.append("建议降低仓位比例")

        # 风险提示
        if indicator.atr_percent > 2:
            recommendations.append("市场波动较大，注意风险")

        return recommendations

    def _generate_final_message(
        self,
        signal_type: str,
        confidence: float,
        passed: bool,
        warnings: List[str],
        recommendations: List[str],
    ) -> str:
        """生成最终消息"""
        parts = []

        if not passed:
            parts.append(f"❌ 信号验证未通过")
            parts.append(f"置信度: {confidence:.2%}")
            if warnings:
                parts.append(f"警告: {'; '.join(warnings[:2])}")
            return "\n".join(parts)

        emoji = "🟢" if signal_type == "buy" else "🔴"
        parts.append(f"{emoji} {signal_type.upper()} 信号验证通过")
        parts.append(f"置信度: {confidence:.2%}")

        if warnings:
            parts.append(f"⚠️ 注意: {'; '.join(warnings[:2])}")

        if recommendations:
            parts.append(f"💡 建议: {recommendations[0]}")

        return "\n".join(parts)

    def should_use_ai(self, validation_result: ValidationResult) -> bool:
        """
        判断是否需要使用AI验证

        Args:
            validation_result: 验证结果

        Returns:
            是否需要AI验证
        """
        if not self.config.use_ai_validation:
            return False

        # 中等置信度时使用AI验证
        if 0.5 <= validation_result.confidence < self.config.min_ai_confidence:
            return True

        # 有警告时使用AI验证
        if len(validation_result.warnings) > 0:
            return True

        return False
