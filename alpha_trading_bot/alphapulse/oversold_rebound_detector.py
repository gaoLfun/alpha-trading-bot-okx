"""
超卖反弹检测器 - 专门检测价格极低位 + RSI超卖时的反弹买入机会

功能：
- 检测价格是否处于极低位（24h/7d < 15%）
- 检测 RSI 是否处于超卖区域（< 30）
- 检测反弹信号（RSI回升、MACD收窄、价格止跌等）
- 在传统 AlphaPulse 系统可能误判时提供独立的 BUY 信号

典型场景：
- 07:15 价格 88131.30，RSI=27.7，24h位置=1.1%
- AlphaPulse 可能给出 SELL（因为 MACD 仍为负值）
- OversoldReboundDetector 应该识别为买入机会
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ReboundCheckResult:
    """超卖反弹检测结果"""

    is_rebound_opportunity: bool
    signal_type: str  # "buy", "hold"
    confidence: float  # 0.0 ~ 1.0
    triggers: List[str]  # 触发原因列表
    risk_level: str  # "low", "medium", "high"
    analysis: Dict[str, Any]  # 详细分析数据
    message: str  # 人类可读的消息


@dataclass
class OversoldReboundDetector:
    """
    超卖反弹检测器

    专门用于检测价格处于极低位 + RSI超卖时的反弹买入机会。

    检测逻辑：
    1. 基础条件：价格位置 < 15% 且 RSI < 30
    2. 反弹条件（满足2个及以上）：
       - RSI 正在回升（与前一根K线比较）
       - MACD 柱状图收窄（从极端值恢复）
       - 价格止跌（接近布林带下轨但未创新低或已回升）
       - ADX > 25（确认趋势方向）
       - 成交量放大（确认资金流入）
    3. 风险控制：
       - 极度超卖仍可能继续下跌
       - 需要多重信号确认
    """

    # 极低位阈值配置
    EXTREME_LOW_THRESHOLD: float = 15.0  # 价格位置 < 15% 视为极低位
    OVERSOLD_RSI_THRESHOLD: float = 30.0  # RSI < 30 视为超卖

    # 反弹信号配置
    MIN_REBOUND_SIGNALS: int = 2  # 最少需要满足的反弹信号数量

    # 反弹信号阈值
    MACD_HISTOGRAM_THRESHOLD: float = 50.0  # MACD柱状图绝对值 < 50 视为收窄
    BB_POSITION_REBOUND: float = 30.0  # BB位置 > 30% 视为脱离底部
    ADX_CONFIRMATION: float = 25.0  # ADX > 25 确认趋势
    VOLUME_SPIKE_RATIO: float = 1.2  # 成交量 > 1.2倍均值视为放大

    # 风险等级配置
    RISK_LEVELS = {
        "low": {
            "min_price_position": 0.0,
            "max_price_position": 10.0,
            "min_rsi": 0.0,
            "max_rsi": 25.0,
            "min_adx": 35.0,
            "confidence_base": 0.75,
        },
        "medium": {
            "min_price_position": 10.0,
            "max_price_position": 15.0,
            "min_rsi": 25.0,
            "max_rsi": 30.0,
            "min_adx": 25.0,
            "confidence_base": 0.65,
        },
        "high": {
            "min_price_position": 0.0,
            "max_price_position": 15.0,
            "min_rsi": 0.0,
            "max_rsi": 30.0,
            "min_adx": 0.0,
            "max_adx": 25.0,  # ADX 太低说明无趋势
            "confidence_base": 0.50,
        },
    }

    # 上一根K线的指标值（用于检测趋势变化）
    _prev_indicators: Dict[str, Any] = field(default_factory=dict)

    def __init__(self) -> None:
        """初始化超卖反弹检测器"""
        self._prev_indicators = {}
        logger.info(
            f"✅ OversoldReboundDetector 已初始化: "
            f"极低位阈值={self.EXTREME_LOW_THRESHOLD}%, "
            f"超卖RSI={self.OVERSOLD_RSI_THRESHOLD}, "
            f"最少反弹信号={self.MIN_REBOUND_SIGNALS}"
        )

    def reset_prev_indicators(self) -> None:
        """重置历史指标（通常在新的监控周期开始时调用）"""
        # 保留历史数据用于趋势比较，不完全清空
        pass

    def _get_risk_level(
        self,
        price_position_24h: float,
        price_position_7d: float,
        rsi: float,
        adx: float,
        rebound_signals: int,
    ) -> Tuple[str, float]:
        """
        计算风险等级

        Args:
            price_position_24h: 24h价格位置 (0-100)
            price_position_7d: 7d价格位置 (0-100)
            rsi: RSI值 (0-100)
            adx: ADX值 (0-100)
            rebound_signals: 满足的反弹信号数量

        Returns:
            (风险等级, 基础信心度)
        """
        # 计算综合价格位置
        composite_position = price_position_24h * 0.7 + price_position_7d * 0.3

        # 判断风险级别
        if (
            composite_position < 10.0
            and rsi < 25.0
            and adx >= self.ADX_CONFIRMATION
            and rebound_signals >= 3
        ):
            return "low", self.RISK_LEVELS["low"]["confidence_base"]
        elif (
            composite_position < self.EXTREME_LOW_THRESHOLD
            and rsi < self.OVERSOLD_RSI_THRESHOLD
            and adx >= self.ADX_CONFIRMATION
        ):
            return "medium", self.RISK_LEVELS["medium"]["confidence_base"]
        else:
            return "high", self.RISK_LEVELS["high"]["confidence_base"]

    def _calculate_rsi_trend(
        self, current_rsi: float, prev_rsi: Optional[float]
    ) -> Tuple[bool, str]:
        """
        计算 RSI 趋势

        Args:
            current_rsi: 当前RSI值
            prev_rsi: 上一根K线的RSI值

        Returns:
            (是否回升, 趋势描述)
        """
        if prev_rsi is None:
            return False, "无历史数据"

        if current_rsi > prev_rsi:
            # 回升幅度
            rise_pct = (current_rsi - prev_rsi) / prev_rsi * 100
            if rise_pct > 5:
                return True, f"RSI明显回升{rise_pct:.1f}%"
            else:
                return True, f"RSI微升{rise_pct:.1f}%"
        elif current_rsi == prev_rsi:
            return False, "RSI持平"
        else:
            return False, (
                f"RSI继续下降{abs(rise_pct):.1f}%"
                if (rise_pct := (prev_rsi - current_rsi) / current_rsi * 100)
                else "RSI下降"
            )

    def _check_macd_trending_up(
        self, macd_histogram: float, prev_macd_histogram: Optional[float]
    ) -> Tuple[bool, str]:
        """
        检查 MACD 柱状图是否正在收窄/转正

        Args:
            macd_histogram: 当前MACD柱状图值
            prev_macd_histogram: 上一根的MACD柱状图值

        Returns:
            (是否收窄, 趋势描述)
        """
        if prev_macd_histogram is None:
            # 无历史数据时，判断绝对值
            if abs(macd_histogram) < self.MACD_HISTOGRAM_THRESHOLD:
                return True, f"MACD柱状图温和({macd_histogram:.2f})"
            return False, f"MACD柱状图仍为负值({macd_histogram:.2f})"

        # 检查是否正在收窄（负值变小或转正）
        if macd_histogram > prev_macd_histogram:
            # 收窄或转正
            improvement = macd_histogram - prev_macd_histogram
            if macd_histogram > 0:
                return True, f"MACD转正(+{macd_histogram:.2f})"
            elif improvement > 10:
                return True, f"MACD柱状图大幅收窄(+{improvement:.1f})"
            else:
                return True, f"MACD柱状图收窄(+{improvement:.1f})"
        else:
            # 继续扩大
            expansion = prev_macd_histogram - macd_histogram
            return False, f"MACD柱状图继续扩大({expansion:.1f})"

    def _check_price_rebound(
        self, bb_position: float, current_price: float, prev_price: Optional[float]
    ) -> Tuple[bool, str]:
        """
        检查价格是否止跌回升

        Args:
            bb_position: 布林带位置 (0-100)
            current_price: 当前价格
            prev_price: 上一根K线的价格

        Returns:
            (是否止跌回升, 趋势描述)
        """
        if bb_position > self.BB_POSITION_REBOUND:
            # 已经脱离底部区域
            if prev_price is not None and current_price > prev_price:
                return (
                    True,
                    f"价格回升(+{(current_price - prev_price) / prev_price * 100:.2f}%)",
                )
            elif prev_price is not None and current_price == prev_price:
                return True, "价格企稳"
            else:
                return True, f"BB位置{bb_position:.1f}%脱离底部"
        elif bb_position > 15:
            # 接近底部但未完全脱离
            return False, f"BB位置{bb_position:.1f}%仍在底部区域"
        else:
            # 处于极端底部
            if prev_price is not None and current_price > prev_price:
                return (
                    True,
                    f"价格开始回升(+{(current_price - prev_price) / prev_price * 100:.2f}%)",
                )
            return False, f"BB位置{bb_position:.1f}%处于极端底部"

    def _check_adx_confirmation(self, adx: float) -> Tuple[bool, str]:
        """
        检查 ADX 是否确认趋势（反弹有方向）

        Args:
            adx: ADX值

        Returns:
            (是否确认趋势, 趋势描述)
        """
        if adx >= self.ADX_CONFIRMATION:
            if adx >= 40:
                return True, f"ADX={adx:.1f} 强趋势确认"
            elif adx >= 30:
                return True, f"ADX={adx:.1f} 中趋势确认"
            else:
                return True, f"ADX={adx:.1f} 弱趋势确认"
        else:
            return False, f"ADX={adx:.1f} 趋势不明确"

    def _check_volume_confirmation(
        self, volume: float, avg_volume: float
    ) -> Tuple[bool, str]:
        """
        检查成交量是否放大（确认资金流入）

        Args:
            volume: 当前成交量
            avg_volume: 平均成交量

        Returns:
            (是否放大, 描述)
        """
        if avg_volume <= 0:
            return False, "无成交量数据"

        volume_ratio = volume / avg_volume

        if volume_ratio >= self.VOLUME_SPIKE_RATIO:
            return True, f"成交量放大{volume_ratio:.1f}倍"
        elif volume_ratio >= 0.8:
            return False, f"成交量正常({volume_ratio:.1f}倍)"
        else:
            return False, f"成交量萎缩({volume_ratio:.1f}倍)"

    def check_rebound(
        self,
        indicator_result: Any,
        prev_indicator_result: Optional[Any] = None,
    ) -> ReboundCheckResult:
        """
        检查是否处于超卖反弹买入时机

        Args:
            indicator_result: 当前技术指标结果 (TechnicalIndicatorResult)
            prev_indicator_result: 上一根K线的指标结果（可选，用于检测趋势）

        Returns:
            ReboundCheckResult: 检测结果
        """
        try:
            # 提取当前指标
            current_price = indicator_result.current_price
            price_position_24h = indicator_result.price_position_24h
            price_position_7d = indicator_result.price_position_7d
            rsi = indicator_result.rsi
            bb_position = indicator_result.bb_position
            macd_histogram = indicator_result.macd_histogram
            adx = indicator_result.adx

            # 提取历史指标（用于趋势比较）
            prev_rsi = (
                prev_indicator_result.rsi
                if prev_indicator_result
                else self._prev_indicators.get("rsi")
            )
            prev_macd = (
                prev_indicator_result.macd_histogram
                if prev_indicator_result
                else self._prev_indicators.get("macd_histogram")
            )
            prev_price = (
                prev_indicator_result.current_price
                if prev_indicator_result
                else self._prev_indicators.get("price")
            )

            # 保存当前指标作为历史
            self._prev_indicators = {
                "rsi": rsi,
                "macd_histogram": macd_histogram,
                "price": current_price,
            }

            # 1. 检查基础条件：极低位 + 超卖
            is_extreme_low = (
                price_position_24h < self.EXTREME_LOW_THRESHOLD
                and price_position_7d < self.EXTREME_LOW_THRESHOLD
            )
            is_oversold = rsi < self.OVERSOLD_RSI_THRESHOLD

            analysis = {
                "current_price": current_price,
                "price_position_24h": price_position_24h,
                "price_position_7d": price_position_7d,
                "composite_position": price_position_24h * 0.7
                + price_position_7d * 0.3,
                "rsi": rsi,
                "bb_position": bb_position,
                "macd_histogram": macd_histogram,
                "adx": adx,
                "is_extreme_low": is_extreme_low,
                "is_oversold": is_oversold,
                "rebound_signals": [],
            }

            # 基础条件不满足，不触发反弹检测
            if not (is_extreme_low and is_oversold):
                return ReboundCheckResult(
                    is_rebound_opportunity=False,
                    signal_type="hold",
                    confidence=0.0,
                    triggers=[],
                    risk_level="none",
                    analysis=analysis,
                    message=f"基础条件不满足: 极低位={is_extreme_low}, 超卖={is_oversold}",
                )

            # 2. 检查反弹信号
            rebound_signals = []
            trend_details = []

            # 2.1 RSI 趋势
            rsi_rising, rsi_detail = self._calculate_rsi_trend(rsi, prev_rsi)
            if rsi_rising:
                rebound_signals.append("RSI回升")
                trend_details.append(rsi_detail)

            # 2.2 MACD 趋势
            macd_narrowing, macd_detail = self._check_macd_trending_up(
                macd_histogram, prev_macd
            )
            if macd_narrowing:
                rebound_signals.append("MACD收窄")
                trend_details.append(macd_detail)

            # 2.3 价格趋势
            price_rebound, price_detail = self._check_price_rebound(
                bb_position, current_price, prev_price
            )
            if price_rebound:
                rebound_signals.append("价格回升")
                trend_details.append(price_detail)

            # 2.4 ADX 确认
            adx_confirmed, adx_detail = self._check_adx_confirmation(adx)
            if adx_confirmed:
                rebound_signals.append("ADX确认趋势")
                trend_details.append(adx_detail)

            # 3. 计算风险等级和信心度
            risk_level, base_confidence = self._get_risk_level(
                price_position_24h,
                price_position_7d,
                rsi,
                adx,
                len(rebound_signals),
            )

            # 4. 根据反弹信号数量调整信心度
            if len(rebound_signals) >= 4:
                confidence = min(base_confidence + 0.2, 0.95)
            elif len(rebound_signals) >= 3:
                confidence = min(base_confidence + 0.1, 0.90)
            elif len(rebound_signals) >= 2:
                confidence = base_confidence
            else:
                confidence = base_confidence - 0.15

            # 更新分析结果
            analysis["rebound_signals"] = rebound_signals
            analysis["trend_details"] = trend_details
            analysis["risk_level"] = risk_level
            analysis["base_confidence"] = base_confidence
            analysis["final_confidence"] = confidence

            # 5. 决定是否触发 BUY 信号
            if len(rebound_signals) >= self.MIN_REBOUND_SIGNALS:
                message = (
                    f"🎯 超卖反弹买入机会！"
                    f"价格位置={price_position_24h:.1f}%, "
                    f"RSI={rsi:.1f}, "
                    f"反弹信号={len(rebound_signals)}个: {'/'.join(rebound_signals)}, "
                    f"信心度={confidence:.2f}"
                )

                logger.info(f"✅ OversoldReboundDetector: {message}")

                return ReboundCheckResult(
                    is_rebound_opportunity=True,
                    signal_type="buy",
                    confidence=confidence,
                    triggers=rebound_signals,
                    risk_level=risk_level,
                    analysis=analysis,
                    message=message,
                )
            else:
                message = (
                    f"⏳ 超卖区域观察中: "
                    f"价格位置={price_position_24h:.1f}%, "
                    f"RSI={rsi:.1f}, "
                    f"反弹信号不足({len(rebound_signals)}/{self.MIN_REBOUND_SIGNALS})"
                )

                logger.debug(f"⏳ OversoldReboundDetector: {message}")

                return ReboundCheckResult(
                    is_rebound_opportunity=False,
                    signal_type="hold",
                    confidence=0.0,
                    triggers=rebound_signals,
                    risk_level=risk_level,
                    analysis=analysis,
                    message=message,
                )

        except Exception as e:
            logger.error(f"❌ OversoldReboundDetector 检测失败: {e}")
            return ReboundCheckResult(
                is_rebound_opportunity=False,
                signal_type="hold",
                confidence=0.0,
                triggers=[],
                risk_level="unknown",
                analysis={},
                message=f"检测失败: {str(e)}",
            )


def create_oversold_rebound_detector() -> OversoldReboundDetector:
    """创建超卖反弹检测器的工厂函数"""
    return OversoldReboundDetector()
