"""
AlphaPulse AI分析器
按需调用AI进行信号验证和优化
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import ConfigManager
from .config import AlphaPulseConfig
from .data_manager import DataManager
from .market_monitor import TechnicalIndicatorResult
from .signal_validator import ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class AIAnalysisResult:
    """AI分析结果"""

    signal: str  # "buy", "sell", "hold"
    confidence: float
    reasoning: str
    risk_assessment: str
    recommended_tp: float
    recommended_sl: float
    recommended_position: float
    market_context: str
    warnings: List[str]
    raw_response: str


class AIAnalyzer:
    """
    AI分析器

    功能:
    - 接收技术指标数据和系统判断
    - 调用AI进行深度分析
    - 返回优化后的交易建议
    """

    def __init__(
        self,
        config: AlphaPulseConfig,
        data_manager: DataManager,
        ai_manager=None,
    ):
        """
        初始化AI分析器

        Args:
            config: AlphaPulse配置
            data_manager: 数据管理器
            ai_manager: AI管理器实例（可选，用于调用现有AI服务）
        """
        self.config = config
        self.data_manager = data_manager
        self.ai_manager = ai_manager

        # 提示词模板
        self._prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """加载提示词模板"""
        return """
你是一个专业的加密货币交易分析师。请根据以下市场数据分析并给出交易建议：

## 当前市场状态
- 交易对: {symbol}
- 当前价格: ${current_price:.2f}
- 24h区间: ${low_24h:.2f} - ${high_24h:.2f} (当前位于{pos_24h:.1f}%)
- 7d区间: ${low_7d:.2f} - ${high_7d:.2f} (当前位于{pos_7d:.1f}%)

## 技术指标
- RSI(14): {rsi:.1f} ({rsi_status})
- MACD: {macd:.4f}, Signal: {macd_signal:.4f}, Histogram: {macd_hist:.4f}
- ADX: {adx:.1f} (趋势强度: {adx_strength})
- 布林带: Upper ${bb_upper:.2f}, Middle ${bb_middle:.2f}, Lower ${bb_lower:.2f}
  - 价格在布林带位置: {bb_position:.1f}%
- ATR: {atr:.2f} ({atr_percent:.2f}%)

## 系统初步判断
- 信号类型: {signal_type}
- 系统置信度: {confidence:.2%}
- 触发因素: {triggers}

## 趋势分析
- 趋势方向: {trend_direction}
- 趋势强度: {trend_strength:.2f}
- 1h价格变化: {1h_change:.2f}%

## 分析要求
请从以下维度进行分析：
1. 验证系统判断是否合理
2. 评估当前市场风险
3. 给出最终交易建议
4. 推荐止盈止损比例
5. 评估仓位大小

请以JSON格式返回分析结果：
{{
    "signal": "buy/sell/hold",
    "confidence": 0.0-1.0,
    "reasoning": "详细分析理由",
    "risk_assessment": "低/中/高",
    "recommended_tp": 百分比,
    "recommended_sl": 百分比,
    "recommended_position": 0.1-1.0,
    "market_context": "市场概况",
    "warnings": ["警告1", "警告2"]
}}
"""

    async def analyze(
        self,
        symbol: str,
        indicator: TechnicalIndicatorResult,
        validation_result: ValidationResult,
    ) -> Optional[AIAnalysisResult]:
        """
        调用AI进行分析

        Args:
            symbol: 交易对
            indicator: 技术指标结果
            validation_result: 系统验证结果

        Returns:
            AI分析结果
        """
        try:
            # 构建提示词
            prompt = self._build_prompt(symbol, indicator, validation_result)

            # 调用AI
            if self.ai_manager:
                response = await self._call_ai_manager(prompt)
            else:
                # 使用备用方式（直接调用模拟分析）
                response = await self._mock_ai_analysis(indicator, validation_result)

            # 解析结果
            result = self._parse_response(response, indicator, validation_result)

            logger.info(
                f"AI分析结果: {symbol} - {result.signal} "
                f"(置信度: {result.confidence:.2f})"
            )

            return result

        except Exception as e:
            logger.error(f"AI分析失败: {e}")
            return None

    def _build_prompt(
        self,
        symbol: str,
        indicator: TechnicalIndicatorResult,
        validation_result: ValidationResult,
    ) -> str:
        """构建AI提示词"""
        # 获取趋势数据
        trend_data = self.data_manager.get_trend_analysis(symbol, "1h", 20)

        # RSI状态
        if indicator.rsi < 30:
            rsi_status = "超卖"
        elif indicator.rsi < 40:
            rsi_status = "偏弱"
        elif indicator.rsi < 60:
            rsi_status = "中性"
        elif indicator.rsi < 70:
            rsi_status = "偏强"
        else:
            rsi_status = "超买"

        # ADX强度
        if indicator.adx < 20:
            adx_strength = "弱"
        elif indicator.adx < 40:
            adx_strength = "中等"
        else:
            adx_strength = "强"

        # 填充模板
        prompt = self._prompt_template.format(
            symbol=symbol,
            current_price=indicator.current_price,
            high_24h=indicator.high_24h,
            low_24h=indicator.low_24h,
            pos_24h=indicator.price_position_24h,
            high_7d=indicator.high_7d,
            low_7d=indicator.low_7d,
            pos_7d=indicator.price_position_7d,
            rsi=indicator.rsi,
            rsi_status=rsi_status,
            macd=indicator.macd,
            macd_signal=indicator.macd_signal,
            macd_hist=indicator.macd_histogram,
            adx=indicator.adx,
            adx_strength=adx_strength,
            bb_upper=indicator.bb_upper,
            bb_middle=indicator.bb_middle,
            bb_lower=indicator.bb_lower,
            bb_position=indicator.bb_position,
            atr=indicator.atr,
            atr_percent=indicator.atr_percent,
            signal_type=validation_result.signal_type.upper(),
            confidence=validation_result.confidence,
            triggers=", ".join(validation_result.score_details.keys()),
            trend_direction=indicator.trend_direction,
            trend_strength=indicator.trend_strength,
            **trend_data,
        )

        return prompt

    async def _call_ai_manager(self, prompt: str) -> str:
        """调用AI管理器"""
        try:
            # 使用现有的AI管理器
            signal = await self.ai_manager.generate_trading_signal(prompt)

            # 构建响应格式
            response = json.dumps(
                {
                    "signal": (
                        signal.signal.value
                        if hasattr(signal.signal, "value")
                        else signal.signal
                    ),
                    "confidence": signal.confidence,
                    "reasoning": signal.reasoning,
                    "risk_assessment": "中",
                    "recommended_tp": 2.0,
                    "recommended_sl": 1.0,
                    "recommended_position": 0.5,
                    "market_context": "AI分析完成",
                    "warnings": [],
                }
            )

            return response

        except Exception as e:
            logger.error(f"调用AI管理器失败: {e}")
            raise

    async def _mock_ai_analysis(
        self,
        indicator: TechnicalIndicatorResult,
        validation_result: ValidationResult,
    ) -> str:
        """模拟AI分析（当没有AI服务时使用）"""
        # 基于系统判断进行简单的AI模拟分析
        signal = validation_result.signal_type
        confidence = validation_result.confidence

        # 根据技术指标调整置信度
        if signal == "buy":
            if indicator.rsi < 30:
                confidence = min(0.95, confidence + 0.1)
            if indicator.bb_position < 15:
                confidence = min(0.95, confidence + 0.05)
            if indicator.trend_direction == "up":
                confidence = min(0.95, confidence + 0.05)
        elif signal == "sell":
            if indicator.rsi > 70:
                confidence = min(0.95, confidence + 0.1)
            if indicator.bb_position > 85:
                confidence = min(0.95, confidence + 0.05)
            if indicator.trend_direction == "down":
                confidence = min(0.95, confidence + 0.05)

        # 生成分析理由
        if signal == "buy":
            reasoning = (
                f"基于技术分析，当前价格{indicator.price_position_24h:.1f}%位于24h区间低位，"
                f"RSI为{indicator.rsi:.1f}显示{'超卖' if indicator.rsi < 30 else '偏弱'}状态，"
                f"布林带位置{indicator.bb_position:.1f}%靠近下轨，整体信号偏多。"
            )
        elif signal == "sell":
            reasoning = (
                f"基于技术分析，当前价格{indicator.price_position_24h:.1f}%位于24h区间高位，"
                f"RSI为{indicator.rsi:.1f}显示{'超买' if indicator.rsi > 70 else '偏强'}状态，"
                f"布林带位置{indicator.bb_position:.1f}%靠近上轨，整体信号偏空。"
            )
        else:
            reasoning = "技术指标显示市场方向不明确，建议观望。"
            confidence = 0.3

        # 风险评估
        if indicator.atr_percent > 2:
            risk = "高"
            warnings = ["市场波动较大"]
        elif indicator.atr_percent > 1:
            risk = "中"
            warnings = []
        else:
            risk = "低"
            warnings = []

        if abs(indicator.macd_histogram) < 0.01:
            warnings.append("MACD信号较弱")

        # 推荐止盈止损
        tp = max(1.5, indicator.atr_percent * 1.5)
        sl = max(0.8, indicator.atr_percent * 0.8)

        # 推荐仓位
        position = 0.5
        if confidence > 0.7:
            position = 0.7
        if confidence > 0.85:
            position = 1.0
        if indicator.trend_strength < 0.3:
            position *= 0.7

        response = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "risk_assessment": risk,
            "recommended_tp": tp,
            "recommended_sl": sl,
            "recommended_position": position,
            "market_context": f"趋势{indicator.trend_direction}，强度{indicator.trend_strength:.2f}",
            "warnings": warnings,
        }

        return json.dumps(response)

    def _parse_response(
        self,
        response: str,
        indicator: TechnicalIndicatorResult,
        validation_result: ValidationResult,
    ) -> AIAnalysisResult:
        """解析AI响应"""
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"AI响应解析失败: {response}")
            # 使用默认结果
            return AIAnalysisResult(
                signal=validation_result.signal_type,
                confidence=validation_result.confidence,
                reasoning="AI响应解析失败，使用系统判断",
                risk_assessment="中",
                recommended_tp=2.0,
                recommended_sl=1.0,
                recommended_position=0.5,
                market_context="解析失败",
                warnings=["AI响应格式异常"],
                raw_response=response,
            )

        return AIAnalysisResult(
            signal=data.get("signal", validation_result.signal_type),
            confidence=data.get("confidence", validation_result.confidence),
            reasoning=data.get("reasoning", "AI分析完成"),
            risk_assessment=data.get("risk_assessment", "中"),
            recommended_tp=data.get("recommended_tp", 2.0),
            recommended_sl=data.get("recommended_sl", 1.0),
            recommended_position=data.get("recommended_position", 0.5),
            market_context=data.get("market_context", ""),
            warnings=data.get("warnings", []),
            raw_response=response,
        )

    def should_execute(
        self,
        validation_result: ValidationResult,
        ai_result: AIAnalysisResult,
    ) -> tuple[bool, str]:
        """
        判断是否应该执行交易

        Args:
            validation_result: 系统验证结果
            ai_result: AI分析结果

        Returns:
            (是否执行, 原因)
        """
        # 1. 检查AI置信度
        if ai_result.confidence < self.config.min_ai_confidence:
            return False, f"AI置信度过低 ({ai_result.confidence:.2f})"

        # 2. 检查系统与AI信号一致性
        if ai_result.signal != validation_result.signal_type:
            # 如果AI信号相反
            if ai_result.signal == "hold":
                return False, "AI建议观望"
            # AI与系统判断不同，以AI为准但降低置信度
            logger.warning(
                f"AI信号与系统不一致: 系统={validation_result.signal_type}, AI={ai_result.signal}"
            )

        # 3. 检查风险
        if ai_result.risk_assessment == "高":
            logger.warning(f"AI评估风险为高: {ai_result.warnings}")
            # 仍然可以执行，但需要更高的确认度
            if ai_result.confidence < 0.8:
                return False, "高风险且置信度不足"

        # 4. 检查警告
        for warning in ai_result.warnings:
            if "极端" in warning or "剧烈" in warning:
                return False, f"AI警告: {warning}"

        return True, "信号验证通过"

    def get_execution_params(
        self, ai_result: AIAnalysisResult, base_params: Dict = None
    ) -> Dict[str, Any]:
        """
        获取执行参数

        Args:
            ai_result: AI分析结果
            base_params: 基础参数

        Returns:
            执行参数
        """
        if base_params is None:
            base_params = {}

        # 合并参数
        params = {
            "take_profit_percent": ai_result.recommended_tp,
            "stop_loss_percent": ai_result.recommended_sl,
            "position_ratio": ai_result.recommended_position,
            "signal": ai_result.signal,
            "confidence": ai_result.confidence,
            "ai_risk_assessment": ai_result.risk_assessment,
        }

        # 合并基础参数
        params.update(base_params)

        return params
