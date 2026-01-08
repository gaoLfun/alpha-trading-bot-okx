"""
AI融合引擎 - 专门负责多AI信号的融合和优化
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .fusion import AIFusion
from .signal_optimizer import SignalOptimizer
from .buy_signal_optimizer import BuySignalOptimizer
from ..config import load_config

logger = logging.getLogger(__name__)


class AIFusionEngine:
    """AI融合引擎 - 专门负责信号融合逻辑"""

    def __init__(self):
        self.ai_fusion = AIFusion()
        self.signal_optimizer = SignalOptimizer()
        self.buy_optimizer = BuySignalOptimizer()

    async def fuse_signals(
        self,
        signals: List[Dict[str, Any]],
        market_data: Dict[str, Any],
        fail_count: int = 0,
        total_providers: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        融合多个AI信号

        Args:
            signals: 原始信号列表
            market_data: 市场数据
            fail_count: 失败的提供商数量
            total_providers: 总提供商数量

        Returns:
            融合后的信号列表
        """
        try:
            if not signals:
                logger.warning("没有可用的信号进行融合")
                return []

            # 记录部分失败的情况
            if fail_count > 0:
                logger.info(
                    f"⚠️  部分提供商失败: {fail_count}/{total_providers}，使用{len(signals)}个成功信号进行融合"
                )

            config = load_config()

            # 信号优化
            optimized_signals = await self._optimize_signals(
                signals, market_data, config
            )

            # BUY信号专项优化
            if optimized_signals:
                optimized_signals = await self._optimize_buy_signals(
                    optimized_signals, market_data
                )

            # 执行融合
            if config.ai.ai_fusion_enabled and len(optimized_signals) > 1:
                return await self._perform_fusion(optimized_signals, config)
            else:
                # 不进行融合，返回优化后的信号
                logger.info("使用单个最优信号")
                return [self._select_best_signal(optimized_signals)]

        except Exception as e:
            logger.error(f"信号融合失败: {e}")
            # 返回原始信号中最优的一个
            if signals:
                return [self._select_best_signal(signals)]
            return []

    async def _optimize_signals(
        self, signals: List[Dict[str, Any]], market_data: Dict[str, Any], config
    ) -> List[Dict[str, Any]]:
        """优化信号"""
        try:
            # 价格位置缩放
            scaled_signals = []
            for signal in signals:
                confidence = signal.get("confidence", 0)
                if confidence > 0:
                    scaled_signal = await self._apply_price_position_scaling(
                        signal, market_data
                    )
                    if scaled_signal:
                        scaled_signals.append(scaled_signal)
                    else:
                        scaled_signals.append(signal)
                else:
                    scaled_signals.append(signal)

            # 动态置信度阈值调整
            dynamic_threshold = self._calculate_dynamic_confidence_threshold(
                market_data
            )
            filtered_signals = []

            for signal in scaled_signals:
                confidence = signal.get("confidence", 0)
                if confidence >= dynamic_threshold:
                    filtered_signals.append(signal)
                else:
                    logger.warning(
                        f"⚠️  信号置信度不足: {confidence:.2f} < {dynamic_threshold}"
                    )

            if not filtered_signals:
                # 如果所有信号都被过滤，保留置信度最高的
                best_signal = max(scaled_signals, key=lambda x: x.get("confidence", 0))
                filtered_signals = [best_signal]
                logger.info("所有信号置信度不足，保留置信度最高的信号")

            # 使用信号优化器
            if config.ai.enable_signal_optimization:
                logger.info("🔧 开始信号优化...")
                optimized_results = await self._call_signal_optimizer(
                    filtered_signals, market_data
                )
                if optimized_results:
                    logger.info(
                        f"✅ 信号优化完成，优化了 {len(optimized_results)} 个信号"
                    )
                    return optimized_results

            return filtered_signals

        except Exception as e:
            logger.error(f"信号优化失败: {e}")
            return signals

    async def _optimize_buy_signals(
        self, signals: List[Dict[str, Any]], market_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """BUY信号专项优化"""
        try:
            logger.info("🎯 开始BUY信号专项优化...")
            buy_optimized_results = self.buy_optimizer.optimize_buy_signals(
                signals, market_data
            )

            if buy_optimized_results:
                # 比较优化前后的变化
                buy_changes = self._compare_buy_changes(signals, buy_optimized_results)
                if buy_changes["changed_count"] > 0:
                    logger.info(
                        f"🎯 BUY信号优化: {buy_changes['changed_count']}个信号被优化"
                    )
                    if buy_changes["buy_to_hold_count"] > 0:
                        logger.info(
                            f"🔄 {buy_changes['buy_to_hold_count']}个BUY转为HOLD"
                        )
                    if buy_changes["confidence_changes"] > 0:
                        logger.info(
                            f"📊 {buy_changes['confidence_changes']}个信号信心度调整"
                        )
                return buy_optimized_results

            return signals

        except Exception as e:
            logger.error(f"BUY信号优化失败: {e}")
            return signals

    async def _perform_fusion(
        self, signals: List[Dict[str, Any]], config
    ) -> List[Dict[str, Any]]:
        """执行信号融合"""
        try:
            # 获取融合配置
            fusion_strategy = config.ai.ai_fusion_strategy
            fusion_threshold = config.ai.ai_fusion_threshold
            fusion_weights = config.ai.ai_fusion_weights

            logger.info(
                f"🔗 开始信号融合 - 策略: {fusion_strategy}, 阈值: {fusion_threshold}"
            )
            if fusion_weights:
                logger.info(f"🎯 融合权重: {fusion_weights}")

            # 执行融合
            fusion_result = await self.ai_fusion.fuse_signals(signals, config)

            if fusion_result:
                fusion_result["provider"] = "fusion"
                fusion_result["fusion_strategy"] = fusion_strategy
                fusion_result["fusion_sources"] = [
                    s.get("provider", "unknown") for s in signals
                ]
                fusion_result["source_count"] = len(signals)

                logger.info(
                    f"✅ 信号融合完成: {fusion_result.get('signal', 'UNKNOWN')} (信心: {fusion_result.get('confidence', 0):.2f})"
                )

                return [fusion_result]
            else:
                logger.warning("信号融合失败，返回最优单个信号")
                return [self._select_best_signal(signals)]

        except Exception as e:
            logger.error(f"执行融合失败: {e}")
            return [self._select_best_signal(signals)]

    def _select_best_signal(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """选择最优信号"""
        if not signals:
            return {
                "signal": "HOLD",
                "confidence": 0.3,
                "reason": "无有效信号",
                "provider": "fallback",
            }

        # 按置信度排序，选择最高的
        best_signal = max(signals, key=lambda x: x.get("confidence", 0))

        # 添加选择信息
        best_signal["selection_reason"] = "highest_confidence"
        best_signal["total_candidates"] = len(signals)

        return best_signal

    async def _apply_price_position_scaling(
        self, signal: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """应用价格位置缩放"""
        try:
            # 这里实现价格位置缩放逻辑
            # 从原始AI管理器中提取相关逻辑

            # 获取价格位置信息
            price_position = market_data.get("price_position", 50)  # 默认中间位置

            # 根据价格位置调整置信度
            confidence = signal.get("confidence", 0)

            if 40 <= price_position <= 60:
                # 中间位置，置信度不变
                scaling_factor = 1.0
            elif price_position < 20:
                # 接近最低价，买入信号增强，卖出信号减弱
                if signal.get("signal") == "BUY":
                    scaling_factor = 1.2
                else:
                    scaling_factor = 0.8
            elif price_position > 80:
                # 接近最高价，卖出信号增强，买入信号减弱
                if signal.get("signal") == "SELL":
                    scaling_factor = 1.2
                else:
                    scaling_factor = 0.8
            else:
                # 其他位置，适度调整
                scaling_factor = 0.9

            # 应用缩放因子
            new_confidence = confidence * scaling_factor
            new_confidence = min(1.0, max(0.0, new_confidence))  # 限制在[0,1]范围内

            scaled_signal = signal.copy()
            scaled_signal["confidence"] = new_confidence
            scaled_signal["original_confidence"] = confidence
            scaled_signal["price_position_scaling"] = scaling_factor
            scaled_signal["price_position"] = price_position

            return scaled_signal

        except Exception as e:
            logger.error(f"价格位置缩放失败: {e}")
            return None

    def _calculate_dynamic_confidence_threshold(
        self, market_data: Dict[str, Any]
    ) -> float:
        """计算动态置信度阈值"""
        try:
            # 基础阈值
            base_threshold = 0.3

            # 根据市场波动性调整
            atr_percentage = market_data.get("atr_percentage", 0.5)

            if atr_percentage > 1.0:
                # 高波动，提高阈值
                return base_threshold + 0.1
            elif atr_percentage < 0.2:
                # 低波动，降低阈值
                return base_threshold - 0.1
            else:
                # 正常波动
                return base_threshold

        except Exception as e:
            logger.error(f"计算动态置信度阈值失败: {e}")
            return 0.3  # 默认值

    async def _call_signal_optimizer(
        self, signals: List[Dict[str, Any]], market_data: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """调用信号优化器"""
        try:
            # 这里应该调用实际的信号优化器
            # 由于原始代码中的信号优化器可能需要异步调用，我们提供一个简化版本

            optimized_signals = []
            for signal in signals:
                # 简单的优化逻辑：基于技术指标二次验证
                if self._validate_signal_with_technicals(signal, market_data):
                    optimized_signals.append(signal)
                else:
                    # 降低置信度而不是完全丢弃
                    optimized_signal = signal.copy()
                    optimized_signal["confidence"] *= 0.8
                    optimized_signals.append(optimized_signal)

            return optimized_signals

        except Exception as e:
            logger.error(f"信号优化器调用失败: {e}")
            return None

    def _validate_signal_with_technicals(
        self, signal: Dict[str, Any], market_data: Dict[str, Any]
    ) -> bool:
        """使用技术指标验证信号"""
        try:
            technical_data = market_data.get("technical_data", {})
            signal_type = signal.get("signal", "HOLD")

            rsi = technical_data.get("rsi", 50)
            macd_hist = technical_data.get("macd_histogram", 0)

            # 简单的验证逻辑
            if signal_type == "BUY":
                # 买入信号：RSI不应过高，MACD不应为负
                return rsi < 70 and macd_hist > -0.01
            elif signal_type == "SELL":
                # 卖出信号：RSI不应过低，MACD不应为正
                return rsi > 30 and macd_hist < 0.01
            else:
                # HOLD信号总是通过验证
                return True

        except Exception as e:
            logger.error(f"技术指标验证失败: {e}")
            return True  # 验证失败时默认通过

    def _compare_buy_changes(
        self,
        original_signals: List[Dict[str, Any]],
        optimized_signals: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """比较BUY信号变化"""
        try:
            buy_to_hold_count = 0
            confidence_changes = 0
            changed_count = 0

            # 创建原始信号的映射
            original_map = {s.get("provider", "unknown"): s for s in original_signals}

            for opt_signal in optimized_signals:
                provider = opt_signal.get("provider", "unknown")
                if provider in original_map:
                    orig_signal = original_map[provider]

                    # 检查信号变化
                    if (
                        orig_signal.get("signal") == "BUY"
                        and opt_signal.get("signal") == "HOLD"
                    ):
                        buy_to_hold_count += 1
                        changed_count += 1

                    # 检查置信度变化
                    if (
                        abs(
                            orig_signal.get("confidence", 0)
                            - opt_signal.get("confidence", 0)
                        )
                        > 0.01
                    ):
                        confidence_changes += 1
                        if changed_count == 0:
                            changed_count += 1

            return {
                "buy_to_hold_count": buy_to_hold_count,
                "confidence_changes": confidence_changes,
                "changed_count": changed_count,
            }

        except Exception as e:
            logger.error(f"比较BUY信号变化失败: {e}")
            return {"buy_to_hold_count": 0, "confidence_changes": 0, "changed_count": 0}
