"""
AI信号生成器 - 专门负责从各个AI提供商生成信号
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .client import AIClient
from ..config import load_config
from .cache_monitor import cache_monitor

logger = logging.getLogger(__name__)


class AISignalGenerator:
    """AI信号生成器 - 专门负责信号生成逻辑"""

    def __init__(self, ai_client: AIClient):
        self.ai_client = ai_client
        self.providers: List[str] = []

    async def initialize(self) -> bool:
        """初始化信号生成器"""
        try:
            logger.info("正在初始化AI信号生成器...")

            # 获取配置
            config = load_config()

            # 根据AI模式选择提供商
            if config.ai.use_multi_ai_fusion:
                # 多AI融合模式
                available_providers = set(config.ai.models.keys())
                fusion_providers = set(config.ai.ai_fusion_providers)

                # 只保留同时有API密钥且在融合配置中的提供商
                self.providers = list(available_providers & fusion_providers)

                if not self.providers:
                    logger.warning(
                        f"配置的融合提供商 {fusion_providers} 没有可用的API密钥，将使用回退模式"
                    )
                    self.providers = ["fallback"]
                else:
                    logger.info(f"AI融合模式已启用，使用提供商: {self.providers}")
            else:
                # 单一AI模式
                default_provider = config.ai.ai_default_provider
                if default_provider in config.ai.models:
                    self.providers = [default_provider]
                    logger.info(f"单一AI模式，使用提供商: {default_provider}")
                else:
                    logger.warning(
                        f"默认提供商 {default_provider} 未配置API密钥，将使用回退模式"
                    )
                    self.providers = ["fallback"]

            logger.info(f"AI信号生成器初始化成功，可用提供商: {self.providers}")
            return True

        except Exception as e:
            logger.error(f"AI信号生成器初始化失败: {e}")
            return False

    async def generate_signals(
        self, market_data: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], int, int, List[str]]:
        """
        生成AI信号

        Returns:
            tuple: (signals, success_count, fail_count, success_providers)
        """
        signals = []
        success_count = 0
        fail_count = 0
        success_providers = []

        try:
            # 记录当前AI决策模式
            config = load_config()
            ai_mode = "融合模式" if config.ai.use_multi_ai_fusion else "单一模式"
            logger.info(f"🤖 AI决策模式: {ai_mode} (提供商: {self.providers})")

            if len(self.providers) > 1 and config.ai.use_multi_ai_fusion:
                # 多AI模式
                logger.info(f"🚀 并行获取多AI信号: {self.providers}")
                (
                    signals,
                    success_count,
                    fail_count,
                    success_providers,
                ) = await self._generate_multi_ai_signals(market_data)
            else:
                # 单AI模式
                provider = self.providers[0] if self.providers else "fallback"
                logger.info(f"🎯 使用单一AI信号: {provider}")
                signal = await self._generate_single_ai_signal(market_data, provider)

                if signal:
                    signals = [signal]
                    success_count = 1
                    success_providers = [provider]

            return signals, success_count, fail_count, success_providers

        except Exception as e:
            logger.error(f"生成AI信号失败: {e}")
            return [], 0, len(self.providers), []

    async def _generate_single_ai_signal(
        self, market_data: Dict[str, Any], provider: str
    ) -> Optional[Dict[str, Any]]:
        """生成单个AI信号"""
        try:
            # 生成信号
            if provider == "fallback":
                logger.info(f"🔄 使用回退信号策略")
                signal = await self._generate_fallback_signal(market_data)
            else:
                logger.info(f"📡 请求 {provider.upper()} 信号...")
                signal = await self.ai_client.generate_signal(provider, market_data)

            # 记录信号详情
            if signal:
                action = signal.get("signal", signal.get("action", "UNKNOWN"))
                confidence = signal.get("confidence", 0)
                reason = signal.get("reason", "")

                # 添加信号理由到日志
                if reason:
                    logger.info(
                        f"✅ {provider.upper()} 成功: {action} (信心: {confidence:.2f}) - {reason}"
                    )
                else:
                    logger.info(
                        f"✅ {provider.upper()} 成功: {action} (信心: {confidence:.2f})"
                    )

                # 记录API调用成本到监控器
                estimated_cost = 0.001  # 估算每次API调用成本
                cache_monitor.record_api_call(provider, estimated_cost)
            else:
                logger.error(f"❌ {provider.upper()} 返回空信号")

            return signal

        except Exception as e:
            logger.error(f"生成单AI信号失败: {e}")
            return await self._generate_fallback_signal(market_data)

    async def _generate_multi_ai_signals(
        self, market_data: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], int, int, List[str]]:
        """生成多AI信号"""
        try:
            # 获取市场趋势信息
            trend_direction = market_data.get("trend_direction", "neutral")
            trend_strength = market_data.get("trend_strength", "normal")

            # 添加趋势过滤警告
            if trend_strength in ["strong", "extreme"]:
                if trend_direction == "down":
                    logger.warning(
                        f"🚨 检测到强势下跌趋势({trend_strength})，将严格过滤买入信号"
                    )
                elif trend_direction == "up":
                    logger.warning(
                        f"🚨 检测到强势上涨趋势({trend_strength})，将严格过滤卖出信号"
                    )

            # 并行获取所有提供商的信号
            tasks = []
            for provider in self.providers:
                if provider == "fallback":
                    task = asyncio.create_task(
                        self._generate_fallback_signal(market_data)
                    )
                else:
                    task = asyncio.create_task(
                        self.ai_client.generate_signal(provider, market_data)
                    )
                tasks.append((provider, task))

            # 等待所有任务完成并记录结果
            results = []
            success_count = 0
            fail_count = 0
            success_providers = []

            for provider, task in tasks:
                try:
                    signal = await task
                    if signal:
                        # 检查置信度阈值
                        confidence = signal.get("confidence", 0)
                        min_confidence = 0.3  # 默认最小置信度

                        if confidence >= min_confidence:
                            signal["provider"] = provider
                            results.append(signal)
                            success_count += 1
                            success_providers.append(provider)

                            # 记录详细的信号信息
                            action = signal.get(
                                "signal", signal.get("action", "UNKNOWN")
                            )
                            reason = signal.get("reason", "")
                            if reason:
                                logger.info(
                                    f"✅ {provider.upper()} 成功: {action} (信心: {confidence:.2f}) - {reason}"
                                )
                            else:
                                logger.info(
                                    f"✅ {provider.upper()} 成功: {action} (信心: {confidence:.2f})"
                                )

                            # 记录API调用成本到监控器
                            estimated_cost = 0.001  # 估算每次API调用成本
                            cache_monitor.record_api_call(provider, estimated_cost)
                        else:
                            logger.warning(
                                f"⚠️  {provider.upper()} 置信度不足: {confidence:.2f} < {min_confidence}"
                            )
                            fail_count += 1
                    else:
                        logger.error(f"❌ {provider.upper()} 返回空信号")
                        fail_count += 1

                except Exception as e:
                    logger.error(f"❌ {provider.upper()} 信号生成失败: {e}")
                    fail_count += 1

            # 记录统计信息
            total = success_count + fail_count
            logger.info(
                f"📊 多AI信号获取统计: 成功={success_count}, 失败={fail_count}, 总计={total}"
            )
            logger.info(
                f"✅ 成功提供商: {success_providers if success_providers else '无'}"
            )

            return results, success_count, fail_count, success_providers

        except Exception as e:
            logger.error(f"生成多AI信号失败: {e}")
            return [], 0, len(self.providers), []

    async def _generate_fallback_signal(
        self, market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """生成回退信号"""
        try:
            # 基于技术指标生成简单回退信号
            technical_data = market_data.get("technical_data", {})
            current_price = market_data.get("price", 0)

            if not technical_data or not current_price:
                return {
                    "signal": "HOLD",
                    "confidence": 0.5,
                    "reason": "技术数据不足，使用中性策略",
                    "provider": "fallback",
                }

            rsi = technical_data.get("rsi", 50)
            macd_hist = technical_data.get("macd_histogram", 0)

            # 简单的回退逻辑
            if rsi < 30 and macd_hist > 0:
                signal = "BUY"
                confidence = 0.6
                reason = f"RSI超卖({rsi:.1f})且MACD转正"
            elif rsi > 70 and macd_hist < 0:
                signal = "SELL"
                confidence = 0.6
                reason = f"RSI超买({rsi:.1f})且MACD转负"
            else:
                signal = "HOLD"
                confidence = 0.5
                reason = f"市场中性(RSI:{rsi:.1f}, MACD:{macd_hist:.4f})"

            return {
                "signal": signal,
                "confidence": confidence,
                "reason": reason,
                "provider": "fallback",
            }

        except Exception as e:
            logger.error(f"生成回退信号失败: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0.3,
                "reason": "回退信号生成失败",
                "provider": "fallback",
            }
