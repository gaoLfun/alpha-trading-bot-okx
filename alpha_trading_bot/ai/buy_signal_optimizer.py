"""
BUY信号专项优化器 - 针对qwen BUY信号导致亏损的优化
基于2025-12-25交易记录分析
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class BuySignalOptimizer:
    """BUY信号专项优化器"""

    def __init__(self):
        # BUY信号专项优化参数
        self.buy_optimizations = {
            # 价格位置限制
            'max_price_position': 0.85,  # 超过85%高位限制BUY
            'min_price_position': 0.15,  # 低于15%低位增强BUY

            # RSI限制
            'max_rsi_for_buy': 65,      # RSI超过65限制BUY
            'min_rsi_for_buy': 35,      # RSI低于35增强BUY

            # ATR波动率限制
            'min_atr_for_buy': 0.15,    # ATR低于0.15%限制BUY（低波动陷阱）
            'max_atr_for_buy': 3.0,     # ATR高于3%限制BUY（高波动风险）

            # 趋势要求
            'min_trend_strength': 0.2,   # 最小趋势强度
            'min_adx': 20,              # 最小ADX值

            # 成交量要求
            'min_volume_ratio': 0.8,    # 最低成交量比例
            'max_volume_spike': 3.0,    # 成交量异常放大限制

            # 时间窗口限制
            'avoid_last_hour': True,    # 避免最后一小时交易
            'cooldown_minutes': 30,     # BUY信号冷却时间
        }

        # 记录BUY信号历史
        self.buy_signal_history = []
        self.recent_buy_signals = []  # 最近30分钟的BUY信号

    def optimize_buy_signals(self, signals: List[Dict[str, Any]],
                           market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """优化BUY信号"""
        optimized_signals = []

        for signal in signals:
            signal_type = signal.get('signal', 'HOLD').upper()
            provider = signal.get('provider', 'unknown')

            # 只对BUY信号进行优化
            if signal_type == 'BUY':
                optimized_signal = self._optimize_buy_signal(signal, market_data, provider)
                optimized_signals.append(optimized_signal)

                # 记录BUY信号历史
                self._record_buy_signal(optimized_signal, market_data)
            else:
                # 非BUY信号直接通过
                optimized_signals.append(signal)

        return optimized_signals

    def _optimize_buy_signal(self, signal: Dict[str, Any],
                           market_data: Dict[str, Any],
                           provider: str) -> Dict[str, Any]:
        """优化单个BUY信号"""
        optimized = signal.copy()
        original_confidence = signal.get('confidence', 0.5)
        reason = signal.get('reason', '')

        # Ensure 'reason' key exists
        if 'reason' not in optimized:
            optimized['reason'] = ''

        # 获取技术指标
        technical_data = market_data.get('technical_data', {})
        price_position = technical_data.get('price_position', 0.5)
        rsi = technical_data.get('rsi', 50)
        adx = technical_data.get('adx', 0)
        trend_strength = technical_data.get('trend_strength', 0)

        # 获取市场数据
        current_price = market_data.get('price', 0)
        atr_percentage = market_data.get('atr_percentage', 0)
        volume = market_data.get('volume', 0)
        avg_volume = market_data.get('avg_volume_24h', volume)

        # 1. 价格位置检查（避免高位接盘）
        if price_position > self.buy_optimizations['max_price_position']:
            # 价格处于高位，降低BUY信号强度或转为HOLD
            optimized['confidence'] = max(original_confidence - 0.15, 0.3)
            optimized['reason'] += f" | ⚠️ 价格处于{price_position*100:.1f}%高位，风险较高"

            # 如果信心度降得太低，考虑转为HOLD
            if optimized['confidence'] < 0.45:
                optimized['signal'] = 'HOLD'
                optimized['reason'] += " | 高位风险过大，建议观望"

        # 2. RSI检查（避免超买买入）
        elif rsi > self.buy_optimizations['max_rsi_for_buy']:
            optimized['confidence'] = max(original_confidence - 0.1, 0.35)
            optimized['reason'] += f" | RSI为{rsi:.1f}，接近超买区域"

        # 3. 低波动率陷阱检查
        elif atr_percentage < self.buy_optimizations['min_atr_for_buy']:
            optimized['confidence'] = max(original_confidence - 0.12, 0.35)
            optimized['reason'] += f" | ATR仅{atr_percentage:.2f}%，低波动可能为陷阱"

        # 4. 趋势强度检查
        elif trend_strength < self.buy_optimizations['min_trend_strength']:
            optimized['confidence'] = max(original_confidence - 0.08, 0.4)
            optimized['reason'] += f" | 趋势强度{trend_strength:.2f}较弱，买入需谨慎"

        # 5. ADX检查（避免无趋势行情）
        elif adx < self.buy_optimizations['min_adx']:
            optimized['confidence'] = max(original_confidence - 0.08, 0.4)
            optimized['reason'] += f" | ADX为{adx:.1f}，市场无明显趋势"

        # 6. 成交量检查
        elif avg_volume > 0:
            volume_ratio = volume / avg_volume
            if volume_ratio < self.buy_optimizations['min_volume_ratio']:
                optimized['confidence'] = max(original_confidence - 0.06, 0.45)
                optimized['reason'] += f" | 成交量仅为均值{volume_ratio:.1f}倍，动能不足"

        # 7. 风险累积检查（多个风险因素叠加）
        risk_factors = 0
        if price_position > 0.75:
            risk_factors += 1
        if rsi > 65:
            risk_factors += 1
        if atr_percentage < 0.2:
            risk_factors += 1
        if trend_strength < 0.3:
            risk_factors += 1

        # 如果存在3个或以上风险因素，强制转为HOLD
        if risk_factors >= 3:
            optimized['signal'] = 'HOLD'
            optimized['confidence'] = min(optimized.get('confidence', original_confidence) - 0.2, 0.4)
            optimized['reason'] += f" | 累积风险过高({risk_factors}个风险因素)"

        # 8. 增强买入信号（满足多个有利条件）
        else:
            # 检查是否有利条件组合
            favorable_conditions = 0

            # 低位买入
            if price_position < 0.35:
                favorable_conditions += 1
                optimized['reason'] += " | 低位买入机会"

            # RSI超卖
            if rsi < 40:
                favorable_conditions += 1
                optimized['reason'] += f" | RSI超卖({rsi:.1f})"

            # 强趋势
            if trend_strength > 0.5 and adx > 25:
                favorable_conditions += 1
                optimized['reason'] += " | 强趋势确认"

            # 成交量放大
            if avg_volume > 0:
                volume_ratio = volume / avg_volume
                if volume_ratio > 1.2:
                    favorable_conditions += 1
                    optimized['reason'] += f" | 成交量放大{volume_ratio:.1f}倍"

            # 根据有利条件数量增强信号
            if favorable_conditions >= 3:
                optimized['confidence'] = min(original_confidence + 0.1, 0.9)
                optimized['reason'] += " | 多重利好确认，强烈买入信号"
            elif favorable_conditions >= 2:
                optimized['confidence'] = min(original_confidence + 0.05, 0.85)
                optimized['reason'] += " | 双重利好确认"

        # 8. 提供商特定优化
        if provider == 'qwen':
            optimized = self._optimize_qwen_buy_signal(optimized, market_data)
        elif provider == 'deepseek':
            optimized = self._optimize_deepseek_buy_signal(optimized, market_data)
        elif provider == 'kimi':
            optimized = self._optimize_kimi_buy_signal(optimized, market_data)
        elif provider == 'openai':
            optimized = self._optimize_openai_buy_signal(optimized, market_data)

        # 9. 时间窗口检查（避免特定时段）
        current_hour = datetime.now().hour
        current_minute = datetime.now().minute

        # 避免最后一小时交易（交易所结算风险）
        if self.buy_optimizations['avoid_last_hour'] and current_hour == 23:
            optimized['confidence'] = max(optimized.get('confidence', original_confidence) - 0.1, 0.3)
            optimized['reason'] += " | 避开最后一小时交易"

        # 冷却期检查
        if self._is_in_cooldown():
            optimized['confidence'] = max(optimized.get('confidence', original_confidence) - 0.15, 0.25)
            optimized['reason'] += " | 买入冷却期内，降低信号强度"

        # 记录优化详情
        if original_confidence != optimized['confidence']:
            change = optimized['confidence'] - original_confidence
            direction = "增强" if change > 0 else "减弱"
            logger.info(f"🔧 {provider.upper()} BUY信号优化: "
                       f"信心 {original_confidence:.2f} → "
                       f"{optimized['confidence']:.2f} ({direction})")

        return optimized

    def _optimize_qwen_buy_signal(self, signal: Dict[str, Any],
                                market_data: Dict[str, Any]) -> Dict[str, Any]:
        """优化qwen的BUY信号（基于历史表现）"""
        optimized = signal.copy()
        reason = signal.get('reason', '')

        # 1. 修正累积变化为0的问题
        if "累积变化为0.00%" in reason:
            change_percent = market_data.get('change_percent', 0)
            if abs(change_percent) > 0.001:  # 有微小变化
                optimized['reason'] = reason.replace("累积变化为0.00%", f"当前变化{change_percent:+.3f}%")

        # 2. 增强连续涨跌识别
        if "连续涨跌次数为0" in reason:
            close_prices = market_data.get('close_prices', [])
            recent_trend = self._calculate_recent_trend(close_prices[-5:]) if len(close_prices) >= 5 else 0
            if recent_trend != 0:
                optimized['reason'] = reason.replace("连续涨跌次数为0", f"连续{recent_trend}个周期同向变化")

        # 3. 增强低位识别
        technical_data = market_data.get('technical_data', {})
        price_position = technical_data.get('price_position', 0.5)
        rsi = technical_data.get('rsi', 50)

        if price_position < 0.25 and rsi < 40:
            # 低位+超卖，增强信号
            current_confidence = optimized.get('confidence', signal.get('confidence', 0.5))
            optimized['confidence'] = min(current_confidence + 0.08, 0.85)
            optimized['reason'] += " | 低位超卖增强信号"

        return optimized

    def _optimize_deepseek_buy_signal(self, signal: Dict[str, Any],
                                    market_data: Dict[str, Any]) -> Dict[str, Any]:
        """优化deepseek的BUY信号"""
        optimized = signal.copy()
        reason = signal.get('reason', '')

        # 1. 平衡过度谨慎的BUY信号
        if "建议谨慎" in reason or "风险" in reason:
            # 检查是否确实有高风险
            technical_data = market_data.get('technical_data', {})
            price_position = technical_data.get('price_position', 0.5)

            if price_position < 0.4:  # 实际处于低位
                # 降低谨慎程度
                current_confidence = optimized.get('confidence', signal.get('confidence', 0.5))
                optimized['confidence'] = min(current_confidence + 0.05, 0.8)
                optimized['reason'] = reason.replace("建议谨慎", "位置相对安全")

        # 2. 增强区间位置判断精度
        import re
        position_matches = re.findall(r'(\d+(?:\.\d+)?)%', reason)
        if position_matches:
            position = float(position_matches[0])
            if position > 80 and price_position < 0.7:  # 判断有误
                optimized['reason'] += f" | 实际位置{price_position*100:.1f}%更安全"

        return optimized

    def _optimize_kimi_buy_signal(self, signal: Dict[str, Any],
                                market_data: Dict[str, Any]) -> Dict[str, Any]:
        """优化kimi的BUY信号"""
        optimized = signal.copy()
        reason = signal.get('reason', '')

        # 1. 验证突破有效性
        if "突破" in reason:
            change_percent = market_data.get('change_percent', 0)
            atr_percentage = market_data.get('atr_percentage', 0)

            # 突破需要超过0.5倍ATR才视为有效
            if abs(change_percent) < atr_percentage * 0.5:
                current_confidence = optimized.get('confidence', signal.get('confidence', 0.5))
                optimized['confidence'] = max(current_confidence - 0.06, 0.45)
                optimized['reason'] += f" | 突破幅度不足({change_percent:+.2f}% < {atr_percentage*0.5:.2f}%)"

        # 2. 验证成交量放大
        if "成交量放大" in reason:
            volume = market_data.get('volume', 0)
            avg_volume = market_data.get('avg_volume_24h', volume)
            if avg_volume > 0:
                actual_ratio = volume / avg_volume
                # 如果实际比例与理由不符，调整信号
                if actual_ratio < 1.2:  # 放大不足
                    current_confidence = optimized.get('confidence', signal.get('confidence', 0.5))
                    optimized['confidence'] = max(current_confidence - 0.05, 0.5)
                    optimized['reason'] += f" | 实际仅{actual_ratio:.1f}倍，放大不足"

        return optimized

    def _optimize_openai_buy_signal(self, signal: Dict[str, Any],
                                  market_data: Dict[str, Any]) -> Dict[str, Any]:
        """优化openai的BUY信号"""
        optimized = signal.copy()
        reason = signal.get('reason', '')

        # 1. 验证概率数值
        import re
        prob_matches = re.findall(r'(\d+(?:\.\d+)?)%', reason)
        if prob_matches:
            claimed_prob = float(prob_matches[0])
            # 检查是否与市场条件匹配
            technical_data = market_data.get('technical_data', {})
            rsi = technical_data.get('rsi', 50)
            trend_strength = technical_data.get('trend_strength', 0)

            # 简单验证：如果RSI>60且声称70%上涨概率，需要谨慎
            if claimed_prob > 70 and rsi > 60:
                current_confidence = optimized.get('confidence', signal.get('confidence', 0.5))
                optimized['confidence'] = max(current_confidence - 0.08, 0.4)
                optimized['reason'] += " | 高概率与超买RSI矛盾"

        # 2. 验证风险回报比
        if "风险回报比" in reason or "回报" in reason:
            price_position = market_data.get('technical_data', {}).get('price_position', 0.5)
            if price_position > 0.7:  # 高位买入，风险较大
                current_confidence = optimized.get('confidence', signal.get('confidence', 0.5))
                optimized['confidence'] = max(current_confidence - 0.1, 0.35)
                optimized['reason'] += " | 高位买入，风险回报比不佳"

        return optimized

    def _record_buy_signal(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> None:
        """记录BUY信号"""
        record = {
            'timestamp': datetime.now(),
            'provider': signal.get('provider', 'unknown'),
            'confidence': signal.get('confidence', 0),
            'price': market_data.get('price', 0),
            'price_position': market_data.get('technical_data', {}).get('price_position', 0.5),
            'rsi': market_data.get('technical_data', {}).get('rsi', 50),
            'atr_percentage': market_data.get('atr_percentage', 0),
            'reason': signal.get('reason', ''),
            'market_data': market_data.copy()
        }

        self.buy_signal_history.append(record)
        self.recent_buy_signals.append(record)

        # 只保留最近30分钟的记录
        cutoff_time = datetime.now() - timedelta(minutes=30)
        self.recent_buy_signals = [
            s for s in self.recent_buy_signals
            if s['timestamp'] > cutoff_time
        ]

        # 只保留最近1000条历史记录
        if len(self.buy_signal_history) > 1000:
            self.buy_signal_history = self.buy_signal_history[-1000:]

    def _is_in_cooldown(self) -> bool:
        """检查是否在买入冷却期内"""
        if not self.recent_buy_signals:
            return False

        # 最近30分钟内是否有BUY信号
        cutoff_time = datetime.now() - timedelta(minutes=30)
        recent_signals = [
            s for s in self.recent_buy_signals
            if s['timestamp'] > cutoff_time
        ]

        return len(recent_signals) > 3  # 30分钟内超过3个BUY信号则进入冷却

    def get_buy_signal_stats(self) -> Dict[str, Any]:
        """获取BUY信号统计"""
        if not self.buy_signal_history:
            return {'total_signals': 0}

        total_signals = len(self.buy_signal_history)
        recent_signals = len(self.recent_buy_signals)

        # 统计提供商分布
        provider_stats = {}
        for signal in self.buy_signal_history:
            provider = signal['provider']
            provider_stats[provider] = provider_stats.get(provider, 0) + 1

        # 平均信心度
        avg_confidence = np.mean([s['confidence'] for s in self.buy_signal_history])

        # 平均价格位置
        avg_price_position = np.mean([s['price_position'] for s in self.buy_signal_history])

        # 平均RSI
        avg_rsi = np.mean([s['rsi'] for s in self.buy_signal_history])

        return {
            'total_signals': total_signals,
            'recent_signals_30min': recent_signals,
            'provider_distribution': provider_stats,
            'avg_confidence': avg_confidence,
            'avg_price_position': avg_price_position,
            'avg_rsi': avg_rsi,
            'in_cooldown': self._is_in_cooldown()
        }