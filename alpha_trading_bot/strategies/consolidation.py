"""
改进的横盘检测模块
基于多种技术指标的横盘状态识别
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# 币种特异性横盘参数（基于波动率调整）
CONSOLIDATION_PARAMS = {
    'BTC/USDT': {
        'atr_threshold': 0.015,      # 1.5%
        'bb_width_threshold': 0.03,  # 3%
        'adx_threshold': 25,         # ADX小于25视为无趋势
        'min_duration_hours': 6,     # 最少6小时确认
        'price_range_threshold': 0.04 # 4%的价格区间
    },
    'ETH/USDT': {
        'atr_threshold': 0.02,       # 2%
        'bb_width_threshold': 0.035, # 3.5%
        'adx_threshold': 25,
        'min_duration_hours': 6,
        'price_range_threshold': 0.05
    },
    'SHIB/USDT': {
        'atr_threshold': 0.05,       # 5%（山寨币波动更大）
        'bb_width_threshold': 0.08,  # 8%
        'adx_threshold': 30,
        'min_duration_hours': 4,
        'price_range_threshold': 0.10
    },
    'DEFAULT': {
        'atr_threshold': 0.025,      # 2.5%
        'bb_width_threshold': 0.04,  # 4%
        'adx_threshold': 25,
        'min_duration_hours': 6,
        'price_range_threshold': 0.06
    }
}

class ConsolidationDetector:
    """改进的横盘检测器"""

    def __init__(self):
        self.consolidation_history = {}
        self.multi_timeframe_data = {}

    def detect_consolidation(self, market_data: Dict[str, Any], symbol: str = 'BTC/USDT') -> Tuple[bool, str, float]:
        """
        检测市场是否处于横盘状态

        Args:
            market_data: 市场数据，包含价格、成交量等信息
            symbol: 交易对符号

        Returns:
            (是否横盘, 原因说明, 置信度)
        """
        try:
            # 获取币种特异性参数
            params = CONSOLIDATION_PARAMS.get(symbol, CONSOLIDATION_PARAMS['DEFAULT'])

            # 1. 基础数据检查
            if not self._validate_market_data(market_data):
                return False, "市场数据不完整", 0.0

            # 2. 多时间框架分析
            consolidation_score = self._multi_timeframe_analysis(market_data, symbol)

            # 3. 技术指标分析
            technical_score = self._technical_indicators_analysis(market_data, params)

            # 4. 波动率分析
            volatility_score = self._volatility_analysis(market_data, params)

            # 5. 成交量分析
            volume_score = self._volume_analysis(market_data)

            # 6. 综合评分
            final_score = (
                consolidation_score * 0.3 +
                technical_score * 0.25 +
                volatility_score * 0.25 +
                volume_score * 0.2
            )

            # 7. 生成结果
            is_consolidation = final_score > 0.7
            confidence = min(final_score, 0.95)
            reason = self._generate_reason(final_score, consolidation_score, technical_score, volatility_score)

            logger.info(f"横盘检测结果: {is_consolidation}, 评分: {final_score:.2f}, 原因: {reason}")

            return is_consolidation, reason, confidence

        except Exception as e:
            logger.error(f"横盘检测失败: {e}")
            return False, f"检测失败: {str(e)}", 0.0

    def _validate_market_data(self, market_data: Dict[str, Any]) -> bool:
        """验证市场数据完整性"""
        required_fields = ['price', 'high', 'low', 'volume', 'timestamp']
        for field in required_fields:
            if field not in market_data or market_data[field] is None:
                return False
        return True

    def _multi_timeframe_analysis(self, market_data: Dict[str, Any], symbol: str) -> float:
        """多时间框架分析"""
        try:
            current_price = float(market_data['price'])

            # 1小时级别（短期）
            hourly_high = float(market_data.get('hourly_high', market_data['high']))
            hourly_low = float(market_data.get('hourly_low', market_data['low']))
            hourly_position = (current_price - hourly_low) / (hourly_high - hourly_low) if hourly_high != hourly_low else 0.5

            # 4小时级别（中期）
            four_hour_high = float(market_data.get('4h_high', market_data['high']))
            four_hour_low = float(market_data.get('4h_low', market_data['low']))
            four_hour_position = (current_price - four_hour_low) / (four_hour_high - four_hour_low) if four_hour_high != four_hour_low else 0.5

            # 24小时级别（长期）
            daily_high = float(market_data['high'])
            daily_low = float(market_data['low'])
            daily_position = (current_price - daily_low) / (daily_high - daily_low) if daily_high != daily_low else 0.5

            # 评分逻辑：价格越接近中间位置，横盘可能性越高
            hourly_score = 1.0 - abs(hourly_position - 0.5) * 2
            four_hour_score = 1.0 - abs(four_hour_position - 0.5) * 2
            daily_score = 1.0 - abs(daily_position - 0.5) * 2

            # 权重：长期更重要
            return hourly_score * 0.2 + four_hour_score * 0.3 + daily_score * 0.5

        except Exception as e:
            logger.error(f"多时间框架分析失败: {e}")
            return 0.0

    def _technical_indicators_analysis(self, market_data: Dict[str, Any], params: Dict[str, float]) -> float:
        """技术指标分析"""
        try:
            score = 0.0

            # 1. ADX趋势强度分析
            if 'adx' in market_data:
                adx = float(market_data['adx'])
                if adx < params['adx_threshold']:  # ADX小于阈值视为无趋势
                    score += 0.3
                elif adx < params['adx_threshold'] + 5:
                    score += 0.15

            # 2. RSI中性区域分析
            if 'rsi' in market_data:
                rsi = float(market_data['rsi'])
                if 40 <= rsi <= 60:  # RSI中性区域
                    score += 0.3
                elif 35 <= rsi <= 65:
                    score += 0.15

            # 3. MACD柱状图分析
            if 'macd_histogram' in market_data:
                histogram = float(market_data['macd_histogram'])
                if abs(histogram) < 0.1:  # MACD柱状图接近0
                    score += 0.2
                elif abs(histogram) < 0.2:
                    score += 0.1

            # 4. 价格与均线关系
            if 'sma_20' in market_data and 'sma_50' in market_data:
                sma_20 = float(market_data['sma_20'])
                sma_50 = float(market_data['sma_50'])
                price = float(market_data['price'])

                # 价格在均线附近徘徊
                if abs(price - sma_20) / price < 0.01 and abs(sma_20 - sma_50) / sma_20 < 0.005:
                    score += 0.2

            return min(score, 0.8)

        except Exception as e:
            logger.error(f"技术指标分析失败: {e}")
            return 0.0

    def _volatility_analysis(self, market_data: Dict[str, Any], params: Dict[str, float]) -> float:
        """波动率分析"""
        try:
            score = 0.0
            current_price = float(market_data['price'])

            # 1. ATR分析
            if 'atr' in market_data:
                atr = float(market_data['atr'])
                atr_ratio = atr / current_price

                if atr_ratio < params['atr_threshold']:
                    score += 0.4
                elif atr_ratio < params['atr_threshold'] * 1.5:
                    score += 0.2

            # 2. 布林带宽度分析
            if 'bb_upper' in market_data and 'bb_lower' in market_data:
                bb_upper = float(market_data['bb_upper'])
                bb_lower = float(market_data['bb_lower'])
                bb_width = (bb_upper - bb_lower) / current_price

                if bb_width < params['bb_width_threshold']:
                    score += 0.4
                elif bb_width < params['bb_width_threshold'] * 1.5:
                    score += 0.2

            # 3. 历史波动率比较
            if 'volatility_30d' in market_data:
                current_vol = float(market_data['volatility_30d'])
                if current_vol < 0.3:  # 低于30%视为低波动
                    score += 0.2

            return min(score, 0.8)

        except Exception as e:
            logger.error(f"波动率分析失败: {e}")
            return 0.0

    def _volume_analysis(self, market_data: Dict[str, Any]) -> float:
        """成交量分析"""
        try:
            score = 0.0

            if 'volume' in market_data and 'avg_volume_24h' in market_data:
                current_volume = float(market_data['volume'])
                avg_volume = float(market_data['avg_volume_24h'])

                # 成交量萎缩通常伴随横盘
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

                if 0.5 <= volume_ratio <= 1.5:  # 正常成交量
                    score += 0.3
                elif volume_ratio < 0.5:  # 成交量萎缩
                    score += 0.4
                elif volume_ratio > 2.0:  # 异常放量但价格不动
                    score += 0.1  # 可能是变盘前兆，降低横盘评分

            return score

        except Exception as e:
            logger.error(f"成交量分析失败: {e}")
            return 0.0

    def _generate_reason(self, final_score: float, consolidation_score: float,
                        technical_score: float, volatility_score: float) -> str:
        """生成横盘原因说明"""
        reasons = []

        if consolidation_score > 0.6:
            reasons.append("价格处于多时间框架的中间区域")

        if technical_score > 0.5:
            reasons.append("技术指标显示无明确趋势")

        if volatility_score > 0.5:
            reasons.append("市场波动率较低")

        if final_score > 0.8:
            reason_level = "高度确认"
        elif final_score > 0.6:
            reason_level = "中度确认"
        else:
            reason_level = "轻度确认"

        if reasons:
            return f"{reason_level}横盘: {'; '.join(reasons)}"
        else:
            return f"横盘评分: {final_score:.2f}"

    def get_consolidation_strength(self, market_data: Dict[str, Any]) -> float:
        """获取横盘强度（0-1）"""
        is_consolidation, _, confidence = self.detect_consolidation(market_data)
        return confidence if is_consolidation else 0.0

    def predict_breakout_direction(self, market_data: Dict[str, Any]) -> str:
        """预测横盘突破方向"""
        try:
            # 基于订单簿、资金流向等预测突破方向
            # 这是一个简化的实现，实际可以更复杂

            if 'order_book_imbalance' in market_data:
                imbalance = float(market_data['order_book_imbalance'])
                if imbalance > 0.1:
                    return "UP"
                elif imbalance < -0.1:
                    return "DOWN"

            # 默认基于价格位置判断
            current_price = float(market_data['price'])
            high = float(market_data['high'])
            low = float(market_data['low'])
            position = (current_price - low) / (high - low)

            if position > 0.6:
                return "UP"
            elif position < 0.4:
                return "DOWN"
            else:
                return "UNCERTAIN"

        except Exception:
            return "UNCERTAIN"