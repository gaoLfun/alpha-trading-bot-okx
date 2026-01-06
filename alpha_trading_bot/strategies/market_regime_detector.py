"""
市场环境识别系统 - 自动识别趋势/震荡/波动状态
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    """市场环境数据结构"""
    regime_type: str  # 'trending_up', 'trending_down', 'ranging', 'volatile', 'chaotic'
    trend_strength: float  # 趋势强度 (0-1)
    volatility_level: str  # 波动率等级 ('low', 'normal', 'high', 'extreme')
    volatility_score: float  # 波动率分数 (0-1)
    adx_value: float  # ADX指标值
    regime_confidence: float  # 环境识别置信度 (0-1)
    expected_duration: int  # 预期持续时间 (周期数)
    recommended_strategy: str  # 推荐策略类型
    last_update: datetime

class MarketRegimeDetector:
    """市场环境识别器"""

    # 市场状态定义
    REGIME_TYPES = {
        'trending_up': {
            'min_adx': 25,
            'min_trend_strength': 0.6,
            'max_volatility': 0.5,
            'description': '上升趋势',
            'recommended_strategy': 'trend_following',
            'expected_duration': 20
        },
        'trending_down': {
            'min_adx': 25,
            'min_trend_strength': 0.6,
            'max_volatility': 0.5,
            'description': '下降趋势',
            'recommended_strategy': 'trend_following',
            'expected_duration': 20
        },
        'ranging': {
            'max_adx': 20,
            'max_trend_strength': 0.4,
            'max_volatility': 0.3,
            'description': '震荡整理',
            'recommended_strategy': 'mean_reversion',
            'expected_duration': 15
        },
        'volatile': {
            'min_volatility': 0.5,
            'max_volatility': 0.8,
            'description': '高波动',
            'recommended_strategy': 'volatility_trading',
            'expected_duration': 10
        },
        'chaotic': {
            'min_volatility': 0.8,
            'description': '极度混乱',
            'recommended_strategy': 'risk_off',
            'expected_duration': 5
        }
    }

    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period
        self.price_history = deque(maxlen=lookback_period)
        self.regime_history = deque(maxlen=20)
        self.current_regime = None
        self._technical_cache = {}

    def detect_market_regime(
        self,
        ohlcv_data: List[List[float]],
        additional_indicators: Optional[Dict[str, float]] = None
    ) -> MarketRegime:
        """
        检测当前市场环境

        Args:
            ohlcv_data: OHLCV数据列表 [timestamp, open, high, low, close, volume]
            additional_indicators: 额外指标字典

        Returns:
            MarketRegime: 当前市场环境信息
        """
        try:
            if len(ohlcv_data) < 20:
                logger.warning("数据不足，使用默认市场环境")
                return self._get_default_regime()

            # 1. 计算基础技术指标
            indicators = self._calculate_indicators(ohlcv_data)

            # 2. 合并额外指标
            if additional_indicators:
                indicators.update(additional_indicators)

            # 3. 计算趋势强度
            trend_strength = self._calculate_trend_strength(ohlcv_data)

            # 4. 计算波动率水平
            volatility_level, volatility_score = self._calculate_volatility(ohlcv_data)

            # 5. 确定市场环境类型
            regime_type = self._determine_regime_type(indicators, trend_strength, volatility_score)

            # 6. 计算置信度
            confidence = self._calculate_confidence(indicators, regime_type)

            # 7. 获取预期持续时间
            expected_duration = self.REGIME_TYPES[regime_type]['expected_duration']

            # 8. 获取推荐策略
            recommended_strategy = self.REGIME_TYPES[regime_type]['recommended_strategy']

            # 创建市场环境对象
            regime = MarketRegime(
                regime_type=regime_type,
                trend_strength=trend_strength,
                volatility_level=volatility_level,
                volatility_score=volatility_score,
                adx_value=indicators.get('adx', 0),
                regime_confidence=confidence,
                expected_duration=expected_duration,
                recommended_strategy=recommended_strategy,
                last_update=datetime.now()
            )

            # 更新历史
            self.current_regime = regime
            self.regime_history.append(regime)

            logger.info(f"市场环境识别完成: {regime_type} (置信度: {confidence:.2f})")
            return regime

        except Exception as e:
            logger.error(f"市场环境识别失败: {e}")
            return self._get_default_regime()

    def _calculate_indicators(self, ohlcv_data: List[List[float]]) -> Dict[str, float]:
        """计算技术指标"""
        try:
            closes = np.array([d[4] for d in ohlcv_data])
            highs = np.array([d[2] for d in ohlcv_data])
            lows = np.array([d[3] for d in ohlcv_data])

            # 确保有足够的数据
            if len(closes) < 14:
                return {'adx': 0, 'rsi': 50, 'macd': 0}

            # 计算ADX
            adx = self._calculate_adx(highs, lows, closes)

            # 计算RSI
            rsi = self._calculate_rsi(closes)

            # 计算MACD
            macd = self._calculate_macd(closes)

            # 计算布林带
            bb_upper, bb_lower, bb_width = self._calculate_bollinger_bands(closes)

            # 计算均线排列
            ma_alignment = self._calculate_ma_alignment(closes)

            return {
                'adx': adx,
                'rsi': rsi,
                'macd': macd,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_width': bb_width,
                'ma_alignment': ma_alignment,
                'current_price': closes[-1],
                'price_change_24h': (closes[-1] - closes[-min(24, len(closes))]) / closes[-min(24, len(closes))]
            }

        except Exception as e:
            logger.error(f"技术指标计算失败: {e}")
            return {'adx': 0, 'rsi': 50, 'macd': 0, 'bb_width': 0, 'ma_alignment': 0}

    def _calculate_adx(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """计算ADX指标"""
        try:
            # 计算真实波幅TR
            tr1 = highs[1:] - lows[1:]
            tr2 = np.abs(highs[1:] - closes[:-1])
            tr3 = np.abs(lows[1:] - closes[:-1])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)

            # 计算方向指标
            plus_dm = np.where((highs[1:] - highs[:-1]) > (lows[:-1] - lows[1:]),
                              np.maximum(0, highs[1:] - highs[:-1]), 0)
            minus_dm = np.where((lows[:-1] - lows[1:]) > (highs[1:] - highs[:-1]),
                               np.maximum(0, lows[:-1] - lows[1:]), 0)

            # 计算平滑值
            atr = self._smooth(tr, period)
            plus_di = 100 * self._smooth(plus_dm, period) / atr
            minus_di = 100 * self._smooth(minus_dm, period) / atr

            # 计算DX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

            # 计算ADX
            adx = self._smooth(dx, period)

            return adx[-1] if len(adx) > 0 else 0

        except Exception as e:
            logger.error(f"ADX计算失败: {e}")
            return 0

    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """计算RSI指标"""
        try:
            if len(closes) < period + 1:
                return 50

            # 计算价格变化
            delta = np.diff(closes)
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)

            # 计算平均收益和损失
            avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
            avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')

            if len(avg_gains) == 0 or len(avg_losses) == 0:
                return 50

            # 计算RSI
            rs = avg_gains[-1] / (avg_losses[-1] + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            return rsi

        except Exception as e:
            logger.error(f"RSI计算失败: {e}")
            return 50

    def _calculate_macd(self, closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
        """计算MACD指标"""
        try:
            # 计算EMA
            ema_fast = self._calculate_ema(closes, fast)
            ema_slow = self._calculate_ema(closes, slow)

            # 计算MACD线
            macd_line = ema_fast - ema_slow

            # 计算信号线
            signal_line = self._calculate_ema(macd_line, signal)

            # 返回当前MACD值
            return macd_line[-1] - signal_line[-1] if len(macd_line) > 0 and len(signal_line) > 0 else 0

        except Exception as e:
            logger.error(f"MACD计算失败: {e}")
            return 0

    def _calculate_bollinger_bands(self, closes: np.ndarray, period: int = 20, std_dev: float = 2) -> tuple:
        """计算布林带"""
        try:
            if len(closes) < period:
                return closes[-1], closes[-1], 0

            # 计算移动平均
            sma = np.convolve(closes, np.ones(period)/period, mode='valid')

            # 计算标准差
            rolling_std = np.array([
                np.std(closes[i:i+period])
                for i in range(len(closes) - period + 1)
            ])

            # 计算布林带
            upper_band = sma + std_dev * rolling_std
            lower_band = sma - std_dev * rolling_std
            band_width = (upper_band - lower_band) / sma

            return upper_band[-1], lower_band[-1], band_width[-1]

        except Exception as e:
            logger.error(f"布林带计算失败: {e}")
            return closes[-1], closes[-1], 0

    def _calculate_ma_alignment(self, closes: np.ndarray) -> float:
        """计算均线排列强度"""
        try:
            # 计算不同周期的均线
            ma5 = np.convolve(closes, np.ones(5)/5, mode='valid')
            ma10 = np.convolve(closes, np.ones(10)/10, mode='valid')
            ma20 = np.convolve(closes, np.ones(20)/20, mode='valid')

            if len(ma5) == 0 or len(ma10) == 0 or len(ma20) == 0:
                return 0

            # 获取最近值
            current_ma5 = ma5[-1]
            current_ma10 = ma10[-1] if len(ma10) > 0 else current_ma5
            current_ma20 = ma20[-1] if len(ma20) > 0 else current_ma10

            # 计算排列强度
            alignment_score = 0
            if current_ma5 > current_ma10 > current_ma20:
                alignment_score = 1.0  # 多头排列
            elif current_ma5 < current_ma10 < current_ma20:
                alignment_score = -1.0  # 空头排列
            else:
                alignment_score = 0.0  # 无明确排列

            return alignment_score

        except Exception as e:
            logger.error(f"均线排列计算失败: {e}")
            return 0

    def _calculate_trend_strength(self, ohlcv_data: List[List[float]]) -> float:
        """计算趋势强度"""
        try:
            closes = np.array([d[4] for d in ohlcv_data])

            if len(closes) < 10:
                return 0.5

            # 使用线性回归计算趋势
            x = np.arange(len(closes))
            slope, intercept = np.polyfit(x, closes, 1)

            # 计算R²值
            y_pred = slope * x + intercept
            ss_res = np.sum((closes - y_pred) ** 2)
            ss_tot = np.sum((closes - np.mean(closes)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))

            # 计算趋势强度（结合斜率和R²）
            trend_strength = min(abs(slope) / np.mean(closes) * 100, 1.0) * r_squared

            return max(0, min(trend_strength, 1.0))

        except Exception as e:
            logger.error(f"趋势强度计算失败: {e}")
            return 0.5

    def _calculate_volatility(self, ohlcv_data: List[List[float]]) -> tuple:
        """计算波动率水平"""
        try:
            closes = np.array([d[4] for d in ohlcv_data])

            if len(closes) < 10:
                return 'normal', 0.5

            # 计算历史波动率
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) * np.sqrt(365 * 96)  # 年化波动率（15分钟周期）

            # 确定波动率等级
            if volatility < 0.2:  # <20%
                level = 'low'
                score = 0.2
            elif volatility < 0.4:  # 20-40%
                level = 'normal'
                score = 0.5
            elif volatility < 0.8:  # 40-80%
                level = 'high'
                score = 0.7
            else:  # >80%
                level = 'extreme'
                score = 0.9

            return level, score

        except Exception as e:
            logger.error(f"波动率计算失败: {e}")
            return 'normal', 0.5

    def _determine_regime_type(self, indicators: Dict[str, float],
                              trend_strength: float,
                              volatility_score: float) -> str:
        """确定市场环境类型"""
        try:
            adx = indicators.get('adx', 0)
            rsi = indicators.get('rsi', 50)
            ma_alignment = indicators.get('ma_alignment', 0)
            bb_width = indicators.get('bb_width', 0)

            # 根据ADX和趋势强度判断趋势
            if adx > 25 and trend_strength > 0.6:
                if ma_alignment > 0.5:  # 多头排列
                    return 'trending_up'
                elif ma_alignment < -0.5:  # 空头排列
                    return 'trending_down'

            # 根据波动率判断
            if volatility_score > 0.8:
                return 'chaotic'
            elif volatility_score > 0.5:
                return 'volatile'

            # 根据ADX和布林带判断震荡
            if adx < 20 and trend_strength < 0.4 and bb_width < 0.05:
                return 'ranging'

            # 默认返回震荡
            return 'ranging'

        except Exception as e:
            logger.error(f"市场环境类型判断失败: {e}")
            return 'ranging'

    def _calculate_confidence(self, indicators: Dict[str, float], regime_type: str) -> float:
        """计算环境识别置信度"""
        try:
            # 基础置信度
            base_confidence = 0.5

            # ADX置信度加成
            adx = indicators.get('adx', 0)
            if regime_type in ['trending_up', 'trending_down']:
                if adx > 30:
                    base_confidence += 0.3
                elif adx > 25:
                    base_confidence += 0.2
            elif regime_type == 'ranging':
                if adx < 15:
                    base_confidence += 0.2
                elif adx < 20:
                    base_confidence += 0.1

            # 均线排列置信度
            ma_alignment = indicators.get('ma_alignment', 0)
            if abs(ma_alignment) > 0.7:
                base_confidence += 0.2
            elif abs(ma_alignment) > 0.5:
                base_confidence += 0.1

            # RSI位置置信度
            rsi = indicators.get('rsi', 50)
            if regime_type == 'ranging':
                if 40 < rsi < 60:
                    base_confidence += 0.1

            return max(0, min(base_confidence, 1.0))

        except Exception as e:
            logger.error(f"置信度计算失败: {e}")
            return 0.5

    def _get_default_regime(self) -> MarketRegime:
        """获取默认市场环境"""
        return MarketRegime(
            regime_type='ranging',
            trend_strength=0.5,
            volatility_level='normal',
            volatility_score=0.5,
            adx_value=15,
            regime_confidence=0.5,
            expected_duration=15,
            recommended_strategy='mean_reversion',
            last_update=datetime.now()
        )

    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算指数移动平均"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema

    def _smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """平滑数据"""
        if len(data) < period:
            return np.array([np.mean(data)] * len(data))

        smoothed = np.zeros(len(data) - period + 1)
        for i in range(len(smoothed)):
            smoothed[i] = np.mean(data[i:i+period])

        return smoothed

    def get_regime_transition_probability(self, from_regime: str, to_regime: str) -> float:
        """获取市场环境转换概率"""
        # 基于历史统计的转换概率矩阵
        transition_matrix = {
            'trending_up': {
                'trending_up': 0.7,
                'trending_down': 0.1,
                'ranging': 0.15,
                'volatile': 0.04,
                'chaotic': 0.01
            },
            'trending_down': {
                'trending_up': 0.1,
                'trending_down': 0.7,
                'ranging': 0.15,
                'volatile': 0.04,
                'chaotic': 0.01
            },
            'ranging': {
                'trending_up': 0.25,
                'trending_down': 0.25,
                'ranging': 0.4,
                'volatile': 0.08,
                'chaotic': 0.02
            },
            'volatile': {
                'trending_up': 0.15,
                'trending_down': 0.15,
                'ranging': 0.3,
                'volatile': 0.3,
                'chaotic': 0.1
            },
            'chaotic': {
                'trending_up': 0.1,
                'trending_down': 0.1,
                'ranging': 0.2,
                'volatile': 0.3,
                'chaotic': 0.3
            }
        }

        return transition_matrix.get(from_regime, {}).get(to_regime, 0.2)

    def should_stay_in_current_regime(self, current_regime: MarketRegime,
                                    new_data: List[List[float]]) -> bool:
        """判断是否应继续当前市场环境"""
        try:
            # 重新检测市场环境
            new_regime = self.detect_market_regime(new_data)

            # 如果环境类型相同，继续当前环境
            if new_regime.regime_type == current_regime.regime_type:
                return True

            # 如果置信度很高，继续当前环境
            if current_regime.regime_confidence > 0.8:
                return True

            # 检查转换概率
            transition_prob = self.get_regime_transition_probability(
                current_regime.regime_type, new_regime.regime_type
            )

            # 如果转换概率低，继续当前环境
            if transition_prob < 0.3:
                return True

            return False

        except Exception as e:
            logger.error(f"市场环境持续判断失败: {e}")
            return True

    def get_strategy_parameters(self, regime: MarketRegime) -> Dict[str, Any]:
        """根据市场环境获取策略参数"""
        base_params = {
            'trending_up': {
                'rsi_oversold': 35,
                'rsi_overbought': 75,
                'macd_threshold': 0.5,
                'stop_loss_pct': 0.025,  # 2.5%
                'take_profit_pct': 0.08,  # 8%
                'position_size_multiplier': 1.2,
                'trailing_stop_distance': 0.02,
                'entry_confidence': 0.7
            },
            'trending_down': {
                'rsi_oversold': 25,
                'rsi_overbought': 65,
                'macd_threshold': -0.5,
                'stop_loss_pct': 0.025,
                'take_profit_pct': 0.08,
                'position_size_multiplier': 1.2,
                'trailing_stop_distance': 0.02,
                'entry_confidence': 0.7
            },
            'ranging': {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'macd_threshold': 0.3,
                'stop_loss_pct': 0.015,  # 1.5% - 更紧的止损
                'take_profit_pct': 0.05,  # 5% - 更早止盈
                'position_size_multiplier': 0.8,
                'trailing_stop_distance': 0.015,
                'entry_confidence': 0.6
            },
            'volatile': {
                'rsi_oversold': 25,
                'rsi_overbought': 75,
                'macd_threshold': 0.8,
                'stop_loss_pct': 0.04,   # 4% - 更宽的止损
                'take_profit_pct': 0.06,  # 6% - 适中止盈
                'position_size_multiplier': 0.6,
                'trailing_stop_distance': 0.025,
                'entry_confidence': 0.8
            },
            'chaotic': {
                'rsi_oversold': 20,
                'rsi_overbought': 80,
                'macd_threshold': 1.0,
                'stop_loss_pct': 0.05,   # 5% - 非常宽的止损
                'take_profit_pct': 0.04,  # 4% - 快速止盈
                'position_size_multiplier': 0.4,
                'trailing_stop_distance': 0.03,
                'entry_confidence': 0.9
            }
        }

        # 根据波动率调整参数
        params = base_params.get(regime.regime_type, base_params['ranging']).copy()

        # 波动率调整
        if regime.volatility_level == 'high':
            params['stop_loss_pct'] *= 1.3
            params['take_profit_pct'] *= 0.9
            params['position_size_multiplier'] *= 0.8
        elif regime.volatility_level == 'extreme':
            params['stop_loss_pct'] *= 1.5
            params['take_profit_pct'] *= 0.8
            params['position_size_multiplier'] *= 0.6

        # 置信度调整
        if regime.regime_confidence < 0.6:
            params['position_size_multiplier'] *= 0.7
            params['entry_confidence'] *= 1.2

        return params