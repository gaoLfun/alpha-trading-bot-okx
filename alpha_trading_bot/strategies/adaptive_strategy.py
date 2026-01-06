"""
自适应策略系统 - 根据市场环境自动调整策略参数
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .market_regime_detector import MarketRegimeDetector, MarketRegime

logger = logging.getLogger(__name__)

@dataclass
class StrategyPerformance:
    """策略绩效数据"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    avg_return: float = 0.0
    volatility: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)

class AdaptiveStrategy:
    """自适应策略管理器"""

    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.current_regime = None
        self.strategy_performance = {}
        self.parameter_history = {}
        self.adaptation_enabled = True
        self.performance_window = 50  # 评估窗口（交易次数）
        self.min_trades_for_adaptation = 20  # 最小交易数才进行自适应
        self.parameter_adjustment_rate = 0.1  # 参数调整幅度

        # 初始化各市场环境的策略绩效
        for regime_type in ['trending_up', 'trending_down', 'ranging', 'volatile', 'chaotic']:
            self.strategy_performance[regime_type] = StrategyPerformance()
            self.parameter_history[regime_type] = []

    def get_adaptive_parameters(
        self,
        ohlcv_data: List[List[float]],
        current_signal: Dict[str, Any],
        account_balance: float,
        position_size: float
    ) -> Dict[str, Any]:
        """
        获取自适应策略参数

        Args:
            ohlcv_data: OHLCV数据
            current_signal: 当前信号
            account_balance: 账户余额
            position_size: 仓位大小

        Returns:
            自适应策略参数
        """
        try:
            # 1. 检测当前市场环境
            regime = self.regime_detector.detect_market_regime(ohlcv_data)
            self.current_regime = regime

            logger.info(f"当前市场环境: {regime.regime_type} (置信度: {regime.regime_confidence:.2f})")

            # 2. 获取基础策略参数
            base_params = self.regime_detector.get_strategy_parameters(regime)

            # 3. 根据历史绩效调整参数
            if self.adaptation_enabled:
                adapted_params = self._adapt_parameters(base_params, regime.regime_type)
            else:
                adapted_params = base_params.copy()

            # 4. 根据当前信号微调
            final_params = self._fine_tune_for_signal(adapted_params, current_signal)

            # 5. 根据账户状态调整
            final_params = self._adjust_for_account_state(final_params, account_balance, position_size)

            # 6. 记录参数历史
            self._record_parameters(regime.regime_type, final_params)

            logger.info(f"最终策略参数: {final_params}")

            return final_params

        except Exception as e:
            logger.error(f"自适应参数获取失败: {e}，使用默认参数")
            return self._get_default_parameters()

    def update_performance(
        self,
        regime_type: str,
        trade_result: Dict[str, Any],
        entry_price: float,
        exit_price: float,
        position_size: float
    ):
        """更新策略绩效"""
        try:
            perf = self.strategy_performance[regime_type]

            # 更新交易计数
            perf.total_trades += 1

            # 计算盈亏
            pnl = (exit_price - entry_price) / entry_price * position_size
            return_pct = pnl / position_size

            if pnl > 0:
                perf.winning_trades += 1
                perf.avg_win = (perf.avg_win * (perf.winning_trades - 1) + return_pct) / perf.winning_trades
            else:
                perf.losing_trades += 1
                perf.avg_loss = (perf.avg_loss * (perf.losing_trades - 1) + abs(return_pct)) / perf.losing_trades

            # 更新胜率
            perf.win_rate = perf.winning_trades / perf.total_trades if perf.total_trades > 0 else 0

            # 更新盈亏比
            if perf.avg_loss > 0:
                perf.profit_factor = perf.avg_win / perf.avg_loss

            # 更新总收益
            perf.total_return += return_pct
            perf.avg_return = perf.total_return / perf.total_trades if perf.total_trades > 0 else 0

            # 更新最后更新时间
            perf.last_update = datetime.now()

            logger.info(f"{regime_type} 策略绩效更新 - 胜率: {perf.win_rate:.2%}, 盈亏比: {perf.profit_factor:.2f}")

        except Exception as e:
            logger.error(f"绩效更新失败: {e}")

    def _adapt_parameters(self, base_params: Dict[str, Any], regime_type: str) -> Dict[str, Any]:
        """根据历史绩效自适应调整参数"""
        try:
            perf = self.strategy_performance[regime_type]

            # 如果交易次数不足，不进行自适应调整
            if perf.total_trades < self.min_trades_for_adaptation:
                return base_params.copy()

            adapted_params = base_params.copy()

            # 1. 基于胜率调整入场门槛
            if perf.win_rate > 0.7:  # 高胜率，降低入场门槛
                adapted_params['entry_confidence'] *= 0.9
            elif perf.win_rate < 0.4:  # 低胜率，提高入场门槛
                adapted_params['entry_confidence'] *= 1.2

            # 2. 基于盈亏比调整止盈止损
            if perf.profit_factor > 2.0:  # 高盈亏比，可以放宽止盈
                adapted_params['take_profit_pct'] *= 1.1
            elif perf.profit_factor < 1.0:  # 低盈亏比，收紧止损
                adapted_params['stop_loss_pct'] *= 0.9

            # 3. 基于最大回撤调整仓位
            if perf.max_drawdown > 0.1:  # 回撤过大，降低仓位
                adapted_params['position_size_multiplier'] *= 0.8
            elif perf.max_drawdown < 0.05:  # 回撤小，可以适当增加仓位
                adapted_params['position_size_multiplier'] *= 1.1

            # 4. 基于夏普比率调整整体风险
            if hasattr(perf, 'sharpe_ratio') and perf.sharpe_ratio:
                if perf.sharpe_ratio > 2.0:
                    adapted_params['position_size_multiplier'] *= 1.1
                elif perf.sharpe_ratio < 0.5:
                    adapted_params['position_size_multiplier'] *= 0.9

            # 确保参数在合理范围内
            adapted_params = self._normalize_parameters(adapted_params)

            logger.info(f"{regime_type} 参数自适应调整完成")

            return adapted_params

        except Exception as e:
            logger.error(f"参数自适应调整失败: {e}")
            return base_params.copy()

    def _fine_tune_for_signal(self, params: Dict[str, Any], signal: Dict[str, Any]) -> Dict[str, Any]:
        """根据当前信号微调参数"""
        try:
            fine_tuned = params.copy()

            # 获取信号强度和置信度
            signal_strength = signal.get('strength', 0.5)
            confidence = signal.get('confidence', 0.5)

            # 强信号微调
            if signal_strength > 0.8 and confidence > 0.8:
                fine_tuned['position_size_multiplier'] *= 1.2
                fine_tuned['take_profit_pct'] *= 1.1
                fine_tuned['entry_confidence'] *= 0.95

            # 弱信号微调
            elif signal_strength < 0.4 or confidence < 0.4:
                fine_tuned['position_size_multiplier'] *= 0.7
                fine_tuned['stop_loss_pct'] *= 0.9
                fine_tuned['entry_confidence'] *= 1.2

            # 中等信号微调
            else:
                fine_tuned['position_size_multiplier'] *= 1.0
                fine_tuned['entry_confidence'] *= 1.0

            return fine_tuned

        except Exception as e:
            logger.error(f"信号微调失败: {e}")
            return params

    def _adjust_for_account_state(self, params: Dict[str, Any],
                                account_balance: float,
                                position_size: float) -> Dict[str, Any]:
        """根据账户状态调整参数"""
        try:
            adjusted = params.copy()

            # 基于账户余额调整（小额账户更保守）
            if account_balance < 1000:  # 小于1000USDT
                adjusted['position_size_multiplier'] *= 0.8
                adjusted['stop_loss_pct'] *= 0.9
                adjusted['take_profit_pct'] *= 0.9

            # 基于当前仓位调整
            position_ratio = position_size / account_balance if account_balance > 0 else 0
            if position_ratio > 0.3:  # 仓位过重
                adjusted['position_size_multiplier'] *= 0.9
                adjusted['stop_loss_pct'] *= 0.95

            return adjusted

        except Exception as e:
            logger.error(f"账户状态调整失败: {e}")
            return params

    def _normalize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """规范化参数到合理范围"""
        normalized = params.copy()

        # 止损百分比范围
        normalized['stop_loss_pct'] = max(0.005, min(normalized['stop_loss_pct'], 0.1))  # 0.5%-10%

        # 止盈百分比范围
        normalized['take_profit_pct'] = max(0.01, min(normalized['take_profit_pct'], 0.2))  # 1%-20%

        # 仓位倍数范围
        normalized['position_size_multiplier'] = max(0.1, min(normalized['position_size_multiplier'], 3.0))  # 10%-300%

        # 入场置信度范围
        normalized['entry_confidence'] = max(0.1, min(normalized['entry_confidence'], 1.0))  # 10%-100%

        # 追踪止损距离范围
        if 'trailing_stop_distance' in normalized:
            normalized['trailing_stop_distance'] = max(0.005, min(normalized['trailing_stop_distance'], 0.05))  # 0.5%-5%

        return normalized

    def _record_parameters(self, regime_type: str, params: Dict[str, Any]):
        """记录参数历史"""
        try:
            record = {
                'timestamp': datetime.now(),
                'parameters': params.copy(),
                'regime_type': regime_type,
                'performance': self.strategy_performance[regime_type].__dict__.copy()
            }
            self.parameter_history[regime_type].append(record)

            # 保持历史记录在合理范围内
            if len(self.parameter_history[regime_type]) > 100:
                self.parameter_history[regime_type] = self.parameter_history[regime_type][-100:]

        except Exception as e:
            logger.error(f"参数记录失败: {e}")

    def _get_default_parameters(self) -> Dict[str, Any]:
        """获取默认策略参数"""
        return {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_threshold': 0.5,
            'stop_loss_pct': 0.02,   # 2%
            'take_profit_pct': 0.06,  # 6%
            'position_size_multiplier': 1.0,
            'trailing_stop_distance': 0.02,  # 2%
            'entry_confidence': 0.6
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取策略绩效总结"""
        summary = {}
        for regime_type, perf in self.strategy_performance.items():
            if perf.total_trades > 0:
                summary[regime_type] = {
                    'total_trades': perf.total_trades,
                    'win_rate': perf.win_rate,
                    'profit_factor': perf.profit_factor,
                    'avg_return': perf.avg_return,
                    'max_drawdown': perf.max_drawdown,
                    'total_return': perf.total_return
                }
        return summary

    def reset_performance(self, regime_type: Optional[str] = None):
        """重置策略绩效"""
        if regime_type:
            self.strategy_performance[regime_type] = StrategyPerformance()
            self.parameter_history[regime_type] = []
        else:
            for regime_type in self.strategy_performance:
                self.strategy_performance[regime_type] = StrategyPerformance()
                self.parameter_history[regime_type] = []

    def enable_adaptation(self, enabled: bool = True):
        """启用/禁用自适应调整"""
        self.adaptation_enabled = enabled
        logger.info(f"策略自适应调整已{'启用' if enabled else '禁用'}")

    def get_current_regime(self) -> Optional[MarketRegime]:
        """获取当前市场环境"""
        return self.current_regime

    def should_trade_in_current_regime(self) -> bool:
        """判断是否应在当前市场环境下交易"""
        if not self.current_regime:
            return True

        # 在极度混乱的市场中暂停交易
        if self.current_regime.regime_type == 'chaotic' and self.current_regime.regime_confidence > 0.7:
            return False

        # 在高波动市场中降低交易频率
        if self.current_regime.regime_type == 'volatile' and self.current_regime.regime_confidence > 0.8:
            # 降低50%的交易频率
            return np.random.random() < 0.5

        return True