"""
回测引擎

功能：
- 基于历史K线数据回测策略表现
- 支持多策略并行回测
- 生成详细的回测报告
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)


class TradeAction(Enum):
    """交易动作"""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class BacktestConfig:
    """回测配置"""

    initial_capital: float = 10000
    position_size: float = 0.1  # 10% 仓位
    stop_loss_percent: float = 0.02
    take_profit_percent: float = 0.06
    fee_percent: float = 0.001  # 0.1% 手续费


@dataclass
class BacktestResult:
    """回测结果"""

    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    trade_history: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "trade_history": self.trade_history,
        }


class BacktestEngine:
    """回测引擎"""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._trade_history: list[Dict[str, Any]] = []
        self._capital_history: list[float] = []

    def run_backtest(
        self,
        market_data: List[Dict[str, Any]],
        strategy_signals: List[Dict[str, Any]],
    ) -> BacktestResult:
        """
        运行回测

        Args:
            market_data: 历史K线数据
            strategy_signals: 策略信号列表

        Returns:
            BacktestResult: 回测结果
        """
        capital = self.config.initial_capital
        position = 0.0  # 持仓数量
        entry_price = 0.0
        in_position = False
        self._trade_history = []
        self._capital_history = [capital]

        for i, candle in enumerate(market_data):
            price = candle.get("close", 0)
            timestamp = candle.get("timestamp", "")

            # 获取对应信号
            signal = strategy_signals[i] if i < len(strategy_signals) else {}

            if not in_position:
                # 无持仓，尝试买入
                if signal.get("signal") == "buy" and signal.get("confidence", 0) > 0.5:
                    # 买入
                    trade_capital = capital * self.config.position_size
                    fee = trade_capital * self.config.fee_percent
                    position = (trade_capital - fee) / price

                    self._trade_history.append(
                        {
                            "action": TradeAction.BUY.value,
                            "price": price,
                            "amount": position,
                            "timestamp": timestamp,
                            "capital_before": capital,
                        }
                    )

                    capital = capital - trade_capital
                    entry_price = price
                    in_position = True

            else:
                # 有持仓，检查是否卖出
                pnl_percent = (price - entry_price) / entry_price

                # 止损检查
                if pnl_percent < -self.config.stop_loss_percent:
                    self._close_position(
                        TradeAction.SELL.value,
                        price,
                        position,
                        timestamp,
                        "stop_loss",
                    )
                    capital = price * position * (1 - self.config.fee_percent)
                    position = 0
                    in_position = False

                # 止盈检查
                elif pnl_percent > self.config.take_profit_percent:
                    self._close_position(
                        TradeAction.SELL.value,
                        price,
                        position,
                        timestamp,
                        "take_profit",
                    )
                    capital = price * position * (1 - self.config.fee_percent)
                    position = 0
                    in_position = False

                # 信号卖出
                elif signal.get("signal") == "sell":
                    self._close_position(
                        TradeAction.SELL.value,
                        price,
                        position,
                        timestamp,
                        "signal",
                    )
                    capital = price * position * (1 - self.config.fee_percent)
                    position = 0
                    in_position = False

            # 记录资本
            portfolio_value = capital + position * price
            self._capital_history.append(portfolio_value)

        # 计算结果
        result = self._calculate_result()
        return result

    def _close_position(
        self,
        action: str,
        price: float,
        amount: float,
        timestamp: str,
        reason: str,
    ) -> None:
        """记录平仓"""
        self._trade_history.append(
            {
                "action": action,
                "price": price,
                "amount": amount,
                "timestamp": timestamp,
                "reason": reason,
            }
        )

    def _calculate_result(self) -> BacktestResult:
        """计算回测结果"""
        capital_history = self._capital_history
        initial_capital = self.config.initial_capital
        final_capital = capital_history[-1]

        # 总收益
        total_return = (final_capital - initial_capital) / initial_capital

        # 年化收益
        n_days = len(capital_history) / 96  # 假设15分钟K线
        annual_return = (1 + total_return) ** (365 / n_days) - 1 if n_days > 0 else 0

        # 夏普比率
        returns = [
            (capital_history[i + 1] - capital_history[i]) / capital_history[i]
            for i in range(len(capital_history) - 1)
        ]
        if returns and len(returns) > 1:
            import statistics

            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 0.001
            sharpe = (mean_return / std_return) * 15 if std_return > 0 else 0
        else:
            sharpe = 0

        # 最大回撤
        max_capital = 0
        max_drawdown = 0
        for capital in capital_history:
            if capital > max_capital:
                max_capital = capital
            drawdown = (max_capital - capital) / max_capital
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # 交易统计
        trades = [
            t for t in self._trade_history if t["action"] == TradeAction.BUY.value
        ]
        closes = [
            t for t in self._trade_history if t["action"] == TradeAction.SELL.value
        ]

        winning_trades = 0
        losing_trades = 0
        total_wins = 0
        total_losses = 0

        for close in closes:
            if close.get("reason") in ["take_profit", "signal"]:
                winning_trades += 1
                total_wins += close.get("pnl_percent", 0)
            elif close.get("reason") == "stop_loss":
                losing_trades += 1
                total_losses += abs(close.get("pnl_percent", 0))

        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        average_win = total_wins / winning_trades if winning_trades > 0 else 0
        average_loss = total_losses / losing_trades if losing_trades > 0 else 0

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            average_win=average_win,
            average_loss=average_loss,
            trade_history=self._trade_history,
        )

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """获取交易历史"""
        return self._trade_history

    def get_capital_history(self) -> List[float]:
        """获取资本曲线"""
        return self._capital_history
