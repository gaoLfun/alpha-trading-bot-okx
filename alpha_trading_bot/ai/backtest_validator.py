"""
回测验证器

功能：
- 基于历史数据验证交易策略表现
- 计算关键指标（胜率、盈亏比、最大回撤等）
- 生成回测报告

作者：AI Trading System
日期：2026-02-04
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from statistics import mean, stdev

logger = logging.getLogger(__name__)


class TradeResult(Enum):
    """交易结果"""

    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    OPEN = "open"


@dataclass
class Trade:
    """交易记录"""

    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    side: str  # buy | sell
    pnl: float
    pnl_percent: float
    result: TradeResult
    confidence: float
    reason: str


@dataclass
class BacktestResult:
    """回测结果"""

    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    open_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    max_drawdown: float
    total_pnl: float
    total_return: float
    sharpe_ratio: float
    trades: List[Trade]
    monthly_returns: Dict[str, float]
    confidence_analysis: Dict[str, Any]


@dataclass
class BacktestConfig:
    """回测配置"""

    initial_capital: float = 10000
    position_size: float = 0.1  # 10%仓位
    stop_loss_percent: float = 0.02
    take_profit_percent: float = 0.06
    min_confidence_threshold: float = 0.5
    fee_percent: float = 0.001  # 0.1%手续费


class BacktestValidator:
    """
    回测验证器

    功能：
    1. 模拟交易执行
    2. 计算交易绩效指标
    3. 置信度分析
    4. 生成回测报告
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        初始化回测验证器

        Args:
            config: 回测配置
        """
        self.config = config or BacktestConfig()
        self.trades: List[Trade] = []
        self.capital_history: List[float] = [self.config.initial_capital]
        self._validate_config()

        logger.info(
            f"[回测验证器] 初始化完成: "
            f"初始资金=${self.config.initial_capital:,.0f}, "
            f"仓位={self.config.position_size * 100}%, "
            f"止损={self.config.stop_loss_percent * 100}%, "
            f"止盈={self.config.take_profit_percent * 100}%"
        )

    def run_backtest(
        self,
        signals: List[Dict[str, Any]],
        prices: List[float],
        timestamps: List[str],
    ) -> BacktestResult:
        """
        运行回测

        Args:
            signals: 信号列表 [{"signal": "buy", "confidence": 0.7, ...}, ...]
            prices: 价格列表
            timestamps: 时间戳列表

        Returns:
            BacktestResult: 回测结果
        """
        if not signals or len(signals) != len(prices):
            raise ValueError("信号、价格、时间戳长度必须一致")

        self.trades.clear()
        self.capital_history = [self.config.initial_capital]

        capital = self.config.initial_capital
        position = None  # 当前持仓

        for i, (signal, price, timestamp) in enumerate(
            zip(signals, prices, timestamps)
        ):
            confidence = signal.get("confidence", 0.6)

            # 跳过置信度过低的信号
            if confidence < self.config.min_confidence_threshold:
                continue

            # 处理当前持仓
            if position:
                # 检查是否触发止损/止盈
                pnl_percent = (price - position["entry_price"]) / position[
                    "entry_price"
                ]
                if position["side"] == "sell":
                    pnl_percent = -pnl_percent

                # 触发止损
                if pnl_percent <= -self.config.stop_loss_percent:
                    position = self._close_position(
                        position, price, timestamp, pnl_percent, "stop_loss", confidence
                    )
                # 触发止盈
                elif pnl_percent >= self.config.take_profit_percent:
                    position = self._close_position(
                        position,
                        price,
                        timestamp,
                        pnl_percent,
                        "take_profit",
                        confidence,
                    )

            # 处理新信号
            if not position and signal["signal"] in ["buy", "sell"]:
                position = self._open_position(
                    signal["signal"], price, timestamp, confidence
                )

        # 处理未平仓的持仓
        if position:
            final_price = prices[-1]
            pnl_percent = (final_price - position["entry_price"]) / position[
                "entry_price"
            ]
            if position["side"] == "sell":
                pnl_percent = -pnl_percent

            trade = self._close_position(
                position,
                final_price,
                timestamps[-1],
                pnl_percent,
                "end_of_backtest",
                position["confidence"],
            )
            trade.result = TradeResult.OPEN
            self.trades.append(trade)

        # 计算结果
        return self._calculate_results()

    def _open_position(
        self, side: str, price: float, timestamp: str, confidence: float
    ) -> Dict[str, Any]:
        """开仓"""
        return {
            "side": side,
            "entry_price": price,
            "entry_time": timestamp,
            "confidence": confidence,
        }

    def _close_position(
        self,
        position: Dict[str, Any],
        price: float,
        timestamp: str,
        pnl_percent: float,
        reason: str,
        confidence: float,
    ) -> Optional[Trade]:
        """平仓"""
        pnl = self.config.initial_capital * self.config.position_size * pnl_percent

        # 判断结果
        if pnl_percent > 0.01:
            result = TradeResult.WIN
        elif pnl_percent < -0.01:
            result = TradeResult.LOSS
        else:
            result = TradeResult.BREAKEVEN

        trade = Trade(
            entry_time=position["entry_time"],
            exit_time=timestamp,
            entry_price=position["entry_price"],
            exit_price=price,
            side=position["side"],
            pnl=pnl,
            pnl_percent=pnl_percent,
            result=result,
            confidence=confidence,
            reason=reason,
        )

        self.trades.append(trade)
        return None

    def _calculate_results(self) -> BacktestResult:
        """计算回测结果"""
        # 统计交易
        winning_trades = [t for t in self.trades if t.result == TradeResult.WIN]
        losing_trades = [t for t in self.trades if t.result == TradeResult.LOSS]
        breakeven_trades = [t for t in self.trades if t.result == TradeResult.BREAKEVEN]
        open_trades = [t for t in self.trades if t.result == TradeResult.CLOSED]
        open_trades = [t for t in self.trades if t.result == TradeResult.OPEN]

        # 计算胜率
        closed_trades = winning_trades + losing_trades + breakeven_trades
        total_closed = len(closed_trades)
        win_rate = len(winning_trades) / total_closed if total_closed > 0 else 0

        # 计算平均盈亏
        avg_win = mean([t.pnl_percent for t in winning_trades]) if winning_trades else 0
        avg_loss = mean([t.pnl_percent for t in losing_trades]) if losing_trades else 0

        # 计算盈亏比
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown()

        # 计算总盈亏
        total_pnl = sum(t.pnl for t in self.trades if t.result != TradeResult.OPEN)
        total_return = total_pnl / self.config.initial_capital

        # 计算夏普比率
        returns = [t.pnl_percent for t in self.trades if t.result != TradeResult.OPEN]
        sharpe_ratio = self._calculate_sharpe_ratio(returns)

        # 月度收益
        monthly_returns = self._calculate_monthly_returns()

        # 置信度分析
        confidence_analysis = self._analyze_confidence()

        return BacktestResult(
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            breakeven_trades=len(breakeven_trades),
            open_trades=len(open_trades),
            win_rate=win_rate,
            average_win=avg_win,
            average_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            total_pnl=total_pnl,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            trades=self.trades,
            monthly_returns=monthly_returns,
            confidence_analysis=confidence_analysis,
        )

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.capital_history:
            return 0

        max_capital = self.capital_history[0]
        max_drawdown = 0

        for capital in self.capital_history:
            if capital > max_capital:
                max_capital = capital
            drawdown = (max_capital - capital) / max_capital
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """计算夏普比率"""
        if len(returns) < 2:
            return 0

        avg_return = mean(returns)
        std_return = stdev(returns) if len(returns) > 1 else 0

        if std_return == 0:
            return 0

        # 假设年化（15分钟周期约为3500个周期/年）
        annualized_return = avg_return * 3500
        annualized_std = std_return * (3500**0.5)

        return annualized_return / annualized_std if annualized_std != 0 else 0

    def _calculate_monthly_returns(self) -> Dict[str, float]:
        """计算月度收益"""
        monthly_pnl: Dict[str, float] = {}

        for trade in self.trades:
            if trade.result == TradeResult.OPEN:
                continue

            # 提取年月
            month = trade.exit_time[:7]  # YYYY-MM

            if month not in monthly_pnl:
                monthly_pnl[month] = 0

            monthly_pnl[month] += trade.pnl_percent

        return monthly_pnl

    def _analyze_confidence(self) -> Dict[str, Any]:
        """分析置信度与交易结果的关系"""
        high_conf_trades = [t for t in self.trades if t.confidence >= 0.7]
        mid_conf_trades = [t for t in self.trades if 0.5 <= t.confidence < 0.7]
        low_conf_trades = [t for t in self.trades if t.confidence < 0.5]

        def calc_stats(trades: List[Trade]) -> Dict[str, Any]:
            if not trades:
                return {"win_rate": 0, "avg_pnl": 0, "count": 0}

            wins = [t for t in trades if t.result == TradeResult.WIN]
            pnls = [t.pnl_percent for t in trades if t.result != TradeResult.OPEN]

            return {
                "win_rate": len(wins) / len(trades) if trades else 0,
                "avg_pnl": mean(pnls) if pnls else 0,
                "count": len(trades),
            }

        return {
            "high_confidence": calc_stats(high_conf_trades),
            "mid_confidence": calc_stats(mid_conf_trades),
            "low_confidence": calc_stats(low_conf_trades),
            "correlation": self._calculate_confidence_correlation(),
        }

    def _calculate_confidence_correlation(self) -> float:
        """计算置信度与交易结果的相关性"""
        if len(self.trades) < 2:
            return 0

        # 简单计算：置信度高的交易胜率 vs 置信度低的交易胜率
        high_conf = [t for t in self.trades if t.confidence >= 0.7]
        low_conf = [t for t in self.trades if t.confidence < 0.6]

        if not high_conf or not low_conf:
            return 0

        high_win_rate = len(
            [t for t in high_conf if t.result == TradeResult.WIN]
        ) / len(high_conf)
        low_win_rate = len([t for t in low_conf if t.result == TradeResult.WIN]) / len(
            low_conf
        )

        return high_win_rate - low_win_rate

    def _validate_config(self) -> None:
        """验证配置"""
        if self.config.initial_capital <= 0:
            raise ValueError("初始资金必须大于0")
        if self.config.position_size <= 0 or self.config.position_size > 1:
            raise ValueError("仓位比例必须在0-1之间")
        if self.config.stop_loss_percent <= 0:
            raise ValueError("止损比例必须大于0")
        if self.config.take_profit_percent <= 0:
            raise ValueError("止盈比例必须大于0")

    def generate_report(self, result: BacktestResult) -> str:
        """生成回测报告"""
        report = []
        report.append("=" * 60)
        report.append("回测报告")
        report.append("=" * 60)
        report.append(f"回测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"初始资金: ${self.config.initial_capital:,.0f}")
        report.append("")
        report.append("【总体统计】")
        report.append(f"总交易次数: {result.total_trades}")
        report.append(f"盈利次数: {result.winning_trades}")
        report.append(f"亏损次数: {result.losing_trades}")
        report.append(f"持平次数: {result.breakeven_trades}")
        report.append(f"持仓次数: {result.open_trades}")
        report.append("")
        report.append("【收益指标】")
        report.append(f"胜率: {result.win_rate:.2%}")
        report.append(f"平均盈利: {result.average_win:.2%}")
        report.append(f"平均亏损: {result.average_loss:.2%}")
        report.append(f"盈亏比: {result.profit_factor:.2f}")
        report.append(f"总盈亏: ${result.total_pnl:,.2f}")
        report.append(f"总收益率: {result.total_return:.2%}")
        report.append(f"夏普比率: {result.sharpe_ratio:.2f}")
        report.append("")
        report.append("【风险指标】")
        report.append(f"最大回撤: {result.max_drawdown:.2%}")
        report.append("")
        report.append("【置信度分析】")
        ca = result.confidence_analysis
        report.append(
            f"高置信度(>=70%): 胜率={ca['high_confidence']['win_rate']:.2%}, "
            f"平均盈亏={ca['high_confidence']['avg_pnl']:.2%}, "
            f"次数={ca['high_confidence']['count']}"
        )
        report.append(
            f"中置信度(50-70%): 胜率={ca['mid_confidence']['win_rate']:.2%}, "
            f"平均盈亏={ca['mid_confidence']['avg_pnl']:.2%}, "
            f"次数={ca['mid_confidence']['count']}"
        )
        report.append(
            f"低置信度(<50%): 胜率={ca['low_confidence']['win_rate']:.2%}, "
            f"平均盈亏={ca['low_confidence']['avg_pnl']:.2%}, "
            f"次数={ca['low_confidence']['count']}"
        )
        report.append(
            f"置信度-胜率相关性: {result.confidence_analysis['correlation']:.2%}"
        )
        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    def print_report(self, result: BacktestResult) -> None:
        """打印回测报告"""
        report = self.generate_report(result)
        print(report)

        # 记录到日志
        logger.info(f"[回测报告]\n{report}")
