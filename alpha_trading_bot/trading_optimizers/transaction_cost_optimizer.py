"""
交易成本优化器 - 最小化交易成本，提高执行效率
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型"""

    MARKET = "market"  # 市价单
    LIMIT = "limit"  # 限价单
    STOP_MARKET = "stop_market"  # 止损市价单
    STOP_LIMIT = "stop_limit"  # 止损限价单
    TRAILING_STOP = "trailing_stop"  # 追踪止损


@dataclass
class CostAnalysisResult:
    """成本分析结果"""

    total_cost: float  # 总成本
    commission: float  # 手续费
    slippage: float  # 滑点成本
    market_impact: float  # 市场冲击成本
    opportunity_cost: float  # 机会成本
    execution_time: float  # 执行时间（秒）
    cost_efficiency: float  # 成本效率评分 (0-100)
    timestamp: Optional[datetime] = None  # 记录时间戳


@dataclass
class OptimizedOrder:
    """优化后的订单"""

    symbol: str
    side: str
    amount: float
    order_type: OrderType
    price: Optional[float]
    stop_price: Optional[float]
    time_in_force: str
    estimated_cost: CostAnalysisResult
    execution_strategy: str
    confidence: float


class TransactionCostOptimizer:
    """交易成本优化器"""

    def __init__(self):
        # 交易所费用配置（OKX为例）
        self.fee_configs = {
            "okx": {
                "maker_fee": 0.0002,  # 0.02%
                "taker_fee": 0.0005,  # 0.05%
                "min_fee": 0.0,
                "max_fee": 0.01,  # 1%
                "vip_discounts": {
                    "regular": 1.0,
                    "vip1": 0.8,
                    "vip2": 0.6,
                    "vip3": 0.4,
                    "vip4": 0.2,
                },
            }
        }

        # 成本阈值
        self.max_acceptable_slippage = 0.001  # 0.1%
        self.max_acceptable_cost = 0.005  # 0.5%
        self.min_profit_threshold = 0.002  # 0.2%

        # 历史成本数据
        self.cost_history: List[CostAnalysisResult] = []

    def optimize_order_execution(
        self,
        symbol: str,
        side: str,
        amount: float,
        market_data: Dict[str, Any],
        account_info: Dict[str, Any],
        time_constraints: Optional[Dict] = None,
    ) -> OptimizedOrder:
        """
        优化订单执行策略

        Args:
            symbol: 交易对
            side: 买卖方向
            amount: 交易数量
            market_data: 市场数据
            account_info: 账户信息
            time_constraints: 时间约束

        Returns:
            优化后的订单
        """
        # 分析当前市场条件
        market_analysis = self._analyze_market_conditions(market_data, amount)

        # 评估不同执行策略的成本
        strategies = self._evaluate_execution_strategies(
            symbol, side, amount, market_data, market_analysis, account_info
        )

        # 选择最优策略
        best_strategy = min(strategies, key=lambda x: x["estimated_cost"].total_cost)

        # 创建优化后的订单
        optimized_order = self._create_optimized_order(
            symbol, side, amount, best_strategy, market_data
        )

        return optimized_order

    def _analyze_market_conditions(
        self, market_data: Dict[str, Any], order_size: float
    ) -> Dict[str, Any]:
        """
        分析市场条件

        Returns:
            市场条件分析结果
        """
        # 流动性分析
        volume_24h = market_data.get("volume_24h", 0)
        spread = market_data.get("spread", 0.001)
        order_book_depth = market_data.get("order_book_depth", 100)

        # 计算订单对市场的影响
        market_cap = market_data.get("market_cap", 1000000000)  # 默认10亿美元
        order_percentage = (order_size * market_data.get("price", 50000)) / market_cap

        # 波动率分析
        volatility = market_data.get("volatility", 0.02)
        atr = market_data.get("atr", 500)

        # 时间因素
        current_hour = datetime.now().hour
        is_peak_hours = 14 <= current_hour <= 21  # UTC时间，美股交易时段

        return {
            "liquidity_score": min(volume_24h / 1000000, 1.0),  # 标准化到0-1
            "spread_cost": spread,
            "market_impact": order_percentage * 0.01,  # 1%的订单规模影响
            "volatility": volatility,
            "atr": atr,
            "is_peak_hours": is_peak_hours,
            "order_book_depth": order_book_depth,
        }

    def _evaluate_execution_strategies(
        self,
        symbol: str,
        side: str,
        amount: float,
        market_data: Dict[str, Any],
        market_analysis: Dict[str, Any],
        account_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        评估不同执行策略的成本

        Returns:
            策略评估结果列表
        """
        strategies = []
        current_price = market_data.get("price", 50000)

        # 策略1: 立即市价执行
        market_cost = self._calculate_market_order_cost(
            symbol, side, amount, current_price, market_analysis, account_info
        )
        strategies.append(
            {
                "type": OrderType.MARKET,
                "estimated_cost": market_cost,
                "execution_time": 2,  # 2秒
                "success_probability": 0.95,
                "strategy": "immediate_market",
            }
        )

        # 策略2: 限价单执行
        limit_cost = self._calculate_limit_order_cost(
            symbol, side, amount, current_price, market_analysis, account_info
        )
        strategies.append(
            {
                "type": OrderType.LIMIT,
                "estimated_cost": limit_cost,
                "execution_time": 30,  # 30秒平均
                "success_probability": 0.7,
                "strategy": "limit_order",
            }
        )

        # 策略3: 分批执行（大单）
        if amount * current_price > 10000:  # 大于1万美元的订单
            batch_cost = self._calculate_batch_execution_cost(
                symbol, side, amount, current_price, market_analysis, account_info
            )
            strategies.append(
                {
                    "type": OrderType.MARKET,
                    "estimated_cost": batch_cost,
                    "execution_time": 120,  # 2分钟
                    "success_probability": 0.9,
                    "strategy": "batch_execution",
                }
            )

        # 策略4: 条件执行（基于时间或价格）
        conditional_cost = self._calculate_conditional_execution_cost(
            symbol, side, amount, current_price, market_analysis, account_info
        )
        strategies.append(
            {
                "type": OrderType.LIMIT,
                "estimated_cost": conditional_cost,
                "execution_time": 300,  # 5分钟
                "success_probability": 0.8,
                "strategy": "conditional_execution",
            }
        )

        return strategies

    def _calculate_market_order_cost(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        market_analysis: Dict[str, Any],
        account_info: Dict[str, Any],
    ) -> CostAnalysisResult:
        """
        计算市价单成本
        """
        # 基础手续费
        order_value = amount * price
        taker_fee_rate = self._get_fee_rate(account_info, "taker")
        commission = order_value * taker_fee_rate

        # 滑点成本（基于市场条件）
        base_slippage = market_analysis["spread_cost"] * 1.5  # 市价单滑点更大
        volatility_adjustment = market_analysis["volatility"] * 0.5
        liquidity_adjustment = (1 - market_analysis["liquidity_score"]) * 0.001

        slippage = base_slippage + volatility_adjustment + liquidity_adjustment
        slippage_cost = order_value * slippage

        # 市场冲击成本
        market_impact = order_value * market_analysis["market_impact"]
        market_impact_cost = market_impact * taker_fee_rate

        # 机会成本（市价单通常为0）
        opportunity_cost = 0

        # 执行时间
        execution_time = 2.0

        # 总成本
        total_cost = commission + slippage_cost + market_impact_cost + opportunity_cost

        # 成本效率评分
        cost_efficiency = self._calculate_cost_efficiency(total_cost, order_value)

        return CostAnalysisResult(
            total_cost=total_cost,
            commission=commission,
            slippage=slippage_cost,
            market_impact=market_impact_cost,
            opportunity_cost=opportunity_cost,
            execution_time=execution_time,
            cost_efficiency=cost_efficiency,
        )

    def _calculate_limit_order_cost(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        market_analysis: Dict[str, Any],
        account_info: Dict[str, Any],
    ) -> CostAnalysisResult:
        """
        计算限价单成本
        """
        order_value = amount * price

        # 限价单可能成为maker或taker
        maker_probability = 0.6  # 60%概率成为maker
        taker_probability = 1 - maker_probability

        maker_fee_rate = self._get_fee_rate(account_info, "maker")
        taker_fee_rate = self._get_fee_rate(account_info, "taker")

        expected_fee_rate = (
            maker_fee_rate * maker_probability + taker_fee_rate * taker_probability
        )
        commission = order_value * expected_fee_rate

        # 限价单滑点较小
        slippage = market_analysis["spread_cost"] * 0.3
        slippage_cost = order_value * slippage

        # 市场冲击较小
        market_impact_cost = order_value * market_analysis["market_impact"] * 0.5

        # 机会成本（等待成交的时间成本）
        opportunity_cost = (
            order_value * 0.0001 * (market_analysis.get("waiting_time", 30) / 60)
        )  # 假设每分钟0.01%的机会成本

        # 执行时间（平均等待时间）
        execution_time = 30.0

        total_cost = commission + slippage_cost + market_impact_cost + opportunity_cost
        cost_efficiency = self._calculate_cost_efficiency(total_cost, order_value)

        return CostAnalysisResult(
            total_cost=total_cost,
            commission=commission,
            slippage=slippage_cost,
            market_impact=market_impact_cost,
            opportunity_cost=opportunity_cost,
            execution_time=execution_time,
            cost_efficiency=cost_efficiency,
        )

    def _calculate_batch_execution_cost(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        market_analysis: Dict[str, Any],
        account_info: Dict[str, Any],
    ) -> CostAnalysisResult:
        """
        计算分批执行成本
        """
        # 分成3批执行
        batch_count = 3
        batch_amount = amount / batch_count
        batch_interval = 20  # 每批间隔20秒

        total_commission = 0
        total_slippage = 0
        total_market_impact = 0

        for i in range(batch_count):
            # 每批的成本计算
            batch_value = batch_amount * price

            # 手续费
            taker_fee_rate = self._get_fee_rate(account_info, "taker")
            total_commission += batch_value * taker_fee_rate

            # 滑点（随着时间推移可能增加）
            time_factor = i * 0.1  # 时间推移增加滑点
            batch_slippage = market_analysis["spread_cost"] * 1.2 + time_factor * 0.0005
            total_slippage += batch_value * batch_slippage

            # 市场冲击（分批减少冲击）
            batch_impact = (
                market_analysis["market_impact"] / batch_count * (1 - i * 0.2)
            )
            total_market_impact += batch_value * batch_impact

        # 机会成本（分批执行的总时间）
        total_time = batch_count * batch_interval
        opportunity_cost = amount * price * 0.00005 * (total_time / 60)  # 时间成本

        execution_time = total_time
        total_cost = (
            total_commission + total_slippage + total_market_impact + opportunity_cost
        )
        cost_efficiency = self._calculate_cost_efficiency(total_cost, amount * price)

        return CostAnalysisResult(
            total_cost=total_cost,
            commission=total_commission,
            slippage=total_slippage,
            market_impact=total_market_impact,
            opportunity_cost=opportunity_cost,
            execution_time=execution_time,
            cost_efficiency=cost_efficiency,
        )

    def _calculate_conditional_execution_cost(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        market_analysis: Dict[str, Any],
        account_info: Dict[str, Any],
    ) -> CostAnalysisResult:
        """
        计算条件执行成本（最佳时机等待）
        """
        order_value = amount * price

        # 条件执行通常能获得更好的价格
        maker_fee_rate = self._get_fee_rate(account_info, "maker")
        commission = order_value * maker_fee_rate

        # 更小的滑点
        slippage = market_analysis["spread_cost"] * 0.1
        slippage_cost = order_value * slippage

        # 更小的市场冲击
        market_impact_cost = order_value * market_analysis["market_impact"] * 0.3

        # 更高的机会成本（等待更长时间）
        opportunity_cost = order_value * 0.0002 * (300 / 60)  # 5分钟等待

        execution_time = 300.0  # 5分钟
        total_cost = commission + slippage_cost + market_impact_cost + opportunity_cost
        cost_efficiency = self._calculate_cost_efficiency(total_cost, order_value)

        return CostAnalysisResult(
            total_cost=total_cost,
            commission=commission,
            slippage=slippage_cost,
            market_impact=market_impact_cost,
            opportunity_cost=opportunity_cost,
            execution_time=execution_time,
            cost_efficiency=cost_efficiency,
        )

    def _get_fee_rate(self, account_info: Dict[str, Any], order_type: str) -> float:
        """
        获取手续费率
        """
        exchange = account_info.get("exchange", "okx")
        account_tier = account_info.get("tier", "regular")

        fee_config = self.fee_configs.get(exchange, self.fee_configs["okx"])

        base_rate = fee_config[f"{order_type}_fee"]
        discount = fee_config["vip_discounts"].get(account_tier, 1.0)

        return base_rate * discount

    def _calculate_cost_efficiency(
        self, total_cost: float, order_value: float
    ) -> float:
        """
        计算成本效率评分

        Returns:
            0-100的评分，100为最优
        """
        cost_percentage = total_cost / order_value if order_value > 0 else 1.0

        # 基于成本百分比计算效率
        if cost_percentage <= 0.001:  # <=0.1%
            efficiency = 95
        elif cost_percentage <= 0.002:  # <=0.2%
            efficiency = 90
        elif cost_percentage <= 0.005:  # <=0.5%
            efficiency = 80
        elif cost_percentage <= 0.01:  # <=1%
            efficiency = 60
        elif cost_percentage <= 0.02:  # <=2%
            efficiency = 40
        else:
            efficiency = max(0, 100 - cost_percentage * 1000)

        return min(100, efficiency)

    def _create_optimized_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        strategy: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> OptimizedOrder:
        """
        创建优化后的订单
        """
        current_price = market_data.get("price", 50000)

        # 根据策略设置订单参数
        if strategy["type"] == OrderType.MARKET:
            price = None
            stop_price = None
            time_in_force = "IOC"

        elif strategy["type"] == OrderType.LIMIT:
            # 设置限价单价格
            spread = market_data.get("spread", 0.001)
            if side == "buy":
                price = current_price * (1 - spread * 0.5)
            else:
                price = current_price * (1 + spread * 0.5)
            stop_price = None
            time_in_force = "GTC"

        else:
            # 默认市价单
            price = None
            stop_price = None
            time_in_force = "IOC"

        return OptimizedOrder(
            symbol=symbol,
            side=side,
            amount=amount,
            order_type=strategy["type"],
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            estimated_cost=strategy["estimated_cost"],
            execution_strategy=strategy["strategy"],
            confidence=strategy.get("success_probability", 0.8),
        )

    def analyze_cost_performance(self, time_window_days: int = 30) -> Dict[str, Any]:
        """
        分析成本表现
        """
        if not self.cost_history:
            return {"total_trades": 0, "avg_cost_efficiency": 0}

        # 计算时间窗口内的数据
        cutoff_time = datetime.now() - timedelta(days=time_window_days)
        recent_costs = [
            cost
            for cost in self.cost_history
            if cost.timestamp is not None and cost.timestamp > cutoff_time
        ]

        if not recent_costs:
            return {"total_trades": 0, "avg_cost_efficiency": 0}

        total_trades = len(recent_costs)
        avg_efficiency = (
            sum(cost.cost_efficiency for cost in recent_costs) / total_trades
        )
        avg_total_cost = sum(cost.total_cost for cost in recent_costs) / total_trades

        # 成本分布
        cost_ranges = {
            "excellent": len([c for c in recent_costs if c.cost_efficiency >= 90]),
            "good": len([c for c in recent_costs if 80 <= c.cost_efficiency < 90]),
            "fair": len([c for c in recent_costs if 60 <= c.cost_efficiency < 80]),
            "poor": len([c for c in recent_costs if c.cost_efficiency < 60]),
        }

        return {
            "total_trades": total_trades,
            "avg_cost_efficiency": avg_efficiency,
            "avg_total_cost": avg_total_cost,
            "cost_distribution": cost_ranges,
            "time_window_days": time_window_days,
        }

    def record_execution_cost(self, cost_result: CostAnalysisResult):
        """
        记录执行成本（用于历史分析）
        """
        # 添加时间戳
        cost_result.timestamp = datetime.now()

        self.cost_history.append(cost_result)

        # 保留最近1000条记录
        if len(self.cost_history) > 1000:
            self.cost_history = self.cost_history[-1000:]

    def get_cost_optimization_recommendations(self) -> List[str]:
        """
        获取成本优化建议
        """
        recommendations = []
        performance = self.analyze_cost_performance()

        if performance["avg_cost_efficiency"] < 70:
            recommendations.append("⚠️ 平均成本效率较低，建议优化执行策略")
        if performance["cost_distribution"]["poor"] > performance["total_trades"] * 0.2:
            recommendations.append("⚠️ 太多交易成本过高，考虑减少交易频率或改进订单类型")
        if performance["avg_total_cost"] > 100:  # 假设平均交易成本
            recommendations.append("💰 交易成本较高，考虑升级VIP账户以获得费率折扣")

        recommendations.extend(
            [
                "✅ 建议使用限价单代替市价单以降低滑点成本",
                "✅ 大额订单建议分批执行以减少市场冲击",
                "✅ 在高波动时期考虑增加冷却时间",
                "✅ 监控执行时间，及时取消未成交订单",
            ]
        )

        return recommendations

    def reset_cost_history(self):
        """重置成本历史记录"""
        self.cost_history = []
        logger.info("交易成本优化器历史已重置")
