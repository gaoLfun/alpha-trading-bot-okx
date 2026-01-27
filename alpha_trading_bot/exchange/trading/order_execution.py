"""
订单执行模块 - 从 trade_executor.py 拆分出来的核心执行逻辑
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ...core.base import BaseComponent
from ..models import (
    TradeResult,
    OrderResult,
    TradeSide,
    OrderStatus,
)

logger = logging.getLogger(__name__)


class OrderExecutionMixin(BaseComponent):
    """订单执行混合类 - 包含核心订单执行逻辑"""

    async def _execute_single_order(
        self,
        symbol: str,
        side: TradeSide,
        amount: float,
        order_type: str,
        price: Optional[float] = None,
    ) -> OrderResult:
        """
        执行单个订单

        Args:
            symbol: 交易对
            side: 买卖方向
            amount: 数量
            order_type: 订单类型 (market/limit)
            price: 价格（限价单需要）

        Returns:
            OrderResult: 订单执行结果
        """
        try:
            if order_type == "market":
                # 市价单
                order_result = await self.exchange_client.create_market_order(
                    symbol=symbol,
                    side=side.value,
                    amount=amount,
                )
            else:
                # 限价单
                if price is None:
                    price = await self._get_current_price(symbol)
                order_result = await self.exchange_client.create_limit_order(
                    symbol=symbol,
                    side=side.value,
                    amount=amount,
                    price=price,
                )

            if order_result:
                logger.info(
                    f"订单执行成功: {symbol} {side.value} {amount} @ {price or '市价'}"
                )
                return OrderResult(
                    success=True,
                    order_id=str(order_result.get("id", "")),
                    status=OrderStatus.FILLED,
                    filled_amount=order_result.get("filled", amount),
                    average_price=order_result.get("average", price or 0),
                    fee=order_result.get("fee", {}).get("cost", 0),
                )
            else:
                logger.error(f"订单执行失败: {symbol} {side.value} {amount}")
                return OrderResult(
                    success=False,
                    order_id="",
                    status=OrderStatus.REJECTED,
                    filled_amount=0,
                    average_price=0,
                    fee=0,
                    error_message="订单执行返回空结果",
                )

        except Exception as e:
            logger.error(f"订单执行异常: {e}")
            return OrderResult(
                success=False,
                order_id="",
                status=OrderStatus.REJECTED,
                filled_amount=0,
                average_price=0,
                fee=0,
                error_message=str(e),
            )

    async def _get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        try:
            ticker = await self.exchange_client.fetch_ticker(symbol)
            return ticker.get("last", 0) or ticker.get("mid", 0) or 0
        except Exception as e:
            logger.error(f"获取当前价格失败: {e}")
            return 0.0

    def _determine_risk_level(self, trade_request: Dict[str, Any]) -> str:
        """
        确定交易风险等级

        Args:
            trade_request: 交易请求

        Returns:
            str: 风险等级 (low/medium/high)
        """
        # 基于信号强度确定风险等级
        confidence = trade_request.get("confidence", 0.5)

        if confidence >= 0.8:
            return "low"
        elif confidence >= 0.5:
            return "medium"
        else:
            return "high"

    def _determine_market_volatility(self, ohlcv_data: list) -> float:
        """
        确定市场波动率

        Args:
            ohlcv_data: K线数据

        Returns:
            float: 波动率 (0-1)
        """
        if not ohlcv_data or len(ohlcv_data) < 2:
            return 0.5

        try:
            closes = [d[4] for d in ohlcv_data]
            if len(closes) < 2:
                return 0.5

            # 计算价格变化的标准差
            import numpy as np

            returns = []
            for i in range(1, len(closes)):
                if closes[i - 1] > 0:
                    returns.append((closes[i] - closes[i - 1]) / closes[i - 1])

            if not returns:
                return 0.5

            volatility = np.std(returns)
            # 将波动率归一化到 0-1 范围
            volatility = min(1.0, volatility * 100)  # 假设10%的波动率是100%

            return max(0.0, min(1.0, volatility))

        except Exception as e:
            logger.warning(f"计算市场波动率失败: {e}")
            return 0.5
