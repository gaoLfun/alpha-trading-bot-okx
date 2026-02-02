"""
订单服务 - 下单、撤单、止损止盈
"""

import asyncio
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class OrderService:
    """订单服务"""

    def __init__(self, exchange, symbol: str):
        self.exchange = exchange
        self.symbol = symbol
        self._stop_orders: Dict[str, str] = {}

    async def create_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        order_type: str = "market",
    ) -> str:
        """创建订单"""
        params = {
            "tdMode": "cross",
            "posMode": "one_way",
        }

        logger.info(
            f"[订单创建] 提交订单: symbol={symbol}, side={side}, "
            f"type={order_type}, amount={amount}, price={price}"
        )

        order = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params,
            ),
        )

        order_id = order["id"]
        logger.info(
            f"[订单创建] 订单成功: ID={order_id}, 符号={symbol}, 方向={side}, 数量={amount}"
        )
        return order_id

    async def create_stop_loss(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float,
    ) -> str:
        """创建止损单"""
        params = {
            "tdMode": "cross",
            "posMode": "one_way",
            "stopLossPrice": stop_price,
            "closePosition": True,
        }

        logger.info(
            f"[止损单创建] 提交止损单: symbol={symbol}, side={side}, "
            f"amount={amount}, stop_price={stop_price}"
        )

        order = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.exchange.create_order(
                symbol=symbol,
                type="limit",
                side=side,
                amount=amount,
                price=stop_price * 0.999,
                params=params,
            ),
        )

        algo_id = order.get("info", {}).get("algoId", order["id"])
        self._stop_orders[str(stop_price)] = algo_id
        logger.info(
            f"[止损单创建] 止损单成功: ID={algo_id}, 止损价={stop_price}, 数量={amount}"
        )
        return algo_id

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消订单"""
        try:
            logger.info(f"[订单取消] 取消订单: ID={order_id}, symbol={symbol}")
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.exchange.cancel_order(order_id, symbol)
            )
            logger.info(f"[订单取消] 订单取消成功: {order_id}")
            return True
        except Exception as e:
            logger.error(f"[订单取消] 取消订单失败: {order_id}, 错误={e}")
            return False

    def get_stop_order_id(self, stop_price: float) -> Optional[str]:
        """获取止损单ID"""
        return self._stop_orders.get(str(stop_price))


def create_order_service(exchange, symbol: str) -> OrderService:
    """创建订单服务实例"""
    return OrderService(exchange, symbol)
