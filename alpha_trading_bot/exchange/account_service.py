"""
账户服务 - 余额和持仓查询
"""

import asyncio
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AccountService:
    """账户服务"""

    def __init__(self, exchange, symbol: str, allow_short_selling: bool = True):
        self.exchange = exchange
        self.symbol = symbol
        self.allow_short_selling = allow_short_selling  # 是否允许做空

    async def get_balance(self) -> float:
        """获取可用USDT余额"""
        try:
            balance = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.exchange.fetch_balance()
            )
            usdt_available = balance.get("free", {}).get("USDT", 0)
            logger.info(f"可用USDT余额: {usdt_available}")
            return float(usdt_available)
        except Exception as e:
            logger.error(f"获取余额失败: {e}")
            return 0.0

    async def get_position_with_retry(
        self, max_retries: int = 3, retry_delay: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """
        P1修复: 获取持仓（带重试机制）
        
        特别优化：当返回"无持仓"时增加额外验证，避免OKX API延迟导致误判

        Args:
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）

        Returns:
            持仓信息字典，无持仓时返回None
        """
        last_error = None
        consecutive_no_position = 0  # 连续无持仓次数
        max_no_position_retries = 2  # 无持仓时的额外验证次数

        for attempt in range(max_retries):
            try:
                result = await self.get_position()
                
                # 如果返回无持仓，增加额外验证
                if result is None:
                    consecutive_no_position += 1
                    logger.warning(
                        f"[账户查询] 第 {attempt + 1} 次返回无持仓，连续 {consecutive_no_position} 次"
                    )
                    
                    # 如果连续多次返回无持仓，才确认真的无持仓
                    if consecutive_no_position <= max_no_position_retries and attempt < max_retries - 1:
                        # 无持仓时增加更长的延迟进行验证
                        extended_delay = retry_delay * 2  # 无持仓时延迟翻倍
                        logger.warning(
                            f"[账户查询] 无持仓待确认，{extended_delay:.1f}秒后再次验证..."
                        )
                        await asyncio.sleep(extended_delay)
                        continue
                    else:
                        logger.info("[账户查询] 确认无持仓")
                        return None
                else:
                    # 找到持仓，重置计数并返回
                    if consecutive_no_position > 0:
                        logger.info(
                            f"[账户查询] 验证成功，之前 {consecutive_no_position} 次误判为无持仓"
                        )
                    return result
                    
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"[账户查询] 获取持仓失败 (尝试 {attempt + 1}/{max_retries}): {e}. "
                        f"{retry_delay:.1f}秒后重试..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"[账户查询] 获取持仓失败 (已重试{max_retries}次): {e}"
                    )

        # 所有重试都失败，返回None但记录错误
        logger.error(f"[账户查询] 多次重试后仍失败: {last_error}")
        return None

    async def get_position(self) -> Optional[Dict[str, Any]]:
        """获取当前持仓（只支持做多，空单自动标记平仓）"""
        try:
            logger.debug(f"[账户查询] 正在获取 {self.symbol} 的持仓信息...")
            positions = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.exchange.fetch_positions([self.symbol])
            )

            for pos in positions:
                if pos["contracts"] and pos["contracts"] != 0:
                    side = pos["side"]
                    # 只有在禁止做空时，才强制平仓空单
                    # 如果允许做空，空单正常持有
                    if side == "short":
                        if not self.allow_short_selling:
                            # 禁止做空时，强制平仓
                            logger.warning(
                                f"[账户查询] 检测到空单(禁止做空): 数量={abs(pos['contracts'])}, "
                                f"入场价={pos['entryPrice']}, 系统将自动平仓"
                            )
                            position_info = {
                                "symbol": self.symbol,
                                "side": "short_to_close",  # 标记为空单需平仓
                                "amount": abs(pos["contracts"]),
                                "entry_price": pos["entryPrice"],
                                "unrealized_pnl": pos.get("unrealizedPnl", 0),
                            }
                        else:
                            # 允许做空时，空单正常持有
                            logger.info(
                                f"[账户查询] 检测到空单(允许做空): 数量={abs(pos['contracts'])}, "
                                f"入场价={pos['entryPrice']}, 正常持有"
                            )
                            position_info = {
                                "symbol": self.symbol,
                                "side": "short",  # 正常空单，不平仓
                                "amount": abs(pos["contracts"]),
                                "entry_price": pos["entryPrice"],
                                "unrealized_pnl": pos.get("unrealizedPnl", 0),
                            }
                    else:
                        position_info = {
                            "symbol": self.symbol,
                            "side": "long",
                            "amount": abs(pos["contracts"]),
                            "entry_price": pos["entryPrice"],
                            "unrealized_pnl": pos.get("unrealizedPnl", 0),
                        }
                    logger.info(
                        f"[账户查询] 找到持仓: 方向={position_info['side']}, "
                        f"数量={position_info['amount']}, 入场价={position_info['entry_price']}, "
                        f"未实现盈亏={position_info['unrealized_pnl']}"
                    )
                    return position_info

            logger.debug(f"[账户查询] {self.symbol} 无持仓")
            return None

        except Exception as e:
            logger.error(f"[账户查询] 获取持仓失败: {e}")
            return None


def create_account_service(exchange, symbol: str, allow_short_selling: bool = True) -> AccountService:
    """创建账户服务实例"""
    return AccountService(exchange, symbol, allow_short_selling)
