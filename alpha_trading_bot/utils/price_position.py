"""
统一价格位置计算工具 - 消除项目中的重复计算逻辑
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PricePositionResult:
    """价格位置计算结果"""

    position_ratio: float  # 位置比例 (0.0 - 1.0)
    position_percentage: float  # 位置百分比 (0.0 - 100.0)
    price: float
    low: float
    high: float
    range_size: float
    period: str
    is_valid: bool = True


class PricePositionCalculator:
    """统一价格位置计算器"""

    @staticmethod
    def calculate_position(
        price: float, low: float, high: float, period: str = "unknown"
    ) -> PricePositionResult:
        """
        计算价格在区间中的位置

        Args:
            price: 当前价格
            low: 区间最低价
            high: 区间最高价
            period: 时间周期描述

        Returns:
            PricePositionResult: 计算结果
        """
        try:
            # 验证输入
            if not all(isinstance(x, (int, float)) for x in [price, low, high]):
                logger.warning(
                    f"价格位置计算输入无效: price={price}, low={low}, high={high}"
                )
                return PricePositionResult(
                    position_ratio=0.5,
                    position_percentage=50.0,
                    price=price,
                    low=low,
                    high=high,
                    range_size=0,
                    period=period,
                    is_valid=False,
                )

            # 处理边界情况
            if high <= low:
                # 如果最高价小于等于最低价，返回中间位置
                logger.warning(f"价格区间无效: high={high} <= low={low}，返回中间位置")
                return PricePositionResult(
                    position_ratio=0.5,
                    position_percentage=50.0,
                    price=price,
                    low=low,
                    high=high,
                    range_size=0,
                    period=period,
                    is_valid=False,
                )

            # 计算价格位置
            range_size = high - low
            position_ratio = (price - low) / range_size
            position_percentage = position_ratio * 100

            # 限制在[0, 1]范围内
            position_ratio = max(0.0, min(1.0, position_ratio))
            position_percentage = max(0.0, min(100.0, position_percentage))

            return PricePositionResult(
                position_ratio=position_ratio,
                position_percentage=position_percentage,
                price=price,
                low=low,
                high=high,
                range_size=range_size,
                period=period,
                is_valid=True,
            )

        except Exception as e:
            logger.error(f"计算价格位置失败: {e}")
            return PricePositionResult(
                position_ratio=0.5,
                position_percentage=50.0,
                price=price,
                low=low,
                high=high,
                range_size=0,
                period=period,
                is_valid=False,
            )

    @staticmethod
    def calculate_from_market_data(
        market_data: Dict[str, Any], period: str = "24h"
    ) -> Optional[PricePositionResult]:
        """
        从市场数据中计算价格位置

        Args:
            market_data: 包含价格数据的字典
            period: 时间周期 ("24h", "7d", "daily" 等)

        Returns:
            PricePositionResult or None
        """
        try:
            price = market_data.get("price", 0)

            # 根据周期选择高低价字段
            if period == "24h":
                low = market_data.get("low", 0)
                high = market_data.get("high", 0)
            elif period == "7d":
                low = market_data.get("low_7d", 0)
                high = market_data.get("high_7d", 0)
            elif period == "daily":
                low = market_data.get("daily_low", 0)
                high = market_data.get("daily_high", 0)
            else:
                # 默认尝试多个字段
                low = market_data.get("low", market_data.get("low_24h", 0))
                high = market_data.get("high", market_data.get("high_24h", 0))

            if price <= 0 or low <= 0 or high <= 0:
                logger.warning(
                    f"市场数据中的价格信息无效: price={price}, low={low}, high={high}"
                )
                return None

            return PricePositionCalculator.calculate_position(price, low, high, period)

        except Exception as e:
            logger.error(f"从市场数据计算价格位置失败: {e}")
            return None

    @staticmethod
    def calculate_from_ohlcv(
        ohlcv_data: list, period: str = "ohlcv"
    ) -> Optional[PricePositionResult]:
        """
        从OHLCV数据计算价格位置

        Args:
            ohlcv_data: K线数据列表 [[timestamp, open, high, low, close, volume], ...]
            period: 时间周期描述

        Returns:
            PricePositionResult or None
        """
        try:
            if not ohlcv_data or len(ohlcv_data) < 2:
                logger.warning("OHLCV数据不足，无法计算价格位置")
                return None

            # 获取最新K线的收盘价作为当前价格
            current_candle = ohlcv_data[-1]
            price = current_candle[4]  # close price

            # 计算整个周期的高低价
            all_highs = [candle[2] for candle in ohlcv_data]  # high prices
            all_lows = [candle[3] for candle in ohlcv_data]  # low prices

            high = max(all_highs)
            low = min(all_lows)

            if price <= 0 or high <= low:
                logger.warning(
                    f"OHLCV数据中的价格信息无效: price={price}, low={low}, high={high}"
                )
                return None

            return PricePositionCalculator.calculate_position(price, low, high, period)

        except Exception as e:
            logger.error(f"从OHLCV数据计算价格位置失败: {e}")
            return None

    @staticmethod
    def get_position_category(position_percentage: float) -> str:
        """
        根据位置百分比获取分类

        Args:
            position_percentage: 位置百分比 (0.0 - 100.0)

        Returns:
            str: 位置分类
        """
        if position_percentage <= 20:
            return "低位"
        elif position_percentage <= 40:
            return "中低位"
        elif position_percentage <= 60:
            return "中位"
        elif position_percentage <= 80:
            return "中高位"
        else:
            return "高位"

    @staticmethod
    def get_trading_signal_suggestion(
        position_result: PricePositionResult,
    ) -> Dict[str, Any]:
        """
        根据价格位置提供交易信号建议

        Args:
            position_result: 价格位置计算结果

        Returns:
            Dict: 交易建议
        """
        if not position_result.is_valid:
            return {
                "signal": "HOLD",
                "confidence": 0.3,
                "reason": "价格位置数据无效",
                "position_info": position_result,
            }

        position_pct = position_result.position_percentage
        category = PricePositionCalculator.get_position_category(position_pct)

        # 基于位置的简单建议逻辑
        if category == "低位":
            return {
                "signal": "BUY",
                "confidence": 0.6,
                "reason": f"价格处于{category}({position_pct:.1f}%)，考虑买入",
                "position_info": position_result,
                "position_category": category,
            }
        elif category == "高位":
            return {
                "signal": "SELL",
                "confidence": 0.6,
                "reason": f"价格处于{category}({position_pct:.1f}%)，考虑卖出",
                "position_info": position_result,
                "position_category": category,
            }
        else:
            return {
                "signal": "HOLD",
                "confidence": 0.5,
                "reason": f"价格处于{category}({position_pct:.1f}%)，观望",
                "position_info": position_result,
                "position_category": category,
            }

    @staticmethod
    def batch_calculate_positions(
        market_data: Dict[str, Any],
    ) -> Dict[str, PricePositionResult]:
        """
        批量计算多个时间周期的价格位置

        Args:
            market_data: 市场数据

        Returns:
            Dict: 各周期的价格位置结果
        """
        results = {}

        # 计算24小时位置
        position_24h = PricePositionCalculator.calculate_from_market_data(
            market_data, "24h"
        )
        if position_24h:
            results["24h"] = position_24h

        # 计算7日位置
        position_7d = PricePositionCalculator.calculate_from_market_data(
            market_data, "7d"
        )
        if position_7d:
            results["7d"] = position_7d

        # 计算日线位置
        position_daily = PricePositionCalculator.calculate_from_market_data(
            market_data, "daily"
        )
        if position_daily:
            results["daily"] = position_daily

        # 从OHLCV计算
        ohlcv_data = market_data.get("ohlcv")
        if ohlcv_data:
            position_ohlcv = PricePositionCalculator.calculate_from_ohlcv(
                ohlcv_data, "ohlcv"
            )
            if position_ohlcv:
                results["ohlcv"] = position_ohlcv

        return results

    @staticmethod
    def log_position_info(
        position_result: PricePositionResult, prefix: str = ""
    ) -> None:
        """
        记录价格位置信息

        Args:
            position_result: 价格位置计算结果
            prefix: 日志前缀
        """
        if not position_result.is_valid:
            logger.warning(
                f"{prefix}价格位置数据无效 - 价格: ${position_result.price:.2f}, 区间: ${position_result.low:.2f} - ${position_result.high:.2f}"
            )
            return

        category = PricePositionCalculator.get_position_category(
            position_result.position_percentage
        )
        logger.info(f"{prefix}价格位置分析 - {position_result.period}:")
        logger.info(f"  💰 当前价格: ${position_result.price:.2f}")
        logger.info(
            f"  📊 价格区间: ${position_result.low:.2f} - ${position_result.high:.2f}"
        )
        logger.info(f"  📍 位置比例: {position_result.position_ratio:.3f}")
        logger.info(f"  📍 位置百分比: {position_result.position_percentage:.1f}%")
        logger.info(f"  🏷️  位置分类: {category}")
        logger.info(f"  📏 区间宽度: ${position_result.range_size:.2f}")


# 便捷的全局函数
def calculate_price_position(
    price: float, low: float, high: float, period: str = "unknown"
) -> PricePositionResult:
    """便捷的价格位置计算函数"""
    return PricePositionCalculator.calculate_position(price, low, high, period)


def calculate_price_position_from_market_data(
    market_data: Dict[str, Any], period: str = "24h"
) -> Optional[PricePositionResult]:
    """便捷的市场数据价格位置计算函数"""
    return PricePositionCalculator.calculate_from_market_data(market_data, period)
