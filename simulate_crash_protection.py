#!/usr/bin/env python3
"""
模拟暴跌保护场景 - 验证AI Prompt中的暴跌保护规则
"""

import asyncio
import logging
from typing import Dict, Any

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def test_crash_protection():
    """测试暴跌保护场景"""

    print("\n" + "=" * 70)
    print("  暴跌保护功能测试")
    print("=" * 70 + "\n")

    # 场景1：正常市场
    print("-" * 70)
    print("  场景1: 正常市场（1小时跌幅 -0.5%）")
    print("-" * 70)

    market_data_normal: Dict[str, Any] = {
        "price": 76800.50,
        "change_percent": 1.85,
        "recent_drop_percent": -0.005,  # -0.5%
        "technical": {
            "rsi": 42.35,
            "macd": 15.20,
            "macd_histogram": 0.5,
            "adx": 25.0,
            "atr_percent": 1.5,
            "bb_position": 0.35,
            "trend_direction": "up",
            "trend_strength": 0.45,
        },
        "position": {},
    }

    from alpha_trading_bot.ai.prompt_builder import build_prompt

    prompt = build_prompt(market_data_normal)
    print("\n[Prompt摘要]")
    for line in prompt.split("\n"):
        if (
            "1小时" in line
            or "趋势方向" in line
            or "买入条件" in line
            or "暴跌保护" in line
        ):
            print(line)

    # 场景2：暴跌市场
    print("\n" + "-" * 70)
    print("  场景2: 暴跌市场（1小时跌幅 -2.5%）")
    print("-" * 70)

    market_data_crash: Dict[str, Any] = {
        "price": 75000.00,
        "change_percent": -2.5,
        "recent_drop_percent": -0.025,  # -2.5% 暴跌！
        "technical": {
            "rsi": 28.5,  # 超卖
            "macd": -5.0,
            "macd_histogram": -1.2,
            "adx": 35.0,
            "atr_percent": 4.0,  # 高波动
            "bb_position": 0.15,  # 接近下轨
            "trend_direction": "down",  # 趋势反转
            "trend_strength": 0.55,
        },
        "position": {
            "side": "long",
            "amount": 0.05,
            "entry_price": 76000.0,
            "unrealized_pnl": -50.0,
        },
    }

    prompt = build_prompt(market_data_crash)
    print("\n[Prompt摘要]")
    for line in prompt.split("\n"):
        if (
            "1小时" in line
            or "趋势方向" in line
            or "买入条件" in line
            or "暴跌保护" in line
            or "警告" in line
        ):
            print(line)

    # 场景3：RSI超卖但无暴跌
    print("\n" + "-" * 70)
    print("  场景3: RSI超卖但无暴跌（1小时跌幅 -0.8%）")
    print("-" * 70)

    market_data_oversold: Dict[str, Any] = {
        "price": 76500.00,
        "change_percent": 0.5,
        "recent_drop_percent": -0.008,  # -0.8%
        "technical": {
            "rsi": 28.0,  # 超卖
            "macd": 5.0,
            "macd_histogram": 0.3,
            "adx": 20.0,
            "atr_percent": 1.2,
            "bb_position": 0.18,
            "trend_direction": "up",
            "trend_strength": 0.25,
        },
        "position": {},
    }

    prompt = build_prompt(market_data_oversold)
    print("\n[Prompt摘要]")
    for line in prompt.split("\n"):
        if (
            "1小时" in line
            or "趋势方向" in line
            or "买入条件" in line
            or "暴跌保护" in line
        ):
            print(line)

    print("\n" + "=" * 70)
    print("  测试完成")
    print("=" * 70 + "\n")


async def main():
    """主函数"""
    await test_crash_protection()


if __name__ == "__main__":
    asyncio.run(main())
