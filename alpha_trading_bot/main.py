#!/usr/bin/env python3
"""
自适应交易机器人入口 v2.0

支持模式：
- 测试模式 (TEST_MODE=true): 不执行真实交易
- 实盘模式: 执行真实交易
"""

import asyncio
import logging
import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from alpha_trading_bot.core.adaptive_bot import AdaptiveTradingBot
from alpha_trading_bot.config.models import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Alpha Trading Bot v2.0 - AI驱动自适应交易机器人"
    )
    parser.add_argument(
        "--test", action="store_true", help="强制使用测试模式（不执行真实交易）"
    )
    parser.add_argument(
        "--real", action="store_true", help="强制使用实盘模式（执行真实交易）"
    )
    parser.add_argument("--symbol", type=str, help="交易品种 (例如: BTC/USDT)")
    return parser.parse_args()


def is_test_mode():
    """判断是否测试模式"""
    # 命令行参数优先
    args = parse_args()
    if args.test:
        return True
    if args.real:
        return False
    # 环境变量次之
    return os.getenv("TEST_MODE", "true").lower() == "true"


async def main():
    """主入口"""
    args = parse_args()

    print("=" * 60)
    print("Alpha Trading Bot v2.0 - AI驱动自适应交易系统")
    print("=" * 60)
    print()
    print("核心特性:")
    print("  • 市场环境感知 - 实时检测8种市场状态")
    print("  • 策略自动选择 - 根据市场状态动态切换策略")
    print("  • 风险边界控制 - 5%硬止损 + 动态熔断")
    print("  • 自适应学习 - 基于交易结果自动优化策略权重")
    print("  • 后台优化 - 每日贝叶斯参数优化")
    print()
    print("-" * 60)

    # 加载配置
    config = Config.from_env()

    # 确定运行模式
    test_mode = is_test_mode()
    print(f"[模式] {'测试模式' if test_mode else '实盘模式'}")

    if args.symbol:
        config.exchange.symbol = args.symbol
        print(f"[交易对] {args.symbol}")

    if not test_mode:
        print("⚠️ 警告: 实盘模式将执行真实交易！")
        print("-" * 60)

    print("-" * 60)

    # 初始化自适应交易机器人
    bot = AdaptiveTradingBot(config)

    try:
        await bot.run()
    except KeyboardInterrupt:
        print("\n收到停止信号...")
        bot._running = False  # 设置停止标志
        await bot.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序退出")
