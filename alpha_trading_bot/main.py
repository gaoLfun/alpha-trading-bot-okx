#!/usr/bin/env python3
"""
精简版交易机器人入口
"""

import asyncio
import logging
from core import TradingBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def main():
    """主入口"""
    print("=" * 50)
    print("Alpha Trading Bot - AI驱动加密货币交易机器人入口")
    print("=" * 50)
    
    bot = TradingBot()
    try:
        await bot.run()
    except KeyboardInterrupt:
        print("\n收到停止信号...")
        await bot.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序退出")
