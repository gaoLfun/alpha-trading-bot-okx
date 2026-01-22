"""
K线数据持久化管理模块

功能：
1. 将 K 线数据保存到本地文件（JSON 格式）
2. 启动时从本地加载历史数据
3. 实现增量更新，只获取新 K 线
4. 自动维护数据文件，清理过期数据

作者: Alpha Trading Bot
日期: 2026-01-23
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import time

logger = logging.getLogger(__name__)

# 数据目录
DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class OHLCVData:
    """K线数据"""

    timestamp: int  # 时间戳 (毫秒)
    open_time: str  # 开放时间字符串
    open_price: float  # 开盘价
    high_price: float  # 最高价
    low_price: float  # 最低价
    close_price: float  # 收盘价
    volume: float  # 成交量

    def to_list(self) -> List:
        """转换为列表格式（保持timestamp为整数，open_time为字符串）"""
        return [
            self.timestamp,  # 保持整数时间戳
            self.open_time,  # 字符串格式
            self.open_price,
            self.high_price,
            self.low_price,
            self.close_price,
            self.volume,
        ]

    @classmethod
    def from_list(cls, data: List) -> "OHLCVData":
        """从列表创建（兼容整数时间戳和字符串open_time）"""
        return cls(
            timestamp=int(data[0]),  # 确保是整数
            open_time=str(data[1]),  # 确保是字符串
            open_price=float(data[2]),
            high_price=float(data[3]),
            low_price=float(data[4]),
            close_price=float(data[5]),
            volume=float(data[6]),
        )

    @classmethod
    def from_ccxt(cls, candle: List) -> "OHLCVData":
        """从 CCXT 格式创建"""
        return cls(
            timestamp=candle[0],
            open_time=datetime.fromtimestamp(candle[0] / 1000).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            open_price=float(candle[1]),
            high_price=float(candle[2]),
            low_price=float(candle[3]),
            close_price=float(candle[4]),
            volume=float(candle[5]),
        )


@dataclass
class KLineFileMetadata:
    """K线文件元数据"""

    symbol: str
    timeframe: str
    last_update: str  # 最后更新时间
    last_timestamp: int  # 最后一条 K 线的时间戳
    count: int  # K 线数量
    file_size: int  # 文件大小（字节）

    def to_dict(self) -> Dict:
        return asdict(self)


class KLinePersistenceManager:
    """K线数据持久化管理器"""

    def __init__(self, data_dir: Path = None):
        """
        初始化 K 线持久化管理器

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 文件缓存 (symbol:timeframe -> file_path)
        self._file_cache: Dict[str, Path] = {}

        # 内存缓存 (symbol:timeframe -> List[OHLCVData])
        self._memory_cache: Dict[str, List[OHLCVData]] = {}

        # 最大保存天数
        self.max_days = 30  # 保存 30 天历史数据

        # 每个时间段的 K 线数量上限
        self.max_candles = {
            "5m": 30 * 24 * 12,  # 30天 * 24小时 * 12 (5分钟)
            "15m": 30 * 24 * 4,  # 30天 * 24小时 * 4 (15分钟)
            "1h": 30 * 24,  # 30天 * 24小时
            "4h": 30 * 6,  # 30天 * 6 (4小时)
            "1d": 30,  # 30天
        }

        logger.info(f"KLinePersistenceManager 初始化完成，数据目录: {self.data_dir}")

    def _get_file_path(self, symbol: str, timeframe: str) -> Path:
        """
        获取 K 线数据文件路径

        Args:
            symbol: 交易对 (如 BTC/USDT:USDT)
            timeframe: 时间周期 (如 5m, 15m, 1h)

        Returns:
            文件路径
        """
        cache_key = f"{symbol}:{timeframe}"
        if cache_key in self._file_cache:
            return self._file_cache[cache_key]

        # 清理特殊字符
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        filename = f"kline_{safe_symbol}_{timeframe}.json"
        file_path = self.data_dir / filename

        self._file_cache[cache_key] = file_path
        return file_path

    def save_klines(self, symbol: str, timeframe: str, klines: List[List]) -> bool:
        """
        保存 K 线数据到文件

        Args:
            symbol: 交易对
            timeframe: 时间周期
            klines: K 线数据列表（支持 CCXT 格式或已转换格式）

        Returns:
            是否保存成功
        """
        try:
            file_path = self._get_file_path(symbol, timeframe)

            # 转换数据（兼容 CCXT 格式和已保存格式）
            ohlcv_data = []
            for k in klines:
                if isinstance(k[0], int) and isinstance(k[1], str):
                    # 已经是 OHLCVData 格式（从文件加载的）
                    ohlcv_data.append(k)
                else:
                    # CCXT 格式，需要转换
                    ohlcv_data.append(OHLCVData.from_ccxt(k).to_list())

            if not ohlcv_data:
                logger.warning(f"没有 K 线数据需要保存: {symbol} {timeframe}")
                return False

            # 按时间戳排序
            ohlcv_data.sort(key=lambda x: x[0])

            # 限制数量，保留最近的 max_candles 条
            max_count = self.max_candles.get(timeframe, 2000)
            if len(ohlcv_data) > max_count:
                ohlcv_data = ohlcv_data[-max_count:]
                logger.info(f"已截取最近 {max_count} 根 K 线")

            # 构建文件数据
            file_data = {
                "metadata": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "last_update": datetime.now().isoformat(),
                    "last_timestamp": ohlcv_data[-1][0],
                    "count": len(ohlcv_data),
                    "version": "1.0",
                },
                "klines": ohlcv_data,
            }

            # 写入文件
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(file_data, f, indent=2, ensure_ascii=False)

            # 清理内存缓存
            cache_key = f"{symbol}:{timeframe}"
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]

            logger.info(
                f"✅ K线数据已保存: {symbol} {timeframe} - "
                f"{len(ohlcv_data)} 根, 文件: {file_path.name}"
            )
            return True

        except Exception as e:
            logger.error(f"保存 K 线数据失败: {symbol} {timeframe} - {e}")
            return False

    def load_klines(
        self, symbol: str, timeframe: str
    ) -> Tuple[List[List], Optional[KLineFileMetadata]]:
        """
        从文件加载 K 线数据

        Args:
            symbol: 交易对
            timeframe: 时间周期

        Returns:
            (K线数据列表, 元数据) - 如果没有本地数据则返回空列表
        """
        try:
            file_path = self._get_file_path(symbol, timeframe)

            if not file_path.exists():
                logger.info(f"本地 K 线数据文件不存在: {file_path}")
                return [], None

            # 检查文件大小
            file_size = file_path.stat().st_size
            if file_size == 0:
                logger.warning(f"本地 K 线数据文件为空: {file_path}")
                return [], None

            with open(file_path, "r", encoding="utf-8") as f:
                file_data = json.load(f)

            # 解析元数据
            metadata_dict = file_data.get("metadata", {})
            metadata = KLineFileMetadata(
                symbol=metadata_dict.get("symbol", symbol),
                timeframe=metadata_dict.get("timeframe", timeframe),
                last_update=metadata_dict.get("last_update", ""),
                last_timestamp=metadata_dict.get("last_timestamp", 0),
                count=metadata_dict.get("count", 0),
                file_size=file_size,
            )

            # 解析 K 线数据
            klines = file_data.get("klines", [])

            # 更新内存缓存
            cache_key = f"{symbol}:{timeframe}"
            self._memory_cache[cache_key] = klines

            logger.info(
                f"📂 已加载本地 K 线数据: {symbol} {timeframe} - "
                f"{len(klines)} 根, 更新时间: {metadata.last_update}"
            )

            return klines, metadata

        except json.JSONDecodeError as e:
            logger.error(f"解析 K 线数据文件失败: {file_path} - {e}")
            return [], None
        except Exception as e:
            logger.error(f"加载 K 线数据失败: {symbol} {timeframe} - {e}")
            return [], None

    def get_klines(
        self, symbol: str, timeframe: str, limit: int = None, since: int = None
    ) -> Tuple[List[List], bool]:
        """
        获取 K 线数据（支持增量更新）

        Args:
            symbol: 交易对
            timeframe: 时间周期
            limit: 限制数量 (默认使用配置文件或 max_candles)
            since: 起始时间戳 (毫秒)

        Returns:
            (K线数据列表, 是否为增量更新)
        """
        cache_key = f"{symbol}:{timeframe}"

        # 1. 先尝试从本地加载
        local_klines, metadata = self.load_klines(symbol, timeframe)

        # 2. 判断是否需要获取新数据
        need_fetch = True
        current_timestamp = int(time.time() * 1000)

        if local_klines and metadata:
            # 计算时间范围
            timeframe_ms = self._timeframe_to_ms(timeframe)
            since_timestamp = since or (
                current_timestamp - 7 * 24 * 60 * 60 * 1000
            )  # 默认 7 天

            # 检查本地数据是否足够
            oldest_local_timestamp = local_klines[0][0] if local_klines else 0

            if since and since >= oldest_local_timestamp:
                # 只请求指定时间之后的数据
                need_fetch = True
            elif since is None and len(local_klines) >= (
                limit or self.max_candles.get(timeframe, 2000)
            ):
                # 本地数据足够，且是最新的
                need_fetch = False
            else:
                # 检查本地数据是否过期（超过 5 分钟）
                last_update = datetime.fromisoformat(metadata.last_update)
                if (datetime.now() - last_update).total_seconds() < 300:
                    need_fetch = False

        # 3. 如果需要获取新数据
        if need_fetch:
            # 这里返回本地数据 + 需要获取的起始时间戳
            # 实际获取由调用方完成
            pass

        # 过滤和截取数据
        result_klines = local_klines

        if since:
            result_klines = [k for k in result_klines if k[0] >= since]

        if limit:
            result_klines = result_klines[-limit:]

        return result_klines, need_fetch

    def merge_klines(
        self, symbol: str, timeframe: str, new_klines: List[List]
    ) -> List[List]:
        """
        合并新旧 K 线数据

        Args:
            symbol: 交易对
            timeframe: 时间周期
            new_klines: 新获取的 K 线数据

        Returns:
            合并后的 K 线数据
        """
        # 加载本地数据
        local_klines, _ = self.load_klines(symbol, timeframe)

        if not local_klines:
            # 没有本地数据，直接保存
            self.save_klines(symbol, timeframe, new_klines)
            return new_klines

        # 合并数据
        all_klines = {}

        # 添加本地数据
        for k in local_klines:
            all_klines[k[0]] = k

        # 添加新数据
        for k in new_klines:
            all_klines[k[0]] = k

        # 转换为列表并排序
        merged = list(all_klines.values())
        merged.sort(key=lambda x: x[0])

        # 限制数量
        max_count = self.max_candles.get(timeframe, 2000)
        if len(merged) > max_count:
            merged = merged[-max_count:]

        return merged

    def update_klines(
        self, symbol: str, timeframe: str, new_klines: List[List]
    ) -> bool:
        """
        更新 K 线数据（增量更新）

        Args:
            symbol: 交易对
            timeframe: 时间周期
            new_klines: 新获取的 K 线数据

        Returns:
            是否更新成功
        """
        # 合并数据
        merged_klines = self.merge_klines(symbol, timeframe, new_klines)

        # 保存
        return self.save_klines(symbol, timeframe, merged_klines)

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """将时间周期转换为毫秒"""
        unit = timeframe[-1]
        value = int(timeframe[:-1])

        multipliers = {
            "m": 60 * 1000,  # 分钟
            "h": 60 * 60 * 1000,  # 小时
            "d": 24 * 60 * 60 * 1000,  # 天
            "w": 7 * 24 * 60 * 60 * 1000,  # 周
        }

        return value * multipliers.get(unit, 60 * 1000)

    def get_data_info(self, symbol: str, timeframe: str) -> Dict:
        """
        获取 K 线数据信息

        Args:
            symbol: 交易对
            timeframe: 时间周期

        Returns:
            数据信息字典
        """
        file_path = self._get_file_path(symbol, timeframe)

        if not file_path.exists():
            return {
                "exists": False,
                "symbol": symbol,
                "timeframe": timeframe,
                "count": 0,
                "file_path": str(file_path),
            }

        # 加载元数据
        _, metadata = self.load_klines(symbol, timeframe)

        return {
            "exists": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "count": metadata.count if metadata else 0,
            "last_update": metadata.last_update if metadata else "",
            "last_timestamp": metadata.last_timestamp if metadata else 0,
            "file_path": str(file_path),
            "file_size": metadata.file_size if metadata else 0,
        }

    def cleanup_old_data(self, symbol: str = None, timeframe: str = None):
        """
        清理过期数据

        Args:
            symbol: 交易对 (None 则清理所有)
            timeframe: 时间周期 (None 则清理所有)
        """
        if symbol and timeframe:
            # 清理单个文件
            file_path = self._get_file_path(symbol, timeframe)
            if file_path.exists():
                # 重新加载并保存（会触发截断）
                klines, _ = self.load_klines(symbol, timeframe)
                if klines:
                    self.save_klines(symbol, timeframe, klines)
                    logger.info(f"已清理数据: {symbol} {timeframe}")
        else:
            # 清理所有文件
            for file_path in self.data_dir.glob("kline_*.json"):
                try:
                    # 提取 symbol 和 timeframe
                    parts = file_path.stem.replace("kline_", "").split("_")
                    if len(parts) >= 3:
                        s = parts[0] + "/" + parts[1].replace("USDT", ":USDT")
                        t = "_".join(parts[2:])
                        self.cleanup_old_data(s, t)
                except Exception as e:
                    logger.warning(f"清理数据文件失败: {file_path} - {e}")

    def clear_cache(self):
        """清理内存缓存"""
        self._memory_cache.clear()
        logger.info("K线数据内存缓存已清理")


# 全局实例
kline_persistence_manager = KLinePersistenceManager()


def get_kline_manager() -> KLinePersistenceManager:
    """获取 K 线持久化管理器实例"""
    return kline_persistence_manager
