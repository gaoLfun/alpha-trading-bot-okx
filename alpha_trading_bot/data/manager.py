"""
数据管理器 - 统一的数据持久化接口
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from ..core.base import BaseComponent, BaseConfig
from .database import DatabaseManager
from .models import AISignalRecord, TradeRecord, MarketDataRecord, EquityRecord

logger = logging.getLogger(__name__)


class DataManagerConfig(BaseConfig):
    """数据管理器配置"""
    db_path: str = "data_json/trading_data.db"
    enable_json_backup: bool = True
    json_backup_path: str = "data_json"
    max_records_in_memory: int = 1000
    cleanup_old_data_days: int = 90


class DataManager(BaseComponent):
    """数据管理器 - 统一的数据持久化接口"""

    def __init__(self, config: Optional[DataManagerConfig] = None):
        """初始化数据管理器"""
        super().__init__(config or DataManagerConfig(name="DataManager"))
        self.db_manager: Optional[DatabaseManager] = None
        self._memory_cache: Dict[str, List] = {
            'ai_signals': [],
            'trades': [],
            'market_data': [],
            'equity': []
        }

    async def initialize(self) -> bool:
        """初始化数据管理器"""
        try:
            logger.info("正在初始化数据管理器...")

            # 确保数据目录存在
            Path(self.config.json_backup_path).mkdir(parents=True, exist_ok=True)

            # 初始化数据库管理器
            self.db_manager = DatabaseManager(self.config.db_path)

            # 加载最近的内存缓存
            await self._load_memory_cache()

            logger.info("数据管理器初始化成功")
            return True

        except Exception as e:
            logger.error(f"数据管理器初始化失败: {e}")
            return False

    async def cleanup(self) -> None:
        """清理资源"""
        if self.db_manager:
            # 清理旧数据
            self.db_manager.cleanup_old_data(self.config.cleanup_old_data_days)

    async def _load_memory_cache(self):
        """加载内存缓存"""
        try:
            # 加载最近的AI信号
            signals = self.db_manager.get_ai_signals(limit=self.config.max_records_in_memory // 4)
            self._memory_cache['ai_signals'] = signals

            # 加载最近的交易记录
            trades = self.db_manager.get_trades(limit=self.config.max_records_in_memory // 4)
            self._memory_cache['trades'] = trades

            # 加载最近的资产记录
            equity_records = self.db_manager.get_equity_history(limit=self.config.max_records_in_memory // 4)
            self._memory_cache['equity'] = equity_records

            logger.info(f"内存缓存加载完成 - AI信号: {len(signals)}, 交易: {len(trades)}, 资产: {len(equity_records)}")

        except Exception as e:
            logger.error(f"加载内存缓存失败: {e}")

    def _add_to_memory_cache(self, data_type: str, record):
        """添加到内存缓存"""
        if data_type in self._memory_cache:
            cache = self._memory_cache[data_type]
            cache.insert(0, record)  # 添加到开头（最新的）

            # 限制缓存大小
            if len(cache) > self.config.max_records_in_memory:
                cache.pop()  # 移除最旧的记录

    # AI信号相关方法
    async def save_ai_signal(self, signal_data: Dict[str, Any]) -> int:
        """保存AI信号"""
        try:
            # 创建AI信号记录
            signal = AISignalRecord(
                timestamp=signal_data.get('timestamp', datetime.now()),
                provider=signal_data.get('provider', 'unknown'),
                signal=signal_data.get('signal', 'HOLD'),
                confidence=signal_data.get('confidence', 0.0),
                reason=signal_data.get('reason', ''),
                market_price=signal_data.get('market_price', 0.0),
                market_data=signal_data.get('market_data', {}),
                used_in_trade=False
            )

            # 保存到数据库
            signal_id = self.db_manager.save_ai_signal(signal)

            # 更新内存缓存
            signal.id = signal_id
            self._add_to_memory_cache('ai_signals', signal)

            # 可选：备份到JSON文件
            if self.config.enable_json_backup:
                await self._backup_ai_signal_to_json(signal)

            logger.debug(f"AI信号已保存 - 提供商: {signal.provider}, 信号: {signal.signal}, 信心: {signal.confidence}")
            return signal_id

        except Exception as e:
            logger.error(f"保存AI信号失败: {e}")
            return -1

    async def _backup_ai_signal_to_json(self, signal: AISignalRecord):
        """备份AI信号到JSON文件"""
        try:
            backup_file = Path(self.config.json_backup_path) / "ai_signals.json"

            # 读取现有数据，处理可能的文件损坏
            data = []
            if backup_file.exists():
                try:
                    with open(backup_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        # 检查文件是否为空或只有空白
                        if content:
                            # 尝试修复不完整的JSON（缺少闭合括号）
                            if not content.endswith(']'):
                                # 找到最后一个完整的对象结尾
                                last_valid_end = content.rfind('}')
                                if last_valid_end != -1:
                                    # 保留到完整对象结束的部分
                                    content = content[:last_valid_end + 1] + ']'
                                else:
                                    # 如果没有找到完整对象，清空数据
                                    content = '[]'

                            # 解析JSON
                            data = json.loads(content)
                        else:
                            data = []
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"读取现有JSON文件失败，将创建新文件: {e}")
                    # 备份损坏的文件
                    backup_corrupted = backup_file.with_suffix('.json.corrupted')
                    backup_file.rename(backup_corrupted)
                    logger.info(f"已备份损坏的文件到: {backup_corrupted}")
                    data = []

            # 添加新数据
            data.append(signal.to_dict())

            # 限制数据量（保留最近1000条）
            if len(data) > 1000:
                data = data[-1000:]

            # 写回文件（使用临时文件确保原子性）
            temp_file = backup_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # 原子替换原文件
            temp_file.replace(backup_file)

            logger.info(f"成功备份AI信号到JSON: {backup_file}")

        except Exception as e:
            logger.error(f"备份AI信号到JSON失败: {e}")

    def get_ai_signals(self, limit: int = 100, provider: Optional[str] = None) -> List[AISignalRecord]:
        """获取AI信号历史"""
        return self.db_manager.get_ai_signals(limit, provider)

    async def update_ai_signal_trade_result(self, signal_id: int, trade_result: str, pnl: float):
        """更新AI信号的交易结果"""
        try:
            self.db_manager.update_ai_signal_trade_result(signal_id, trade_result, pnl)

            # 更新内存缓存
            for signal in self._memory_cache['ai_signals']:
                if signal.id == signal_id:
                    signal.trade_result = trade_result
                    signal.pnl = pnl
                    signal.used_in_trade = True
                    break

        except Exception as e:
            logger.error(f"更新AI信号交易结果失败: {e}")

    # 交易记录相关方法
    async def save_trade(self, trade_data: Dict[str, Any]) -> int:
        """保存交易记录"""
        try:
            # 创建交易记录
            trade = TradeRecord(
                timestamp=trade_data.get('timestamp', datetime.now()),
                symbol=trade_data.get('symbol', ''),
                side=trade_data.get('side', ''),
                price=trade_data.get('price', 0.0),
                amount=trade_data.get('amount', 0.0),
                cost=trade_data.get('cost', 0.0),
                fee=trade_data.get('fee', 0.0),
                status=trade_data.get('status', 'pending'),
                order_id=trade_data.get('order_id', ''),
                signal_source=trade_data.get('signal_source', ''),
                signal_confidence=trade_data.get('signal_confidence', 0.0),
                notes=trade_data.get('notes', '')
            )

            # 保存到数据库
            trade_id = self.db_manager.save_trade(trade)

            # 更新内存缓存
            trade.id = trade_id
            self._add_to_memory_cache('trades', trade)

            logger.info(f"交易记录已保存 - {trade.symbol} {trade.side} @ ${trade.price:.2f}")
            return trade_id

        except Exception as e:
            logger.error(f"保存交易记录失败: {e}")
            return -1

    async def update_trade(self, trade_id: int, **kwargs):
        """更新交易记录"""
        try:
            self.db_manager.update_trade(trade_id, **kwargs)

            # 更新内存缓存
            for trade in self._memory_cache['trades']:
                if trade.id == trade_id:
                    for key, value in kwargs.items():
                        if hasattr(trade, key):
                            setattr(trade, key, value)
                    break

        except Exception as e:
            logger.error(f"更新交易记录失败: {e}")

    def get_trades(self, limit: int = 100, symbol: Optional[str] = None) -> List[TradeRecord]:
        """获取交易历史"""
        return self.db_manager.get_trades(limit, symbol)

    def get_trade_statistics(self, days: int = 30) -> Dict[str, Any]:
        """获取交易统计信息"""
        return self.db_manager.get_trade_statistics(days)

    # 市场数据相关方法
    async def save_market_data(self, market_data: Dict[str, Any]):
        """保存市场数据快照"""
        try:
            # 创建市场数据记录
            record = MarketDataRecord(
                timestamp=market_data.get('timestamp', datetime.now()),
                symbol=market_data.get('symbol', ''),
                price=market_data.get('price', 0.0),
                bid=market_data.get('bid', 0.0),
                ask=market_data.get('ask', 0.0),
                volume=market_data.get('volume', 0.0),
                high=market_data.get('high', 0.0),
                low=market_data.get('low', 0.0),
                open=market_data.get('open', 0.0),
                close=market_data.get('close', 0.0),
                change_percent=market_data.get('change_percent', 0.0),
                market_state=market_data.get('market_state', ''),
                technical_indicators=market_data.get('technical_indicators', {})
            )

            # 保存到数据库
            self.db_manager.save_market_data(record)

            logger.debug(f"市场数据已保存 - {record.symbol} @ ${record.price:.2f}")

        except Exception as e:
            logger.error(f"保存市场数据失败: {e}")

    def get_market_data_history(self, symbol: str, limit: int = 100) -> List[MarketDataRecord]:
        """获取市场数据历史"""
        return self.db_manager.get_market_data(symbol, limit)

    # 资产记录相关方法
    async def save_equity_record(self, equity_data: Dict[str, Any]):
        """保存资产记录"""
        try:
            # 创建资产记录
            equity = EquityRecord(
                timestamp=equity_data.get('timestamp', datetime.now()),
                total_equity=equity_data.get('total_equity', 0.0),
                available_balance=equity_data.get('available_balance', 0.0),
                used_margin=equity_data.get('used_margin', 0.0),
                unrealized_pnl=equity_data.get('unrealized_pnl', 0.0),
                realized_pnl=equity_data.get('realized_pnl', 0.0),
                total_pnl=equity_data.get('total_pnl', 0.0),
                total_pnl_percent=equity_data.get('total_pnl_percent', 0.0),
                position_count=equity_data.get('position_count', 0),
                open_position_value=equity_data.get('open_position_value', 0.0)
            )

            # 保存到数据库
            self.db_manager.save_equity_record(equity)

            # 更新内存缓存
            self._add_to_memory_cache('equity', equity)

            # 可选：备份到JSON文件
            if self.config.enable_json_backup:
                await self._backup_equity_to_json(equity)

            logger.debug(f"资产记录已保存 - 总资产: ${equity.total_equity:.2f}, 盈亏: ${equity.total_pnl:.2f}")

        except Exception as e:
            logger.error(f"保存资产记录失败: {e}")

    async def _backup_equity_to_json(self, equity: EquityRecord):
        """备份资产记录到JSON文件"""
        try:
            backup_file = Path(self.config.json_backup_path) / "equity_history.json"

            # 读取现有数据
            if backup_file.exists():
                with open(backup_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = []

            # 添加新数据
            data.append(equity.to_dict())

            # 限制数据量（保留最近1000条）
            if len(data) > 1000:
                data = data[-1000:]

            # 写回文件
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"备份资产记录到JSON失败: {e}")

    def get_equity_history(self, limit: int = 100) -> List[EquityRecord]:
        """获取资产历史"""
        return self.db_manager.get_equity_history(limit)

    # 数据分析方法
    def get_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        """获取性能指标"""
        try:
            # 获取交易统计
            trade_stats = self.db_manager.get_trade_statistics(days)

            # 获取最近的资产记录
            equity_history = self.get_equity_history(limit=days * 24)  # 假设每天24条记录

            # 计算额外指标
            if equity_history:
                initial_equity = equity_history[-1].total_equity
                current_equity = equity_history[0].total_equity
                total_return = (current_equity - initial_equity) / initial_equity * 100 if initial_equity > 0 else 0

                # 计算最大回撤
                max_equity = max(eq.total_equity for eq in equity_history)
                current_drawdown = (max_equity - current_equity) / max_equity * 100 if max_equity > 0 else 0
            else:
                total_return = 0
                current_drawdown = 0

            return {
                'trade_statistics': trade_stats,
                'total_return_percent': total_return,
                'current_drawdown_percent': current_drawdown,
                'equity_records_count': len(equity_history),
                'analysis_period_days': days
            }

        except Exception as e:
            logger.error(f"获取性能指标失败: {e}")
            return {}

    # 工具方法
    def export_to_json(self, data_type: str, filename: str):
        """导出数据到JSON文件"""
        try:
            data = []

            if data_type == 'ai_signals':
                data = [signal.to_dict() for signal in self.get_ai_signals(limit=10000)]
            elif data_type == 'trades':
                data = [trade.to_dict() for trade in self.get_trades(limit=10000)]
            elif data_type == 'equity':
                data = [equity.to_dict() for equity in self.get_equity_history(limit=10000)]

            export_path = Path(self.config.json_backup_path) / filename
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"数据已导出到 {export_path}")

        except Exception as e:
            logger.error(f"导出数据失败: {e}")

    def get_status(self) -> Dict[str, Any]:
        """获取数据管理器状态"""
        base_status = super().get_status()
        base_status.update({
            'db_path': str(self.config.db_path),
            'json_backup_enabled': self.config.enable_json_backup,
            'json_backup_path': self.config.json_backup_path,
            'memory_cache_size': {
                'ai_signals': len(self._memory_cache['ai_signals']),
                'trades': len(self._memory_cache['trades']),
                'equity': len(self._memory_cache['equity'])
            }
        })
        return base_status


# 全局数据管理器实例
_data_manager: Optional[DataManager] = None


async def create_data_manager(config: Optional[DataManagerConfig] = None) -> DataManager:
    """创建数据管理器实例"""
    global _data_manager
    _data_manager = DataManager(config)
    await _data_manager.initialize()
    return _data_manager


async def get_data_manager() -> DataManager:
    """获取全局数据管理器实例"""
    global _data_manager
    if _data_manager is None:
        raise RuntimeError("数据管理器尚未初始化，请先调用 create_data_manager()")
    return _data_manager