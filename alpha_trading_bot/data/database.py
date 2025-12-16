"""
SQLite数据库管理
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .models import AISignalRecord, TradeRecord, MarketDataRecord, EquityRecord

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite数据库管理器"""

    def __init__(self, db_path: str = "data_json/trading_data.db"):
        """初始化数据库管理器"""
        self.db_path = Path(db_path)
        # 确保目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """初始化数据库和表结构"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # AI信号表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reason TEXT,
                    market_price REAL NOT NULL,
                    market_data TEXT,
                    used_in_trade BOOLEAN DEFAULT FALSE,
                    trade_result TEXT,
                    pnl REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 交易记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    amount REAL NOT NULL,
                    cost REAL NOT NULL,
                    fee REAL DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    order_id TEXT,
                    signal_source TEXT,
                    signal_confidence REAL DEFAULT 0,
                    pnl REAL,
                    pnl_percent REAL,
                    exit_price REAL,
                    exit_time TEXT,
                    holding_period INTEGER,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 市场数据表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    bid REAL,
                    ask REAL,
                    volume REAL,
                    high REAL,
                    low REAL,
                    open REAL,
                    close REAL,
                    change_percent REAL,
                    market_state TEXT,
                    technical_indicators TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 资产记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS equity_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_equity REAL NOT NULL,
                    available_balance REAL NOT NULL,
                    used_margin REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    total_pnl_percent REAL DEFAULT 0,
                    position_count INTEGER DEFAULT 0,
                    open_position_value REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_signals_timestamp ON ai_signals(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_signals_provider ON ai_signals(provider)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity_records(timestamp)")

            conn.commit()
            logger.info("数据库初始化完成")

    # AI信号相关方法
    def save_ai_signal(self, signal: AISignalRecord) -> int:
        """保存AI信号"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ai_signals (
                    timestamp, provider, signal, confidence, reason,
                    market_price, market_data, used_in_trade, trade_result, pnl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.timestamp.isoformat(),
                signal.provider,
                signal.signal,
                signal.confidence,
                signal.reason,
                signal.market_price,
                json.dumps(signal.market_data),
                signal.used_in_trade,
                signal.trade_result,
                signal.pnl
            ))
            return cursor.lastrowid

    def get_ai_signals(self, limit: int = 100, provider: Optional[str] = None) -> List[AISignalRecord]:
        """获取AI信号历史"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM ai_signals"
            params = []

            if provider:
                query += " WHERE provider = ?"
                params.append(provider)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            signals = []
            for row in rows:
                signal = AISignalRecord(
                    id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    provider=row[2],
                    signal=row[3],
                    confidence=row[4],
                    reason=row[5],
                    market_price=row[6],
                    market_data=json.loads(row[7]) if row[7] else {},
                    used_in_trade=bool(row[8]),
                    trade_result=row[9],
                    pnl=row[10]
                )
                signals.append(signal)

            return signals

    def update_ai_signal_trade_result(self, signal_id: int, trade_result: str, pnl: float):
        """更新AI信号的交易结果"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE ai_signals
                SET trade_result = ?, pnl = ?, used_in_trade = TRUE
                WHERE id = ?
            """, (trade_result, pnl, signal_id))

    # 交易记录相关方法
    def save_trade(self, trade: TradeRecord) -> int:
        """保存交易记录"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (
                    timestamp, symbol, side, price, amount, cost, fee,
                    status, order_id, signal_source, signal_confidence,
                    pnl, pnl_percent, exit_price, exit_time, holding_period, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.timestamp.isoformat(),
                trade.symbol,
                trade.side,
                trade.price,
                trade.amount,
                trade.cost,
                trade.fee,
                trade.status.value,
                trade.order_id,
                trade.signal_source,
                trade.signal_confidence,
                trade.pnl,
                trade.pnl_percent,
                trade.exit_price,
                trade.exit_time.isoformat() if trade.exit_time else None,
                trade.holding_period,
                trade.notes
            ))
            return cursor.lastrowid

    def update_trade(self, trade_id: int, **kwargs):
        """更新交易记录"""
        allowed_fields = {
            'status', 'pnl', 'pnl_percent', 'exit_price',
            'exit_time', 'holding_period', 'notes'
        }

        update_fields = []
        values = []

        for key, value in kwargs.items():
            if key in allowed_fields:
                update_fields.append(f"{key} = ?")
                if key == 'exit_time' and value:
                    values.append(value.isoformat())
                else:
                    values.append(value)

        if not update_fields:
            return

        values.append(trade_id)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = f"UPDATE trades SET {', '.join(update_fields)} WHERE id = ?"
            cursor.execute(query, values)

    def get_trades(self, limit: int = 100, symbol: Optional[str] = None) -> List[TradeRecord]:
        """获取交易历史"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM trades"
            params = []

            if symbol:
                query += " WHERE symbol = ?"
                params.append(symbol)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            trades = []
            for row in rows:
                trade = TradeRecord(
                    id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    symbol=row[2],
                    side=row[3],
                    price=row[4],
                    amount=row[5],
                    cost=row[6],
                    fee=row[7],
                    status=TradeStatus(row[8]),
                    order_id=row[9],
                    signal_source=row[10],
                    signal_confidence=row[11],
                    pnl=row[12],
                    pnl_percent=row[13],
                    exit_price=row[14],
                    exit_time=datetime.fromisoformat(row[15]) if row[15] else None,
                    holding_period=row[16],
                    notes=row[17]
                )
                trades.append(trade)

            return trades

    def get_trade_statistics(self, days: int = 30) -> Dict[str, Any]:
        """获取交易统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 获取日期范围
            cursor.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                    COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                    AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
                    SUM(fee) as total_fees
                FROM trades
                WHERE timestamp >= datetime('now', '-{} days')
                AND status = 'executed'
            """.format(days))

            row = cursor.fetchone()

            if row and row[0] > 0:
                total_trades = row[0]
                winning_trades = row[1] or 0
                losing_trades = row[2] or 0
                total_pnl = row[3] or 0
                avg_pnl = row[4] or 0
                avg_win = row[5] or 0
                avg_loss = row[6] or 0
                total_fees = row[7] or 0

                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

                return {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'avg_pnl': avg_pnl,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'total_fees': total_fees,
                    'period_days': days
                }
            else:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'profit_factor': 0,
                    'total_fees': 0,
                    'period_days': days
                }

    # 市场数据相关方法
    def save_market_data(self, market_data: MarketDataRecord) -> int:
        """保存市场数据"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO market_data (
                    timestamp, symbol, price, bid, ask, volume,
                    high, low, open, close, change_percent,
                    market_state, technical_indicators
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                market_data.timestamp.isoformat(),
                market_data.symbol,
                market_data.price,
                market_data.bid,
                market_data.ask,
                market_data.volume,
                market_data.high,
                market_data.low,
                market_data.open,
                market_data.close,
                market_data.change_percent,
                market_data.market_state,
                json.dumps(market_data.technical_indicators)
            ))
            return cursor.lastrowid

    def get_market_data(self, symbol: str, limit: int = 100) -> List[MarketDataRecord]:
        """获取市场数据历史"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM market_data
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (symbol, limit))

            rows = cursor.fetchall()
            market_data_list = []

            for row in rows:
                market_data = MarketDataRecord(
                    id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    symbol=row[2],
                    price=row[3],
                    bid=row[4],
                    ask=row[5],
                    volume=row[6],
                    high=row[7],
                    low=row[8],
                    open=row[9],
                    close=row[10],
                    change_percent=row[11],
                    market_state=row[12],
                    technical_indicators=json.loads(row[13]) if row[13] else {}
                )
                market_data_list.append(market_data)

            return market_data_list

    # 资产记录相关方法
    def save_equity_record(self, equity: EquityRecord) -> int:
        """保存资产记录"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO equity_records (
                    timestamp, total_equity, available_balance, used_margin,
                    unrealized_pnl, realized_pnl, total_pnl, total_pnl_percent,
                    position_count, open_position_value
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                equity.timestamp.isoformat(),
                equity.total_equity,
                equity.available_balance,
                equity.used_margin,
                equity.unrealized_pnl,
                equity.realized_pnl,
                equity.total_pnl,
                equity.total_pnl_percent,
                equity.position_count,
                equity.open_position_value
            ))
            return cursor.lastrowid

    def get_equity_history(self, limit: int = 100) -> List[EquityRecord]:
        """获取资产历史"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM equity_records
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            equity_history = []

            for row in rows:
                equity = EquityRecord(
                    id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    total_equity=row[2],
                    available_balance=row[3],
                    used_margin=row[4],
                    unrealized_pnl=row[5],
                    realized_pnl=row[6],
                    total_pnl=row[7],
                    total_pnl_percent=row[8],
                    position_count=row[9],
                    open_position_value=row[10]
                )
                equity_history.append(equity)

            return equity_history

    def cleanup_old_data(self, days: int = 90):
        """清理旧数据"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 删除90天前的市场数据（保留交易和信号数据）
            cursor.execute("""
                DELETE FROM market_data
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days))

            # 删除30天前的资产记录（保留最新状态）
            cursor.execute("""
                DELETE FROM equity_records
                WHERE timestamp < datetime('now', '-30 days')
                AND id NOT IN (
                    SELECT id FROM equity_records
                    ORDER BY timestamp DESC
                    LIMIT 1
                )
            """)

            conn.commit()
            logger.info(f"清理了 {cursor.rowcount} 条旧数据")