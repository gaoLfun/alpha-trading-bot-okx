"""
优化器模块

提供离线参数优化能力：
- 贝叶斯优化 (Optuna)
- 回测引擎
- 配置热更新

此模块在后台运行，不影响实时交易
"""

from .bayesian_optimizer import BayesianOptimizer, OptimizationResult
from .backtest_engine import BacktestEngine, BacktestResult
from .config_updater import ConfigUpdater, ConfigChange

__all__ = [
    "BayesianOptimizer",
    "OptimizationResult",
    "BacktestEngine",
    "BacktestResult",
    "ConfigUpdater",
    "ConfigUpdate",
]
