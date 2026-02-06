"""
贝叶斯优化器

使用 Optuna 进行参数自动优化
每日收盘后运行，找出最优参数组合
"""

import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """优化结果"""

    best_params: Dict[str, float]
    best_value: float
    n_trials: int
    optimization_time_seconds: float
    study_name: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": self.n_trials,
            "optimization_time_seconds": self.optimization_time_seconds,
            "study_name": self.study_name,
            "timestamp": self.timestamp,
        }


class BayesianOptimizer:
    """
    贝叶斯参数优化器

    使用 Optuna 进行高效的参数搜索
    """

    def __init__(
        self,
        study_name: str = "trading_bot_optimization",
        storage_path: str = "data_json/optuna_study.db",
        n_trials: int = 100,
    ):
        """
        初始化优化器

        Args:
            study_name: 研究名称
            storage_path: SQLite 存储路径
            n_trials: 优化试验次数
        """
        self.study_name = study_name
        self.storage_path = storage_path
        self.n_trials = n_trials
        self._objective_func: Optional[Callable] = None
        self._search_space: Dict[str, Dict[str, Any]] = {}

    def define_search_space(self) -> Dict[str, Dict[str, Any]]:
        """定义参数搜索空间"""
        search_space = {
            "fusion_threshold": {
                "type": "float",
                "low": 0.3,
                "high": 0.8,
                "log": False,
            },
            "stop_loss_percent": {
                "type": "float",
                "low": 0.002,
                "high": 0.015,
                "log": False,
            },
            "stop_loss_profit_percent": {
                "type": "float",
                "low": 0.001,
                "high": 0.01,
                "log": False,
            },
            "weight_deepseek": {
                "type": "float",
                "low": 0.3,
                "high": 0.7,
                "log": False,
            },
            "buy_rsi_threshold": {
                "type": "float",
                "low": 50,
                "high": 80,
                "log": False,
            },
        }

        self._search_space = search_space
        return search_space

    def set_objective(
        self,
        objective_func: Callable[[Dict[str, float]], float],
    ) -> None:
        """
        设置优化目标函数

        Args:
            objective_func: 目标函数，输入参数字典，返回优化目标值
        """
        self._objective_func = objective_func

    def _create_optuna_objective(self):
        """创建 Optuna 目标函数"""
        import optuna

        search_space = self._search_space

        def objective(trial):
            params = {}

            for name, config in search_space.items():
                if config["type"] == "float":
                    if config.get("log", False):
                        params[name] = trial.suggest_float(
                            name,
                            config["low"],
                            config["high"],
                            log=True,
                        )
                    else:
                        params[name] = trial.suggest_float(
                            name, config["low"], config["high"]
                        )
                elif config["type"] == "int":
                    params[name] = trial.suggest_int(
                        name, int(config["low"]), int(config["high"])
                    )
                elif config["type"] == "categorical":
                    params[name] = trial.suggest_categorical(name, config["choices"])

            return self._objective_func(params)

        return objective

    def optimize(self) -> OptimizationResult:
        """
        运行优化

        Returns:
            OptimizationResult: 优化结果
        """
        import optuna
        import time

        start_time = time.time()

        try:
            # 创建或加载研究
            storage = f"sqlite:///{self.storage_path}"
            study = optuna.create_study(
                study_name=self.study_name,
                storage=storage,
                load_if_exists=True,
                direction="maximize",
            )

            # 运行优化
            study.optimize(
                self._create_optuna_objective(),
                n_trials=self.n_trials,
                show_progress_bar=False,
            )

            optimization_time = time.time() - start_time

            # 获取最优参数
            best_params = study.best_params
            best_value = study.best_value

            logger.info(
                f"[贝叶斯优化] 完成: 最优值={best_value:.4f}, 试验次数={study.n_trials}"
            )

            return OptimizationResult(
                best_params=best_params,
                best_value=best_value,
                n_trials=study.n_trials,
                optimization_time_seconds=optimization_time,
                study_name=self.study_name,
                timestamp=datetime.now().isoformat(),
            )

        except ImportError:
            logger.error("[贝叶斯优化] Optuna 未安装，请运行: pip install optuna")
            return OptimizationResult(
                best_params={},
                best_value=0,
                n_trials=0,
                optimization_time_seconds=0,
                study_name=self.study_name,
                timestamp=datetime.now().isoformat(),
            )

    def get_best_params(self) -> Dict[str, float]:
        """获取当前最优参数"""
        try:
            import optuna

            storage = f"sqlite:///{self.storage_path}"
            study = optuna.load_study(study_name=self.study_name, storage=storage)
            return study.best_params
        except ImportError:
            return {}
        except Exception as e:
            logger.error(f"[贝叶斯优化] 获取最优参数失败: {e}")
            return {}

    def get_optimization_history(self, n_top: int = 10) -> list[Dict[str, Any]]:
        """获取优化历史"""
        try:
            import optuna

            storage = f"sqlite:///{self.storage_path}"
            study = optuna.load_study(study_name=self.study_name, storage=storage)

            trials = study.get_trials(
                deepcopy=False, order_by="value", descending=True
            )[:n_top]

            return [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": t.state.name,
                    "duration": t.duration.total_seconds(),
                }
                for t in trials
            ]
        except ImportError:
            return []
        except Exception as e:
            logger.error(f"[贝叶斯优化] 获取历史失败: {e}")
            return []
