"""
配置热更新模块

功能：
- 支持运行时动态更新配置
- 配置版本管理
- 配置变更通知
- 回滚能力
"""

import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)


class UpdateType(Enum):
    """更新类型"""

    PARAMETER = "parameter"  # 参数调整
    RULE = "rule"  # 规则变更
    STRATEGY = "strategy"  # 策略开关
    RISK = "risk"  # 风险参数


@dataclass
class ConfigChange:
    """配置变更"""

    update_type: UpdateType
    key: str
    old_value: Any
    new_value: Any
    timestamp: str
    reason: str
    approved: bool = False
    applied: bool = False


@dataclass
class ConfigVersion:
    """配置版本"""

    version: int
    timestamp: str
    changes: list[ConfigChange]
    snapshot: Dict[str, Any]


class ConfigUpdater:
    """
    配置热更新器

    安全管理配置变更，支持热更新和回滚
    """

    def __init__(self, config_path: str = "data_json/config.json"):
        self.config_path = config_path
        self._current_config: Dict[str, Any] = {}
        self._change_history: list[ConfigChange] = []
        self._version_history: list[ConfigVersion] = []
        self._current_version = 0
        self._listeners: list[Callable] = []

        # 监听器列表
        self._change_listeners: list[Callable[[ConfigChange], None]] = []

        # 加载配置
        self._load_config()

    def _load_config(self) -> None:
        """加载配置"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._current_config = json.load(f)
                logger.info(f"[配置] 加载配置: {len(self._current_config)} 项")
            except Exception as e:
                logger.error(f"[配置] 加载失败: {e}")
                self._current_config = {}
        else:
            logger.info("[配置] 使用默认配置")
            self._current_config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "ai": {
                "fusion_threshold": 0.5,
                "fusion_weights": {"deepseek": 0.5, "kimi": 0.5},
            },
            "risk": {
                "hard_stop_loss_percent": 0.05,
                "max_position_percent": 0.1,
            },
            "strategies": {
                "trend_following": {"enabled": True, "weight": 1.0},
                "mean_reversion": {"enabled": True, "weight": 1.0},
                "breakout": {"enabled": True, "weight": 0.8},
                "safe_mode": {"enabled": True, "weight": 2.0},
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键（使用点号分隔，如 "ai.fusion_threshold"）

        Returns:
            配置值
        """
        keys = key.split(".")
        value = self._current_config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default

    def set(
        self,
        key: str,
        value: Any,
        reason: str = "",
        update_type: UpdateType = UpdateType.PARAMETER,
    ) -> bool:
        """
        设置配置值

        Args:
            key: 配置键
            value: 新值
            reason: 变更原因
            update_type: 更新类型

        Returns:
            是否成功
        """
        keys = key.split(".")
        old_value = self.get(key)

        # 创建变更记录
        change = ConfigChange(
            update_type=update_type,
            key=key,
            old_value=old_value,
            new_value=value,
            timestamp=datetime.now().isoformat(),
            reason=reason,
        )

        # 更新配置
        config = self._current_config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

        # 保存
        self._save_config()
        self._change_history.append(change)

        # 通知监听器
        for listener in self._change_listeners:
            try:
                listener(change)
            except Exception as e:
                logger.error(f"[配置] 监听器回调失败: {e}")

        logger.info(f"[配置] 更新: {key} = {value} ({reason})")
        return True

    def _save_config(self) -> None:
        """保存配置"""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self._current_config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[配置] 保存失败: {e}")

    def apply_optimized_params(
        self, params: Dict[str, Any], reason: str = "贝叶斯优化"
    ) -> bool:
        """
        应用优化后的参数

        Args:
            params: 优化后的参数字典
            reason: 变更原因

        Returns:
            是否成功
        """
        success = True

        for key, value in params.items():
            if key.startswith("ai_"):
                # AI相关参数
                sub_key = key[3:]  # 去掉 "ai_" 前缀
                if sub_key == "fusion_threshold":
                    success = success and self.set(
                        "ai.fusion_threshold",
                        value,
                        f"{reason}: {value}",
                        UpdateType.PARAMETER,
                    )
                elif sub_key == "weight_deepseek":
                    success = success and self.set(
                        "ai.fusion_weights.deepseek",
                        value,
                        f"{reason}: {value}",
                        UpdateType.PARAMETER,
                    )
                    # 同时更新 kimi 权重
                    kimi_value = 1.0 - value
                    success = success and self.set(
                        "ai.fusion_weights.kimi",
                        kimi_value,
                        f"{reason}: 自动调整",
                        UpdateType.PARAMETER,
                    )

            elif key.startswith("risk_") or key.startswith("stop_loss"):
                # 风险相关参数
                if "stop_loss" in key:
                    success = success and self.set(
                        "risk.hard_stop_loss_percent",
                        value,
                        f"{reason}: {value}",
                        UpdateType.RISK,
                    )

        return success

    def update_strategy_weight(
        self,
        strategy_name: str,
        weight: float,
        enabled: bool = True,
    ) -> bool:
        """
        更新策略权重

        Args:
            strategy_name: 策略名称
            weight: 新权重
            enabled: 是否启用

        Returns:
            是否成功
        """
        key = f"strategies.{strategy_name}"
        return self.set(
            key,
            {"enabled": enabled, "weight": weight},
            f"策略权重调整: {weight}",
            UpdateType.STRATEGY,
        )

    def create_version_snapshot(self, reason: str = "手动保存") -> int:
        """
        创建配置版本快照

        Args:
            reason: 快照原因

        Returns:
            版本号
        """
        self._current_version += 1

        version = ConfigVersion(
            version=self._current_version,
            timestamp=datetime.now().isoformat(),
            changes=self._change_history.copy(),
            snapshot=self._current_config.copy(),
        )

        self._version_history.append(version)

        # 限制历史长度
        if len(self._version_history) > 50:
            self._version_history = self._version_history[-50:]

        logger.info(f"[配置] 创建版本快照: v{self._current_version} ({reason})")
        return self._current_version

    def rollback(self, version: Optional[int] = None) -> bool:
        """
        回滚配置

        Args:
            version: 回滚到的版本（None表示上一个版本）

        Returns:
            是否成功
        """
        if version is None:
            if len(self._version_history) < 2:
                logger.warning("[配置] 无可回滚的版本")
                return False
            target = self._version_history[-2]
        else:
            target = None
            for v in self._version_history:
                if v.version == version:
                    target = v
                    break

            if target is None:
                logger.warning(f"[配置] 版本 {version} 不存在")
                return False

        self._current_config = target.snapshot.copy()
        self._save_config()

        logger.info(f"[配置] 回滚到版本 v{version}")
        return True

    def add_change_listener(self, listener: Callable[[ConfigChange], None]) -> None:
        """添加配置变更监听器"""
        self._change_listeners.append(listener)

    def get_change_history(self, limit: int = 20) -> list[Dict[str, Any]]:
        """获取变更历史"""
        changes = self._change_history[-limit:]
        return [
            {
                "type": c.update_type.value,
                "key": c.key,
                "old": str(c.old_value),
                "new": str(c.new_value),
                "timestamp": c.timestamp,
                "reason": c.reason,
            }
            for c in changes
        ]

    def export_config(self) -> str:
        """导出配置为JSON字符串"""
        return json.dumps(self._current_config, indent=2, ensure_ascii=False)

    def get_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            "version": self._current_version,
            "total_changes": len(self._change_history),
            "strategies": list(self._current_config.get("strategies", {}).keys()),
            "recent_changes": self.get_change_history(5),
        }


class ConfigMonitor:
    """
    配置监控器

    监控配置变更，支持自动热更新
    """

    def __init__(self, updater: Optional[ConfigUpdater] = None):
        self.updater = updater or ConfigUpdater()
        self._update_callbacks: list[Callable[[str, Any], None]] = []

    def watch(self, key: str) -> None:
        """监控配置变更"""
        self.updater.add_change_listener(lambda change: self._on_change(key, change))

    def _on_change(self, watched_key: str, change: ConfigChange) -> None:
        """配置变更回调"""
        if change.key == watched_key or watched_key in change.key:
            for callback in self._update_callbacks:
                try:
                    callback(change.key, change.new_value)
                except Exception as e:
                    logger.error(f"[监控] 回调失败: {e}")

    def on_update(self, callback: Callable[[str, Any], None]) -> None:
        """注册更新回调"""
        self._update_callbacks.append(callback)

    def check_and_apply(
        self,
        expected_key: str,
        expected_value: Any,
        action: Callable[[], None],
    ) -> bool:
        """
        检查配置并执行操作

        Args:
            expected_key: 期望的配置键
            expected_value: 期望的配置值
            action: 要执行的操作

        Returns:
            是否执行
        """
        current = self.updater.get(expected_key)
        if current == expected_value:
            action()
            return True
        return False
