"""
配置访问统一模块
集中管理所有配置参数的访问，避免重复读取
"""


class ConfigAccessor:
    """统一配置访问器"""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            # 这里需要导入ConfigManager，但为了避免循环导入，我们使用延迟导入
            from ..config.manager import ConfigManager

            self._config = ConfigManager().get_config()

    def get_config(self):
        """获取配置对象"""
        return self._config

    # AI配置相关
    def get_ai_provider(self) -> str:
        """获取AI提供商"""
        return getattr(self._config.ai, "default_provider", "deepseek")

    def get_ai_fusion_weights(self):
        """获取AI融合权重"""
        return getattr(
            self._config.ai, "fusion_weights", {"deepseek": 0.6, "qwen": 0.4}
        )

    def get_ai_fusion_enabled(self) -> bool:
        """获取AI融合启用状态"""
        return getattr(self._config.ai, "fusion_enabled", True)

    def get_ai_min_confidence(self) -> float:
        """获取AI最小置信度"""
        return getattr(self._config.ai, "min_confidence", 0.5)

    # 策略配置相关
    def get_take_profit_enabled(self) -> bool:
        """获取止盈启用状态"""
        return getattr(self._config.strategies, "take_profit_enabled", True)

    def get_take_profit_mode(self) -> str:
        """获取止盈模式"""
        return getattr(self._config.strategies, "take_profit_mode", "smart")

    def get_stop_loss_enabled(self) -> bool:
        """获取止损启用状态"""
        return getattr(self._config.strategies, "stop_loss_enabled", True)

    def get_normal_take_profit_percent(self) -> float:
        """获取普通止盈百分比"""
        return getattr(self._config.strategies, "normal_take_profit_percent", 0.06)

    def get_smart_take_profit_percent(self) -> float:
        """获取智能止盈百分比"""
        return getattr(self._config.strategies, "smart_fixed_take_profit_percent", 0.06)

    def get_multi_level_tp_levels(self):
        """获取多级止盈级别"""
        return getattr(self._config.strategies, "smart_multi_take_profit_levels", None)

    def get_multi_level_tp_ratios(self):
        """获取多级止盈比例"""
        return getattr(self._config.strategies, "smart_multi_take_profit_ratios", None)

    # 风险控制配置相关
    def get_max_daily_loss(self) -> float:
        """获取每日最大亏损"""
        return getattr(self._config.risk, "max_daily_loss", 100.0)

    def get_max_position_risk(self) -> float:
        """获取单仓位最大风险"""
        return getattr(self._config.risk, "max_position_risk", 0.05)

    def get_trailing_stop_enabled(self) -> bool:
        """获取追踪止损启用状态"""
        return getattr(self._config.risk, "trailing_stop_enabled", True)

    def get_trailing_stop_loss_enabled(self) -> bool:
        """获取追踪止损启用状态"""
        return getattr(self._config.risk, "trailing_stop_loss_enabled", True)

    def get_trailing_distance(self) -> float:
        """获取追踪距离"""
        return getattr(self._config.risk, "trailing_distance", 0.015)

    # 交易引擎配置相关
    def get_cycle_interval(self) -> int:
        """获取交易周期"""
        return getattr(self._config, "cycle_interval", 15)

    def get_random_offset_enabled(self) -> bool:
        """获取随机偏移启用状态"""
        return getattr(self._config, "random_offset_enabled", True)

    def get_random_offset_range(self) -> int:
        """获取随机偏移范围"""
        return getattr(self._config, "random_offset_range", 180)

    def get_use_leverage(self) -> bool:
        """获取杠杆使用状态"""
        return getattr(self._config, "use_leverage", True)

    def get_leverage(self) -> int:
        """获取杠杆倍数"""
        return getattr(self._config, "leverage", 10)

    # 缓存配置相关
    def get_enable_dynamic_cache(self) -> bool:
        """获取动态缓存启用状态"""
        return getattr(self._config, "enable_dynamic_cache", True)

    def get_cache_duration(self) -> int:
        """获取缓存持续时间"""
        return getattr(self._config, "cache_duration", 300)

    # 便捷方法
    def is_production_mode(self) -> bool:
        """判断是否为生产模式"""
        return getattr(self._config, "production_mode", False)

    def get_log_level(self) -> str:
        """获取日志级别"""
        return getattr(self._config, "log_level", "INFO")

    def get_max_workers(self) -> int:
        """获取最大工作线程数"""
        return getattr(self._config, "max_workers", 4)


# 全局配置访问器实例
config_accessor = ConfigAccessor()
