"""
精简版配置模型 - 支持单AI/多AI融合
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ExchangeConfig:
    """交易所配置"""

    api_key: str = ""
    secret: str = ""
    password: str = ""
    symbol: str = "BTC/USDT:USDT"
    leverage: int = 10


@dataclass
class TradingConfig:
    """交易配置"""

    cycle_minutes: int = 15
    random_offset_range: int = 180


@dataclass
class AIConfig:
    """AI配置"""

    mode: str = "single"  # single=单AI, fusion=多AI融合
    default_provider: str = "deepseek"

    # 多AI融合配置
    fusion_providers: List[str] = field(default_factory=lambda: ["deepseek", "kimi"])
    fusion_strategy: str = "weighted"
    fusion_weights: Dict[str, float] = field(
        default_factory=lambda: {"deepseek": 0.5, "kimi": 0.5}
    )
    fusion_threshold: float = 0.5

    # 各提供商API Keys
    api_keys: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "AIConfig":
        import os

        fusion_providers_str = os.getenv("AI_FUSION_PROVIDERS", "deepseek,kimi")
        fusion_providers = [
            p.strip() for p in fusion_providers_str.split(",") if p.strip()
        ]

        fusion_weights_str = os.getenv("AI_FUSION_WEIGHTS", "deepseek:0.5,kimi:0.5")
        fusion_weights = {}
        for item in fusion_weights_str.split(","):
            if ":" in item:
                k, v = item.split(":")
                fusion_weights[k.strip()] = float(v.strip())

        return cls(
            mode=os.getenv("AI_MODE", "single"),
            default_provider=os.getenv("AI_DEFAULT_PROVIDER", "deepseek"),
            fusion_providers=fusion_providers,
            fusion_strategy=os.getenv("AI_FUSION_STRATEGY", "weighted"),
            fusion_weights=fusion_weights,
            fusion_threshold=float(os.getenv("AI_FUSION_THRESHOLD", "0.6")),
            api_keys={
                "deepseek": os.getenv("DEEPSEEK_API_KEY", ""),
                "kimi": os.getenv("KIMI_API_KEY", ""),
                "openai": os.getenv("OPENAI_API_KEY", ""),
                "qwen": os.getenv("QWEN_API_KEY", ""),
            },
        )


@dataclass
class StopLossConfig:
    """止损配置"""

    stop_loss_percent: float = 0.005  # 亏损时止损比例 (如 0.005 = 0.5%)
    stop_loss_profit_percent: float = 0.002  # 盈利时止损比例 (如 0.002 = 0.2%)
    stop_loss_tolerance_percent: float = (
        0.001  # 止损价容错比例 (如 0.001 = 0.1%, 约77美元对于BTC)
    )


@dataclass
class SystemConfig:
    """系统配置"""

    log_level: str = "INFO"  # 日志级别: DEBUG/INFO/WARNING/ERROR


@dataclass
class Config:
    """主配置"""

    exchange: ExchangeConfig = None
    trading: TradingConfig = None
    ai: AIConfig = None
    stop_loss: StopLossConfig = None
    system: SystemConfig = None

    def __post_init__(self):
        if self.exchange is None:
            self.exchange = ExchangeConfig()
        if self.trading is None:
            self.trading = TradingConfig()
        if self.ai is None:
            self.ai = AIConfig()
        if self.stop_loss is None:
            self.stop_loss = StopLossConfig()
        if self.system is None:
            self.system = SystemConfig()

    @classmethod
    def from_env(cls) -> "Config":
        import os

        return cls(
            exchange=ExchangeConfig(
                api_key=os.getenv("OKX_API_KEY", ""),
                secret=os.getenv("OKX_SECRET", ""),
                password=os.getenv("OKX_PASSWORD", ""),
                symbol=os.getenv("OKX_SYMBOL", "BTC/USDT:USDT"),
                leverage=int(os.getenv("OKX_LEVERAGE", "10")),
            ),
            trading=TradingConfig(
                cycle_minutes=int(os.getenv("CYCLE_MINUTES", "15")),
                random_offset_range=int(os.getenv("RANDOM_OFFSET_RANGE", "180")),
            ),
            ai=AIConfig.from_env(),
            stop_loss=StopLossConfig(
                stop_loss_percent=float(os.getenv("STOP_LOSS_PERCENT", "0.005")),
                stop_loss_profit_percent=float(
                    os.getenv("STOP_LOSS_PROFIT_PERCENT", "0.002")
                ),
            ),
            system=SystemConfig(
                log_level=os.getenv("LOG_LEVEL", "INFO"),
            ),
        )
