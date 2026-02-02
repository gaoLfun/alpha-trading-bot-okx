"""
AI模块 - 支持单AI/多AI融合
"""

from .client import AIClient, get_signal
from .providers import PROVIDERS, get_provider_config
from .prompt_builder import PromptBuilder, build_prompt
from .response_parser import ResponseParser, parse_response, extract_signal
from .fusion import (
    FusionStrategy,
    WeightedFusion,
    MajorityFusion,
    ConsensusFusion,
    ConfidenceFusion,
)

__version__ = "1.0.0"

__all__ = [
    # 客户端
    "AIClient",
    "get_signal",
    # 提供商
    "PROVIDERS",
    "get_provider_config",
    # Prompt构建
    "PromptBuilder",
    "build_prompt",
    # 响应解析
    "ResponseParser",
    "parse_response",
    "extract_signal",
    # 融合策略
    "FusionStrategy",
    "WeightedFusion",
    "MajorityFusion",
    "ConsensusFusion",
    "ConfidenceFusion",
]
