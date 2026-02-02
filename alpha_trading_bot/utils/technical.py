"""
技术指标计算工具（向后兼容包装器）
纯Python实现，不依赖外部库
"""

# 重新导出所有功能，保持向后兼容
from .technical import (
    calculate_rsi,
    calculate_macd,
    calculate_ema,
    calculate_adx,
    calculate_trend,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_true_range,
    calculate_all_indicators,
)

__all__ = [
    "calculate_rsi",
    "calculate_macd",
    "calculate_ema",
    "calculate_adx",
    "calculate_trend",
    "calculate_atr",
    "calculate_bollinger_bands",
    "calculate_true_range",
    "calculate_all_indicators",
]
