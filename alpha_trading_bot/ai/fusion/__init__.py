"""
AI信号融合策略模块
"""

from .base import FusionStrategy, get_fusion_strategy
from .weighted import WeightedFusion
from .majority import MajorityFusion
from .consensus import ConsensusFusion
from .confidence import ConfidenceFusion

__all__ = [
    "FusionStrategy",
    "WeightedFusion",
    "MajorityFusion",
    "ConsensusFusion",
    "ConfidenceFusion",
    "get_fusion_strategy",
]
