"""
共识融合策略
"""

import logging
from typing import Dict, List

from .base import FusionStrategy

logger = logging.getLogger(__name__)


class ConsensusFusion(FusionStrategy):
    """共识融合 - 所有AI必须一致"""

    def fuse(
        self,
        signals: List[Dict[str, str]],
        weights: Dict[str, float],
        threshold: float,
    ) -> str:
        if not signals:
            logger.warning("无有效信号，默认hold")
            return "hold"

        unique_signals = set(s["signal"] for s in signals)
        if len(unique_signals) == 1:
            sig = list(unique_signals)[0]
            self._log_result("共识", sig, "all agreed")
            return sig
        else:
            logger.warning(f"未达成共识: {unique_signals}，默认hold")
            return "hold"
