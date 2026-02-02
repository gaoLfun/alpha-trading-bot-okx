"""
置信度优先融合策略
"""

import logging
from typing import Dict, List

from .base import FusionStrategy

logger = logging.getLogger(__name__)


class ConfidenceFusion(FusionStrategy):
    """置信度优先融合"""

    def fuse(
        self,
        signals: List[Dict[str, str]],
        weights: Dict[str, float],
        threshold: float,
    ) -> str:
        if not signals:
            logger.warning("无有效信号，默认hold")
            return "hold"

        signal_counts = {}
        total = len(signals)

        for s in signals:
            sig = s["signal"]
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

        buy_count = signal_counts.get("buy", 0)
        sell_count = signal_counts.get("sell", 0)

        if buy_count > sell_count and buy_count / total >= threshold:
            return "buy"
        elif sell_count > buy_count and sell_count / total >= threshold:
            return "sell"
        return "hold"
