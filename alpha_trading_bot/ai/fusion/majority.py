"""
多数表决融合策略
"""

import logging
from typing import Dict, List

from .base import FusionStrategy

logger = logging.getLogger(__name__)


class MajorityFusion(FusionStrategy):
    """多数表决融合"""

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
        for s in signals:
            sig = s["signal"]
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

        total = len(signals)

        for sig, count in signal_counts.items():
            if count / total >= threshold:
                self._log_result("多数表决", sig, f"{count}/{total} >= {threshold}")
                return sig

        # 未达阈值，取最多的
        max_sig = max(signal_counts, key=signal_counts.get)
        self._log_result("多数表决-降级", max_sig, f"max count")
        return max_sig
