"""
加权平均融合策略
"""

import logging
from typing import Dict, List

from .base import FusionStrategy

logger = logging.getLogger(__name__)


class WeightedFusion(FusionStrategy):
    """加权平均融合"""

    def fuse(
        self,
        signals: List[Dict[str, str]],
        weights: Dict[str, float],
        threshold: float,
    ) -> str:
        if not signals:
            logger.warning("无有效信号，默认hold")
            return "hold"

        weighted_scores = {"buy": 0, "hold": 0, "sell": 0}
        total_weight = 0

        for s in signals:
            provider = s["provider"]
            sig = s["signal"]
            weight = weights.get(provider, 1.0)

            if sig == "buy":
                weighted_scores["buy"] += weight
            elif sig == "sell":
                weighted_scores["sell"] += weight
            else:
                weighted_scores["hold"] += weight

            total_weight += weight

        for sig in weighted_scores:
            weighted_scores[sig] /= total_weight if total_weight > 0 else 1

        max_sig = max(weighted_scores, key=weighted_scores.get)
        self._log_result(
            "加权平均",
            max_sig,
            f"buy:{weighted_scores['buy']:.2f}, hold:{weighted_scores['hold']:.2f}, sell:{weighted_scores['sell']:.2f}",
        )
        return max_sig
