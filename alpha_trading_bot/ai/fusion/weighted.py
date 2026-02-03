"""
加权平均融合策略 - 支持置信度加权
"""

import logging
from typing import Dict, List, Optional

from .base import FusionStrategy

logger = logging.getLogger(__name__)


class WeightedFusion(FusionStrategy):
    """加权平均融合 - 支持置信度加权"""

    def fuse(
        self,
        signals: List[Dict[str, str]],
        weights: Dict[str, float],
        threshold: float,
        confidences: Optional[Dict[str, int]] = None,
    ) -> str:
        """
        融合信号（带置信度加权）

        Args:
            signals: [{"provider": "deepseek", "signal": "buy"}, ...]
            weights: {"deepseek": 0.5, "kimi": 0.5, ...}
            threshold: 融合阈值
            confidences: {"deepseek": 70, "kimi": 75, ...} 置信度（可选）
        """
        if not signals:
            logger.warning("无有效信号，默认hold")
            return "hold"

        weighted_scores = {"buy": 0, "hold": 0, "sell": 0}
        total_weight = 0

        for s in signals:
            provider = s["provider"]
            sig = s["signal"]
            weight = weights.get(provider, 1.0)

            # 获取置信度，如果未提供则使用默认值70
            confidence = confidences.get(provider, 70) if confidences else 70
            confidence_factor = confidence / 100.0  # 归一化到 0-1

            # 置信度加权：score = weight * confidence_factor
            adjusted_weight = weight * confidence_factor

            if sig == "buy":
                weighted_scores["buy"] += adjusted_weight
            elif sig == "sell":
                weighted_scores["sell"] += adjusted_weight
            else:
                weighted_scores["hold"] += adjusted_weight

            total_weight += adjusted_weight

        for sig in weighted_scores:
            weighted_scores[sig] /= total_weight if total_weight > 0 else 1

        max_sig = max(weighted_scores, key=weighted_scores.get)
        max_score = weighted_scores[max_sig]

        # 信号有效性判断
        is_valid = max_score >= threshold

        self._log_result(
            "加权平均(置信度加权)",
            max_sig,
            f"buy:{weighted_scores['buy']:.2f}, hold:{weighted_scores['hold']:.2f}, sell:{weighted_scores['sell']:.2f}, threshold:{threshold}, valid:{is_valid}",
        )
        return max_sig
