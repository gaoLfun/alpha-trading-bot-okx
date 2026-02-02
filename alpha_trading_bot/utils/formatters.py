"""
格式化工具
"""

from typing import Dict, Any


def format_indicators_for_ai(indicators: Dict[str, Any], current_price: float) -> str:
    """格式化技术指标为AI可读格式"""
    lines = []

    lines.append(f"当前价格: {current_price:.2f}")

    # RSI
    rsi = indicators.get("rsi", 50)
    rsi_state = indicators.get("rsi_state", "normal")
    lines.append(f"RSI({rsi:.1f}) - {rsi_state}")

    # MACD
    macd = indicators.get("macd", 0)
    hist = indicators.get("macd_histogram", 0)
    macd_state = indicators.get("macd_state", "neutral")
    lines.append(f"MACD({macd:.2f}, 柱状:{hist:+.4f}) - {macd_state}")

    # ADX
    adx = indicators.get("adx", 0)
    adx_state = indicators.get("adx_state", "weak")
    lines.append(f"ADX({adx:.1f}) - {adx_state}")

    # ATR
    atr_percent = indicators.get("atr_percent", 0)
    vol_state = indicators.get("volatility_state", "normal")
    lines.append(f"ATR({atr_percent:.2f}%) - {vol_state}")

    # 布林带
    bb_pos = indicators.get("bb_position", 0.5) * 100
    lines.append(f"布林带位置: {bb_pos:.1f}%")

    # 趋势
    trend_dir = indicators.get("trend_direction", "neutral")
    trend_strength = indicators.get("trend_strength", 0)
    lines.append(f"趋势: {trend_dir} (强度:{trend_strength:.2f})")

    return "\n".join(lines)
