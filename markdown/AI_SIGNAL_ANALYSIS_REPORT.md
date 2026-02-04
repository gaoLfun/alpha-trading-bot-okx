# AI交易信号分析报告

## 一、统计摘要

### 1.1 日志数据概览

| 指标 | 数值 |
|------|------|
| 分析时段 | 2026-02-04 00:00 - 05:32 |
| 交易周期数 | 50+ 完整周期 |
| AI响应总数 | 152 次 |
| BUY信号数 | 0 次（0%） |
| HOLD信号数 | 152 次（100%） |
| SELL信号数 | 0 次（0%） |
| 平均置信度 | 约 66% |

### 1.2 市场环境分析

| 时间段 | 价格区间 | 24h涨跌幅 | RSI范围 | ATR范围 | 市场状态 |
|--------|----------|-----------|---------|---------|----------|
| 00:00-02:00 | 78031→73912 | -5%至-6% | 52→20 | 659→862 | 持续下跌 |
| 02:00-04:00 | 73912→74956 | -4%至-6% | 20→32 | 862→1080 | 震荡筑底 |
| 04:00-05:32 | 74956→76552 | -1%至-4% | 32→40 | 1080→1151 | 小幅反弹 |

### 1.3 AI提供商响应分布

| 提供商 | BUY信号 | HOLD信号 | SELL信号 | 平均置信度 |
|--------|---------|----------|----------|------------|
| deepseek | 0 | 152 | 0 | 65%-75% |
| kimi | 0 | 152 | 0 | 60% |

**核心问题**：两个AI提供商在所有周期中均返回HOLD信号，即使在RSI低于30（超卖区域）时也是如此。

---

## 二、问题深度分析

### 2.1 融合策略缺陷

**当前配置**：
- 融合策略：`weighted`（置信度加权平均）
- 融合阈值：`0.4`（40%）
- 权重配置：`deepseek:0.5, kimi:0.5`

**问题诊断**：

融合逻辑中存在根本性缺陷。当所有AI返回HOLD时，加权平均计算如下：

```
buy_score = 0.5 * 0.70 = 0.35
hold_score = 0.5 * 0.70 = 0.35
sell_score = 0.5 * 0.70 = 0.35
```

**关键发现**：
- `hold`信号的得分被归一化后始终为 `1.00`（100%）
- 这意味着即使所有AI返回相同信号，融合结果也是必然的HOLD
- 缺乏「当所有信号一致时提高信号强度」的机制
- 融合阈值的设置与归一化后的得分不匹配

### 2.2 Prompt设计问题

**买入条件过于严格**（`alpha_trading_bot/ai/prompt_builder.py`）：

```python
BUY_TREND_STRENGTH = 0.2  # 趋势强度要求
BUY_RSI_THRESHOLD = 68     # RSI必须低于68
BUY_BB_POSITION = 65       # 布林带位置低于65%
BUY_ADX_THRESHOLD = 15     # ADX必须大于15
```

**实际市场环境下的矛盾**：

| 市场条件 | 理想信号 | 实际信号 | 原因 |
|----------|----------|----------|------|
| RSI=20（超卖） | BUY | HOLD | 趋势方向为"down"时禁止买入 |
| 1h涨幅=2.47% | BUY | HOLD | 下跌趋势中动量不被认可 |
| 价格从高点下跌15% | BUY | HOLD | 24h仍为负值 |

**Prompt置信度计算逻辑问题**：

买入置信度计算规则中，`BUY`信号需要满足所有条件才能触发。但以下问题导致系统无法产生BUY信号：

1. **趋势方向限制过于刚性**：「趋势为down时禁止买入」的规则过于绝对
2. **RSI阈值设置不合理**：RSI低于30时通常意味着超卖，是潜在的买入机会，但系统反而因趋势限制而不买入
3. **动量信号未被有效利用**：1小时涨幅达到2%以上时，系统未能给予足够的买入权重
4. **缺乏逆向思维**：在超卖+动量反转的情况下，系统应当考虑试探性买入

### 2.3 缺乏交易机会识别

**关键缺失的信号场景**：

| 场景 | 技术指标特征 | 当前系统响应 | 期望响应 |
|------|--------------|--------------|----------|
| 超卖反弹 | RSI<30，1h涨幅>1% | HOLD | 试探性BUY |
| 强势支撑位 | 价格回撤15%+，RSI<35 | HOLD | 支撑位BUY |
| 趋势反转 | 1h涨幅持续>1% | HOLD | 趋势确认BUY |
| 低波动筑底 | ATR下降，RSI低位 | HOLD | 区间BUY |

---

## 三、业务逻辑优化方案

### 3.1 融合策略重构

**方案A：引入「一致性强化机制」**

```python
def fuse_with_consensus_boost(
    self,
    signals: List[Dict[str, str]],
    weights: Dict[str, float],
    threshold: float,
    confidences: Optional[Dict[str, int]] = None,
) -> str:
    """
    带一致性强化的加权融合策略
    
    规则：
    - 当所有AI返回相同信号时，该信号得分 * 1.5
    - 当有2/3以上AI返回相同信号时，该信号得分 * 1.25
    - 当AI意见分裂时，保持原始加权
    """
    if not signals:
        return "hold"
    
    # 统计信号分布
    signal_counts = {"buy": 0, "hold": 0, "sell": 0}
    for s in signals:
        signal_counts[s["signal"]] += 1
    
    total_ai = len(signals)
    consensus_ratio = max(signal_counts.values()) / total_ai
    
    # 计算加权得分
    weighted_scores = {"buy": 0, "hold": 0, "sell": 0}
    for s in signals:
        provider = s["provider"]
        sig = s["signal"]
        weight = weights.get(provider, 1.0)
        confidence = confidences.get(provider, 70) if confidences else 70
        adjusted_weight = weight * (confidence / 100.0)
        weighted_scores[sig] += adjusted_weight
    
    # 一致性强化
    if consensus_ratio >= 1.0:
        # 所有AI一致：强化因子1.5
        max_sig = max(weighted_scores, key=weighted_scores.get)
        weighted_scores[max_sig] *= 1.5
    elif consensus_ratio >= 0.67:
        # 2/3以上一致：强化因子1.25
        max_sig = max(weighted_scores, key=weighted_scores.get)
        weighted_scores[max_sig] *= 1.25
    
    # 归一化
    total = sum(weighted_scores.values())
    for sig in weighted_scores:
        weighted_scores[sig] /= total
    
    return max(weighted_scores, key=weighted_scores.get)
```

**方案B：引入「趋势反转检测」模块**

```python
class TrendReversalDetector:
    """趋势反转检测器"""
    
    def __init__(self):
        self.momentum_window = 3  # 使用最近3个周期
        self.reversal_thresholds = {
            "momentum_shift": 0.015,      # 1.5%动量反转
            "rsi_oversold": 30,           # RSI超卖阈值
            "consecutive_up": 2,           # 连续上涨周期数
        }
    
    def detect_reversal(
        self,
        price_changes: List[float],
        rsi: float,
        trend_direction: str,
    ) -> Dict[str, Any]:
        """
        检测趋势反转信号
        
        Returns:
            {
                "reversal_detected": bool,
                "reversal_type": "momentum|rsi|pattern",
                "confidence": float,
                "suggested_signal": "buy|hold|sell",
            }
        """
        # 动量反转检测
        recent_momentum = sum(price_changes[-self.momentum_window:]) / self.momentum_window
        
        # RSI超卖反弹检测
        rsi_reversal = rsi < self.reversal_thresholds["rsi_oversold"]
        
        # 连续上涨检测
        up_count = sum(1 for c in price_changes[-self.momentum_window:] if c > 0)
        consecutive_up = up_count >= self.reversal_thresholds["consecutive_up"]
        
        # 综合判断
        reversal_signals = []
        if recent_momentum > self.reversal_thresholds["momentum_shift"]:
            reversal_signals.append("momentum")
        if rsi_reversal and recent_momentum > 0:
            reversal_signals.append("rsi_rebound")
        if consecutive_up:
            reversal_signals.append("recovering")
        
        if len(reversal_signals) >= 2 and trend_direction == "down":
            return {
                "reversal_detected": True,
                "reversal_type": "|".join(reversal_signals),
                "confidence": min(0.5 + len(reversal_signals) * 0.15, 0.85),
                "suggested_signal": "buy",
            }
        
        return {
            "reversal_detected": False,
            "reversal_type": None,
            "confidence": 0.0,
            "suggested_signal": "hold",
        }
```

### 3.2 买入条件动态调整

**引入「超卖反弹模式」**：

```python
class AdaptiveBuyCondition:
    """自适应买入条件判断"""
    
    # 常规模式参数
    REGULAR_PARAMS = {
        "trend_strength_min": 0.2,
        "rsi_max": 68,
        "bb_position_max": 65,
        "adx_min": 15,
        "momentum_threshold": 0.005,
    }
    
    # 超卖反弹模式参数（更宽松）
    OVERSOLD_PARAMS = {
        "trend_strength_min": 0.1,      # 降低趋势要求
        "rsi_max": 35,                   # RSI超卖时放宽
        "bb_position_max": 40,           # 布林带低位
        "adx_min": 10,                   # 降低ADX要求
        "momentum_threshold": 0.01,        # 动量要求更高
    }
    
    def should_buy(
        self,
        trend_direction: str,
        trend_strength: float,
        rsi: float,
        bb_position: float,
        adx: float,
        recent_change: float,
        price_position: float,  # 综合价格位置
    ) -> Dict[str, Any]:
        """
        判断是否满足买入条件
        
        Returns:
            {
                "can_buy": bool,
                "mode": "regular|oversold|reversal",
                "confidence": float,
                "reason": str,
            }
        """
        # 检测是否处于超卖反弹模式
        is_oversold = rsi < 30
        is_low_price = price_position < 25
        has_momentum = recent_change > 0.01  # 1%以上涨幅
        
        if is_oversold and is_low_price and has_momentum:
            # 超卖反弹模式：大幅放宽条件
            params = self.OVERSOLD_PARAMS
            mode = "oversold_rebound"
            base_confidence = 0.5
        elif is_low_price and has_momentum:
            # 低位反弹模式
            params = self.OVERSOLD_PARAMS.copy()
            params["rsi_max"] = 45
            mode = "low_price"
            base_confidence = 0.45
        else:
            # 常规模式
            params = self.REGULAR_PARAMS
            mode = "regular"
            base_confidence = 0.4
        
        # 条件检查
        checks = {
            "trend": trend_direction != "down" and trend_strength >= params["trend_strength_min"],
            "rsi": rsi < params["rsi_max"],
            "bb": bb_position < params["bb_position_max"],
            "adx": adx >= params["adx_min"],
            "momentum": recent_change >= params["momentum_threshold"],
        }
        
        # 计算通过率
        passed = sum(1 for c in checks.values() if c)
        pass_rate = passed / len(checks)
        
        # 特殊加分项
        bonus = 0
        if rsi < 25:
            bonus += 0.15  # 严重超卖
        if trend_strength > 0.4:
            bonus += 0.1   # 趋势明确
        if recent_change > 0.02:
            bonus += 0.1   # 强动量
        
        final_confidence = min(base_confidence + pass_rate * 0.3 + bonus, 0.9)
        
        # 至少需要4个条件通过
        can_buy = passed >= 4 or (mode == "oversold_rebound" and passed >= 3)
        
        return {
            "can_buy": can_buy,
            "mode": mode,
            "confidence": final_confidence,
            "reason": f"{mode}: {passed}/{len(checks)}条件通过" if can_buy else "条件不足",
            "checks": checks,
        }
```

### 3.3 卖出条件优化

**引入「动态止盈止损触发」**：

```python
class DynamicSellCondition:
    """动态卖出条件判断"""
    
    def should_sell(
        self,
        position_pnl_percent: float,
        rsi: float,
        bb_position: float,
        macd_hist: float,
        trend_direction: str,
        trend_strength: float,
        has_reached_take_profit: bool,
        has_reached_stop_loss: bool,
    ) -> Dict[str, Any]:
        """
        判断是否应该卖出
        
        Returns:
            {
                "should_sell": bool,
                "reason": str,
                "sell_type": "stop_loss|take_profit|risk_management",
            }
        """
        reasons = []
        
        # 止损触发（优先级最高）
        if has_reached_stop_loss:
            return {
                "should_sell": True,
                "reason": "触发止损",
                "sell_type": "stop_loss",
            }
        
        # 止盈触发
        if has_reached_take_profit:
            return {
                "should_sell": True,
                "reason": "触发止盈",
                "sell_type": "take_profit",
            }
        
        # 风险信号检查
        risk_signals = 0
        
        # RSI超买
        if rsi > 80:
            risk_signals += 1
            reasons.append("RSI超买(>80)")
        elif rsi > 75:
            risk_signals += 0.5
            reasons.append("RSI偏高(>75)")
        
        # 布林带超买
        if bb_position > 90:
            risk_signals += 1
            reasons.append("布林带超买(>90%)")
        elif bb_position > 85:
            risk_signals += 0.5
            reasons.append("布林带偏高(>85%)")
        
        # 趋势反转信号
        if trend_direction == "down" and trend_strength > 0.4:
            risk_signals += 1
            reasons.append("趋势反转下行")
        
        # MACD转空
        if macd_hist < -0.001:
            risk_signals += 0.5
            reasons.append("MACD转空")
        
        # 浮盈回撤检测
        if position_pnl_percent < -1.0:
            risk_signals += 0.5
            reasons.append("浮盈大幅回撤(>-1%)")
        
        # 综合判断
        should_sell = risk_signals >= 2
        
        return {
            "should_sell": should_sell,
            "reason": "; ".join(reasons) if reasons else "无风险信号",
            "sell_type": "risk_management" if should_sell else None,
            "risk_score": risk_signals,
        }
```

---

## 四、配置优化建议

### 4.1 融合策略配置

```bash
# 建议配置（新增环境变量）
# 融合策略选择：weighted_consensus（带一致性强化）
export AI_FUSION_STRATEGY="weighted_consensus"

# 融合阈值：提高至0.5，使信号更明确
export AI_FUSION_THRESHOLD="0.5"

# 一致性强化的阈值配置
export AI_CONSENSUS_FULL_BOOST="1.0"   # 全部一致时强化因子
export AI_CONSENSUS_PARTIAL_BOOST="0.67"  # 部分一致时强化因子

# 提供商权重：根据历史表现调整
export AI_FUSION_WEIGHTS="deepseek:0.55,kimi:0.45"
```

### 4.2 买入条件配置

```bash
# 新增环境变量配置
# 趋势强度要求（常规）
export AI_BUY_TREND_STRENGTH="0.2"

# RSI阈值（常规）
export AI_BUY_RSI_THRESHOLD="68"

# 超卖反弹模式配置
export AI_OVERSOLD_RSI_THRESHOLD="30"
export AI_OVERSOLD_BUY_ENABLED="true"
export AI_OVERSOLD_TREND_RELAXED="true"

# 动量增强配置
export AI_MOMENTUM_BOOST_ENABLED="true"
export AI_MOMENTUM_THRESHOLD="0.01"   # 1%涨幅启用动量增强
export AI_MOMENTUM_BOOST_BONUS="0.1"

# 价格位置配置
export AI_LOW_PRICE_THRESHOLD="25"
export AI_LOW_PRICE_BUY_ENABLED="true"
export AI_LOW_PRICE_BONUS="0.15"
```

### 4.3 趋势检测配置

```bash
# 趋势分析配置
export TREND_PERIODS="10,20,50"
export TREND_STRENGTH_MIN="0.15"
export TREND_REVERSAL_DETECTION="true"
export TREND_REVERSAL_WINDOW="3"
export TREND_REVERSAL_MOMENTUM_THRESHOLD="0.015"
```

### 4.4 建议的配置文件模板

```yaml
# ai_config.yaml
ai:
  mode: fusion
  fusion:
    providers:
      - deepseek
      - kimi
    strategy: weighted_consensus
    threshold: 0.5
    weights:
      deepseek: 0.55
      kimi: 0.45
    consensus_boost:
      full: 1.5
      partial: 1.25
  
  buy_conditions:
    regular:
      trend_strength_min: 0.2
      rsi_max: 68
      bb_position_max: 65
      adx_min: 15
    oversold:
      enabled: true
      rsi_max: 35
      trend_strength_min: 0.1
      bb_position_max: 40
      adx_min: 10
      require_momentum: true
      momentum_threshold: 0.01
    
  sell_conditions:
    regular:
      rsi_overbought: 75
      bb_position_max: 85
      stop_loss_percent: 2.0
      take_profit_percent: 6.0
    
  trend_analysis:
    periods: [10, 20, 50]
    reversal_detection:
      enabled: true
      window: 3
      momentum_threshold: 0.015
```

---

## 五、Prompt优化方案

### 5.1 核心Prompt重构

```python
TRADING_PROMPT_TEMPLATE = """
你是一个专业的加密货币量化交易决策引擎。

【当前市场状态】（所有指标基于1小时周期计算）
- 当前价格: {current_price:.2f}
- 1小时涨跌幅: {recent_change:.2%}
- 24小时涨跌幅: {daily_change:.2%}
- RSI: {rsi:.1f} （<30严重超卖, 30-50偏弱, 50中性, 50-70偏强, >70超买）
- MACD: {macd:.2f}, Histogram: {macd_hist:+.4f} （>0多头, <0空头, 绝对值越大动能越强）
- ADX: {adx:.1f} （<20无趋势, 20-40有趋势, >40强趋势）
- ATR: {atr_pct:.2f}% （波动率，>3%高波动需谨慎）
- 布林带位置: {bb_pos:.1f}% （<20严重超卖, 20-40超卖, 40-60中性, 60-80超买, >80严重超买）
- 趋势方向: {trend_dir}
- 趋势强度: {trend_strength:.2f} （0-1，>0.3为有效趋势）
- 综合价格位置: {price_position:.1f}% （0=7日最低, 100=7日最高）

【持仓状态】
- 持仓方向: {pos_side}
- 持仓数量: {pos_amount:.4f} 张
- 入场价格: {entry_price:.2f} USDT
- 当前浮盈: {unrealized_pnl:.2f} USDT ({pnl_percent:.2f}%)

【核心决策逻辑】

1. **买入信号触发条件**（满足以下任一组合即可）：
   
   A. 常规买入条件（需满足全部）：
      - 趋势不为"down" 且 趋势强度 > {BUY_TREND_STRENGTH}
      - RSI < {BUY_RSI_THRESHOLD}
      - MACD Histogram > 0
      - 布林带位置 < {BUY_BB_POSITION}%
   
   B. 超卖反弹模式（满足3/4即可，更激进）：
      - RSI < 30（严重超卖）
      - 1小时涨幅 > 1%（动量反转）
      - 趋势强度 > 0.1
      - 布林带位置 < 45%
   
   C. 强势支撑位（满足2/3即可）：
      - 综合价格位置 < 25%（处于相对低位）
      - RSI < 45
      - 1小时涨幅 > 0.5%

2. **卖出信号触发条件**：
   
   A. 止损条件（满足任一）：
      - 浮亏 > 2%
      - 浮亏 > 1% 且 趋势为"down"
   
   B. 止盈条件（满足任一）：
      - 浮盈 > 6%
      - 浮盈 > 4% 且 RSI > 75
      - 浮盈 > 3% 且 趋势转"down"
   
   C. 风险规避（满足任一）：
      - RSI > 80
      - 布林带位置 > 90%
      - 趋势为"down" 且 趋势强度 > 0.4
      - MACD Histogram < -0.002

3. **持仓观望条件**：
   - 多指标信号冲突
   - 趋势强度 < 0.15
   - ADX < 20
   - 布林带位置在35%-65%区间（震荡整理）
   - ATR > 4%（高波动市场）

【关键决策规则】

1. **超卖反弹优先**：当RSI<30且1小时涨幅>1%时，优先考虑买入
2. **趋势为王但不教条**：趋势为down时减少买入，但超卖+动量反转可突破限制
3. **动量确认**：短期涨幅>1%时，可适当放宽RSI和趋势强度要求
4. **低位机会**：价格位置<25%时，增加买入权重
5. **风险第一**：浮亏>2%立即止损，不抱幻想

【置信度计算规则】

买入置信度：
- 基础：55%
- +10% RSI < 50
- +10% RSI < 30（严重超卖）
- +10% MACD Histogram > 0
- +10% 趋势明确向上（strength > 0.4）
- +10% 布林带位置 < 40%
- +10% 1小时涨幅 > 0.5%
- +15% 1小时涨幅 > 1%（强动量）
- +10% 价格位置 < 25%
- +10% 持仓浮盈 > 2%
- -20% 趋势为"down"
- -20% RSI > 65
- 范围：50%-95%

卖出置信度：
- 基础：55%
- +10% 趋势明确向下（strength > 0.4）
- +10% RSI > 70
- +10% RSI > 80（严重超买）
- +10% MACD Histogram < 0
- +10% 布林带位置 > 70%
- +10% 持仓浮亏 > -2%
- +15% 浮亏 > -3%
- -20% 趋势为"up"
- -20% RSI < 35
- 范围：50%-95%

持有置信度：
- 基础：55%
- +10% 多指标信号冲突
- +10% 趋势强度 < 0.2
- +10% ADX < 20
- +10% ATR > 4%
- +10% 持仓浮盈在 -1%~2%区间
- -10% 趋势明确且有持仓
- 范围：50%-90%

【强制输出格式】

只输出最终结果，不要任何解释：

buy | confidence: XX%
或
hold | confidence: XX%
或
sell | confidence: XX%
"""
```

### 5.2 条件Prompt对比

| 条件 | 原版 | 优化版 | 改进说明 |
|------|------|--------|----------|
| 趋势为down时买入 | ❌ 绝对禁止 | ⚠️ 超卖+动量可突破 | 允许逆向机会 |
| RSI超卖处理 | 忽略 | +10%置信度加分 | 识别超卖价值 |
| 1小时涨幅>1% | 动量增强 | 动量增强+条件放宽 | 更积极捕捉反弹 |
| 低位（<25%） | 无特殊处理 | +10%置信度 | 逆向投资机会 |
| 卖出触发 | RSI>75 | RSI>80+多条件组合 | 减少假突破 |

---

## 六、优先级实施计划

### P0（立即实施）

1. **修复融合策略归一化问题**
   - 问题：HOLD信号始终得分为1.0
   - 方案：引入一致性强化机制
   - 预期效果：允许更多BUY/SELL信号产生

2. **放宽超卖反弹的买入条件**
   - 问题：趋势为down时绝对禁止买入
   - 方案：RSI<30 + 1h涨幅>1%时允许试探性买入
   - 预期效果：在超卖区域能够建仓

3. **调整融合阈值**
   - 问题：阈值0.4过低导致信号无效
   - 方案：提高至0.5
   - 预期效果：信号更明确

### P1（1周内实施）

4. **引入趋势反转检测模块**
   - 检测动量反转、RSI反弹、连续上涨
   - 自动触发买入信号

5. **优化Prompt置信度计算**
   - 增加超卖、低位、强动量的正向加分
   - 降低趋势限制的负面影响

6. **配置化参数调整**
   - 将硬编码阈值移至配置
   - 支持环境变量覆盖

### P2（2周内实施）

7. **历史表现回测**
   - 基于历史信号验证优化效果
   - 调整参数至最优

8. **引入机器学习优化**
   - 基于交易结果自动调整权重
   - 持续学习改进

---

## 七、风险提示

### 7.1 潜在风险

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 过度交易 | 中 | 高 | 保留HOLD倾向，宽松但不激进 |
| 假突破 | 中 | 中 | 多条件确认，避免单点触发 |
| API不稳定 | 低 | 高 | 保留现有错误处理机制 |
| 参数量级爆炸 | 低 | 中 | 配置化管理，渐进式调整 |

### 7.2 建议监控指标

```python
# 建议新增的监控指标
MONITORING_METRICS = {
    "signal_distribution": "BUY/HOLD/SELL比例",
    "buy_opportunity_rate": "买入机会识别率",
    "oversold_buy_rate": "超卖反弹买入率",
    "false_break_rate": "假突破率",
    "average_hold_duration": "平均持仓时间",
    "consecutive_hold_count": "连续HOLD次数",
    "confidence_distribution": "置信度分布",
}
```

---

## 八、总结

本次分析揭示了当前AI交易系统的核心问题：**信号过度保守**。在50+个交易周期、152次AI响应中，竟然没有产生任何BUY或SELL信号，这在超卖+动量反转的市场环境下是明显的策略失效。

### 核心发现

1. **融合策略缺陷**：加权平均+归一化导致HOLD信号必然胜出
2. **Prompt过于保守**：趋势为down时绝对禁止买入的规则过于刚性
3. **缺乏逆向思维**：未能识别超卖+动量反转的买入机会
4. **置信度系统失效**：大量加分项但仍无法突破趋势限制

### 关键优化方向

1. **引入一致性强化**：当所有AI一致时，提高信号可信度
2. **增加超卖反弹模式**：RSI<30 + 动量反转 = 试探性买入
3. **趋势限制软化**：趋势为down时减少买入权重，但不绝对禁止
4. **融合阈值调整**：从0.4提高至0.5，使信号更明确

### 预期效果

| 指标 | 当前 | 预期（优化后） |
|------|------|----------------|
| BUY信号占比 | 0% | 5%-15% |
| SELL信号占比 | 0% | 3%-10% |
| HOLD信号占比 | 100% | 75%-90% |
| 平均持仓时间 | N/A | 2-4周期 |
| 买入胜率 | N/A | >60% |

---

**报告生成时间**：2026-02-04
**分析周期**：2026-02-04 00:00 - 05:32
**建议实施周期**：P0 立即实施，P1 1周，P2 2周
