# 杠杆设置调试增强总结

## 问题现象
系统日志显示：
```
[WARNING] 设置杠杆失败，存在活跃算法订单: okx {"code":"59669","data":[],"msg":"Cancel cross-margin TP/SL, trailing, trigger, and chase orders or stop bots before adjusting your leverage."}
```

但增强的杠杆设置方法似乎没有被调用。

## 调试增强

### 1. 初始化阶段日志增强
- 添加了详细的配置信息日志
- 显示交易所、符号和杠杆配置
- 添加了异常捕获的详细错误信息

### 2. set_leverage方法日志增强
- 添加了方法入口日志：`[Enhanced set_leverage] 开始设置杠杆`
- 添加了成功/失败的明确日志
- 添加了详细的错误分析日志

### 3. 错误检测逻辑增强
- 添加了错误码59669的直接检测
- 增强了关键词匹配（包括正则表达式）
- 添加了详细的错误分析日志

### 4. 算法订单处理日志增强
- 添加了符号转换日志
- 添加了订单发现和取消的详细日志
- 添加了订单恢复的详细日志

## 关键改进

### 错误检测逻辑
```python
# 检查是否是因为存在算法订单导致的错误
# OKX错误码59669表示存在活跃的算法订单
if '59669' in error_msg or any(keyword in error_lower for keyword in [
    'cancel cross-margin tp/sl',
    'trailing, trigger, and chase orders',
    'stop bots before adjusting your leverage',
    'cancel.*orders.*before.*adjusting.*leverage'
]):
```

### 容错设计
```python
# 即使杠杆设置失败，系统仍继续运行
logger.warning("杠杆设置失败，但系统将继续初始化...")
```

## 预期日志输出

当杠杆设置因算法订单失败时，应该看到：
```
[INFO] 准备设置杠杆: 10x for BTC/USDT:USDT
[INFO] 当前配置: exchange=okx, symbol=BTC/USDT:USDT, leverage=10
[INFO] [Enhanced set_leverage] 开始设置杠杆: 10x for BTC/USDT:USDT
[INFO] 杠杆设置失败详情: [完整错误信息]
[INFO] 错误码分析: code=59669 在错误中: True
[INFO] 算法订单关键词检测: True
[WARNING] 设置杠杆失败，存在活跃算法订单: [错误信息]
[INFO] 尝试取消算法订单后重新设置杠杆...
[INFO] [_save_and_cancel_algo_orders] 转换符号: BTC/USDT:USDT -> BTC-USDT-SWAP
[INFO] 发现 X 个活跃算法订单，正在取消...
[INFO] 已取消 X 个算法订单
[INFO] [Enhanced set_leverage] 杠杆设置成功: 10x
[INFO] 正在恢复 X 个算法订单...
[INFO] 恢复算法订单成功: [订单ID]
[INFO] 杠杆设置成功: 10x
```

## 下一步

等待系统下次初始化，观察新的日志输出，确认：
1. 增强的set_leverage方法是否被调用
2. 错误检测是否正确识别59669错误码
3. 算法订单取消和恢复流程是否正常执行
4. 杠杆设置最终是否成功

如果仍然有问题，可以根据新的日志进一步调试。