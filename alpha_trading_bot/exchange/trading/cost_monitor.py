"""
成本监控报告 - 定期生成交易成本分析报告
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from .transaction_cost_analyzer import TransactionCostAnalyzer

logger = logging.getLogger(__name__)

class CostMonitor:
    """成本监控器"""

    def __init__(self, cost_analyzer: TransactionCostAnalyzer):
        self.cost_analyzer = cost_analyzer
        self.last_report_time = None
        self.report_interval = timedelta(hours=4)  # 每4小时生成一次报告
        self.daily_report_time = datetime.now().replace(hour=23, minute=59, second=0, microsecond=0)
        self.weekly_report_sent = False

    async def check_and_generate_report(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """检查是否需要生成成本报告"""
        try:
            now = datetime.now()

            # 强制生成报告
            if force:
                return await self.generate_daily_report()

            # 定期报告（每4小时）
            if (self.last_report_time is None or
                now - self.last_report_time >= self.report_interval):
                report = await self.generate_periodic_report()
                self.last_report_time = now
                return report

            # 每日报告（23:59）
            if now.hour == 23 and now.minute >= 55:
                if not hasattr(self, 'daily_report_sent_today') or not self.daily_report_sent_today:
                    report = await self.generate_daily_report()
                    self.daily_report_sent_today = True
                    return report
            else:
                self.daily_report_sent_today = False

            # 每周报告（周日）
            if now.weekday() == 6 and now.hour == 23 and now.minute >= 55:
                if not self.weekly_report_sent:
                    report = await self.generate_weekly_report()
                    self.weekly_report_sent = True
                    return report
            else:
                self.weekly_report_sent = False

            return None

        except Exception as e:
            logger.error(f"成本报告检查失败: {e}")
            return None

    async def generate_periodic_report(self) -> Dict[str, Any]:
        """生成定期成本报告"""
        try:
            # 生成4小时报告
            report = self.cost_analyzer.get_cost_analysis_report(days=1)

            # 添加时间戳和报告类型
            report['report_type'] = 'periodic'
            report['report_time'] = datetime.now().isoformat()
            report['next_report'] = (datetime.now() + self.report_interval).isoformat()

            # 简化的关键指标
            key_metrics = {
                'avg_cost_24h': report.get('average_cost_percentage', 0),
                'total_cost_24h': report.get('total_cost', 0),
                'total_volume_24h': report.get('total_volume', 0),
                'execution_quality': report.get('execution_quality', {}),
                'recommendations': report.get('recommendations', [])[:3]  # 只显示前3条建议
            }

            report['key_metrics'] = key_metrics

            logger.info(f"定期成本报告生成完成 - 24小时平均成本: {key_metrics['avg_cost_24h']:.3%}")

            return report

        except Exception as e:
            logger.error(f"定期成本报告生成失败: {e}")
            return {'error': str(e), 'report_type': 'periodic'}

    async def generate_daily_report(self) -> Dict[str, Any]:
        """生成每日成本报告"""
        try:
            # 生成完整日报告
            report = self.cost_analyzer.get_cost_analysis_report(days=1)

            # 添加时间戳和报告类型
            report['report_type'] = 'daily'
            report['report_time'] = datetime.now().isoformat()
            report['period'] = '24h'

            # 计算关键指标
            key_metrics = {
                'total_trades': report.get('total_trades', 0),
                'total_fees': report.get('total_fees', 0),
                'total_slippage': report.get('total_slippage', 0),
                'avg_cost_pct': report.get('average_cost_percentage', 0),
                'best_execution_side': self._get_best_execution_side(report),
                'worst_execution_side': self._get_worst_execution_side(report),
                'cost_efficiency': self._calculate_cost_efficiency(report)
            }

            report['key_metrics'] = key_metrics

            # 生成执行摘要
            report['executive_summary'] = self._generate_executive_summary(report)

            # 添加趋势分析
            report['trend_analysis'] = await self._analyze_cost_trends(days=7)

            logger.info(f"每日成本报告生成完成 - 总交易: {key_metrics['total_trades']}, "
                       f"平均成本: {key_metrics['avg_cost_pct']:.3%}")

            return report

        except Exception as e:
            logger.error(f"每日成本报告生成失败: {e}")
            return {'error': str(e), 'report_type': 'daily'}

    async def generate_weekly_report(self) -> Dict[str, Any]:
        """生成本周成本报告"""
        try:
            # 生成7天报告
            report = self.cost_analyzer.get_cost_analysis_report(days=7)

            # 添加时间戳和报告类型
            report['report_type'] = 'weekly'
            report['report_time'] = datetime.now().isoformat()
            report['period'] = '7d'
            report['week_start'] = (datetime.now() - timedelta(days=7)).isoformat()

            # 计算周度关键指标
            key_metrics = {
                'total_trades_week': report.get('total_trades', 0),
                'total_fees_week': report.get('total_fees', 0),
                'total_slippage_week': report.get('total_slippage', 0),
                'avg_cost_pct_week': report.get('average_cost_percentage', 0),
                'cost_per_trade': report.get('total_cost', 0) / max(report.get('total_trades', 1), 1),
                'fee_efficiency': self._calculate_fee_efficiency(report),
                'execution_consistency': self._calculate_execution_consistency(report)
            }

            report['key_metrics'] = key_metrics

            # 生成深度分析
            report['deep_analysis'] = await self._generate_deep_analysis(report)

            # 添加改进建议
            report['improvement_suggestions'] = await self._generate_improvement_suggestions(report)

            logger.info(f"周度成本报告生成完成 - 本周交易: {key_metrics['total_trades_week']}, "
                       f"总成本: ${key_metrics['total_fees_week'] + key_metrics['total_slippage_week']:.2f}")

            return report

        except Exception as e:
            logger.error(f"周度成本报告生成失败: {e}")
            return {'error': str(e), 'report_type': 'weekly'}

    def _get_best_execution_side(self, report: Dict[str, Any]) -> str:
        """获取执行质量最好的一侧"""
        try:
            by_side = report.get('by_side', {})
            if not by_side:
                return 'unknown'

            best_side = 'buy'
            best_cost = float('inf')

            for side, data in by_side.items():
                cost = data.get('avg_cost', float('inf'))
                if cost < best_cost:
                    best_cost = cost
                    best_side = side

            return best_side

        except:
            return 'unknown'

    def _get_worst_execution_side(self, report: Dict[str, Any]) -> str:
        """获取执行质量最差的一侧"""
        try:
            by_side = report.get('by_side', {})
            if not by_side:
                return 'unknown'

            worst_side = 'buy'
            worst_cost = 0

            for side, data in by_side.items():
                cost = data.get('avg_cost', 0)
                if cost > worst_cost:
                    worst_cost = cost
                    worst_side = side

            return worst_side

        except:
            return 'unknown'

    def _calculate_cost_efficiency(self, report: Dict[str, Any]) -> float:
        """计算成本效率（成本占盈利的百分比）"""
        try:
            avg_cost = report.get('average_cost_percentage', 0)
            # 假设目标盈利率为1%（可根据实际调整）
            target_profit = 0.01
            efficiency = (target_profit - avg_cost) / target_profit if target_profit > 0 else 0
            return max(0, min(1, efficiency))  # 限制在0-1之间

        except:
            return 0.0

    def _calculate_fee_efficiency(self, report: Dict[str, Any]) -> float:
        """计算手续费效率"""
        try:
            cost_breakdown = report.get('cost_breakdown', {})
            fee_pct = cost_breakdown.get('fees_percentage', 50)
            # 理想情况下，手续费应占总成本的大部分（60-80%）
            if 60 <= fee_pct <= 80:
                return 1.0
            elif fee_pct > 80:
                return 0.8  # 手续费占比过高
            else:
                return fee_pct / 60  # 线性计算

        except:
            return 0.5

    def _calculate_execution_consistency(self, report: Dict[str, Any]) -> float:
        """计算执行一致性"""
        try:
            execution_quality = report.get('execution_quality', {})
            fill_rate = execution_quality.get('avg_fill_rate', 0)
            quality_score = execution_quality.get('quality_score', 0)

            # 综合成交率和质量评分
            consistency = (fill_rate * 0.6 + (quality_score / 100) * 0.4)
            return max(0, min(1, consistency))

        except:
            return 0.5

    def _generate_executive_summary(self, report: Dict[str, Any]) -> str:
        """生成执行摘要"""
        try:
            key_metrics = report.get('key_metrics', {})
            total_trades = key_metrics.get('total_trades', 0)
            avg_cost = key_metrics.get('avg_cost_pct', 0)
            total_cost = key_metrics.get('total_cost', 0)

            if total_trades == 0:
                return "今日无交易记录"

            summary = f"今日执行{total_trades}笔交易，平均交易成本{avg_cost:.3%}，总成本${total_cost:.2f}。"

            # 成本评估
            if avg_cost < 0.005:  # <0.5%
                summary += "成本表现优秀。"
            elif avg_cost < 0.01:  # <1%
                summary += "成本表现良好。"
            else:
                summary += "成本偏高，建议优化。"

            return summary

        except:
            return "报告生成失败"

    async def _analyze_cost_trends(self, days: int) -> Dict[str, Any]:
        """分析成本趋势"""
        try:
            # 获取多日数据
            trend_data = {}
            for i in range(days):
                date = datetime.now() - timedelta(days=i)
                daily_report = self.cost_analyzer.get_cost_analysis_report(days=1)
                if daily_report.get('total_trades', 0) > 0:
                    trend_data[date.strftime('%Y-%m-%d')] = {
                        'avg_cost': daily_report.get('average_cost_percentage', 0),
                        'total_trades': daily_report.get('total_trades', 0),
                        'total_cost': daily_report.get('total_cost', 0)
                    }

            # 计算趋势
            if len(trend_data) >= 3:
                costs = [data['avg_cost'] for data in trend_data.values()]
                trend = 'stable'
                if len(costs) >= 3:
                    if costs[0] > costs[-1] * 1.1:  # 下降10%以上
                        trend = 'improving'
                    elif costs[0] * 1.1 < costs[-1]:  # 上升10%以上
                        trend = 'worsening'

                return {
                    'trend': trend,
                    'data': trend_data,
                    'avg_cost_trend': np.mean(costs) if costs else 0
                }

            return {'trend': 'insufficient_data', 'data': trend_data}

        except Exception as e:
            logger.error(f"趋势分析失败: {e}")
            return {'trend': 'error', 'error': str(e)}

    async def _generate_deep_analysis(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """生成深度分析"""
        try:
            key_metrics = report.get('key_metrics', {})

            analysis = {
                'cost_distribution': self._analyze_cost_distribution(report),
                'execution_patterns': self._analyze_execution_patterns(report),
                'market_impact': self._analyze_market_impact(report),
                'optimization_opportunities': self._identify_optimization_opportunities(report)
            }

            return analysis

        except Exception as e:
            logger.error(f"深度分析失败: {e}")
            return {'error': str(e)}

    def _analyze_cost_distribution(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """分析成本分布"""
        try:
            by_side = report.get('by_side', {})
            by_order_type = report.get('by_order_type', {})

            return {
                'buy_vs_sell_efficiency': self._compare_buy_sell_costs(by_side),
                'maker_vs_taker_efficiency': self._compare_maker_taker_costs(by_order_type),
                'cost_concentration': self._analyze_cost_concentration(report)
            }

        except:
            return {}

    def _compare_buy_sell_costs(self, by_side: Dict[str, Any]) -> str:
        """比较买卖成本效率"""
        try:
            buy_cost = by_side.get('buy', {}).get('avg_cost', 0)
            sell_cost = by_side.get('sell', {}).get('avg_cost', 0)

            if buy_cost < sell_cost:
                return f"买入成本更优（{buy_cost:.3%} vs {sell_cost:.3%}）"
            elif sell_cost < buy_cost:
                return f"卖出成本更优（{sell_cost:.3%} vs {buy_cost:.3%}）"
            else:
                return "买卖成本相当"

        except:
            return "分析失败"

    def _compare_maker_taker_costs(self, by_order_type: Dict[str, Any]) -> str:
        """比较Maker和Taker成本效率"""
        try:
            maker_fee = by_order_type.get('maker', {}).get('avg_fee', 0)
            taker_fee = by_order_type.get('taker', {}).get('avg_fee', 0)

            if maker_fee < taker_fee:
                efficiency = (taker_fee - maker_fee) / taker_fee if taker_fee > 0 else 0
                return f"Maker订单节省{efficiency:.1%}手续费"
            else:
                return "费率异常，建议检查"

        except:
            return "分析失败"

    def _analyze_cost_concentration(self, report: Dict[str, Any]) -> str:
        """分析成本集中度"""
        try:
            # 简单分析：如果成本差异很大，说明存在优化空间
            avg_cost = report.get('average_cost_percentage', 0)
            median_cost = report.get('median_cost_percentage', 0)

            if avg_cost > median_cost * 1.5:
                return "成本分布不均，存在异常高成本交易"
            else:
                return "成本分布相对均匀"

        except:
            return "分析失败"

    def _analyze_execution_patterns(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """分析执行模式"""
        try:
            execution_quality = report.get('execution_quality', {})

            return {
                'fill_rate_consistency': self._assess_fill_rate_consistency(execution_quality),
                'slippage_stability': self._assess_slippage_stability(execution_quality),
                'timing_efficiency': self._assess_timing_efficiency(execution_quality)
            }

        except:
            return {}

    def _assess_fill_rate_consistency(self, execution_quality: Dict[str, Any]) -> str:
        """评估成交率一致性"""
        try:
            fill_rate = execution_quality.get('avg_fill_rate', 0)
            if fill_rate > 0.95:
                return "优秀（>95%）"
            elif fill_rate > 0.85:
                return "良好（85-95%）"
            else:
                return "需改进（<85%）"

        except:
            return "评估失败"

    def _assess_slippage_stability(self, execution_quality: Dict[str, Any]) -> str:
        """评估滑点稳定性"""
        try:
            slippage = execution_quality.get('avg_slippage_bps', 0)
            if slippage < 3:
                return "稳定（<3基点）"
            elif slippage < 5:
                return "可接受（3-5基点）"
            else:
                return "不稳定（>5基点）"

        except:
            return "评估失败"

    def _assess_timing_efficiency(self, execution_quality: Dict[str, Any]) -> str:
        """评估时机效率"""
        try:
            exec_time = execution_quality.get('avg_execution_time', 0)
            if exec_time < 0.5:
                return "高效（<0.5秒）"
            elif exec_time < 1.0:
                return "正常（0.5-1秒）"
            else:
                return "较慢（>1秒）"

        except:
            return "评估失败"

    def _analyze_market_impact(self, report: Dict[str, Any]) -> str:
        """分析市场影响"""
        try:
            # 基于滑点和成交率评估市场影响
            execution_quality = report.get('execution_quality', {})
            slippage = execution_quality.get('avg_slippage_bps', 0)
            fill_rate = execution_quality.get('avg_fill_rate', 0)

            if slippage < 2 and fill_rate > 0.9:
                return "市场影响较小"
            elif slippage < 5 and fill_rate > 0.8:
                return "市场影响适中"
            else:
                return "市场影响较大，建议减小单笔交易规模"

        except:
            return "分析失败"

    def _identify_optimization_opportunities(self, report: Dict[str, Any]) -> List[str]:
        """识别优化机会"""
        opportunities = []

        try:
            # 基于报告内容识别具体优化机会
            avg_cost = report.get('average_cost_percentage', 0)
            execution_quality = report.get('execution_quality', {})
            recommendations = report.get('recommendations', [])

            # 成本优化
            if avg_cost > 0.008:  # >0.8%
                opportunities.append("成本偏高，考虑优化订单类型和时机")

            # 执行质量优化
            fill_rate = execution_quality.get('avg_fill_rate', 0)
            if fill_rate < 0.85:
                opportunities.append("成交率偏低，建议调整下单策略")

            # 滑点优化
            slippage = execution_quality.get('avg_slippage_bps', 0)
            if slippage > 5:
                opportunities.append("滑点较大，建议使用限价单")

            # 添加具体建议
            opportunities.extend(recommendations[:3])  # 最多3条

            return opportunities

        except:
            return ["分析失败，无法识别优化机会"]

    async def _generate_improvement_suggestions(self, report: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        try:
            suggestions = []

            # 基于深度分析结果生成具体建议
            deep_analysis = report.get('deep_analysis', {})
            key_metrics = report.get('key_metrics', {})

            # 费用效率建议
            fee_efficiency = key_metrics.get('fee_efficiency', 0)
            if fee_efficiency < 0.8:
                suggestions.append("提高Maker订单比例以降低手续费")

            # 执行一致性建议
            consistency = key_metrics.get('execution_consistency', 0)
            if consistency < 0.8:
                suggestions.append("优化执行策略以提高一致性")

            # 基于深度分析的具体建议
            optimization_opportunities = deep_analysis.get('optimization_opportunities', [])
            suggestions.extend(optimization_opportunities[:5])  # 最多5条

            return suggestions

        except:
            return ["建议生成失败"]

    def get_cost_statistics(self) -> Dict[str, Any]:
        """获取成本统计信息"""
        try:
            # 获取最新的成本分析报告
            report = self.cost_analyzer.get_cost_analysis_report(days=30)

            return {
                'last_report': report,
                'report_count': len(self.cost_analyzer.transaction_history),
                'last_report_time': self.last_report_time.isoformat() if self.last_report_time else None,
                'next_report_time': (datetime.now() + self.report_interval).isoformat()
            }

        except Exception as e:
            logger.error(f"获取成本统计失败: {e}")
            return {'error': str(e)}

    def reset(self):
        """重置监控器"""
        self.last_report_time = None
        self.daily_report_sent_today = False
        self.weekly_report_sent = False
        logger.info("成本监控器已重置")