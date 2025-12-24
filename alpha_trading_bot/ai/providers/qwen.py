"""
Qwen AI提供商实现
"""

import aiohttp
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .base import BaseAIProvider
from ...core.exceptions import NetworkError, RateLimitError

logger = logging.getLogger(__name__)

class QwenProvider(BaseAIProvider):
    """Qwen AI提供商"""

    def __init__(self, api_key: str, model: str = "qwen-plus"):
        super().__init__(api_key, model)
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation"
        self.timeout = 20.0

    async def generate_signal(self, prompt: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """生成交易信号"""
        logger.info(f"开始生成Qwen信号，模型: {self.model}")
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                #'X-DashScope-SSE': 'enable'  # 非流式输出，简化调用
            }

            data = {
                'model': self.model,
                'input': {
                    'messages': [
                        {'role': 'system', 'content': f'''你是一个专业的加密货币交易分析师，擅长技术分析和市场预测。请基于提供的市场数据给出准确的交易建议。

重要指导原则：
1. 必须综合考虑趋势方向和强度，不能仅看日内价格位置
2. 在下跌趋势中（趋势强度>0.5），即使价格处于日内低位，也应谨慎做多或考虑做空
3. 在上升趋势中（趋势强度>0.5），即使价格处于日内高位，也应谨慎做空或考虑做多
4. 注意趋势的一致性：如果短、中、长期趋势都指向同一方向，应给予更高权重
5. 当趋势强度较弱（<0.3）时，可以更多考虑区间交易机会

当前趋势分析: {market_data.get("technical_data", {}).get("trend_analysis", {})}'''},
                        {'role': 'user', 'content': prompt}
                    ]
                },
                'parameters': {
                    'temperature': 0.3,
                    'max_tokens': 500,
                    'top_p': 0.95,
                    'result_format': 'message'
                }
            }

            # 创建独立的HTTP会话
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(
                limit=30,
                limit_per_host=10,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30
            )

            async with aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'User-Agent': 'AlphaTradingBot/3.0'}
            ) as session:
                async with session.post(
                    f'{self.base_url}/generation',
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    # 记录详细的响应信息
                    logger.debug(f"Qwen API响应状态: {response.status}")
                    logger.debug(f"Qwen API响应头: {response.headers}")

                    if response.status == 429:
                        raise RateLimitError("Qwen API速率限制")
                    elif response.status != 200:
                        # 尝试获取错误详情
                        error_detail = ""
                        try:
                            error_data = await response.json()
                            error_detail = f" - {error_data}"
                        except:
                            try:
                                error_text = await response.text()
                                error_detail = f" - {error_text[:200]}"
                            except:
                                pass
                        raise NetworkError(f"Qwen API错误: {response.status}{error_detail}")

                    # 非流式响应，直接获取JSON
                    result = await response.json()
                    logger.debug(f"完整响应: {result}")

                    # 提取内容
                    if result.get('output') and result['output'].get('choices'):
                        message = result['output']['choices'][0].get('message', {})
                        content = message.get('content', '')
                        logger.info(f"收到内容: {content[:100]}...")
                        return self._parse_response(content)
                    else:
                        logger.warning("响应格式不符合预期")
                        return None

        except Exception as e:
            logger.error(f"Qwen信号生成失败: {e}")
            return None

    def _parse_response(self, content: str) -> Optional[Dict[str, Any]]:
        """解析响应内容"""
        try:
            logger.debug(f"开始解析响应内容: {content[:200]}...")
            import re

            # 查找JSON内容
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                logger.debug(f"提取的JSON: {json_str}")
                ai_data = json.loads(json_str)
                logger.debug(f"解析后的数据: {ai_data}")

                # 验证必需字段
                signal = ai_data.get('signal', 'HOLD').upper()
                confidence = float(ai_data.get('confidence', 0.5))
                reason = ai_data.get('reason', 'Qwen AI分析')

                logger.debug(f"提取的信号: {signal}, 信心度: {confidence}, 理由: {reason}")

                # 验证信号有效性
                if signal not in ['BUY', 'SELL', 'HOLD']:
                    signal = 'HOLD'

                # 验证置信度范围
                confidence = max(0.0, min(1.0, confidence))

                result = {
                    'signal': signal,
                    'confidence': confidence,
                    'reason': reason,
                    'timestamp': datetime.now().isoformat(),
                    'provider': 'qwen',
                    'raw_response': content
                }
                logger.info(f"解析成功: 信号={signal}, 信心度={confidence}")
                return result
            else:
                logger.warning(f"未找到JSON格式内容，尝试直接解析文本: {content[:100]}...")
                # 如果没有JSON，尝试从文本中提取信号
                if 'BUY' in content.upper():
                    signal = 'BUY'
                    confidence = 0.7
                elif 'SELL' in content.upper():
                    signal = 'SELL'
                    confidence = 0.7
                else:
                    signal = 'HOLD'
                    confidence = 0.5

                result = {
                    'signal': signal,
                    'confidence': confidence,
                    'reason': content[:200],
                    'timestamp': datetime.now().isoformat(),
                    'provider': 'qwen',
                    'raw_response': content
                }
                logger.info(f"文本解析成功: 信号={signal}, 信心度={confidence}")
                return result

        except Exception as e:
            logger.error(f"解析Qwen响应失败: {e}")
            return None

    def get_provider_name(self) -> str:
        """获取提供商名称"""
        return "qwen"