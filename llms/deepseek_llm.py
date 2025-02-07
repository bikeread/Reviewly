import logging
import requests
from typing import Optional, Dict, Any
from .base_llm import LLMInterface
from config import config

logger = logging.getLogger(__name__)

class DeepSeekClient(LLMInterface):
    def __init__(self):
        """初始化 DeepSeek API 客户端"""
        self.api_key = config.DEEPSEEK.api_key
        self.api_base = config.DEEPSEEK.api_base
        self.model = config.DEEPSEEK.model
        
        if not self.api_key:
            raise ValueError("DeepSeek API key not configured")
            
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        通过 DeepSeek API 生成回复
        
        Args:
            prompt: 输入提示
            **kwargs: 其他生成参数
            
        Returns:
            生成的回复文本
        """
        try:
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": config.DEEPSEEK.temperature,
                "max_tokens": config.DEEPSEEK.max_length,
                "top_p": config.DEEPSEEK.top_p,
                "stream": False,
                **kwargs
            }
            
            logger.debug(f"Sending request to DeepSeek API: {self.api_base}/chat/completions")
            logger.debug(f"Request data: {data}")
            
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json=data,
                timeout=30  # 添加超时设置
            )
            
            if response.status_code != 200:
                error_msg = f"DeepSeek API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                logger.error(f"Response headers: {response.headers}")
                return error_msg
            
            try:
                result = response.json()
                if not result or 'choices' not in result:
                    error_msg = f"Invalid response format from DeepSeek API: {response.text}"
                    logger.error(error_msg)
                    return error_msg
                return result['choices'][0]['message']['content']
            except ValueError as e:
                error_msg = f"Failed to parse JSON response: {str(e)}, Response text: {response.text}"
                logger.error(error_msg)
                return error_msg
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error when calling DeepSeek API: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
        except Exception as e:
            error_msg = f"DeepSeek generation error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg 