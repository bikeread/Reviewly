import logging
from openai import OpenAI
from config import config
from .base_llm import LLMInterface

logger = logging.getLogger(__name__)

class OpenAIClient(LLMInterface):
    def __init__(self):
        """初始化 OpenAI API 客户端"""
        self.api_key = config.OPENAI.api_key
        self.model = config.OPENAI.model
        self.api_base = config.OPENAI.api_base
        
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base if self.api_base else None
        )
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        通过 OpenAI API 生成回复
        
        Args:
            prompt: 输入提示
            **kwargs: 其他生成参数
            
        Returns:
            生成的回复文本
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.OPENAI.temperature,
                max_tokens=config.OPENAI.max_length,
                top_p=config.OPENAI.top_p,
                **kwargs
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = f"OpenAI generation error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
