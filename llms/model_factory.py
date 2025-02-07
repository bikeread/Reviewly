from typing import Optional
import logging
from config import config
from .base_llm import LLMInterface
from .transformers_llm import TransformersLLMClient
from .deepseek_llm import DeepSeekClient
from .openai_llm import OpenAIClient

logger = logging.getLogger(__name__)

class ModelFactory:
    _instance = None
    _model_client: Optional[LLMInterface] = None
    
    @classmethod
    def get_model(cls) -> LLMInterface:
        """获取模型客户端单例"""
        if cls._model_client is None:
            if config.MODEL_TYPE == "local":
                logger.info("使用本地模型")
                cls._model_client = TransformersLLMClient()
            elif config.MODEL_TYPE == "deepseek":
                logger.info("使用 DeepSeek API")
                cls._model_client = DeepSeekClient()
            elif config.MODEL_TYPE == "openai":
                logger.info("使用 OpenAI API")
                cls._model_client = OpenAIClient()
            else:
                raise ValueError(f"Unsupported model type: {config.MODEL_TYPE}")
                
        return cls._model_client 