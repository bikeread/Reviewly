import torch
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from .base_llm import LLMInterface
from config import config
import logging
import gc

logger = logging.getLogger(__name__)

class TransformersLLMClient(LLMInterface):
    def __init__(self):
        """初始化本地模型客户端"""
        try:
            logger.info(f"正在加载本地模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                config.LOCAL_MODEL.path,
                device_map="auto" if config.LOCAL_MODEL.device == "cuda" else None,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                load_in_8bit=config.LOCAL_MODEL.load_8bit
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.LOCAL_MODEL.path,
                trust_remote_code=True
            )
            
            # 设置设备
            if config.LOCAL_MODEL.device == "cuda":
                self.model = self.model.cuda()
            logger.info("模型加载完成")
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise

    def _clear_cuda_cache(self):
        """清理CUDA缓存"""
        if config.LOCAL_MODEL.device == "cuda":
            torch.cuda.empty_cache()
            
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成回复文本
        
        Args:
            prompt: 输入提示
            **kwargs: 其他生成参数
            
        Returns:
            生成的回复文本
        """
        if not prompt.strip():
            logger.warning("收到空提示")
            return "请提供有效的输入文本。"
            
        try:
            logger.debug(f"开始处理提示，长度: {len(prompt)}")
            
            # 1. 编码输入
            logger.debug("tokenizing输入...")
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.LOCAL_MODEL.max_length,
            ).to(config.LOCAL_MODEL.device)
            
            # 2. 更新生成配置
            logger.debug("更新生成配置...")
            generation_config = GenerationConfig(**self.generation_config.to_dict())
            if kwargs:
                generation_config.update(**kwargs)
                logger.debug(f"使用自定义参数: {kwargs}")
            
            # 3. 生成回复
            logger.info("开始生成回复...")
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                )
            logger.info("生成完成")
            
            # 4. 解码输出
            logger.debug("解码输出...")
            response = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # 5. 清理缓存
            logger.debug("清理CUDA缓存...")
            self._clear_cuda_cache()
            
            # 6. 移除prompt部分
            prompt_length = len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
            response = response[prompt_length:].strip()
            
            logger.debug(f"最终响应长度: {len(response)}")
            return response
            
        except Exception as e:
            logger.error(f"生成过程中发生错误: {str(e)}", exc_info=True)
            return f"生成过程中发生错误: {str(e)}"
        
    def __del__(self):
        """析构函数，确保资源正确释放"""
        self._clear_cuda_cache()
