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
        # 1. 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_PATH,
            trust_remote_code=True,
            padding_side="left",
            use_fast=True
        )
        
        # 2. 配置模型加载参数
        model_kwargs = {
            "device_map": "auto" if config.MODEL_DEVICE == "cuda" else "cpu",
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if config.MODEL_DEVICE == "cuda" else torch.float32,
        }
        
        if config.LOAD_8BIT and config.MODEL_DEVICE == "cuda":
            model_kwargs.update({
                "load_in_8bit": True,
                "load_in_4bit": False,
            })
        
        # 3. 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_PATH,
            **model_kwargs
        )
        
        # 4. 配置生成参数
        self.generation_config = GenerationConfig(
            max_new_tokens=config.MODEL_MAX_LENGTH,
            temperature=config.MODEL_TEMPERATURE,
            top_p=config.MODEL_TOP_P,
            top_k=config.MODEL_TOP_K,
            repetition_penalty=config.MODEL_REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        
        # 5. 模型评估模式
        self.model.eval()
        
        logging.info(f"Successfully initialized model from {config.MODEL_PATH}")
            
    def _clear_cuda_cache(self):
        """清理CUDA缓存"""
        if config.MODEL_DEVICE == "cuda":
            gc.collect()
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
                max_length=config.MODEL_MAX_LENGTH,
            ).to(config.MODEL_DEVICE)
            
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
