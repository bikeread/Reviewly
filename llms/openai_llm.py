import openai
from .base_llm import LLMInterface
from config import OPENAI_API_KEY

class OpenAIModelClient(LLMInterface):
    def __init__(self, model_name="gpt-3.5-turbo"):
        openai.api_key = OPENAI_API_KEY
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> str:
        """
        调用OpenAI的聊天模型接口，得到生成结果。
        """
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            **kwargs
        )
        # 假设只返回第一条
        return response["choices"][0]["message"]["content"]
