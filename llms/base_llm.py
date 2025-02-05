from abc import ABC, abstractmethod

class LLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        输入Prompt字符串，返回模型生成的文本结果。
        """
        pass
