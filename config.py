import os
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings
import torch
from dotenv import load_dotenv
from pydantic import validator
import logging

# 根据环境变量加载对应的配置文件
env = os.getenv('ENVIRONMENT', 'dev')
env_file = f".env.{env}"
if os.path.exists(env_file):
    load_dotenv(env_file)
else:
    load_dotenv()  # 如果没有找到特定环境的配置文件，就加载默认的 .env

logger = logging.getLogger(__name__)
logger.info(f"Loading configuration from {env_file if os.path.exists(env_file) else '.env'}")

class ModelConfig(BaseModel):
    """模型基础配置"""
    max_length: int = Field(default=2048, description="模型最大输入长度")
    temperature: float = Field(default=0.7, description="生成温度")
    top_p: float = Field(default=0.9, description="Top-p采样参数")
    top_k: int = Field(default=40, description="Top-k采样参数")
    repetition_penalty: float = Field(default=1.1, description="重复惩罚系数")

class LocalModelConfig(ModelConfig):
    """本地模型配置"""
    path: Path = Field(
        default=Path("models/Qwen2-7B-Chat"),
        description="本地模型路径"
    )
    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="模型运行设备"
    )
    load_8bit: bool = Field(default=False, description="是否使用8位量化")

class OpenAIConfig(ModelConfig):
    """OpenAI API配置"""
    api_key: Optional[str] = Field(default=None, description="OpenAI API密钥")
    model: str = Field(default="gpt-3.5-turbo", description="OpenAI模型名称")
    api_base: Optional[str] = Field(default=None, description="OpenAI API基础URL")

class DeepSeekConfig(ModelConfig):
    """DeepSeek API配置"""
    api_key: Optional[str] = Field(default=None, description="DeepSeek API密钥")
    model: str = Field(default="deepseek-chat", description="DeepSeek模型名称")
    api_base: str = Field(
        default="https://api.deepseek.ai/v1",
        description="DeepSeek API基础URL"
    )

# 首先定义代码审查配置的模型
class CodeReviewConfig(BaseModel):
    """代码审查配置模型"""
    FILE_PATTERNS: List[str] = Field(
        default=["*.py", "*.js", "*.java", "*.go", "*.ts"],
        description="支持的文件类型"
    )
    IGNORE_DIRS: List[str] = Field(
        default=[".git", ".venv", "venv", "__pycache__", "node_modules", "vector_db"],
        description="忽略的目录"
    )
    MAX_FILE_SIZE: int = Field(
        default=1024 * 1024,
        description="最大文件大小(1MB)"
    )
    MAX_RELATED_FILES: int = Field(
        default=3,
        description="相关文件最大数量"
    )
    CONTEXT_LINES: int = Field(
        default=5,
        description="上下文显示行数"
    )
    MAX_REVIEW_LENGTH: int = Field(
        default=2000,
        description="评审结果最大长度"
    )
    REVIEW_LANGUAGE: str = Field(
        default="zh_CN",
        description="评审结果语言"
    )
    GITHUB_TOKEN: Optional[str] = Field(
        default=None,
        description="GitHub API Token"
    )
    GITLAB_TOKEN: Optional[str] = Field(
        default=None,
        description="GitLab API Token"
    )
    AUTO_COMMENT: bool = Field(
        default=False,
        description="是否自动评论"
    )
    WEBHOOK_SECRET: Optional[str] = Field(
        default=None,
        description="Webhook密钥"
    )

class ServiceConfig(BaseSettings):
    """服务配置类"""
    
    class Config:
        env_file = f".env.{os.getenv('ENVIRONMENT', 'dev')}"
        env_file_encoding = 'utf-8'
        case_sensitive = True
        env_nested_delimiter = '__'
        
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                env_settings,  # 环境变量优先级高于配置文件
                file_secret_settings,
            )

    # === 服务基础配置 ===
    HOST: str = Field(default="0.0.0.0", description="服务监听地址")
    PORT: int = Field(default=8000, description="服务端口")
    DEBUG: bool = Field(default=False, description="是否开启调试模式")
    
    # === 代码审查配置 ===
    CODE_REVIEW: CodeReviewConfig = Field(
        default_factory=CodeReviewConfig,
        description="代码审查配置"
    )
    
    # 添加模型配置验证
    @validator('CODE_REVIEW', pre=True)
    def build_code_review_config(cls, v, values):
        if isinstance(v, dict):
            return v
        return CodeReviewConfig(
            GITHUB_TOKEN=os.getenv('GITHUB_TOKEN'),
            WEBHOOK_SECRET=os.getenv('WEBHOOK_SECRET'),
            AUTO_COMMENT=os.getenv('AUTO_COMMENT', 'false').lower() == 'true',
            FILE_PATTERNS=os.getenv('CODE_REVIEW__FILE_PATTERNS', '*.py,*.js,*.java,*.go,*.ts').split(','),
            MAX_FILE_SIZE=int(os.getenv('CODE_REVIEW__MAX_FILE_SIZE', str(1024 * 1024))),
            MAX_RELATED_FILES=int(os.getenv('CODE_REVIEW__MAX_RELATED_FILES', '3')),
            CONTEXT_LINES=int(os.getenv('CODE_REVIEW__CONTEXT_LINES', '5')),
            MAX_REVIEW_LENGTH=int(os.getenv('CODE_REVIEW__MAX_REVIEW_LENGTH', '2000')),
            REVIEW_LANGUAGE=os.getenv('CODE_REVIEW__REVIEW_LANGUAGE', 'zh_CN')
        ).dict()
    
    # === 模型配置 ===
    MODEL_TYPE: Literal["local", "openai", "deepseek"] = Field(
        default="deepseek",
        description="使用的模型类型"
    )
    
    # 各类模型配置
    LOCAL_MODEL: LocalModelConfig = Field(
        default_factory=LocalModelConfig,
        description="本地模型配置"
    )
    OPENAI: OpenAIConfig = Field(
        default_factory=OpenAIConfig,
        description="OpenAI配置"
    )
    DEEPSEEK: DeepSeekConfig = Field(
        default_factory=DeepSeekConfig,
        description="DeepSeek配置"
    )
    
    # === 向量化配置 ===
    EMBEDDING_MODEL: str = Field(
        default="microsoft/codebert-base",
        description="代码向量化模型"
    )
    VECTOR_DB_PATH: Path = Field(
        default=Path("./vector_db"),
        description="向量数据库存储路径"
    )
    VECTOR_COLLECTION: str = Field(
        default="code_embeddings",
        description="向量集合名称"
    )
    BATCH_SIZE: int = Field(default=32, description="向量化批处理大小")
    
    # === 系统配置 ===
    NUM_THREADS: int = Field(default=8, description="PyTorch线程数")
    CACHE_TTL: int = Field(default=3600, description="文件缓存有效期(秒)")
    
    # === 日志配置 ===
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        description="日志格式"
    )
    LOG_FILE: Optional[Path] = Field(
        default=Path("app.log"),
        description="日志文件路径"
    )
    
    # === 代码仓库配置 ===
    REPO_ROOT: Path = Field(
        default=Path(os.getcwd()),
        description="代码仓库根目录"
    )
    REPO_FILE_PATTERNS: List[str] = Field(
        default=["*.py", "*.js", "*.java", "*.go", "*.ts"],
        description="需要加载的文件类型"
    )

    @validator('DEEPSEEK', pre=True)
    def build_deepseek_config(cls, v, values):
        if isinstance(v, dict):
            return v
        return {
            'api_key': os.getenv('DEEPSEEK__API_KEY'),
            'model': os.getenv('DEEPSEEK__MODEL', 'deepseek-chat'),
            'api_base': os.getenv('DEEPSEEK__API_BASE', 'https://api.deepseek.ai/v1'),
            'max_length': int(os.getenv('DEEPSEEK__MAX_LENGTH', '2048')),
            'temperature': float(os.getenv('DEEPSEEK__TEMPERATURE', '0.7')),
            'top_p': float(os.getenv('DEEPSEEK__TOP_P', '0.9')),
            'top_k': int(os.getenv('DEEPSEEK__TOP_K', '40')),
            'repetition_penalty': float(os.getenv('DEEPSEEK__REPETITION_PENALTY', '1.1'))
        }

    @validator('OPENAI', pre=True)
    def build_openai_config(cls, v, values):
        if isinstance(v, dict):
            return v
        return {
            'api_key': os.getenv('OPENAI__API_KEY'),
            'model': os.getenv('OPENAI__MODEL', 'gpt-3.5-turbo'),
            'api_base': os.getenv('OPENAI__API_BASE'),
            'max_length': 2048,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'repetition_penalty': 1.1
        }

    @validator('LOCAL_MODEL', pre=True)
    def build_local_model_config(cls, v, values):
        if isinstance(v, dict):
            return v
        return {
            'path': os.getenv('LOCAL_MODEL__PATH', 'models/Qwen2-7B-Chat'),
            'device': os.getenv('LOCAL_MODEL__DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'),
            'load_8bit': os.getenv('LOCAL_MODEL__LOAD_8BIT', 'false').lower() == 'true',
            'max_length': 2048,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'repetition_penalty': 1.1
        }

# 创建全局配置实例
config = ServiceConfig()
