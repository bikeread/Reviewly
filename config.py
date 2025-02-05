import os
from pathlib import Path
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings
import torch
from dotenv import load_dotenv

load_dotenv()

class ServiceConfig(BaseSettings):
    """服务配置类"""
    
    # 服务基础配置
    HOST: str = Field(default="0.0.0.0", description="服务监听地址")
    PORT: int = Field(default=8000, description="服务端口")
    DEBUG: bool = Field(default=False, description="是否开启调试模式")
    
    # 系统资源配置
    NUM_THREADS: int = Field(
        default=8, 
        description="PyTorch线程数"
    )
    
    # 代码向量化配置
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
    BATCH_SIZE: int = Field(
        default=32,
        description="向量化批处理大小"
    )
    
    # 代码评审配置
    MODEL_PATH: Path = Field(
        default=Path("C:/Users/biker/.cache/modelscope/hub/qwen/Qwen2___5-Coder-0___5B"),
        description="代码评审本地模型路径"
    )
    MODEL_TYPE: str = Field(
        default="local",  # 可选 "local" 或 "openai"
        description="使用的模型类型"
    )
    MODEL_DEVICE: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="模型运行设备"
    )
    MODEL_MAX_LENGTH: int = Field(
        default=2048,
        description="模型最大输入长度"
    )
    MODEL_TEMPERATURE: float = Field(
        default=0.7,
        description="生成温度"
    )
    MODEL_TOP_P: float = Field(
        default=0.9,
        description="Top-p采样参数"
    )
    MODEL_TOP_K: int = Field(
        default=40,
        description="Top-k采样参数"
    )
    MODEL_REPETITION_PENALTY: float = Field(
        default=1.1,
        description="重复惩罚系数"
    )
    LOAD_8BIT: bool = Field(
        default=False,
        description="是否使用8位量化"
    )
    
    # OpenAI配置
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API密钥"
    )
    OPENAI_MODEL: str = Field(
        default="gpt-3.5-turbo",
        description="OpenAI模型名称"
    )
    
    # 缓存配置
    CACHE_TTL: int = Field(
        default=3600,
        description="文件缓存有效期(秒)"
    )
    
    # 日志配置
    LOG_LEVEL: str = Field(
        default="INFO",
        description="日志级别"
    )
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        description="日志格式"
    )
    LOG_FILE: Optional[Path] = Field(
        default=Path("app.log"),
        description="日志文件路径"
    )
    
    # 代码仓库配置
    REPO_ROOT: Path = Field(
        default=Path(os.getcwd()),
        description="代码仓库根目录"
    )
    REPO_FILE_PATTERNS: List[str] = Field(
        default=["*.py", "*.js", "*.java", "*.go", "*.ts"],
        description="需要加载的文件类型"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# 创建全局配置实例
config = ServiceConfig()
