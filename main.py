from fastapi import FastAPI
import logging
import sys
import torch
from config import config
from services.webhook_handler import router as webhook_router
from services.github_webhook_handler import router as github_router
import os

# ===== 全局设置日志 =====
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE) if config.LOG_FILE else logging.NullHandler()
    ]
)

# 设置第三方库的日志级别
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

logger.info("开始初始化应用...")
logger.info(f"当前环境: {os.getenv('ENVIRONMENT', 'dev')}")
logger.info(f"使用配置文件: {config.Config.env_file}")

# 验证关键配置
if config.MODEL_TYPE == "deepseek":
    if not config.DEEPSEEK.api_key:
        logger.error("DeepSeek API key not configured!")
        sys.exit(1)
    logger.info(f"DeepSeek API Base: {config.DEEPSEEK.api_base}")
    logger.info(f"DeepSeek Model: {config.DEEPSEEK.model}")

# ===== 创建 FastAPI 应用 =====
app = FastAPI()
logger.info("FastAPI 应用创建完成")

# ===== 初始化服务 =====
logger.info("设置 PyTorch 线程数...")
torch.set_num_threads(config.NUM_THREADS)
logger.info(f"PyTorch线程数设置为: {config.NUM_THREADS}")

# 注册路由
logger.info("开始注册路由...")
app.include_router(webhook_router, prefix="/webhook", tags=["webhook"])     # 会变成 /webhook/direct
app.include_router(github_router, prefix="/webhook", tags=["github"])       # 会变成 /webhook/github
logger.info("路由注册完成")

@app.get("/")
async def home():
    logger.info("访问根路径")
    return {"message": "Hello, this is Code Review Service"}

if __name__ == "__main__":
    try:
        # 检查必要的依赖
        try:
            import uvicorn
        except ImportError:
            logger.error("缺少必要的依赖 'uvicorn'，请运行: pip install uvicorn")
            sys.exit(1)
            
        logger.info("启动 uvicorn 服务器...")
        uvicorn.run(
            "main:app",
            host=config.HOST,
            port=config.PORT,
            reload=config.DEBUG,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}", exc_info=True)
        sys.exit(1)
