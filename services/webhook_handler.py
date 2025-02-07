from fastapi import APIRouter, Request, Body
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from services.code_review_service import CodeReviewService
from llms.model_factory import ModelFactory
# 也可换成 openai_llm.OpenAIModelClient
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class WebhookRequest(BaseModel):
    """通用Webhook请求模型"""
    repo_url: Optional[str] = Field(
        default="https://example.com",
        description="代码仓库URL"
    )
    commit_id: Optional[str] = Field(
        default="dummy_commit_id",
        description="提交ID"
    )
    code_diff: Optional[str] = Field(
        default=None,
        description="代码差异内容"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "repo_url": "https://github.com/user/repo",
                "commit_id": "abc123",
                "code_diff": "diff --git a/file.py b/file.py\n..."
            }
        }

logger.info("初始化 webhook_handler...")
# 使用模型工厂获取正确的模型客户端
logger.info("获取模型客户端...")
llm_client = ModelFactory.get_model()
logger.info("模型客户端初始化完成")

# 在这里创建一个全局的 CodeReviewService 实例
logger.info("初始化 CodeReviewService...")
review_service = CodeReviewService(llm_client)
logger.info("CodeReviewService 初始化完成")

@router.post("/direct")
async def handle_direct_webhook(request: WebhookRequest = Body(...)):
    """处理直接传入代码差异的webhook"""
    logger.info(f"收到直接webhook请求: repo={request.repo_url}, commit={request.commit_id}")
    
    try:
        if request.code_diff:
            logger.info("使用直接传入的代码差异进行评审")
            review_result = review_service.review_diff(request.code_diff)
        else:
            logger.info("从仓库获取代码差异进行评审")
            review_result = review_service.review_commit(request.repo_url, request.commit_id)
        
        logger.info("评审完成")
        logger.debug(f"评审结果长度: {len(review_result)}")
        
        return {
            "status": "ok",
            "review_result": review_result,
            "source": "direct_diff" if request.code_diff else "repo_fetch"
        }
    except Exception as e:
        logger.error(f"处理webhook请求失败: {str(e)}", exc_info=True)
        raise
