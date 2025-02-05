from fastapi import APIRouter, Request, Body
from typing import Optional
from pydantic import BaseModel, Field
from services.code_review_service import CodeReviewService
from llms.transformers_llm import TransformersLLMClient
# 也可换成 openai_llm.OpenAIModelClient
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# 定义请求模型
class WebhookRequest(BaseModel):
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

# 在这里创建一个全局的 CodeReviewService 实例
# 你也可以用依赖注入的方式，每次请求都重新初始化
logger.info("初始化 TransformersLLMClient...")
llm_client = TransformersLLMClient()
logger.info("初始化 CodeReviewService...")
review_service = CodeReviewService(llm_client)

@router.post("/")
async def handle_webhook(request: WebhookRequest = Body(...)):
    """
    接收 Webhook 事件,支持直接传入代码差异。
    """
    logger.info(f"收到webhook请求: repo={request.repo_url}, commit={request.commit_id}")
    logger.debug(f"代码差异长度: {len(request.code_diff) if request.code_diff else 0}")
    
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
        logger.error(f"处理webhook请求时发生错误: {str(e)}", exc_info=True)
        raise
