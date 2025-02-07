import logging
from typing import Optional, Dict, Any
import requests
import hmac
import hashlib
from fastapi import APIRouter, Request, HTTPException
from config import config
from .webhook_handler import review_service

logger = logging.getLogger(__name__)
router = APIRouter()

class GitHubWebhookHandler:
    """GitHub Webhook处理器"""
    
    def __init__(self):
        self.api_base = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {config.CODE_REVIEW.GITHUB_TOKEN}"
        }
        self.webhook_secret = config.CODE_REVIEW.WEBHOOK_SECRET.encode() if config.CODE_REVIEW.WEBHOOK_SECRET else None
    
    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """验证 GitHub webhook 签名"""
        if not self.webhook_secret or not signature:
            return False
            
        try:
            # GitHub 签名格式: sha256=hash
            algo, received_hash = signature.split('=')
            if algo != 'sha256':
                return False
                
            # 计算预期的哈希
            expected_hash = hmac.new(
                self.webhook_secret,
                payload,
                hashlib.sha256
            ).hexdigest()
            
            # 使用 hmac.compare_digest 防止时序攻击
            return hmac.compare_digest(received_hash, expected_hash)
        except Exception as e:
            logger.error(f"验证签名时出错: {str(e)}")
            return False
    
    def handle_event(self, event_type: str, payload: Dict[Any, Any]) -> Optional[Dict[str, str]]:
        """处理GitHub webhook事件"""
        try:
            if event_type == "push":
                return self._handle_push_event(payload)
            elif event_type == "pull_request":
                return self._handle_pr_event(payload)
            else:
                logger.warning(f"不支持的事件类型: {event_type}")
                return None
        except Exception as e:
            logger.error(f"处理GitHub事件失败: {str(e)}")
            raise
    
    def _handle_push_event(self, payload: Dict[Any, Any]) -> Optional[Dict[str, str]]:
        """处理push事件"""
        try:
            repo_url = payload["repository"]["html_url"]
            commit_id = payload["after"]
            
            # 获取commit的diff
            diff_url = f"{self.api_base}/repos/{payload['repository']['full_name']}/commits/{commit_id}"
            response = requests.get(diff_url, headers=self.headers)
            
            if response.status_code != 200:
                logger.error(f"获取commit diff失败: {response.status_code}")
                return None
                
            return {
                "repo_url": repo_url,
                "commit_id": commit_id,
                "code_diff": response.json()["files"]
            }
        except Exception as e:
            logger.error(f"处理push事件失败: {str(e)}")
            return None
    
    def _handle_pr_event(self, payload: Dict[Any, Any]) -> Optional[Dict[str, str]]:
        """处理pull request事件"""
        try:
            if payload["action"] not in ["opened", "synchronize"]:
                return None
                
            repo_url = payload["repository"]["html_url"]
            pr_number = payload["number"]
            
            # 获取PR的diff
            diff_url = f"{self.api_base}/repos/{payload['repository']['full_name']}/pulls/{pr_number}"
            response = requests.get(
                diff_url, 
                headers={**self.headers, "Accept": "application/vnd.github.v3.diff"}
            )
            
            if response.status_code != 200:
                logger.error(f"获取PR diff失败: {response.status_code}")
                return None
                
            return {
                "repo_url": repo_url,
                "pr_number": pr_number,
                "code_diff": response.text
            }
        except Exception as e:
            logger.error(f"处理PR事件失败: {str(e)}")
            return None
            
    def post_review_comment(self, repo_full_name: str, pr_number: int, review_content: str):
        """在PR上发表评论"""
        if not config.CODE_REVIEW.AUTO_COMMENT:
            logger.info("自动评论功能未启用")
            return
            
        try:
            url = f"{self.api_base}/repos/{repo_full_name}/issues/{pr_number}/comments"
            response = requests.post(
                url,
                headers=self.headers,
                json={"body": review_content}
            )
            
            if response.status_code == 201:
                logger.info(f"成功发表评论在PR #{pr_number}")
            else:
                logger.error(f"发表评论失败: {response.status_code}")
        except Exception as e:
            logger.error(f"发表评论时发生错误: {str(e)}")

github_handler = GitHubWebhookHandler()

@router.post("/github")
async def handle_github_webhook(request: Request):
    """处理GitHub webhook请求"""
    # 获取并验证签名
    signature = request.headers.get("X-Hub-Signature-256")
    if not signature:
        raise HTTPException(status_code=401, detail="缺少签名")
        
    # 获取事件类型
    event_type = request.headers.get("X-GitHub-Event")
    if not event_type:
        raise HTTPException(status_code=400, detail="非GitHub webhook请求")
    
    try:
        # 读取原始请求体
        payload_bytes = await request.body()
        # 验证签名
        if not github_handler.verify_signature(payload_bytes, signature):
            raise HTTPException(status_code=401, detail="签名验证失败")
        
        # 解析 JSON
        payload = await request.json()
        logger.info(f"收到GitHub {event_type}事件: repo={payload.get('repository', {}).get('full_name')}")
        
        webhook_data = github_handler.handle_event(event_type, payload)
        if not webhook_data:
            raise HTTPException(status_code=400, detail="无法处理的事件类型")
            
        review_result = review_service.review_diff(webhook_data['code_diff'])
        
        if event_type == 'pull_request' and config.CODE_REVIEW.AUTO_COMMENT:
            github_handler.post_review_comment(
                payload['repository']['full_name'],
                webhook_data['pr_number'],
                review_result
            )
        
        return {
            "status": "ok",
            "review_result": review_result,
            "event_type": event_type
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理GitHub webhook失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 