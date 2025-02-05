from flask import Flask, request, jsonify
import logging
import sys
import torch
from config import config
from services.code_review_service import CodeReviewService
from llms.transformers_llm import TransformersLLMClient

# ===== 全局设置日志 =====
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE) if config.LOG_FILE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== 创建 Flask 应用 =====
app = Flask(__name__)

# ===== 初始化服务 =====
torch.set_num_threads(config.NUM_THREADS)
logger.info(f"PyTorch线程数设置为: {config.NUM_THREADS}")

# 初始化代码评审服务
llm_client = TransformersLLMClient()
review_service = CodeReviewService(llm_client)
logger.info("代码评审服务已初始化")

@app.route('/')
def home():
    logger.info("访问根路径")
    return jsonify({"message": "Hello, this is Code Review Service"})

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    data = request.get_json()
    logger.info(f"收到webhook请求: repo={data.get('repo_url')}, commit={data.get('commit_id')}")
    
    try:
        if data.get('code_diff'):
            logger.info("使用直接传入的代码差异进行评审")
            review_result = review_service.review_diff(data['code_diff'])
        else:
            logger.info("从仓库获取代码差异进行评审")
            review_result = review_service.review_commit(data['repo_url'], data['commit_id'])
        
        logger.info("评审完成")
        return jsonify({
            "status": "ok",
            "review_result": review_result,
            "source": "direct_diff" if data.get('code_diff') else "repo_fetch"
        })
    except Exception as e:
        logger.error(f"处理webhook请求时发生错误: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("启动服务...")
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )
