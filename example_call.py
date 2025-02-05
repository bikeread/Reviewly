import requests
import json
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 示例代码差异
code_diff = """diff --git a/services/code_review_service.py b/services/code_review_service.py
--- a/services/code_review_service.py
+++ b/services/code_review_service.py
@@ -15,8 +15,12 @@ class CodeReviewService:
     def review_diff(self, diff_text: str) -> str:
-        # 简单调用模型
-        prompt = CODE_REVIEW_PROMPT.format(code_diff=diff_text)
-        return self.llm.generate(prompt)
+        # 增加输入验证
+        if not diff_text or diff_text.isspace():
+            return "无效的代码差异"
+            
+        # 格式化并调用模型
+        formatted_diff = self._format_diff(diff_text)
+        prompt = CODE_REVIEW_PROMPT.format(code_diff=formatted_diff)
+        return self.llm.generate(prompt)"""

try:
    logger.info("开始发送代码评审请求...")
    logger.debug(f"请求数据: {code_diff[:200]}...") # 只记录前200个字符避免日志过长

    # 发送请求
    response = requests.post(
        "http://localhost:8000/webhook",
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "code_diff": code_diff
        })
    )
    
    logger.info(f"请求完成，状态码: {response.status_code}")
    
    # 打印结果
    print("Status Code:", response.status_code)
    if response.status_code == 200:
        result = response.json()
        logger.info("成功获取评审结果")
        logger.debug(f"评审结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        print("Response:", json.dumps(result, indent=2, ensure_ascii=False))
    else:
        logger.error(f"请求失败: {response.text}")
        print("Error Response:", response.text)

except Exception as e:
    logger.error(f"发生错误: {str(e)}", exc_info=True)
    raise 