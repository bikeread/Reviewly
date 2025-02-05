import requests
from llms.base_llm import LLMInterface
from prompts.code_review_prompt import CODE_REVIEW_PROMPT
import logging
from services.code_context_service import CodeContextService
from typing import List

logger = logging.getLogger(__name__)


def fetch_diff_from_repo(repo_url: str, commit_id: str) -> str:
    """
    从代码仓库获取差异
    """
    try:
        # 这里添加实际的获取逻辑
        # 例如通过 GitHub API:
        # headers = {"Authorization": f"token {github_token}"}
        # diff_url = f"{repo_url}/commit/{commit_id}.diff"
        # response = requests.get(diff_url, headers=headers)
        # return response.text if response.status_code == 200 else ""
        logger.info(f"尝试获取差异: {repo_url} - {commit_id}")
        return "示例代码差异"
    except Exception as e:
        logger.error(f"获取代码差异失败: {str(e)}")
        return ""


def _format_diff(diff_text: str) -> str:
    """
    格式化代码差异,使其更易于阅读和分析
    """
    # 移除过长的空行
    lines = [line for line in diff_text.splitlines() if line.strip()]
    # 确保每个文件的差异有清晰的分隔
    formatted_lines = []
    for line in lines:
        if line.startswith('diff --git'):
            formatted_lines.extend(['', '---', line, '---', ''])
        else:
            formatted_lines.append(line)
    return '\n'.join(formatted_lines)


class CodeReviewService:
    def __init__(self, llm: LLMInterface):
        """
        :param llm: 一个实现了 LLMInterface 的对象
        """
        self.llm = llm
        self.context_service = CodeContextService()

    def review_diff(self, diff_text: str) -> str:
        """
        评审代码差异
        """
        if not diff_text or diff_text.isspace():
            return "未提供有效的代码差异内容"
            
        try:
            # 解析差异中涉及的文件
            changed_files = self._extract_changed_files(diff_text)
            
            # 获取相关文件的上下文
            context = self._build_context(changed_files)
            
            # 格式化差异内容
            formatted_diff = _format_diff(diff_text)
            
            # 生成带上下文的评审提示
            prompt = CODE_REVIEW_PROMPT.format(
                code_diff=formatted_diff,
                context=context
            )
            
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"代码评审失败: {str(e)}")
            return f"评审过程发生错误: {str(e)}"

    def _extract_changed_files(self, diff_text: str) -> List[str]:
        """从diff中提取改动的文件路径"""
        files = []
        for line in diff_text.splitlines():
            if line.startswith('diff --git'):
                # 解析 "diff --git a/path/to/file b/path/to/file" 格式
                file_path = line.split()[-1][2:]  # 取 b/path/to/file 的路径部分
                files.append(file_path)
        return files

    def _build_context(self, changed_files: List[str]) -> str:
        """构建代码上下文信息"""
        context_parts = []
        
        for file_path in changed_files:
            # 获取当前文件内容
            current_content = self.context_service.get_file_context(file_path)
            if current_content:
                context_parts.append(f"\n=== {file_path} 的当前内容 ===\n{current_content}")
            
            # 获取相关文件
            related_files = self.context_service.get_related_files(file_path)
            for related_file in related_files:
                content = self.context_service.get_file_context(related_file)
                if content:
                    context_parts.append(f"\n=== 相关文件 {related_file} ===\n{content}")
        
        return "\n".join(context_parts)

    def review_commit(self, repo_url: str, commit_id: str) -> str:
        """
        评审指定提交
        """
        diff_text = fetch_diff_from_repo(repo_url, commit_id)
        if not diff_text:
            return "无法获取代码差异"
        return self.review_diff(diff_text)
