# 示例的审阅提示，可根据需要调整或用更加复杂的系统/human 模板
CODE_REVIEW_PROMPT = """作为代码审阅者，请基于以下信息评估代码改动：

【仓库上下文】
{context}

【代码改动】
{code_diff}

请回答：
1. 主要改动：涉及哪些内容，目的何在，与现有代码如何关联？
2. 关键问题：存在哪些潜在 bug、风险或性能隐患？是否与整体架构一致？
3. 改进建议：列出1-2个最关键的优化点。

请用简明扼要的语言给出评审意见，重点突出最重要的发现。"""
