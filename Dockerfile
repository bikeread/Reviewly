# 第一阶段：下载模型
FROM python:3.10-slim as model-downloader
WORKDIR /app

# 安装必要的包
COPY requirements.txt .
RUN pip install --no-cache-dir transformers torch

# 复制并运行下载脚本
COPY scripts/download_models.py .
RUN python download_models.py

# 第二阶段：构建应用
FROM python:3.10-slim
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY .env.example .env
COPY . .

# 从第一阶段复制模型文件
COPY --from=model-downloader /app/models /app/models

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 设置环境变量
ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV EMBEDDING_MODEL=/app/models/codebert-base

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "main.py"] 