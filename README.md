# Code Review Service

本项目是一个基于 AI 的代码审查服务，支持多种模型（本地模型/OpenAI/DeepSeek）和多种代码托管平台（GitHub/GitLab）。

## 功能特点

- 支持多种 AI 模型：本地模型、OpenAI API、DeepSeek API
- 支持多种代码托管平台：GitHub、GitLab
- 支持 Push 事件和 PR/MR 自动评审
- 支持自动评论功能
- 支持容器化部署
- 支持向量化搜索相关代码

## 快速开始

### 本地运行

1. 克隆项目并安装依赖：
   ```bash
   git clone <repo-url>
   cd code-review-service
   pip install -r requirements.txt
   ```

2. 配置环境：
   ```bash
   cp .env.example .env
   # 编辑 .env 文件，设置必要的配置项
   ```

3. 启动服务：
   ```bash
   python main.py
   ```

## 部署指南

### 环境配置方案

本项目使用 `.env.{environment}` 文件管理不同环境的配置：

1. **配置模板**：
   - `.env.example`: 包含所有配置项的说明和默认值
   - 不包含实际的密钥或敏感信息
   - 用于生成各环境的配置文件

2. **环境配置**：
   ```bash
   # 开发环境
   cp .env.example .env.dev
   vim .env.dev    # 设置开发环境的配置

   # 生产环境
   cp .env.example .env.prod
   vim .env.prod   # 设置生产环境的配置
   ```

3. **启动服务**：
   ```bash
   # 开发环境
   ENVIRONMENT=dev docker-compose up -d

   # 生产环境
   ENVIRONMENT=prod docker-compose up -d
   ```

### 配置文件说明

1. **.env.example**：
   - 配置模板文件
   - 包含所有配置项的说明
   - 用于创建环境特定的配置

2. **.env.dev**：
   - 开发环境配置
   - 可以使用测试密钥
   - 通常启用调试模式

3. **.env.prod**：
   - 生产环境配置
   - 使用正式密钥
   - 禁用调试模式
   - 需要安全保管

### 必要配置项

1. **模型配置（必选其一）**：
   ```bash
   # OpenAI
   MODEL_TYPE=openai
   OPENAI_API_KEY=your_key_here

   # 或 DeepSeek
   MODEL_TYPE=deepseek
   DEEPSEEK_API_KEY=your_key_here
   ```

2. **GitHub配置（可选）**：
   ```bash
   GITHUB_TOKEN=your_token_here
   WEBHOOK_SECRET=your_secret_here
   ```

## 服务管理

### 启动服务

```bash
# 开发环境
ENVIRONMENT=dev docker-compose up -d

# 生产环境
ENVIRONMENT=prod docker-compose up -d
```

### 查看日志

```bash
# 查看服务日志
docker-compose logs -f

# 查看特定环境日志
ENVIRONMENT=prod docker-compose logs -f
```

### 更新服务

```bash
# 拉取最新代码
git pull

# 重建并启动服务
docker-compose down
ENVIRONMENT=prod docker-compose up -d --build
```

## 数据管理

### 持久化目录

- `vector_db/`: 向量数据库文件
- `logs/`: 应用日志
- `.env.{environment}`: 环境配置

### 备份恢复

```bash
# 创建备份
tar -czf backup-$(date +%Y%m%d).tar.gz vector_db/ logs/ .env.*

# 恢复备份
tar -xzf backup-20240101.tar.gz
```

## API 接口

1. **直接评审接口**：
   - URL: `/webhook/direct`
   - 方法: POST
   - 用途: 直接代码评审

2. **GitHub Webhook**：
   - URL: `/webhook/github`
   - 方法: POST
   - 用途: GitHub 事件处理

## 故障排除

1. **配置问题**：
   - 检查环境配置文件是否正确
   - 验证 API 密钥是否有效
   - 确认端口映射是否正确

2. **资源问题**：
   - 检查 CPU/内存限制
   - 验证磁盘空间
   - 确认目录权限

## License

[MIT License](LICENSE)

### 模型管理

本项目使用 CodeBERT 模型进行代码向量化，有两种部署方式：

1. **在线下载（默认）**：
   - 设置 `EMBEDDING_MODEL=microsoft/codebert-base`
   - 首次启动时自动下载模型
   - 需要确保网络可访问 HuggingFace

2. **离线部署（推荐）**：
   - 模型文件已包含在 Docker 镜像中
   - 设置 `EMBEDDING_MODEL=/app/models/codebert-base`
   - 无需额外下载，启动更快

### 自定义模型

如果需要使用其他向量化模型：

1. 修改 `scripts/download_models.py`
2. 重新构建 Docker 镜像：
   ```bash
   docker-compose build --no-cache
   ```

### 本地开发

如果需要在本地开发环境使用：

1. 下载模型：
   ```bash
   python scripts/download_models.py
   ```

2. 修改配置：
   ```bash
   # .env.dev
   EMBEDDING_MODEL=./models/codebert-base
   ```

