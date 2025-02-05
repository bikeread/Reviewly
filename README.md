# Code Review Service

本项目演示如何使用本地模型 / OpenAI 模型做代码审阅，并可对接GitLab/GitHub。

## 安装 & 运行

1. 克隆本项目
2. 安装依赖
   ```bash
   pip install -r requirements.txt
3. 在 config.py 或 .env 文件中配置模型路径、OpenAI Key等
4. 启动服务
    ```bash
   python main.py
   # 或者使用 Flask CLI
   # export FLASK_APP=main.py
   # flask run --host=0.0.0.0 --port=8000
5. 配置 GitLab/GitHub Webhook:
    指向 http://<your-host>:8000/webhook
    触发类型可以是 push events / pull request events


---

# 如何使用这份代码

1. **克隆或拷贝**上述结构到你的项目目录中。  
2. **安装依赖**：`pip install -r requirements.txt`。  
3. 修改 `config.py` 中的 `MODEL_PATH`、`OPENAI_API_KEY` 等参数，保证能成功加载到模型或调用 OpenAI。  
4. 在 `services/webhook_handler.py` 中，根据你的 GitLab/GitHub Webhook 格式，实现真正的 "获取 repo_url / commit_id / diff" 逻辑。  
5. 如需在 PR 下自动评论，需要再调 GitHub/GitLab API（在 `handle_webhook` 最后加入相应逻辑）。  
6. 启动服务：
    ```bash
    # 方式1：直接运行
    python main.py
    
    # 方式2：使用 Flask CLI（支持更多选项）
    export FLASK_APP=main.py
    # 开发模式
    export FLASK_ENV=development  
    flask run --host=0.0.0.0 --port=8000
    ```
7. 在 GitLab/GitHub 中，将 Webhook 指向 `http://<your-deployed-ip>:8000/webhook/` 并选择对应事件（Push/MR/PR等）。  
8. 提交代码后，Webhook 触发，服务会调用本地/远程 LLM 生成审阅文本，并可在 CI 流程或 PR 讨论里输出结果。

这样就可以拥有一个**可替换模型、可对接 Git 平台**的自动 Code Review 服务雏形。你可以在此基础上进一步完善"多轮对话"、"分段审阅"、"精细化行级提示"等高级功能。祝你开发顺利!

## 配置说明

本服务支持通过环境变量或 `.env` 文件进行配置。主要配置项包括：

### 基础配置
- `HOST`: 服务监听地址，默认 "0.0.0.0"
- `PORT`: 服务端口，默认 8000
- `DEBUG`: 调试模式，默认 false

### 模型配置
- `MODEL_PATH`: 本地模型路径
- `MODEL_TYPE`: 模型类型，可选 "local" 或 "openai"
- `OPENAI_API_KEY`: OpenAI API密钥（使用OpenAI时需要）
- `EMBEDDING_MODEL`: 代码向量化模型，默认 "microsoft/codebert-base"

### 系统配置
- `NUM_THREADS`: PyTorch线程数，默认 8
- `CACHE_TTL`: 文件缓存有效期(秒)，默认 3600
- `REPO_ROOT`: 代码仓库根目录，默认 "./"
- `REPO_FILE_PATTERNS`: 需要加载的文件类型，默认 "*.py,*.js,*.java,*.go,*.ts"

### 日志配置
- `LOG_LEVEL`: 日志级别，默认 "INFO"
- `LOG_FILE`: 日志文件路径，默认 "app.log"

### 向量化配置
- `VECTOR_DB_PATH`: 向量数据库存储路径，默认 "./vector_db"
- `VECTOR_ON_DISK`: 是否使用磁盘存储向量，默认 true
- `VECTOR_CACHE_DIR`: 向量缓存目录，可选
- `BATCH_SIZE`: 向量化批处理大小，默认 32

详细配置请参考 `.env.example` 文件。

## 环境配置

1. 复制配置模板：
   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env` 文件，根据需要修改配置：
   - 如果使用本地模型，设置 `MODEL_TYPE=local` 并配置 `MODEL_PATH`
   - 如果使用OpenAI，设置 `MODEL_TYPE=openai` 并配置 `OPENAI_API_KEY`
   - 根据机器资源调整 `NUM_THREADS` 和 `BATCH_SIZE`
   - 配置日志相关参数

3. 确保 `.env` 文件不会被提交到版本控制系统：
   ```bash
   echo ".env" >> .gitignore
   ```

