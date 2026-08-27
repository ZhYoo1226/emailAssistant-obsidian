# Email Assistant — 基于 CrewAI + Qdrant + Obsidian 的 Agentic RAG

一个 agentic RAG 系统：把你的 [Obsidian](https://obsidian.md/) 笔记库变成知识库，自动为 Gmail 收件箱里的**提问**起草回复。

新邮件进来时，系统先分类，**只有看起来是「问题」的邮件**才会去知识库检索相关内容，并写成一条 **Gmail 草稿**（绝不自动发送，你始终掌握最终决定权）。

---

## 工作原理

两个循环并行运行：

```
┌─────────────────────────────┐        ┌──────────────────────────────────────┐
│  Obsidian 笔记库             │        │  Gmail 收件箱                        │
│  .md 文件                   │        │  （每 60 秒轮询一次）                 │
└──────────────┬──────────────┘        └──────────────┬───────────────────────┘
               │  watchdog（实时监听）                  │  新的未读邮件
               ▼                                       ▼
        切块 + 上下文化                          分类（flash）
        （flash LLM）                             QUESTION / NOTIFICATION /
               │                                 NEWSLETTER / SPAM
               ▼                                       │
        嵌入（fastembed，本地）                        │  仅当是 QUESTION
               │                                       ▼
               ▼                                检索知识库
        Qdrant 知识库                           （QdrantSearchTool）
                                               生成回复（pro LLM）
                                                       │
                                                       ▼
                                               保存为 Gmail 草稿
```

### 两个 CrewAI Crew

- **`KnowledgeOrganizingCrew`** —— 读一篇 Markdown 笔记，切成语义块，为每块补充检索上下文，本地嵌入后写入 Qdrant。
- **`AutoResponderCrew`** —— 给邮件线程分类，并且**仅当分类为 `QUESTION`** 时才基于知识库写回复。

### 模型分配

| Agent | 任务 | 模型 |
|---|---|---|
| `chunks_extractor` | 语义切块 | 网关 `deepseek-v4-flash` |
| `contextualizer` | 块上下文补全 | 网关 `deepseek-v4-flash` |
| `categorizer` | 邮件分类 | 网关 `deepseek-v4-flash` |
| `response_writer` | 回复生成 | 网关 `deepseek-v4-pro` |

### 知识库同步保证

- **内容哈希幂等** —— 每个 chunk 都存源文件的 SHA-256。启动时，只有内容变了才会重新入库。
- **孤儿清理** —— 启动时，源文件已不在笔记库里的 chunk 会被删除，保证 Obsidian 是 Qdrant 的唯一事实来源。
- **类型过滤** —— 跳过 Excalidraw 绘图和 `.trash/` 回收站文件（它们不是正文笔记）。

---

## 技术栈

- [CrewAI](https://www.crewai.com/) `0.95` —— agent 编排
- [Qdrant](https://qdrant.tech/) —— 向量库 / 知识库
- [fastembed](https://github.com/qdrant/fastembed) —— 本地 ONNX 嵌入（`BAAI/bge-small-en-v1.5`）
- **OpenAI 兼容网关**承载 DeepSeek 模型（所有 LLM 调用）
- Gmail API（OAuth 2.0）

无需 GPU。LLM 推理走网关 API，嵌入在本地 CPU 上运行。

---

## 前提条件

- Python **3.10–3.12**
- [Poetry](https://python-poetry.org/) 用于依赖管理
- Docker（本地跑 Qdrant），**或** 免费的 [Qdrant Cloud](https://cloud.qdrant.io/) 账号
- 一个 OpenAI 兼容网关的 URL + API key
- Gmail API 凭据（见下文）

---

## 配置

复制 `.env.example` 为 `.env` 并填写：

```dotenv
# LLM 网关（OpenAI 兼容）
GATEWAY_BASE_URL=https://your-gateway.example.com/v1
GATEWAY_API_KEY=YOUR_API_KEY
GATEWAY_MAIN_MODEL=deepseek-v4-pro
GATEWAY_FAST_MODEL=deepseek-v4-flash

# 本地嵌入模型（fastembed）
# 若 Hugging Face 无法访问，设 HF_ENDPOINT=https://hf-mirror.com
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5

# Qdrant（本地 Docker 无需 API key）
QDRANT_LOCATION=http://localhost:6333
QDRANT_API_KEY=

# Obsidian 笔记库路径（知识库来源）
OBSIDIAN_VAULT_PATH=/path/to/your/vault
```

### Gmail OAuth 凭据

1. 在 [Google Cloud Console](https://console.cloud.google.com/) 创建项目并**启用 Gmail API**。
2. 把 OAuth 同意屏幕配成 **外部（External）**，并把你的 Gmail 加为测试用户。
3. 创建类型为 **桌面应用（Desktop app）** 的 OAuth 客户端，下载 JSON，重命名为 `credentials.json` 放到项目根目录。
4. 首次运行会弹出浏览器让你授权，之后生成 `token.json` 复用。

> 若处于防火墙/GFW 环境，Python 进程需要代理才能访问 `www.googleapis.com`：
> `export HTTPS_PROXY=http://127.0.0.1:7890`（建议写入 shell 配置文件）。

---

## 运行

```bash
poetry install              # 安装依赖
bash scripts/run-qdrant.sh  # 启动本地 Qdrant（或用 Qdrant Cloud）
poetry run python main.py
```

`main.py` 会启动两个线程：

1. 一个文件系统监听器，把 Obsidian 笔记库同步进 Qdrant；
2. 一个 Gmail 监听器，每 60 秒轮询收件箱并处理新的未读线程。

用 `Ctrl+C` 退出 —— Gmail 游标（`last_history_id`）会被保存，下次接着处理。

---

## 迁移到另一台机器

知识库是**从笔记库重建的**，所以无需拷贝 Qdrant 数据：

1. `git clone` 仓库，`poetry install`
2. 拷贝你的 `.md` 笔记库，并把 `OBSIDIAN_VAULT_PATH` 指向它
3. 补上 `.env`（网关 key + 笔记库路径）和 `credentials.json`
4. 启动一个空的 Qdrant，跑 `main.py` —— 知识库会自动重新入库

注意：切块/上下文化是 LLM 生成的（非确定性），所以重新入库是「语义等价」但非「逐字节一致」。若要逐字节一致，改用 Qdrant collection 快照导出/导入。
