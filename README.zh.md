# Email Assistant — 基于 CrewAI + Qdrant + Obsidian 的 Agentic RAG

一个 agentic RAG 系统：把你的 [Obsidian](https://obsidian.md/) 笔记库变成知识库，自动回复 Gmail 收件箱里的**提问**。

新邮件进来时，系统先分类，**只有看起来是「问题」的邮件**才会去知识库检索相关内容，并直接发送回复。若知识库不足以支撑一个忠实回答，则发送一条礼貌的「超出能力范围」说明。

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
        Qdrant 知识库                           （QdrantHybridSearchTool）
                                               生成回复（pro LLM）
                                                       │
                                                       ▼
                                               发送回复（或兜底）
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

- [CrewAI](https://www.crewai.com/) `1.15` —— agent 编排
- [Qdrant](https://qdrant.tech/) —— 向量库 / 知识库，**混合检索**（稠密 + 稀疏向量 RRF 融合，再经交叉编码器重排）
- [fastembed](https://github.com/qdrant/fastembed) —— 本地 ONNX 嵌入：`BAAI/bge-small-en-v1.5`（稠密）+ `Qdrant/bm25` 接 jieba 分词（稀疏）+ `BAAI/bge-reranker-base`（重排）
- [jieba](https://github.com/fxsjy/jieba) —— 中文分词，喂给 BM25 稀疏检索路径
- **OpenAI 兼容网关**承载 DeepSeek 模型（所有 LLM 调用）
- Gmail API（OAuth 2.0）

无需 GPU。LLM 推理走网关 API，嵌入、稀疏向量与重排在本地 CPU 上运行。

> **从旧版本（纯稠密检索）升级？** collection 结构变了（新增了名为 `sparse` 的稀疏向量）。删除现有的 `knowledge-base` collection 再重启即可，笔记库会自动重新入库。

---

## 前提条件

- Python **3.10–3.12**
- 依赖管理：[uv](https://docs.astral.sh/uv/) —— Python 版本和依赖一把抓，缺啥装啥
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

# 代理（防火墙/GFW 环境必填，程序启动时自动加载，无需在 shell 里 export）
# 端口改成你代理软件的 HTTP 端口；网络环境不需要代理时删掉或注释这行
HTTPS_PROXY=http://127.0.0.1:7890

# 可选调优
# 每次 LLM 调用的超时（秒，默认 180）
LLM_TIMEOUT_SEC=180
# 拒绝 tool_choice 的网关模型（思考型模型），逗号分隔的子串匹配。
# 命中的模型会回退到 JSON 模式的结构化输出。
NO_TOOL_CHOICE_MODELS=deepseek-v4,qwen3.8
```

### Gmail OAuth 凭据

1. 在 [Google Cloud Console](https://console.cloud.google.com/) 创建项目并**启用 Gmail API**。
2. 把 OAuth 同意屏幕配成 **外部（External）**，并把你的 Gmail 加为测试用户。
3. 创建类型为 **桌面应用（Desktop app）** 的 OAuth 客户端，下载 JSON，重命名为 `credentials.json` 放到项目根目录。
4. 首次运行会弹出浏览器让你授权，之后生成 `token.json` 复用。

> 若处于防火墙/GFW 环境，Python 进程需要代理才能访问 `www.googleapis.com`：
> 在 `.env` 里配置 `HTTPS_PROXY`（见上方配置示例，启动时自动加载，改端口在 `.env` 里改）。

---

## 运行

```bash
# 任意系统（uv）—— 缺 Python 的话 uv 会自动装
bash scripts/run-qdrant.sh   # 启动本地 Qdrant（或用 Qdrant Cloud）
uv sync                      # 创建 .venv 并安装依赖
uv run main.py               # 启动助手
```

防火墙/GFW 环境下，把代理写进 `.env` 即可，程序启动时会自动加载：

```dotenv
HTTPS_PROXY=http://127.0.0.1:7890
```

（端口号改成你代理软件的实际 HTTP 端口；网络环境不需要代理时删掉这行。）

`main.py` 会启动两个线程：

1. 一个文件系统监听器，把 Obsidian 笔记库同步进 Qdrant；
2. 一个 Gmail 监听器，每 60 秒轮询收件箱并处理新的未读线程。

用 `Ctrl+C` 退出 —— Gmail 游标（`last_history_id`）会被保存，下次接着处理。

### 在 Windows 上作为后台任务运行

安装一个计划任务：登录时自启、崩溃后自动重启：

```powershell
.\scripts\install-task.ps1                # 安装
Start-ScheduledTask -TaskName EmailAssistant
.\scripts\install-task.ps1 -Remove       # 卸载
```

### 开发

```bash
uv run ruff check src tests config.py main.py   # 代码检查
uv run pytest tests/ -q                          # 单元测试
```

每次 push 和 pull request 时，GitHub Actions CI 也会自动跑这两步（见仓库的 **Actions** 标签页）。

---

## 迁移到另一台机器

知识库是**从笔记库重建的**，所以无需拷贝 Qdrant 数据：

1. `git clone` 仓库，`uv sync`
2. 拷贝你的 `.md` 笔记库，并把 `OBSIDIAN_VAULT_PATH` 指向它
3. 补上 `.env`（网关 key + 笔记库路径）和 `credentials.json`
4. 启动一个空的 Qdrant，跑 `main.py` —— 知识库会自动重新入库

注意：切块/上下文化是 LLM 生成的（非确定性），所以重新入库是「语义等价」但非「逐字节一致」。若要逐字节一致，改用 Qdrant collection 快照导出/导入。

---

## 常见问题（FAQ）

**Q：邮件明明还是未读，系统却一直不处理？**

A：系统用 `last_history_id` 游标（存在 `gmail_inbox_state.json`）防重复，而不是看「未读/已读」标记。一个线程处理过后，即使还是未读也不会回头处理。要强制重扫所有未读邮件，删掉 `gmail_inbox_state.json` 再重启。

**Q：每次重启都弹 Gmail 授权？**

A：access token 1 小时就过期。现在代码会通过 refresh token 静默刷新，7 天内重启不会弹。Google「测试模式」的 refresh token 7 天过期，所以大约每周会被要求重新授权一次——这是 Google 的限制，不是 bug。

**Q：Excalidraw 绘图 / 回收站的笔记被灌进知识库了？**

A：两者都已被过滤：Excalidraw 的 `.md`（通过 `excalidraw-plugin` frontmatter 识别）和 `.trash/` 文件在入库时被跳过。

**Q：入库很慢？**

A：每篇笔记要两次 LLM 调用（切块 + 上下文化）。入库是一次性成本，之后只有内容变化的笔记才会重新处理。

**Q：为什么系统没回复某封邮件？**

A：它只回复分类为 `QUESTION` 的邮件，`NOTIFICATION` / `NEWSLETTER` / `SPAM` 会故意跳过。

**Q：回复是自动发送的（还是存草稿）？**

A：回复通过 Gmail API **直接发送**（不存草稿）。如果你希望发送前人工确认，把 `gmail/handlers.py` 里的 `send_message` 改回草稿即可。

**Q：知识库答不了问题时会发生什么？**

A：回复在发送前要过几层校验：

1. **来源校验** —— 回复引用的每个来源路径必须真实存在于笔记库中；
2. **忠实性校验** —— 用第二个 LLM 验证回复的每句话是否被检索到的来源支持，模型幻觉出的答案会在这里被拦下；
3. 校验失败时，改发一条礼貌的兜底说明：「我是智能邮件助手，我的知识库暂时没有足够资料支撑，请等待本人回复或通过其他方式联系」。

这能防止幻觉内容发给对方。

**Q：回复里会出现 [来源1] 或 [1] 这样的引用标记吗？**

A：不会。来源只作为内部校验依据记录，不会展示给收件人。即使生成文本中漏出了引用标记，发送前也会被自动清除。
