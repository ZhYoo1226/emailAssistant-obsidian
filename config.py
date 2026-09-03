import os

from dotenv import load_dotenv

# 加载 .env 文件（写入进程的环境变量里，在os.environ.get之前）
load_dotenv(".env")

# 本文件是所有配置的唯一定义者：环境变量只在这里读取，其余模块一律
# `import config` 取值，不再直接碰 os.environ。

# 出网代理。访问 Google/Qdrant Cloud 需要它在任何客户端库建立连接
# 之前就绪，所以放这里而不是 shell 里。在 .env 中设置 HTTPS_PROXY
# （不需要代理的网络环境可直接删掉这一行）。
https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
if https_proxy:
    os.environ.setdefault("HTTP_PROXY", https_proxy)

# ---------------------------------------------------------------------------
# LLM 网关配置（OpenAI 兼容）
#
# 网关通过 OpenAI 兼容 API 暴露 glm / DeepSeek / Qwen 等模型。CrewAI（经由
# LiteLLM）用 OPENAI_BASE_URL / OPENAI_API_KEY 来解析 "openai/<model>"。
# ---------------------------------------------------------------------------
gateway_base_url = os.environ["GATEWAY_BASE_URL"]
gateway_api_key = (
    os.environ.get("GATEWAY_API_KEY")
    or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    or os.environ.get("OPENAI_API_KEY")
)

# 把网关暴露给 LiteLLM，"openai/<model>" 字符串才能正确路由。
# 注意：LiteLLM 1.x 的 OpenAI provider 读取的是 OPENAI_API_BASE（而非
# OPENAI_BASE_URL），所以两个都设置。
os.environ.setdefault("OPENAI_API_BASE", gateway_base_url)
os.environ.setdefault("OPENAI_BASE_URL", gateway_base_url)
if gateway_api_key:
    os.environ.setdefault("OPENAI_API_KEY", gateway_api_key)

# 网关提供的模型名。
gateway_main_model = os.environ["GATEWAY_MAIN_MODEL"]
gateway_fast_model = os.environ["GATEWAY_FAST_MODEL"]

# 每次 LLM 调用的超时（秒），防止挂起的网关连接把邮件处理或摄取流程
# 永久阻塞。思考型模型 legitimately 可能较慢，所以给得宽裕。
LLM_TIMEOUT_SEC = int(os.environ["LLM_TIMEOUT_SEC"])

# 回复写作 agent 的可选 temperature 覆盖（调试幻觉问题时用）。
# 不设置则使用模型默认值。
response_temperature = (
    float(os.environ["RESPONSE_TEMPERATURE"])
    if os.environ.get("RESPONSE_TEMPERATURE")
    else None
)

# 拒绝 tool_choice 的网关模型（思考型），逗号分隔的子串匹配。命中的
# 模型回退到 JSON 模式的结构化输出（见文件末尾的补丁）；未列出的模型
# （如 glm-5.3）保持原生函数调用不变。
NO_TOOL_CHOICE_MODELS = os.environ["NO_TOOL_CHOICE_MODELS"]

# ---------------------------------------------------------------------------
# 嵌入模型配置（本地，基于 fastembed）
#
# fastembed 通过 ONNX 在本地跑嵌入模型，不需要外部嵌入 API。模型在首次
# 使用时下载；如果网络访问不了 Hugging Face，可设置 HF_ENDPOINT 指向镜像。
#
# 只定义模型名——实例化在 main.py（组装处）完成，避免 import config 时
# 就加载 ONNX 模型。
# ---------------------------------------------------------------------------
embedding_model_name = os.environ["EMBEDDING_MODEL_NAME"]

# 稀疏（词法）模型用于混合检索；交叉编码器对融合后的候选结果重排。
# 两者都通过 fastembed ONNX 在本地运行，仅用 CPU。
sparse_model_name = os.environ["SPARSE_MODEL_NAME"]
reranker_model_name = os.environ["RERANKER_MODEL_NAME"]

# Qdrant 配置
qdrant_location = os.environ["QDRANT_LOCATION"]
qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
qdrant_collection_name = os.environ["QDRANT_COLLECTION_NAME"]

# Obsidian 配置
obsidian_vault_path = os.environ["OBSIDIAN_VAULT_PATH"]


def _parse_folder_list(raw: str | None) -> list[str]:
    return [name.strip() for name in (raw or "").split(",") if name.strip()]


# 按仓库顶层文件夹（逗号分隔的名称）限制摄取范围。include 列表为空表示
# 全部文件夹；exclude 再从这个集合中剔除。排除哪些文件夹完全由 .env
# 决定，这里不做任何内置——典型条目如 .trash（Obsidian 回收站）和
# .obsidian（程序配置目录）。
obsidian_include_folders = _parse_folder_list(
    os.environ.get("OBSIDIAN_INCLUDE_FOLDERS")
)
obsidian_exclude_folders = sorted(
    set(_parse_folder_list(os.environ.get("OBSIDIAN_EXCLUDE_FOLDERS")))
)

# 跳过 frontmatter 中含有以下任一键（逗号分隔）的笔记。典型条目如
# excalidraw-plugin：Excalidraw 画板存的是压缩的 base64 数据而非正文，
# 会把 LLM 请求撑爆。同样完全由 .env 提供，不设内置默认。
obsidian_exclude_frontmatter = _parse_folder_list(
    os.environ.get("OBSIDIAN_EXCLUDE_FRONTMATTER")
)

# 共库保护：这个 collection 里不属于本仓库摄取范围的 folder（逗号分隔
# 的 metadata.folder 值）。这些 folder 下的点由其他系统管理（例如
# Hermes 用户画像的 user-profile），孤儿清理与范围过滤一律跳过它们，
# 只管理自己摄取的点。留空 = 不保护任何 folder（独占 collection 的
# 传统行为）。增删保护对象只改 .env，不动代码。
orphan_protect_folders = _parse_folder_list(
    os.environ.get("ORPHAN_PROTECT_FOLDERS")
)

# AgentOps 可观测性（留空 = 关闭）
agentops_api_key = os.environ.get("AGENTOPS_API_KEY") or None

# ---------------------------------------------------------------------------
# 针对 OpenAI 兼容网关模型的补丁（workaround）。
#
# crewai 1.15 的 litellm 链路对我们有两个缺口：
#   1. InternalInstructor（用于 Task.output_pydantic）不转发 LLM 实例的
#      timeout，网关连接一旦挂起就会永久阻塞。
#   2. 思考型模型（如 deepseek-v4）拒绝 tool_choice；而 crewai 仍会把
#      结构化输出请求走 instructor 默认的 TOOLS 模式，被网关拒绝。
#
# 下面应用的补丁：
#   1. 思考型模型的 supports_function_calling -> False，强制带工具的
#      agent 走文本（非原生）工具调用路径。
#   2. InternalInstructor 始终拿到注入了 timeout 的 litellm client；
#      思考型模型额外使用 instructor MD_JSON 模式——从回复文本中提取
#      JSON，而不是发送 tool_choice。
#
# 补丁用到的 NO_TOOL_CHOICE_MODELS / LLM_TIMEOUT_SEC 已在上方网关配置
# 一节定义。
# ---------------------------------------------------------------------------
import functools

import instructor as _instructor
from crewai.llm import LLM as _CrewAILLM
from crewai.utilities.internal_instructor import (
    InternalInstructor as _InternalInstructor,
)
from litellm import completion as _litellm_completion


def _rejects_tool_choice(model: str) -> bool:
    return any(
        name.strip() and name.strip() in model
        for name in NO_TOOL_CHOICE_MODELS.split(",")
    )


_original_supports_function_calling = _CrewAILLM.supports_function_calling


def _supports_function_calling(self) -> bool:
    model = getattr(self, "model", "") or ""
    if _rejects_tool_choice(model):
        return False
    return _original_supports_function_calling(self)

# 设置function_calling的值为true或者false
_CrewAILLM.supports_function_calling = _supports_function_calling

# 内部构造器的init
_original_ii_init = _InternalInstructor.__init__


def _ii_init(self, content, model, agent=None, llm=None):
    _original_ii_init(self, content, model, agent=agent, llm=llm)
    if not getattr(self.llm, "is_litellm", False):  # self指的是internal_instructor
        return
    kwargs = {}
    if _rejects_tool_choice(getattr(self.llm, "model", "") or ""):
        kwargs["mode"] = _instructor.Mode.MD_JSON
    timeout = getattr(self.llm, "timeout", None) or LLM_TIMEOUT_SEC
    self._client = _instructor.from_litellm(
        functools.partial(_litellm_completion, timeout=timeout),
        **kwargs,
    )


_InternalInstructor.__init__ = _ii_init
