import os

import jieba
from chromadb import EmbeddingFunction
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

# Load dotenv file
load_dotenv(".env")

# The proxy needed to reach Google/Qdrant Cloud must be set before any client
# library builds its connection, so it lives here rather than in the shell.
# Set HTTPS_PROXY in .env (or delete the line on networks that don't need it).
_os_proxy = os.environ.get("HTTPS_PROXY")
if _os_proxy:
    os.environ.setdefault("HTTP_PROXY", _os_proxy)

# ---------------------------------------------------------------------------
# LLM gateway configuration (OpenAI-compatible)
#
# The gateway exposes DeepSeek models over an OpenAI-compatible API. CrewAI
# (via LiteLLM) resolves "openai/<model>" using OPENAI_BASE_URL / OPENAI_API_KEY.
# ---------------------------------------------------------------------------
gateway_base_url = os.environ.get("GATEWAY_BASE_URL", "https://api.upmore.net/v1")
gateway_api_key = (
    os.environ.get("GATEWAY_API_KEY")
    or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    or os.environ.get("OPENAI_API_KEY")
)

# Expose the gateway to LiteLLM so "openai/<model>" strings are routed correctly.
# NOTE: LiteLLM 1.x reads OPENAI_API_BASE (not OPENAI_BASE_URL) for the OpenAI
# provider, so we set both.
os.environ.setdefault("OPENAI_API_BASE", gateway_base_url)
os.environ.setdefault("OPENAI_BASE_URL", gateway_base_url)
if gateway_api_key:
    os.environ.setdefault("OPENAI_API_KEY", gateway_api_key)

# Model names served by the gateway.
gateway_main_model = os.environ.get("GATEWAY_MAIN_MODEL", "deepseek-v4-pro")
gateway_fast_model = os.environ.get("GATEWAY_FAST_MODEL", "deepseek-v4-flash")

# ---------------------------------------------------------------------------
# Embedder configuration (local, via fastembed)
#
# fastembed runs embedding models locally over ONNX, so no external embedding
# API is needed. The model is downloaded on first use; set HF_ENDPOINT to a
# mirror if Hugging Face is unreachable from your network.
# ---------------------------------------------------------------------------
embedding_model_name = os.environ.get(
    "EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"
)

# Sparse (lexical) model for hybrid search; the cross-encoder reranks the
# fused candidates. Both run locally via fastembed ONNX, CPU-only.
sparse_model_name = "Qdrant/bm25"
reranker_model_name = "BAAI/bge-reranker-base"


class FastEmbedFunction(EmbeddingFunction):
    """A chromadb-compatible EmbeddingFunction backed by a local fastembed model."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def name(self) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        return f"fastembed/{self.model_name}"

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return [vector.tolist() for vector in self._model.embed(input)]


class JiebaBM25Function:
    """
    Sparse embedding for Chinese text: jieba segments the text into
    space-separated tokens, then fastembed's BM25 converts them into sparse
    vectors. Without pre-segmentation BM25 falls back to whitespace
    tokenization, which collapses a Chinese sentence into 1-2 tokens.
    """

    def __init__(self, model_name: str = sparse_model_name):
        self.model_name = model_name
        self._model = SparseTextEmbedding(model_name=model_name)

    @staticmethod
    def _segment(texts: list[str]) -> list[str]:
        return [" ".join(jieba.cut_for_search(text)) for text in texts]

    def embed(self, texts: list[str]):
        return list(self._model.embed(self._segment(texts)))


class BgeRerankFunction:
    """Cross-encoder reranker over (query, candidate) pairs, scored locally."""

    def __init__(self, model_name: str = reranker_model_name):
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [float(score) for score in self._model.rerank(query, documents)]


# CrewAI's EmbeddingConfigurator accepts an EmbeddingFunction instance directly
# when it is placed under the "provider" key.
embedder_config = {"provider": FastEmbedFunction(embedding_model_name)}
sparse_embedder = JiebaBM25Function()
reranker = BgeRerankFunction()

# Qdrant configuration
qdrant_location = os.environ.get("QDRANT_LOCATION", "http://localhost:6333")
qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
qdrant_collection_name = "obsidian-notes"

# Obsidian configuration
obsidian_vault_path = os.environ.get("OBSIDIAN_VAULT_PATH")

# AgentOps configuration
agentops_api_key = os.environ.get("AGENTOPS_API_KEY") or None

# ---------------------------------------------------------------------------
# Workaround patches for the OpenAI-compatible gateway models.
#
# crewai 1.15's litellm path has two gaps for us:
#   1. InternalInstructor (used for Task.output_pydantic) does not forward
#      the LLM instance's timeout, so a hung gateway call blocks forever.
#   2. Thinking-mode models (e.g. deepseek-v4) reject tool_choice; crewai
#      still sends structured-output requests through instructor's default
#      TOOLS mode, which the gateway refuses.
#
# Patches applied below:
#   1. supports_function_calling -> False for thinking-mode models, forcing
#      the text (non-native) tool calling path for agents with tools.
#   2. InternalInstructor always gets a timeout-injected litellm client;
#      thinking-mode models additionally get instructor MD_JSON mode, which
#      extracts JSON from the reply text instead of sending tool_choice.
#
# The thinking-mode model list is configurable: comma-separated substrings
# matched against the model name. Models not listed (e.g. glm-5.3) keep
# native function calling untouched.
# ---------------------------------------------------------------------------
import functools

import instructor as _instructor
from crewai.llm import LLM as _CrewAILLM
from crewai.utilities.internal_instructor import (
    InternalInstructor as _InternalInstructor,
)
from litellm import completion as _litellm_completion

LLM_TIMEOUT_SEC = int(os.environ.get("LLM_TIMEOUT_SEC", "180"))

NO_TOOL_CHOICE_MODELS = os.environ.get(
    "NO_TOOL_CHOICE_MODELS", "deepseek-v4,qwen3.8"
)


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


_CrewAILLM.supports_function_calling = _supports_function_calling

_original_ii_init = _InternalInstructor.__init__


def _ii_init(self, content, model, agent=None, llm=None):
    _original_ii_init(self, content, model, agent=agent, llm=llm)
    if not getattr(self.llm, "is_litellm", False):
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
