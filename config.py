import os

from chromadb import EmbeddingFunction
from dotenv import load_dotenv
from fastembed import TextEmbedding

# Load dotenv file
load_dotenv(".env")

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


class FastEmbedFunction(EmbeddingFunction):
    """A chromadb-compatible EmbeddingFunction backed by a local fastembed model."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def name(self) -> str:
        return f"fastembed/{self.model_name}"

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return [vector.tolist() for vector in self._model.embed(input)]


# CrewAI's EmbeddingConfigurator accepts an EmbeddingFunction instance directly
# when it is placed under the "provider" key.
embedder_config = {"provider": FastEmbedFunction(embedding_model_name)}

# Qdrant configuration
qdrant_location = os.environ.get("QDRANT_LOCATION", "http://localhost:6333")
qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
qdrant_collection_name = "obsidian-notes"

# Obsidian configuration
obsidian_vault_path = os.environ.get("OBSIDIAN_VAULT_PATH")

# AgentOps configuration
agentops_api_key = os.environ.get("AGENTOPS_API_KEY") or None

# ---------------------------------------------------------------------------
# Workaround: the gateway's deepseek-v4-* models are "thinking mode" models that
# reject function calling / tool_choice / response_format ("Thinking mode does
# not support this tool_choice"). CrewAI uses instructor (function calling) for
# structured output when the LLM reports function-calling support, so we force it
# down the plain-prompt JSON path for these models.
# ---------------------------------------------------------------------------
from crewai.llm import LLM as _CrewAILLM  # noqa: E402

_original_supports_function_calling = _CrewAILLM.supports_function_calling


def _supports_function_calling(self) -> bool:
    model = getattr(self, "model", "") or ""
    if "deepseek-v4" in model:
        return False
    return _original_supports_function_calling(self)


_CrewAILLM.supports_function_calling = _supports_function_calling
