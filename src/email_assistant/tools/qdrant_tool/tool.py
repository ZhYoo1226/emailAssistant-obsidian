import logging
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from email_assistant.storage import QdrantStorage

logger = logging.getLogger(__name__)


class SearchInput(BaseModel):
    query: str = Field(description="The query to search in the knowledge base.")
    folder: str | None = Field(
        default=None,
        description=(
            "Optional top-level vault folder name to restrict the search to "
            "(e.g. '实习经历'). Pass None to search every folder."
        ),
    )


class QdrantHybridSearchTool(BaseTool):
    """
    A tool that can be used to search in the knowledge base using Qdrant.
    """

    name: str = "QdrantHybridSearchTool"
    description: str = (
        "A tool that can be used to search in the knowledge base using Qdrant "
        "hybrid retrieval (dense semantic + sparse BM25 lexical, fused by RRF "
        "and reranked by a cross-encoder). The knowledge base acts as a ground "
        "truth for the relevant information. Optionally restrict the search to "
        "one top-level vault folder via the folder argument."
    )
    args_schema: type[BaseModel] = SearchInput

    def __init__(self, qdrant_storage: QdrantStorage, /, **data: Any):
        super().__init__(**data)
        self._qdrant_storage = qdrant_storage

    def _run(self, query: str, folder: str | None = None) -> list[dict]:
        logger.info(
            "Received a query to search in the knowledge base: %s (folder=%s)",
            query,
            folder,
        )
        results = self._qdrant_storage.search(
            query, limit=5, filter=({"folder": folder} if folder else None)
        )
        if not results and folder:
            # 文件夹过滤器可能不对（模型编造的名字），也可能该文件夹
            # 确实没有答案；回退到全部文件夹，而不是返回空结果。
            logger.warning(
                "No results in folder '%s'; retrying without the folder filter.",
                folder,
            )
            results = self._qdrant_storage.search(query, limit=5)
        for index, result in enumerate(results, start=1):
            result["index"] = index
        return results
