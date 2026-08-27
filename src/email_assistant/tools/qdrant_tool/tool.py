import logging
from typing import Type, Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from email_assistant.storage import QdrantStorage

logger = logging.getLogger(__name__)


class SearchInput(BaseModel):
    query: str = Field(description="The query to search in the knowledge base.")
    limit: str = Field(default=10, description="The number of results to return.")


class QdrantSearchTool(BaseTool):
    """
    A tool that can be used to search in the knowledge base using Qdrant.
    """

    name: str = "QdrantSearchTool"
    description: str = (
        "A tool that can be used to search in the knowledge base using Qdrant. "
        "The knowledge base acts as a ground truth for the relevant information."
    )
    args_schema: Type[BaseModel] = SearchInput

    def __init__(self, qdrant_storage: QdrantStorage, /, **data: Any):
        super().__init__(**data)
        self._qdrant_storage = qdrant_storage

    def _run(self, query: str, limit: int = 10) -> list[dict]:
        # This method signature reflects the input schema defined in args_schema.
        # We could also use *args, **kwargs, but this is more explicit.
        logger.info("Received a query to search in the knowledge base: %s", query)
        results = self._qdrant_storage.search(query, limit)
        return results
