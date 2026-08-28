import uuid
from typing import Optional, List, Any, Dict

from crewai.memory.storage.rag_storage import RAGStorage
from qdrant_client import QdrantClient, models


class QdrantStorage(RAGStorage):
    """
    Extends Storage to handle embeddings for memory entries using Qdrant.
    """

    TEST_STRING = "test"
    MAX_LENGTH_BYTES = 8192

    app: QdrantClient | None = None

    def __init__(
        self,
        type: str,
        allow_reset: bool = True,
        embedder_config: Optional[Any] = None,
        crew: Any = None,
        qdrant_location: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ):
        self._qdrant_location = qdrant_location
        self._qdrant_api_key = qdrant_api_key
        super().__init__(type, allow_reset, embedder_config, crew)

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0.5,
    ) -> list[dict]:
        # Limit the text length to avoid the document being too large for the model
        query = self._normalize_text(query)

        # Embed the text and search for similar points
        embedding = self.embedder_config([query])[0]
        response = self.app.query_points(
            self.type,
            query=embedding,
            query_filter=self._to_qdrant_filter(filter),
            limit=limit,
            score_threshold=score_threshold,
        )
        results = [
            {
                "id": point.id,
                "metadata": point.payload.get("metadata"),
                "context": point.payload.get("value"),
                "score": point.score,
            }
            for point in response.points
        ]

        return results

    def reset(self) -> None:
        self.app.delete_collection(self.type)

    def save(self, value: str, metadata: Dict[str, Any]) -> None:
        # Limit the document length to avoid it being too large for the model
        value = self._normalize_text(value)

        # Embed the text and search for similar points
        embedding = self.embedder_config([value])[0]
        self.app.upsert(
            self.type,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector=embedding,
                    payload={"value": value, "metadata": metadata},
                )
            ],
        )

    def delete(self, filter: Optional[dict] = None) -> None:
        self.app.delete(
            self.type,
            points_selector=self._to_qdrant_filter(filter),
        )

    def count(self, filter: Optional[dict] = None) -> int:
        return self.app.count(
            self.type,
            count_filter=self._to_qdrant_filter(filter),
        ).count

    def list_src_paths(self) -> set[str]:
        """
        Return the set of distinct ``src_path`` values currently stored in the
        collection, by scrolling through all points.
        """
        paths: set[str] = set()
        offset = None
        while True:
            points, offset = self.app.scroll(
                collection_name=self.type,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                metadata = point.payload.get("metadata") or {}
                src_path = metadata.get("src_path")
                if src_path:
                    paths.add(src_path)
            if offset is None:
                break
        return paths

    def _initialize_app(self):
        # Initialize the embedder from given config
        self._set_embedder_config()

        # Initialize the Qdrant client and create the collection if it doesn't exist
        client = QdrantClient(self._qdrant_location, api_key=self._qdrant_api_key)
        if not client.collection_exists(self.type):
            # Create an embedding for a dummy value to get the embedding dimensionality
            embedding = self.embedder_config([self.TEST_STRING])[0]

            # Create Qdrant collection with the embedding dimensionality
            client.create_collection(
                collection_name=self.type,
                vectors_config=models.VectorParams(
                    size=len(embedding),
                    distance=models.Distance.COSINE,
                ),
            )

            # Create payload indexes for the fields used in filters
            for field in ("metadata.src_path", "metadata.content_hash"):
                client.create_payload_index(
                    collection_name=self.type,
                    field_name=field,
                    field_schema=models.KeywordIndexParams(
                        type=models.KeywordIndexType.KEYWORD
                    ),
                )
        self.app = client

    def _normalize_text(self, text: str) -> str:
        """
        Normalize the text to be within the maximum length.
        :param text:
        :return:
        """
        encoded = text.encode("utf-8")
        if len(encoded) <= self.MAX_LENGTH_BYTES:
            return text

        # Truncate to the byte limit. Decoding with errors="ignore" drops any
        # trailing partial multi-byte character rather than raising.
        truncated = encoded[: self.MAX_LENGTH_BYTES]
        return truncated.decode("utf-8", errors="ignore")

    def _to_qdrant_filter(self, filter: Optional[dict]) -> Optional[models.Filter]:
        """
        Convert dictionary filter to Qdrant filter. For now only supports exact match.
        :param filter:
        :return:
        """
        if filter is None:
            return None

        must = []
        for key, value in filter.items():
            must.append(
                models.FieldCondition(
                    key=f"metadata.{key}",
                    match=models.MatchValue(value=value),
                )
            )
        return models.Filter(must=must)
