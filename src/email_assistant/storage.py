import os
import uuid
from typing import Optional, List, Any, Dict

from qdrant_client import QdrantClient, models

# Minimum similarity score for a search result to be returned. Kept low
# because the responder should see near-misses too and decide itself; the
# faithfulness check downstream filters unsupported claims. Configurable via
# environment for easy tuning.
DEFAULT_SCORE_THRESHOLD = float(os.environ.get("QDRANT_SCORE_THRESHOLD", "0.3"))


class QdrantStorage:
    """
    Storage for knowledge base entries in Qdrant, embedded with the project's
    local fastembed model. Standalone class: crewai>=1.15 no longer ships
    crewai.memory.storage.rag_storage.
    """

    TEST_STRING = "test"
    MAX_LENGTH_BYTES = 8192

    def __init__(
        self,
        type: str,
        allow_reset: bool = True,
        embedder_config: Optional[Any] = None,
        crew: Any = None,
        qdrant_location: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ):
        self.type = type
        # The project passes config.embedder_config, a dict with the callable
        # EmbeddingFunction under the "provider" key (see config.py).
        self.embedder_config = embedder_config["provider"]
        self._qdrant_location = qdrant_location
        self._qdrant_api_key = qdrant_api_key
        self.app: QdrantClient | None = None
        self._initialize_app()

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: Optional[float] = None,
    ) -> list[dict]:
        # Limit the text length to avoid the document being too large for the model
        query = self._normalize_text(query)

        if score_threshold is None:
            score_threshold = DEFAULT_SCORE_THRESHOLD

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
        """Save a single entry (see save_batch for the actual logic)."""
        self.save_batch([(value, metadata)])
    def save_batch(self, entries: List[tuple[str, Dict[str, Any]]]) -> None:
        """
        Save multiple entries in one go: a single embedding call for all the
        values and a single Qdrant upsert. Much faster than per-chunk round
        trips, especially against a remote (Cloud) Qdrant.
        :param entries: list of (value, metadata) tuples
        """
        if not entries:
            return

        # Limit the document length to avoid it being too large for the model
        values = [self._normalize_text(value) for value, _ in entries]
        metadatas = [metadata for _, metadata in entries]

        # Embed all the texts at once
        embeddings = self.embedder_config(values)

        # Upsert all the points in a single request
        self.app.upsert(
            self.type,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector=embedding,
                    payload={"value": value, "metadata": metadata},
                )
                for embedding, value, metadata in zip(embeddings, values, metadatas)
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

    def get_metadata_value(self, filter: dict, key: str) -> Optional[Any]:
        """
        Return the value of a metadata key from the first point matching the
        filter, or None if no point matches. Used to read per-file bookkeeping
        fields (e.g. total_chunks) without scrolling the whole collection.
        """
        points, _ = self.app.scroll(
            collection_name=self.type,
            scroll_filter=self._to_qdrant_filter(filter),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None
        metadata = points[0].payload.get("metadata") or {}
        return metadata.get(key)

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
